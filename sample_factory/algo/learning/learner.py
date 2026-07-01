from __future__ import annotations

import glob
import os
import time
from abc import ABC, abstractmethod
from os.path import join
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from sample_factory.algo.learning.rnn_utils import build_core_out_from_seq, build_rnn_inputs
from sample_factory.algo.utils.action_distributions import get_action_distribution, is_continuous_action_space
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, POLICY_ID_KEY, STATS_KEY, TRAIN_STATS, memory_stats
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.optimizers import Lamb
from sample_factory.algo.utils.rl_utils import gae_advantages, prepare_and_normalize_obs
from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.algo.utils.tensor_dict import TensorDict, shallow_recursive_copy
from sample_factory.algo.utils.torch_utils import masked_select, synchronize, to_scalar
from sample_factory.cfg.configurable import Configurable
from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.decay import LinearDecay
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import ActionDistribution, Config, InitModelData, PolicyID
from sample_factory.utils.utils import ensure_dir_exists, experiment_dir, log
from sample_factory.utils.state_proxy import StateProxy


class LearningRateScheduler:
    def update(self, current_lr, recent_kls):
        return current_lr

    def invoke_after_each_minibatch(self):
        return False

    def invoke_after_each_epoch(self):
        return False


class KlAdaptiveScheduler(LearningRateScheduler, ABC):
    def __init__(self, cfg: Config):
        self.lr_schedule_kl_threshold = cfg.lr_schedule_kl_threshold
        self.min_lr = cfg.lr_adaptive_min
        self.max_lr = cfg.lr_adaptive_max

    @abstractmethod
    def num_recent_kls_to_use(self) -> int:
        pass

    def update(self, current_lr, recent_kls):
        num_kls_to_use = self.num_recent_kls_to_use()
        kls = recent_kls[-num_kls_to_use:]
        mean_kl = np.mean(kls)
        lr = current_lr
        if mean_kl > 2.0 * self.lr_schedule_kl_threshold:
            lr = max(current_lr / 1.5, self.min_lr)
        if mean_kl < (0.5 * self.lr_schedule_kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr


class KlAdaptiveSchedulerPerMinibatch(KlAdaptiveScheduler):
    def num_recent_kls_to_use(self) -> int:
        return 1

    def invoke_after_each_minibatch(self):
        return True


class KlAdaptiveSchedulerPerEpoch(KlAdaptiveScheduler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_minibatches_per_epoch = cfg.num_batches_per_epoch

    def num_recent_kls_to_use(self) -> int:
        return self.num_minibatches_per_epoch

    def invoke_after_each_epoch(self):
        return True


class LinearDecayScheduler(LearningRateScheduler):
    def __init__(self, cfg):
        num_updates = cfg.train_for_env_steps // cfg.batch_size * cfg.num_epochs
        self.linear_decay = LinearDecay([(0, cfg.learning_rate), (num_updates, 0)])
        self.step = 0

    def invoke_after_each_minibatch(self):
        return True

    def update(self, current_lr, recent_kls):
        self.step += 1
        lr = self.linear_decay.at(self.step)
        return lr


def get_lr_scheduler(cfg) -> LearningRateScheduler:
    if cfg.lr_schedule == "constant":
        return LearningRateScheduler()
    elif cfg.lr_schedule == "kl_adaptive_minibatch":
        return KlAdaptiveSchedulerPerMinibatch(cfg)
    elif cfg.lr_schedule == "kl_adaptive_epoch":
        return KlAdaptiveSchedulerPerEpoch(cfg)
    elif cfg.lr_schedule == "linear_decay":
        return LinearDecayScheduler(cfg)
    else:
        raise RuntimeError(f"Unknown scheduler {cfg.lr_schedule}")


def model_initialization_data(
    cfg: Config, policy_id: PolicyID, actor_critic: Module, policy_version: int, device: torch.device
) -> InitModelData:
    # in serial mode we will just use the same actor_critic directly
    state_dict = None if cfg.serial_mode else actor_critic.state_dict()
    model_state = (policy_id, state_dict, device, policy_version)
    return model_state


class Learner(StateProxy, Configurable):
    _STATE_ATTRS = frozenset(('actor_critic', 'best_performance', 'curr_lr', 'device', 'env_info', 'env_steps', 'exploration_loss_func', 'is_initialized', 'kl_loss_func', 'last_milestone_time', 'last_summary_time', 'lr_scheduler', 'new_cfg', 'optimizer', 'param_server', 'policy_id', 'policy_to_load', 'policy_versions_tensor', 'summary_rate_decay_seconds', 'timing', 'train_step'))

    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        self._init_state_proxy()
        Configurable.__init__(self, cfg)

        self._state.timing = Timing(name=f"Learner {policy_id} profile")

        self._state.policy_id = policy_id

        self._state.env_info = env_info

        self._state.device = None
        self._state.actor_critic: Optional[ActorCritic] = None

        self._state.optimizer = None

        self._state.curr_lr: Optional[float] = None
        self._state.lr_scheduler: Optional[LearningRateScheduler] = None

        self._state.train_step: int = 0  # total number of SGD steps
        self._state.env_steps: int = 0  # total number of environment steps consumed by the learner

        self._state.best_performance = -1e9

        # for configuration updates, i.e. from PBT
        self._state.new_cfg: Optional[Dict] = None

        # for multi-policy learning (i.e. with PBT) when we need to load weights of another policy
        self._state.policy_to_load: Optional[PolicyID] = None

        # decay rate at which summaries are collected
        # save summaries every 5 seconds in the beginning, but decay to every 4 minutes in the limit, because we
        # do not need frequent summaries for longer experiments
        self._state.summary_rate_decay_seconds = LinearDecay([(0, 2), (100000, 60), (1000000, 120)])
        self._state.last_summary_time = 0
        self._state.last_milestone_time = 0

        # shared tensor used to share the latest policy version between processes
        self._state.policy_versions_tensor: Tensor = policy_versions_tensor

        self._state.param_server: ParameterServer = param_server

        self._state.exploration_loss_func: Optional[Callable] = None
        self._state.kl_loss_func: Optional[Callable] = None

        self._state.is_initialized = False

    def init(self) -> InitModelData:
        if self.cfg.exploration_loss_coeff == 0.0:
            self._state.exploration_loss_func = lambda action_distr, valids, num_invalids: 0.0
        elif self.cfg.exploration_loss == "entropy":
            self._state.exploration_loss_func = self._entropy_exploration_loss
        elif self.cfg.exploration_loss == "symmetric_kl":
            self._state.exploration_loss_func = self._symmetric_kl_exploration_loss
        else:
            raise NotImplementedError(f"{self.cfg.exploration_loss} not supported!")

        if self.cfg.kl_loss_coeff == 0.0:
            if is_continuous_action_space(self._state.env_info.action_space):
                log.warning(
                    "WARNING! It is generally recommended to enable Fixed KL loss (https://arxiv.org/pdf/1707.06347.pdf) for continuous action tasks to avoid potential numerical issues. "
                    "I.e. set --kl_loss_coeff=0.1"
                )
            self._state.kl_loss_func = lambda action_space, action_logits, distribution, valids, num_invalids: (None, 0.0)
        else:
            self._state.kl_loss_func = self._kl_loss

        # initialize the Torch modules
        if self.cfg.seed is None:
            log.info("Starting seed is not provided")
        else:
            log.info("Setting fixed seed %d", self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        # initialize device
        self._state.device = policy_device(self.cfg, self._state.policy_id)

        log.debug("Initializing actor-critic model on device %s", self._state.device)

        # trainable torch module
        self._state.actor_critic = create_actor_critic(self.cfg, self._state.env_info.obs_space, self._state.env_info.action_space)
        log.debug("Created Actor Critic model with architecture:")
        log.debug(self._state.actor_critic)
        self._state.actor_critic.model_to_device(self._state.device)

        def share_mem(t):
            if t is not None and not t.is_cuda:
                return t.share_memory_()
            return t

        # noinspection PyProtectedMember
        self._state.actor_critic._apply(share_mem)
        self._state.actor_critic.train()

        params = list(self._state.actor_critic.parameters())

        optimizer_cls = dict(adam=torch.optim.Adam, lamb=Lamb)
        if self.cfg.optimizer not in optimizer_cls:
            raise RuntimeError(f"Unknown optimizer {self.cfg.optimizer}")

        optimizer_cls = optimizer_cls[self.cfg.optimizer]
        log.debug(f"Using optimizer {optimizer_cls}")

        optimizer_kwargs = dict(
            lr=self.cfg.learning_rate,  # use default lr only in ctor, then we use the one loaded from the checkpoint
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
        )

        if self.cfg.optimizer in ["adam", "lamb"]:
            optimizer_kwargs["eps"] = self.cfg.adam_eps

        self._state.optimizer = optimizer_cls(params, **optimizer_kwargs)

        self.load_from_checkpoint(self._state.policy_id)
        self._state.param_server.init(self._state.actor_critic, self._state.train_step, self._state.device)
        self._state.policy_versions_tensor[self._state.policy_id] = self._state.train_step

        self._state.lr_scheduler = get_lr_scheduler(self.cfg)
        self._state.curr_lr = self.cfg.learning_rate if self._state.curr_lr is None else self._state.curr_lr
        self._apply_lr(self._state.curr_lr)

        self._state.is_initialized = True

        return model_initialization_data(self.cfg, self._state.policy_id, self._state.actor_critic, self._state.train_step, self._state.device)

    @staticmethod
    def checkpoint_dir(cfg, policy_id):
        checkpoint_dir = join(experiment_dir(cfg=cfg), f"checkpoint_p{policy_id}")
        return ensure_dir_exists(checkpoint_dir)

    @staticmethod
    def get_checkpoints(checkpoints_dir, pattern="checkpoint_*"):
        checkpoints = glob.glob(join(checkpoints_dir, pattern))
        return sorted(checkpoints)

    @staticmethod
    def load_checkpoint(checkpoints, device):
        if len(checkpoints) <= 0:
            log.warning("No checkpoints found")
            return None
        else:
            latest_checkpoint = checkpoints[-1]

            # this flag was introduced with torch 2.0, and since torch 2.6 its default value is True, which breaks loads
            extra_load_kwargs = {"weights_only": False} if torch.__version__.split(".")[0] >= "2" else {}

            # extra safety mechanism to recover from spurious filesystem errors
            num_attempts = 3
            for attempt in range(num_attempts):
                # noinspection PyBroadException
                try:
                    log.warning("Loading state from checkpoint %s...", latest_checkpoint)
                    checkpoint_dict = torch.load(latest_checkpoint, map_location=device, **extra_load_kwargs)
                    return checkpoint_dict
                except Exception:
                    log.exception(f"Could not load from checkpoint, attempt {attempt}")

    def _load_state(self, checkpoint_dict, load_progress=True):
        if load_progress:
            self._state.train_step = checkpoint_dict["train_step"]
            self._state.env_steps = checkpoint_dict["env_steps"]
            self._state.best_performance = checkpoint_dict.get("best_performance", self._state.best_performance)
        self._state.actor_critic.load_state_dict(checkpoint_dict["model"])
        self._state.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self._state.curr_lr = checkpoint_dict.get("curr_lr", self.cfg.learning_rate)

        log.info(f"Loaded experiment state at {self._state.train_step=}, {self._state.env_steps=}")

    def load_from_checkpoint(self, policy_id: PolicyID, load_progress: bool = True) -> None:
        name_prefix = dict(latest="checkpoint", best="best")[self.cfg.load_checkpoint_kind]
        checkpoints = self.get_checkpoints(self.checkpoint_dir(self.cfg, policy_id), pattern=f"{name_prefix}_*")
        checkpoint_dict = self.load_checkpoint(checkpoints, self._state.device)
        if checkpoint_dict is None:
            log.debug("Did not load from checkpoint, starting from scratch!")
        else:
            log.debug("Loading model from checkpoint")

            # if we're replacing our policy with another policy (under PBT), let's not reload the env_steps
            self._load_state(checkpoint_dict, load_progress=load_progress)

    def _should_save_summaries(self):
        summaries_every_seconds = self._state.summary_rate_decay_seconds.at(self._state.train_step)
        if time.time() - self._state.last_summary_time < summaries_every_seconds:
            return False

        return True

    def _after_optimizer_step(self):
        """A hook to be called after each optimizer step."""
        self._state.train_step += 1

    def _get_checkpoint_dict(self):
        checkpoint = {
            "train_step": self._state.train_step,
            "env_steps": self._state.env_steps,
            "best_performance": self._state.best_performance,
            "model": self._state.actor_critic.state_dict(),
            "optimizer": self._state.optimizer.state_dict(),
            "curr_lr": self._state.curr_lr,
        }
        return checkpoint

    def _save_impl(self, name_prefix, name_suffix, keep_checkpoints, verbose=True) -> bool:
        if not self._state.is_initialized:
            return False

        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None

        checkpoint_dir = self.checkpoint_dir(self.cfg, self._state.policy_id)
        tmp_filepath = join(checkpoint_dir, f"{name_prefix}_temp")
        checkpoint_name = f"{name_prefix}_{self._state.train_step:09d}_{self._state.env_steps}{name_suffix}.pth"
        filepath = join(checkpoint_dir, checkpoint_name)
        if verbose:
            log.info("Saving %s...", filepath)

        # This should protect us from a rare case where something goes wrong mid-save and we end up with a corrupted
        # checkpoint file. It better be a corrupted temp file.
        torch.save(checkpoint, tmp_filepath)
        os.rename(tmp_filepath, filepath)

        while len(checkpoints := self.get_checkpoints(checkpoint_dir, f"{name_prefix}_*")) > keep_checkpoints:
            oldest_checkpoint = checkpoints[0]
            if os.path.isfile(oldest_checkpoint):
                if verbose:
                    log.debug("Removing %s", oldest_checkpoint)
                os.remove(oldest_checkpoint)

        return True

    def save(self) -> bool:
        return self._save_impl("checkpoint", "", self.cfg.keep_checkpoints)

    def save_milestone(self):
        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None
        checkpoint_dir = self.checkpoint_dir(self.cfg, self._state.policy_id)
        checkpoint_name = f"checkpoint_{self._state.train_step:09d}_{self._state.env_steps}.pth"

        milestones_dir = ensure_dir_exists(join(checkpoint_dir, "milestones"))
        milestone_path = join(milestones_dir, f"{checkpoint_name}")
        log.info("Saving a milestone %s", milestone_path)
        torch.save(checkpoint, milestone_path)

    def save_best(self, policy_id, metric, metric_value) -> bool:
        if policy_id != self._state.policy_id:
            return False
        p = 3  # precision, number of significant digits
        if metric_value - self._state.best_performance > 1 / 10**p:
            log.info(f"Saving new best policy, {metric}={metric_value:.{p}f}!")
            self._state.best_performance = metric_value
            name_suffix = f"_{metric}_{metric_value:.{p}f}"
            return self._save_impl("best", name_suffix, 1, verbose=False)

        return False

    def set_new_cfg(self, new_cfg: Dict) -> None:
        self._state.new_cfg = new_cfg

    def set_policy_to_load(self, policy_to_load: PolicyID) -> None:
        self._state.policy_to_load = policy_to_load

    def _maybe_update_cfg(self) -> None:
        if self._state.new_cfg is not None:
            for key, value in self._state.new_cfg.items():
                if self.cfg[key] != value:
                    log.debug("Learner %d replacing cfg parameter %r with new value %r", self._state.policy_id, key, value)
                    self.cfg[key] = value

            if self.cfg.lr_schedule == "constant" and self._state.curr_lr != self.cfg.learning_rate:
                # PBT-optimized learning rate, only makes sense if we use constant LR
                # in case of more advanced LR scheduling we should update the parameters of the scheduler, not the
                # learning rate directly
                log.debug(f"Updating learning rate from {self._state.curr_lr} to {self.cfg.learning_rate}")
                self._state.curr_lr = self.cfg.learning_rate
                self._apply_lr(self._state.curr_lr)

            for param_group in self._state.optimizer.param_groups:
                param_group["betas"] = (self.cfg.adam_beta1, self.cfg.adam_beta2)
                log.debug("Optimizer lr value %.7f, betas: %r", param_group["lr"], param_group["betas"])

            self._state.new_cfg = None

    def _maybe_load_policy(self) -> None:
        if self._state.policy_to_load is not None:
            with self._state.param_server.policy_lock:
                # don't re-load progress if we are loading from another policy checkpoint
                self.load_from_checkpoint(self._state.policy_to_load, load_progress=False)

            # make sure everything (such as policy weights) is committed to shared device memory
            synchronize(self.cfg, self._state.device)
            # this will force policy update on the inference worker (policy worker)
            # we add max_policy_lag steps so that all experience currently in batches is invalidated
            self._state.train_step += self.cfg.max_policy_lag + 1
            self._state.policy_versions_tensor[self._state.policy_id] = self._state.train_step

            self._state.policy_to_load = None

    @staticmethod
    def _policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids: int):
        clipped_ratio = torch.clamp(ratio, clip_ratio_low, clip_ratio_high)
        loss_unclipped = ratio * adv
        loss_clipped = clipped_ratio * adv
        loss = torch.min(loss_unclipped, loss_clipped)
        loss = masked_select(loss, valids, num_invalids)
        loss = -loss.mean()

        return loss

    def _value_loss(
        self,
        new_values: Tensor,
        old_values: Tensor,
        target: Tensor,
        clip_value: float,
        valids: Tensor,
        num_invalids: int,
    ) -> Tensor:
        value_clipped = old_values + torch.clamp(new_values - old_values, -clip_value, clip_value)
        value_original_loss = (new_values - target).pow(2)
        value_clipped_loss = (value_clipped - target).pow(2)
        value_loss = torch.max(value_original_loss, value_clipped_loss)
        value_loss = masked_select(value_loss, valids, num_invalids)
        value_loss = value_loss.mean()

        value_loss *= self.cfg.value_loss_coeff

        return value_loss

    def _kl_loss(
        self, action_space, action_logits, action_distribution, valids, num_invalids: int
    ) -> Tuple[Tensor, Tensor]:
        old_action_distribution = get_action_distribution(action_space, action_logits)
        kl_old = action_distribution.kl_divergence(old_action_distribution)
        kl_old = masked_select(kl_old, valids, num_invalids)
        kl_loss = kl_old.mean()

        kl_loss *= self.cfg.kl_loss_coeff

        return kl_old, kl_loss

    def _entropy_exploration_loss(self, action_distribution, valids, num_invalids: int) -> Tensor:
        entropy = action_distribution.entropy()
        entropy = masked_select(entropy, valids, num_invalids)
        entropy_loss = -self.cfg.exploration_loss_coeff * entropy.mean()
        return entropy_loss

    def _symmetric_kl_exploration_loss(self, action_distribution, valids, num_invalids: int) -> Tensor:
        kl_prior = action_distribution.symmetric_kl_with_uniform_prior()
        kl_prior = masked_select(kl_prior, valids, num_invalids).mean()
        if not torch.isfinite(kl_prior):
            kl_prior = torch.zeros(kl_prior.shape)
        kl_prior = torch.clamp(kl_prior, max=30)
        kl_prior_loss = self.cfg.exploration_loss_coeff * kl_prior
        return kl_prior_loss

    def _optimizer_lr(self):
        for param_group in self._state.optimizer.param_groups:
            return param_group["lr"]

    def _apply_lr(self, lr: float) -> None:
        """Change learning rate in the optimizer."""
        if lr != self._optimizer_lr():
            for param_group in self._state.optimizer.param_groups:
                param_group["lr"] = lr

    def _get_minibatches(self, batch_size, experience_size):
        """Generating minibatches for training."""
        assert self.cfg.rollout % self.cfg.recurrence == 0
        assert experience_size % batch_size == 0, f"experience size: {experience_size}, batch size: {batch_size}"
        minibatches_per_epoch = self.cfg.num_batches_per_epoch

        if minibatches_per_epoch == 1:
            return [None]  # single minibatch is actually the entire buffer, we don't need indices

        if self.cfg.shuffle_minibatches:
            # indices that will start the mini-trajectories from the same episode (for bptt)
            indices = np.arange(0, experience_size, self.cfg.recurrence)
            indices = np.random.permutation(indices)

            # complete indices of mini trajectories, e.g. with recurrence==4: [4, 16] -> [4, 5, 6, 7, 16, 17, 18, 19]
            indices = [np.arange(i, i + self.cfg.recurrence) for i in indices]
            indices = np.concatenate(indices)

            assert len(indices) == experience_size

            num_minibatches = experience_size // batch_size
            minibatches = np.split(indices, num_minibatches)
        else:
            minibatches = list(slice(i * batch_size, (i + 1) * batch_size) for i in range(0, minibatches_per_epoch))

            # this makes sense but I'd like to do some testing before enabling it
            # random.shuffle(minibatches)  # same minibatches between epochs, but in random order

        return minibatches

    @staticmethod
    def _get_minibatch(buffer, indices):
        if indices is None:
            # handle the case of a single batch, where the entire buffer is a minibatch
            return buffer

        mb = buffer[indices]
        return mb

    def _loss_clip_settings(self, mb: AttrDict) -> Tuple[int, float, float, float, Tensor]:
        with torch.no_grad(), self._state.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids

        return recurrence, clip_ratio_low, clip_ratio_high, clip_value, valids

    def _prepare_bptt_inputs(
        self, mb: AttrDict, head_outputs: Tensor, valids: Tensor, recurrence: int
    ) -> Tuple[Optional[Tensor], Tensor, Optional[Tensor]]:
        with self._state.timing.add_time("bptt_initial"):
            if self.cfg.use_rnn:
                # this is the only way to stop RNNs from backpropagating through invalid timesteps
                # (i.e. experience collected by another policy)
                done_or_invalid = torch.logical_or(mb.dones_cpu, ~valids.cpu()).float()
                head_output_seq, rnn_states, inverted_select_inds = build_rnn_inputs(
                    head_outputs,
                    done_or_invalid,
                    mb.rnn_states,
                    recurrence,
                )
                return head_output_seq, rnn_states, inverted_select_inds

            rnn_states = mb.rnn_states[::recurrence]
            return None, rnn_states, None

    def _forward_core_outputs(
        self,
        head_outputs: Tensor,
        head_output_seq: Optional[Tensor],
        rnn_states: Tensor,
        inverted_select_inds: Optional[Tensor],
    ) -> Tensor:
        with self._state.timing.add_time("bptt"):
            if self.cfg.use_rnn:
                assert head_output_seq is not None
                assert inverted_select_inds is not None

                with self._state.timing.add_time("bptt_forward_core"):
                    core_output_seq, _ = self._state.actor_critic.forward_core(head_output_seq, rnn_states)
                core_outputs = build_core_out_from_seq(core_output_seq, inverted_select_inds)
                del core_output_seq
            else:
                core_outputs, _ = self._state.actor_critic.forward_core(head_outputs, rnn_states)

        return core_outputs

    def _forward_tail_outputs(
        self, mb: AttrDict, core_outputs: Tensor
    ) -> Tuple[Dict, ActionDistribution, Tensor, Tensor]:
        with self._state.timing.add_time("tail"):
            result = self._state.actor_critic.forward_tail(core_outputs, values_only=False, sample_actions=False)
            action_distribution = self._state.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values = result["values"].squeeze()

        return result, action_distribution, ratio, values

    def _calculate_vtrace_advantages(
        self, mb: AttrDict, ratio: Tensor, values: Tensor, recurrence: int, num_trajectories: int
    ) -> Tuple[Tensor, Tensor]:
        rho_hat = torch.Tensor([self.cfg.vtrace_rho])
        c_hat = torch.Tensor([self.cfg.vtrace_c])

        ratios_cpu = ratio.cpu()
        values_cpu = values.cpu()
        rewards_cpu = mb.rewards_cpu
        dones_cpu = mb.dones_cpu

        vtrace_rho = torch.min(rho_hat, ratios_cpu)
        vtrace_c = torch.min(c_hat, ratios_cpu)

        vs = torch.zeros((num_trajectories * recurrence))
        adv = torch.zeros((num_trajectories * recurrence))

        next_values = values_cpu[recurrence - 1 :: recurrence] - rewards_cpu[recurrence - 1 :: recurrence]
        next_values /= self.cfg.gamma
        next_vs = next_values

        for i in reversed(range(self.cfg.recurrence)):
            rewards = rewards_cpu[i::recurrence]
            dones = dones_cpu[i::recurrence]
            not_done = 1.0 - dones
            not_done_gamma = not_done * self.cfg.gamma

            curr_values = values_cpu[i::recurrence]
            curr_vtrace_rho = vtrace_rho[i::recurrence]
            curr_vtrace_c = vtrace_c[i::recurrence]

            delta_s = curr_vtrace_rho * (rewards + not_done_gamma * next_values - curr_values)
            adv[i::recurrence] = curr_vtrace_rho * (rewards + not_done_gamma * next_vs - curr_values)
            next_vs = curr_values + delta_s + not_done_gamma * curr_vtrace_c * (next_vs - next_values)
            vs[i::recurrence] = next_vs

            next_values = curr_values

        targets = vs.to(self._state.device)
        adv = adv.to(self._state.device)
        return adv, targets

    def _calculate_advantages_and_returns(
        self,
        mb: AttrDict,
        ratio: Tensor,
        values: Tensor,
        valids: Tensor,
        num_invalids: int,
        recurrence: int,
        num_trajectories: int,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        with torch.no_grad(), self._state.timing.add_time("advantages_returns"):
            if self.cfg.with_vtrace:
                adv, targets = self._calculate_vtrace_advantages(mb, ratio, values, recurrence, num_trajectories)
            else:
                adv = mb.advantages
                targets = mb.returns

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)

        return adv, targets, adv_std, adv_mean

    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        recurrence, clip_ratio_low, clip_ratio_high, clip_value, valids = self._loss_clip_settings(mb)

        # calculate policy head outside of recurrent loop
        with self._state.timing.add_time("forward_head"):
            head_outputs = self._state.actor_critic.forward_head(mb.normalized_obs)
            minibatch_size: int = head_outputs.size(0)

        head_output_seq, rnn_states, inverted_select_inds = self._prepare_bptt_inputs(
            mb, head_outputs, valids, recurrence
        )
        core_outputs = self._forward_core_outputs(head_outputs, head_output_seq, rnn_states, inverted_select_inds)
        del head_outputs

        num_trajectories = minibatch_size // recurrence
        assert core_outputs.shape[0] == minibatch_size

        result, action_distribution, ratio, values = self._forward_tail_outputs(mb, core_outputs)
        del core_outputs

        adv, targets, adv_std, adv_mean = self._calculate_advantages_and_returns(
            mb, ratio, values, valids, num_invalids, recurrence, num_trajectories
        )

        with self._state.timing.add_time("losses"):
            # noinspection PyTypeChecker
            policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
            exploration_loss = self._state.exploration_loss_func(action_distribution, valids, num_invalids)
            kl_old, kl_loss = self._state.kl_loss_func(
                self._state.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
            )
            old_values = mb["values"]
            value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=result["values"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
        )

        return action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries

    def _train_loop_initial_state(self) -> AttrDict:
        with torch.no_grad():
            if self.cfg.with_vtrace:
                assert (
                    self.cfg.recurrence == self.cfg.rollout and self.cfg.recurrence > 1
                ), "V-trace requires to recurrence and rollout to be equal"

            with_summaries = self._should_save_summaries()
            if np.random.rand() < 0.5:
                summaries_epoch = np.random.randint(0, self.cfg.num_epochs)
                summaries_batch = np.random.randint(0, self.cfg.num_batches_per_epoch)
            else:
                summaries_epoch = self.cfg.num_epochs - 1
                summaries_batch = self.cfg.num_batches_per_epoch - 1

            assert self._state.actor_critic.training

        return AttrDict(
            early_stopping_tolerance=1e-6,
            early_stop=False,
            prev_epoch_actor_loss=1e9,
            epoch_actor_losses=[0] * self.cfg.num_batches_per_epoch,
            recent_kls=[],
            num_sgd_steps=0,
            stats_and_summaries=None,
            with_summaries=with_summaries,
            summaries_epoch=summaries_epoch,
            summaries_batch=summaries_batch,
            force_summaries=False,
        )

    def _get_train_minibatch(self, gpu_buffer: TensorDict, minibatches, batch_num: int) -> AttrDict:
        with torch.no_grad(), self._state.timing.add_time("minibatch_init"):
            indices = minibatches[batch_num]
            mb = self._get_minibatch(gpu_buffer, indices)
            return AttrDict(mb)

    def _postprocess_losses(
        self,
        policy_loss: Tensor,
        exploration_loss: Tensor,
        kl_loss: Tensor | float,
        value_loss: Tensor,
        batch_num: int,
        epoch_actor_losses,
    ) -> Tuple[Tensor, Tensor, Tensor, bool]:
        with self._state.timing.add_time("losses_postprocess"):
            actor_loss: Tensor = policy_loss + exploration_loss + kl_loss
            critic_loss = value_loss
            loss: Tensor = actor_loss + critic_loss

            epoch_actor_losses[batch_num] = float(actor_loss)

            force_summaries = False
            high_loss = 30.0
            if torch.abs(loss) > high_loss:
                log.warning(
                    "High loss value: l:%.4f pl:%.4f vl:%.4f exp_l:%.4f kl_l:%.4f "
                    "(recommended to adjust the --reward_scale parameter)",
                    to_scalar(loss),
                    to_scalar(policy_loss),
                    to_scalar(value_loss),
                    to_scalar(exploration_loss),
                    to_scalar(kl_loss),
                )
                force_summaries = True

        return actor_loss, critic_loss, loss, force_summaries

    def _update_recent_kls(
        self,
        mb: AttrDict,
        action_distribution: ActionDistribution,
        kl_old: Optional[Tensor],
        num_invalids: int,
        recent_kls,
    ) -> Tuple[Tensor, float]:
        with torch.no_grad(), self._state.timing.add_time("kl_divergence"):
            if kl_old is None:
                old_action_distribution = get_action_distribution(self._state.actor_critic.action_space, mb.action_logits)
                kl_old = action_distribution.kl_divergence(old_action_distribution)
                kl_old = masked_select(kl_old, mb.valids, num_invalids)

            kl_old_mean = float(kl_old.mean().item())
            recent_kls.append(kl_old_mean)
            if kl_old.numel() > 0 and kl_old.max().item() > 100:
                log.warning(f"KL-divergence is very high: {kl_old.max().item():.4f}")

        return kl_old, kl_old_mean

    def _optimizer_step(self, loss: Tensor, num_invalids: int, experience_size: int) -> Tuple[int, float]:
        with self._state.timing.add_time("update"):
            for p in self._state.actor_critic.parameters():
                p.grad = None

            loss.backward()

            if self.cfg.max_grad_norm > 0.0:
                with self._state.timing.add_time("clip"):
                    torch.nn.utils.clip_grad_norm_(self._state.actor_critic.parameters(), self.cfg.max_grad_norm)

            curr_policy_version = self._state.train_step
            actual_lr = self._state.curr_lr
            if num_invalids > 0:
                actual_lr = self._state.curr_lr * (experience_size - num_invalids) / experience_size
            self._apply_lr(actual_lr)

            with self._state.param_server.policy_lock:
                self._state.optimizer.step()

        return curr_policy_version, actual_lr

    def _after_train_minibatch(
        self,
        state: AttrDict,
        epoch: int,
        batch_num: int,
        minibatches,
        summary_vars: Dict,
    ) -> Optional[AttrDict]:
        with torch.no_grad(), self._state.timing.add_time("after_optimizer"):
            self._after_optimizer_step()

            if self._state.lr_scheduler.invoke_after_each_minibatch():
                self._state.curr_lr = self._state.lr_scheduler.update(self._state.curr_lr, state.recent_kls)

            stats_and_summaries = None
            should_record_summaries = state.with_summaries
            should_record_summaries &= epoch == state.summaries_epoch and batch_num == state.summaries_batch
            should_record_summaries |= state.force_summaries
            if should_record_summaries:
                state.force_summaries = False
                stats_and_summaries = self._record_summaries(AttrDict(summary_vars))

            synchronize(self.cfg, self._state.device)
            self._state.policy_versions_tensor[self._state.policy_id] = self._state.train_step
            return stats_and_summaries

    def _update_early_stop(self, state: AttrDict, epoch: int) -> None:
        new_epoch_actor_loss = float(np.mean(state.epoch_actor_losses))
        loss_delta_abs = abs(state.prev_epoch_actor_loss - new_epoch_actor_loss)
        if loss_delta_abs < state.early_stopping_tolerance:
            state.early_stop = True
            log.debug(
                "Early stopping after %d epochs (%d sgd steps), loss delta %.7f",
                epoch + 1,
                state.num_sgd_steps,
                loss_delta_abs,
            )
            return

        state.prev_epoch_actor_loss = new_epoch_actor_loss

    def _train(
        self, gpu_buffer: TensorDict, batch_size: int, experience_size: int, num_invalids: int
    ) -> Optional[AttrDict]:
        timing = self._state.timing
        state = self._train_loop_initial_state()

        for epoch in range(self.cfg.num_epochs):
            with timing.add_time("epoch_init"):
                if state.early_stop:
                    break

                minibatches = self._get_minibatches(batch_size, experience_size)

            for batch_num in range(len(minibatches)):
                mb = self._get_train_minibatch(gpu_buffer, minibatches, batch_num)

                with timing.add_time("calculate_losses"):
                    (
                        action_distribution,
                        policy_loss,
                        exploration_loss,
                        kl_old,
                        kl_loss,
                        value_loss,
                        loss_summaries,
                    ) = self._calculate_losses(mb, num_invalids)

                actor_loss, critic_loss, loss, force_summaries = self._postprocess_losses(
                    policy_loss, exploration_loss, kl_loss, value_loss, batch_num, state.epoch_actor_losses
                )
                state.force_summaries |= force_summaries

                kl_old, kl_old_mean = self._update_recent_kls(
                    mb, action_distribution, kl_old, num_invalids, state.recent_kls
                )
                curr_policy_version, actual_lr = self._optimizer_step(loss, num_invalids, experience_size)
                state.num_sgd_steps += 1

                summary_vars = {**locals(), **loss_summaries, **state}
                stats_and_summaries = self._after_train_minibatch(state, epoch, batch_num, minibatches, summary_vars)
                if stats_and_summaries is not None:
                    state.stats_and_summaries = stats_and_summaries

            # end of an epoch
            if self._state.lr_scheduler.invoke_after_each_epoch():
                self._state.curr_lr = self._state.lr_scheduler.update(self._state.curr_lr, state.recent_kls)

            self._update_early_stop(state, epoch)
            if state.early_stop:
                break

        return state.stats_and_summaries

    def _record_basic_summaries(self, stats: AttrDict, var: AttrDict) -> None:
        stats.env_steps = self._state.env_steps
        stats.lr = self._state.curr_lr
        stats.actual_lr = var.actual_lr

        stats.update(self._state.actor_critic.summaries())

        stats.valids_fraction = var.mb.valids.float().mean()
        stats.same_policy_fraction = (var.mb.policy_id == self._state.policy_id).float().mean()

        grad_norm = (
            sum(p.grad.data.norm(2).item() ** 2 for p in self._state.actor_critic.parameters() if p.grad is not None) ** 0.5
        )
        stats.grad_norm = grad_norm
        stats.loss = var.loss
        stats.value = var.values.mean()
        stats.entropy = var.action_distribution.entropy().mean()
        stats.policy_loss = var.policy_loss
        stats.kl_loss = var.kl_loss
        stats.value_loss = var.value_loss
        stats.exploration_loss = var.exploration_loss

        stats.act_min = var.mb.actions.min()
        stats.act_max = var.mb.actions.max()

    @staticmethod
    def _record_advantage_summaries(stats: AttrDict, var: AttrDict) -> None:
        if "adv_mean" in stats:
            stats.adv_min = var.mb.advantages.min()
            stats.adv_max = var.mb.advantages.max()
            stats.adv_std = var.adv_std
            stats.adv_mean = var.adv_mean

    def _record_distribution_summaries(self, stats: AttrDict, var: AttrDict) -> None:
        stats.max_abs_logprob = torch.abs(var.mb.action_logits).max()

        if hasattr(var.action_distribution, "summaries"):
            stats.update(var.action_distribution.summaries())

    def _record_last_batch_summaries(self, stats: AttrDict, var: AttrDict) -> None:
        if var.epoch != self.cfg.num_epochs - 1 or var.batch_num != len(var.minibatches) - 1:
            return

        valid_ratios = masked_select(var.ratio, var.mb.valids, var.num_invalids)
        ratio_mean = torch.abs(1.0 - valid_ratios).mean().detach()
        ratio_min = valid_ratios.min().detach()
        ratio_max = valid_ratios.max().detach()

        value_delta = torch.abs(var.values - var.mb.values)
        value_delta_avg, value_delta_max = value_delta.mean(), value_delta.max()

        stats.kl_divergence = var.kl_old_mean
        stats.kl_divergence_max = var.kl_old.max()
        stats.value_delta = value_delta_avg
        stats.value_delta_max = value_delta_max
        stats.fraction_clipped = (
            (valid_ratios < var.clip_ratio_low).float() + (valid_ratios > var.clip_ratio_high).float()
        ).mean()
        stats.ratio_mean = ratio_mean
        stats.ratio_min = ratio_min
        stats.ratio_max = ratio_max
        stats.num_sgd_steps = var.num_sgd_steps

    def _record_optimizer_summaries(self, stats: AttrDict) -> None:
        adam_max_second_moment = 0.0
        for tensor_state in self._state.optimizer.state.values():
            if "exp_avg_sq" in tensor_state:
                adam_max_second_moment = max(tensor_state["exp_avg_sq"].max().item(), adam_max_second_moment)
        stats.adam_max_second_moment = adam_max_second_moment

    def _record_policy_version_summaries(self, stats: AttrDict, var: AttrDict) -> None:
        version_diff = (var.curr_policy_version - var.mb.policy_version)[var.mb.policy_id == self._state.policy_id]
        stats.version_diff_avg = version_diff.mean()
        stats.version_diff_min = version_diff.min()
        stats.version_diff_max = version_diff.max()

    @staticmethod
    def _scalarize_summaries(stats: AttrDict) -> AttrDict:
        for key, value in stats.items():
            stats[key] = to_scalar(value)
        return stats

    def _record_summaries(self, train_loop_vars) -> AttrDict:
        var = train_loop_vars

        self._state.last_summary_time = time.time()
        stats = AttrDict()

        self._record_basic_summaries(stats, var)
        self._record_advantage_summaries(stats, var)
        self._record_distribution_summaries(stats, var)
        self._record_last_batch_summaries(stats, var)
        self._record_optimizer_summaries(stats)
        self._record_policy_version_summaries(stats, var)
        return self._scalarize_summaries(stats)

    def _prepare_and_normalize_obs(self, obs: TensorDict) -> TensorDict:
        og_shape = dict()

        # assuming obs is a flat dict, collapse time and envs dimensions into a single batch dimension
        for key, x in obs.items():
            og_shape[key] = x.shape
            obs[key] = x.view((x.shape[0] * x.shape[1],) + x.shape[2:])

        # hold the lock while we alter the state of the normalizer since they can be used in other processes too
        with self._state.param_server.policy_lock:
            normalized_obs = prepare_and_normalize_obs(self._state.actor_critic, obs)

        # restore original shape
        for key, x in normalized_obs.items():
            normalized_obs[key] = x.view(og_shape[key])

        return normalized_obs

    def _prepare_batch(self, batch: TensorDict) -> Tuple[TensorDict, int, int]:
        with torch.no_grad():
            # create a shallow copy so we can modify the dictionary
            # we still reference the same buffers though
            buff = shallow_recursive_copy(batch)

            # ignore experience from other agents (i.e. on episode boundary) and from inactive agents
            valids: Tensor = buff["policy_id"] == self._state.policy_id
            # ignore experience that was older than the threshold even before training started
            curr_policy_version: int = self._state.train_step
            buff["valids"][:, :-1] = valids & (curr_policy_version - buff["policy_version"] < self.cfg.max_policy_lag)
            # for last T+1 step, we want to use the validity of the previous step
            buff["valids"][:, -1] = buff["valids"][:, -2]

            # ensure we're in train mode so that normalization statistics are updated
            if not self._state.actor_critic.training:
                self._state.actor_critic.train()

            buff["normalized_obs"] = self._prepare_and_normalize_obs(buff["obs"])
            del buff["obs"]  # don't need non-normalized obs anymore

            # calculate estimated value for the next step (T+1)
            normalized_last_obs = buff["normalized_obs"][:, -1]
            next_values = self._state.actor_critic(normalized_last_obs, buff["rnn_states"][:, -1], values_only=True)["values"]
            buff["values"][:, -1] = next_values

            if self.cfg.normalize_returns:
                # Since our value targets are normalized, the values will also have normalized statistics.
                # We need to denormalize them before using them for GAE caculation and value bootstrapping.
                # rl_games PPO uses a similar approach, see:
                # https://github.com/Denys88/rl_games/blob/7b5f9500ee65ae0832a7d8613b019c333ecd932c/rl_games/algos_torch/models.py#L51
                denormalized_values = buff["values"].clone()  # need to clone since normalizer is in-place
                self._state.actor_critic.returns_normalizer(denormalized_values, denormalize=True)
            else:
                # values are not normalized in this case, so we can use them as is
                denormalized_values = buff["values"]

            if self.cfg.value_bootstrap:
                # Value bootstrapping is a technique that reduces the surprise for the critic in case
                # we're ending the episode by timeout. Intuitively, in this case the cumulative return for the last step
                # should not be zero, but rather what the critic expects. This improves learning in many envs
                # because otherwise the critic cannot predict the abrupt change in rewards in a timed-out episode.
                # What we really want here is v(t+1) which we don't have because we don't have obs(t+1) (since
                # the episode ended). Using v(t) is an approximation that requires that rew(t) can be generally ignored.

                # Multiply by both time_out and done flags to make sure we count only timeouts in terminal states.
                # There was a bug in older versions of isaacgym where timeouts were reported for non-terminal states.
                buff["rewards"].add_(self.cfg.gamma * denormalized_values[:, :-1] * buff["time_outs"] * buff["dones"])

            if not self.cfg.with_vtrace:
                # calculate advantage estimate (in case of V-trace it is done separately for each minibatch)
                buff["advantages"] = gae_advantages(
                    buff["rewards"],
                    buff["dones"],
                    denormalized_values,
                    buff["valids"],
                    self.cfg.gamma,
                    self.cfg.gae_lambda,
                )
                # here returns are not normalized yet, so we should use denormalized values
                buff["returns"] = buff["advantages"] + buff["valids"][:, :-1] * denormalized_values[:, :-1]

            # remove next step obs, rnn_states, and values from the batch, we don't need them anymore
            for key in ["normalized_obs", "rnn_states", "values", "valids"]:
                buff[key] = buff[key][:, :-1]

            dataset_size = buff["actions"].shape[0] * buff["actions"].shape[1]
            for d, k, v in iterate_recursively(buff):
                # collapse first two dimensions (batch and time) into a single dimension
                d[k] = v.reshape((dataset_size,) + tuple(v.shape[2:]))

            buff["dones_cpu"] = buff["dones"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)
            buff["rewards_cpu"] = buff["rewards"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)

            # return normalization parameters are only used on the learner, no need to lock the mutex
            if self.cfg.normalize_returns:
                self._state.actor_critic.returns_normalizer(buff["returns"])  # in-place

            num_invalids = dataset_size - buff["valids"].sum().item()
            if num_invalids > 0:
                invalid_fraction = num_invalids / dataset_size
                if invalid_fraction > 0.5:
                    log.warning(f"{self._state.policy_id=} batch has {invalid_fraction:.2%} of invalid samples")

                # invalid action values can cause problems when we calculate logprobs
                # here we set them to 0 just to be safe
                invalid_indices = (buff["valids"] == 0).nonzero().squeeze()
                buff["actions"][invalid_indices] = 0
                # likewise, some invalid values of log_prob_actions can cause NaNs or infs
                buff["log_prob_actions"][invalid_indices] = -1  # -1 seems like a safe value

            return buff, dataset_size, num_invalids

    def train(self, batch: TensorDict) -> Optional[Dict]:
        with self._state.timing.add_time("misc"):
            self._maybe_update_cfg()
            self._maybe_load_policy()

        with self._state.timing.add_time("prepare_batch"):
            buff, experience_size, num_invalids = self._prepare_batch(batch)

        if num_invalids >= experience_size:
            if self.cfg.with_pbt:
                log.warning("No valid samples in the batch, with PBT this must mean we just replaced weights")
            else:
                log.error(f"Learner {self._state.policy_id=} received an entire batch of invalid data, skipping...")
            return None
        else:
            with self._state.timing.add_time("train"):
                train_stats = self._train(buff, self.cfg.batch_size, experience_size, num_invalids)

            # multiply the number of samples by frameskip so that FPS metrics reflect the number
            # of environment steps actually simulated
            if self.cfg.summaries_use_frameskip:
                self._state.env_steps += experience_size * self._state.env_info.frameskip
            else:
                self._state.env_steps += experience_size

            stats = {LEARNER_ENV_STEPS: self._state.env_steps, POLICY_ID_KEY: self._state.policy_id}
            if train_stats is not None:
                if train_stats is not None:
                    stats[TRAIN_STATS] = train_stats
                stats[STATS_KEY] = memory_stats("learner", self._state.device)

            return stats
