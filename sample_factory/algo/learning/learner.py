from __future__ import annotations

import copy
import glob
import os
import time
from abc import ABC, abstractmethod
from os.path import join
from threading import Thread
from typing import Optional, Tuple, Dict

import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.learning.batcher import Batcher
from sample_factory.algo.learning.rnn_utils import build_rnn_inputs, build_core_out_from_seq
from sample_factory.algo.utils.context import SampleFactoryContext, set_global_context
from sample_factory.algo.utils.misc import memory_stats
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.optimizers import Lamb
from sample_factory.algo.utils.rl_utils import gae_advantages_returns
from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.algo.utils.torch_utils import init_torch_runtime, to_scalar
from sample_factory.model.model import create_actor_critic
from sample_factory.algo.utils.action_distributions import get_action_distribution, is_continuous_action_space
from sample_factory.cfg.configurable import Configurable
from sample_factory.signal_slot.signal_slot import signal, EventLoopObject
from sample_factory.utils.decay import LinearDecay
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.utils.gpu_utils import cuda_envvars_for_policy
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import PolicyID
from sample_factory.utils.utils import log, experiment_dir, ensure_dir_exists, AttrDict


def init_learner_process(sf_context: SampleFactoryContext, learner: Learner):
    set_global_context(sf_context)
    log.info(f'{learner.object_id}\tpid {os.getpid()}\tparent {os.getppid()}')

    # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
    import signal as os_signal
    os_signal.signal(os_signal.SIGINT, os_signal.SIG_IGN)

    cfg = learner.cfg

    import psutil
    try:
        psutil.Process().nice(cfg.default_niceness)
    except psutil.AccessDenied:
        log.error('Low niceness requires sudo!')

    if cfg.device == 'gpu':
        cuda_envvars_for_policy(learner.policy_id, 'learning')

    init_torch_runtime(cfg)


class LearningRateScheduler:
    def update(self, current_lr, recent_kls):
        return current_lr

    def invoke_after_each_minibatch(self):
        return False

    def invoke_after_each_epoch(self):
        return False


class KlAdaptiveScheduler(LearningRateScheduler, ABC):
    def __init__(self, cfg):
        self.lr_schedule_kl_threshold = cfg.lr_schedule_kl_threshold
        self.min_lr = 1e-6
        self.max_lr = 1e-2

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


def get_lr_scheduler(cfg) -> LearningRateScheduler:
    if cfg.lr_schedule == 'constant':
        return LearningRateScheduler()
    elif cfg.lr_schedule == 'kl_adaptive_minibatch':
        return KlAdaptiveSchedulerPerMinibatch(cfg)
    elif cfg.lr_schedule == 'kl_adaptive_epoch':
        return KlAdaptiveSchedulerPerEpoch(cfg)
    else:
        raise RuntimeError(f'Unknown scheduler {cfg.lr_schedule}')


class Learner(EventLoopObject, Configurable):
    def __init__(self, evt_loop, cfg, env_info, buffer_mgr, batcher: Batcher, policy_id: PolicyID, mp_ctx):
        Configurable.__init__(self, cfg)

        unique_name = f'{Learner.__name__}_p{policy_id}'
        EventLoopObject.__init__(self, evt_loop, unique_name)

        self.timing = Timing(name=f'Learner {policy_id} profile')

        self.policy_id = policy_id

        self.env_info = env_info
        self.param_server = ParameterServer(policy_id, buffer_mgr.policy_versions, cfg.serial_mode, mp_ctx)

        self.device = None
        self.actor_critic = None

        self.optimizer = None

        self.lr_scheduler: Optional[LearningRateScheduler] = None

        self.train_step = self.env_steps = 0
        self.best_performance = -1e9

        # decay rate at which summaries are collected
        # save summaries every 5 seconds in the beginning, but decay to every 4 minutes in the limit, because we
        # do not need frequent summaries for longer experiments
        self.summary_rate_decay_seconds = LinearDecay([(0, 5), (100000, 120), (1000000, 240)])
        self.last_summary_time = 0

        # TODO: fix milestone mechanism
        self.last_milestone_time = 0

        self.buffer_mgr = buffer_mgr
        self.batcher: Batcher = batcher
        self.batcher_thread: Optional[Thread] = None

        self.exploration_loss_func = self.kl_loss_func = None

        self.is_initialized = False

    @signal
    def initialized(self): pass

    @signal
    def model_initialized(self): pass

    @signal
    def report_msg(self): pass

    @signal
    def training_batch_released(self): pass

    @signal
    def finished_training_iteration(self): pass

    @signal
    def stop(self): pass

    def init(self):
        if not self.cfg.serial_mode:
            self.start_batcher_thread()

        if self.cfg.exploration_loss_coeff == 0.0:
            self.exploration_loss_func = lambda action_distr, valids: 0.0
        elif self.cfg.exploration_loss == 'entropy':
            self.exploration_loss_func = self._entropy_exploration_loss
        elif self.cfg.exploration_loss == 'symmetric_kl':
            self.exploration_loss_func = self._symmetric_kl_exploration_loss
        else:
            raise NotImplementedError(f'{self.cfg.exploration_loss} not supported!')

        if self.cfg.kl_loss_coeff == 0.0:
            if is_continuous_action_space(self.env_info.action_space):
                log.warning(
                    'WARNING! It is generally recommended to enable Fixed KL loss (https://arxiv.org/pdf/1707.06347.pdf) for continuous action tasks to avoid potential numerical issues. '
                    'I.e. set --kl_loss_coeff=0.1'
                )
            self.kl_loss_func = lambda action_space, action_logits, distribution, valids: (None, 0.0)
        else:
            self.kl_loss_func = self._kl_loss

        # initialize the Torch modules
        if self.cfg.seed is None:
            log.info('Starting seed is not provided')
        else:
            log.info('Setting fixed seed %d', self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)

        # initialize device
        self.device = policy_device(self.cfg, self.policy_id)

        log.debug('Initializing actor-critic model on device %s', self.device)
        with self.param_server.policy_lock:
            # trainable torch module
            self.actor_critic = create_actor_critic(self.cfg, self.env_info.obs_space, self.env_info.action_space, self.timing)
            self.actor_critic.model_to_device(self.device)
            self.actor_critic.share_memory()
            self.actor_critic.train()

            params = list(self.actor_critic.parameters())

            optimizer_cls = dict(adam=torch.optim.Adam, lamb=Lamb)
            if self.cfg.optimizer not in optimizer_cls:
                raise RuntimeError(f'Unknown optimizer {self.cfg.optimizer}')

            optimizer_cls = optimizer_cls[self.cfg.optimizer]
            log.debug(f'Using optimizer {optimizer_cls}')

            self.optimizer = optimizer_cls(
                params,
                lr=self.cfg.learning_rate,
                betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
                eps=self.cfg.adam_eps,
            )

            self.lr_scheduler = get_lr_scheduler(self.cfg)

            self.load_from_checkpoint(self.policy_id)
            self.param_server.init(self.actor_critic, self.train_step)

            # in serial mode we will just use the same actor_critic directly
            state_dict = None if self.cfg.serial_mode else self.actor_critic.state_dict()
            model_state = (state_dict, self.device, self.train_step)  # TODO: probably should not send device since we set CUDA vars to only have one device visible
            # signal other components that the model is ready
            self.model_initialized.emit(model_state)

        self.is_initialized = True
        self.initialized.emit()
        log.debug(f'{self.object_id} finished initialization!')

    def start_batcher_thread(self):
        self.batcher.event_loop.process = self.event_loop.process
        self.batcher_thread = Thread(target=self.batcher.event_loop.exec)
        self.batcher_thread.start()

    def join_batcher_thread(self):
        self.batcher_thread.join()

    @staticmethod
    def checkpoint_dir(cfg, policy_id):
        checkpoint_dir = join(experiment_dir(cfg=cfg), f'checkpoint_p{policy_id}')
        return ensure_dir_exists(checkpoint_dir)

    @staticmethod
    def get_checkpoints(checkpoints_dir, pattern='checkpoint_*'):
        checkpoints = glob.glob(join(checkpoints_dir, pattern))
        return sorted(checkpoints)

    @staticmethod
    def load_checkpoint(checkpoints, device):
        if len(checkpoints) <= 0:
            log.warning('No checkpoints found')
            return None
        else:
            latest_checkpoint = checkpoints[-1]

            # extra safety mechanism to recover from spurious filesystem errors
            num_attempts = 3
            for attempt in range(num_attempts):
                # noinspection PyBroadException
                try:
                    log.warning('Loading state from checkpoint %s...', latest_checkpoint)
                    checkpoint_dict = torch.load(latest_checkpoint, map_location=device)
                    return checkpoint_dict
                except Exception:
                    log.exception(f'Could not load from checkpoint, attempt {attempt}')

    def _load_state(self, checkpoint_dict, load_progress=True):
        if load_progress:
            self.train_step = checkpoint_dict['train_step']
            self.env_steps = checkpoint_dict['env_steps']
            self.best_performance = checkpoint_dict.get('best_performance', self.best_performance)
        self.actor_critic.load_state_dict(checkpoint_dict['model'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])

        log.info('Loaded experiment state at training iteration %d, env step %d', self.train_step, self.env_steps)

    def load_from_checkpoint(self, policy_id):
        name_prefix = dict(latest='checkpoint', best='best')[self.cfg.load_checkpoint_kind]
        checkpoints = self.get_checkpoints(self.checkpoint_dir(self.cfg, policy_id), pattern=f'{name_prefix}_*')
        checkpoint_dict = self.load_checkpoint(checkpoints, self.device)
        if checkpoint_dict is None:
            log.debug('Did not load from checkpoint, starting from scratch!')
        else:
            log.debug('Loading model from checkpoint')

            # if we're replacing our policy with another policy (under PBT), let's not reload the env_steps
            # TODO: learner shouldn't know anything about PBT
            load_progress = policy_id == self.policy_id
            self._load_state(checkpoint_dict, load_progress=load_progress)

    def _should_save_summaries(self):
        summaries_every_seconds = self.summary_rate_decay_seconds.at(self.train_step)
        if time.time() - self.last_summary_time < summaries_every_seconds:
            return False

        return True

    def _after_optimizer_step(self):
        """A hook to be called after each optimizer step."""
        self.train_step += 1

    def _get_checkpoint_dict(self):
        checkpoint = {
            'train_step': self.train_step,
            'env_steps': self.env_steps,
            'best_performance': self.best_performance,
            'model': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        return checkpoint

    def _save_impl(self, name_prefix, name_suffix, keep_checkpoints, verbose=True):
        if not self.is_initialized:
            return

        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None

        checkpoint_dir = self.checkpoint_dir(self.cfg, self.policy_id)
        tmp_filepath = join(checkpoint_dir, f'.{name_prefix}_temp')
        checkpoint_name = f'{name_prefix}_{self.train_step:09d}_{self.env_steps}{name_suffix}.pth'
        filepath = join(checkpoint_dir, checkpoint_name)
        if verbose:
            log.info('Saving %s...', filepath)

        # This should protect us from a rare case where something goes wrong mid-save and we end up with a corrupted
        # checkpoint file. It better be a corrupted temp file.
        torch.save(checkpoint, tmp_filepath)
        os.rename(tmp_filepath, filepath)

        while len(checkpoints := self.get_checkpoints(checkpoint_dir, f'{name_prefix}_*')) > keep_checkpoints:
            oldest_checkpoint = checkpoints[0]
            if os.path.isfile(oldest_checkpoint):
                if verbose:
                    log.debug('Removing %s', oldest_checkpoint)
                os.remove(oldest_checkpoint)

    def save(self):
        self._save_impl('checkpoint', '', self.cfg.keep_checkpoints)

        # TODO: move milestone logic to the runner?
        # if self.cfg.save_milestones_sec > 0:
        #     # milestones enabled
        #     if time.time() - self.last_milestone_time >= self.cfg.save_milestones_sec:
        #         milestones_dir = ensure_dir_exists(join(checkpoint_dir, 'milestones'))
        #         milestone_path = join(milestones_dir, f'{checkpoint_name}.milestone')
        #         log.debug('Saving a milestone %s', milestone_path)
        #         shutil.copy(filepath, milestone_path)
        #         self.last_milestone_time = time.time()

    def save_best(self, policy_id, metric, metric_value):
        # TODO it seems that the Runner is broadcasting the signals to all learners, so it won't pass the assertion in multi-policy env, we may add an if instead of assert?
        # assert policy_id == self.policy_id
        if policy_id != self.policy_id:
            return
        p = 3  # precision, number of significant digits
        if metric_value - self.best_performance > 1 / 10**p:
            log.info(f'Saving new best policy, {metric}={metric_value:.{p}f}!')
            self.best_performance = metric_value
            name_suffix = f'_{metric}_{metric_value:.{p}f}'
            self._save_impl('best', name_suffix, 1, verbose=False)

    @staticmethod
    def _policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids):
        clipped_ratio = torch.clamp(ratio, clip_ratio_low, clip_ratio_high)
        loss_unclipped = ratio * adv
        loss_clipped = clipped_ratio * adv
        loss = torch.min(loss_unclipped, loss_clipped)
        loss = torch.masked_select(loss, valids)
        loss = -loss.mean()

        return loss

    def _value_loss(self, new_values, old_values, target, clip_value, valids):
        value_clipped = old_values + torch.clamp(new_values - old_values, -clip_value, clip_value)
        value_original_loss = (new_values - target).pow(2)
        value_clipped_loss = (value_clipped - target).pow(2)
        value_loss = torch.max(value_original_loss, value_clipped_loss)
        value_loss = torch.masked_select(value_loss, valids)
        value_loss = value_loss.mean()

        value_loss *= self.cfg.value_loss_coeff

        return value_loss

    def _kl_loss(self, action_space, action_logits, action_distribution, valids) -> Tuple[Tensor, Tensor]:
        old_action_distribution = get_action_distribution(action_space, action_logits)
        kl_old = action_distribution.kl_divergence(old_action_distribution)
        kl_loss = torch.masked_select(kl_old, valids)
        kl_loss = kl_loss.mean()

        kl_loss *= self.cfg.kl_loss_coeff

        return kl_old, kl_loss

    def _entropy_exploration_loss(self, action_distribution, valids):
        entropy = action_distribution.entropy()
        entropy = torch.masked_select(entropy, valids)
        entropy_loss = -self.cfg.exploration_loss_coeff * entropy.mean()
        return entropy_loss

    def _symmetric_kl_exploration_loss(self, action_distribution, valids):
        kl_prior = action_distribution.symmetric_kl_with_uniform_prior()
        kl_prior = torch.masked_select(kl_prior, valids).mean()
        if not torch.isfinite(kl_prior):
            kl_prior = torch.zeros(kl_prior.shape)
        kl_prior = torch.clamp(kl_prior, max=30)
        kl_prior_loss = self.cfg.exploration_loss_coeff * kl_prior
        return kl_prior_loss

    def _curr_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _update_lr(self, new_lr):
        if new_lr != self._curr_lr():
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

    def _get_minibatches(self, batch_size, experience_size):
        """Generating minibatches for training."""
        assert self.cfg.rollout % self.cfg.recurrence == 0
        assert experience_size % batch_size == 0, f'experience size: {experience_size}, batch size: {batch_size}'
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
            minibatches = tuple(slice(i * batch_size, (i + 1) * batch_size) for i in range(0, minibatches_per_epoch))

        return minibatches

    @staticmethod
    def _get_minibatch(buffer, indices):
        if indices is None:
            # handle the case of a single batch, where the entire buffer is a minibatch
            return buffer

        # TODO: make sure we use this code everywhere. Here we're relying on TensorDict's indexing features
        mb = buffer[indices]
        return mb

    def _train(self, gpu_buffer, batch_size, experience_size, timing):
        with torch.no_grad(), timing.add_time('prepare_train'):
            policy_version_before_train = self.train_step

            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_actor_loss = 1e9
            epoch_actor_losses = []

            # recent mean KL-divergences per minibatch, this used by LR schedulers
            recent_kls = []

            # V-trace parameters
            # noinspection PyArgumentList
            rho_hat = torch.Tensor([self.cfg.vtrace_rho])
            # noinspection PyArgumentList
            c_hat = torch.Tensor([self.cfg.vtrace_c])

            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high

            clip_value = self.cfg.ppo_clip_value
            gamma = self.cfg.gamma
            recurrence = self.cfg.recurrence

            if self.cfg.with_vtrace:
                assert recurrence == self.cfg.rollout and recurrence > 1, \
                    'V-trace requires to recurrence and rollout to be equal'

            num_sgd_steps = 0
            stats_and_summaries = None

            # When it is time to record train summaries, we randomly sample epoch/batch for which the summaries are
            # collected to get equal representation from different stages of training.
            # Half the time, we record summaries from the very large step of training. There we will have the highest
            # KL-divergence and ratio of PPO-clipped samples, which makes this data even more useful for analysis.
            # Something to consider: maybe we should have these last-batch metrics in a separate summaries category?
            with_summaries = self._should_save_summaries()
            if np.random.rand() < 0.5:
                summaries_epoch = np.random.randint(0, self.cfg.num_epochs)
                summaries_batch = np.random.randint(0, self.cfg.num_batches_per_epoch)
            else:
                summaries_epoch = self.cfg.num_epochs - 1
                summaries_batch = self.cfg.num_batches_per_epoch - 1

            assert self.actor_critic.training

        for epoch in range(self.cfg.num_epochs):
            with timing.add_time('epoch_init'):
                if early_stop:
                    break

                force_summaries = False
                minibatches = self._get_minibatches(batch_size, experience_size)

            for batch_num in range(len(minibatches)):
                with timing.add_time('minibatch_init'):
                    indices = minibatches[batch_num]

                    # current minibatch consisting of short trajectory segments with length == recurrence
                    mb = self._get_minibatch(gpu_buffer, indices)

                    # enable syntactic sugar that allows us to access dict's keys as object attributes
                    mb = AttrDict(mb)

                # calculate policy head outside of recurrent loop
                with timing.add_time('forward_head'):
                    head_outputs = self.actor_critic.forward_head(mb.normalized_obs)

                # initial rnn states
                with timing.add_time('bptt_initial'):
                    if self.cfg.use_rnn:
                        head_output_seq, rnn_states, inverted_select_inds = build_rnn_inputs(
                            head_outputs, mb.dones_cpu, mb.rnn_states, recurrence,
                        )
                    else:
                        rnn_states = mb.rnn_states[::recurrence]

                # calculate RNN outputs for each timestep in a loop
                with timing.add_time('bptt'):
                    if self.cfg.use_rnn:
                        with timing.add_time('bptt_forward_core'):
                            core_output_seq, _ = self.actor_critic.forward_core(head_output_seq, rnn_states)
                        core_outputs = build_core_out_from_seq(core_output_seq, inverted_select_inds)
                    else:
                        core_outputs, _ = self.actor_critic.forward_core(head_outputs, rnn_states)

                num_trajectories = head_outputs.size(0) // recurrence

                with timing.add_time('tail'):
                    assert core_outputs.shape[0] == head_outputs.shape[0]

                    # calculate policy tail outside of recurrent loop
                    result = self.actor_critic.forward_tail(core_outputs, sample_actions=False)
                    action_distribution = self.actor_critic.action_distribution()
                    log_prob_actions = action_distribution.log_prob(mb.actions)
                    ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

                    # super large/small values can cause numerical problems and are probably noise anyway
                    ratio = torch.clamp(ratio, 0.05, 20.0)

                    values = result['values'].squeeze()

                with torch.no_grad():  # these computations are not the part of the computation graph
                    # ignore experience from other agents (i.e. on episode boundary) and from inactive agents
                    valids = mb.policy_id == self.policy_id

                    # ignore experience that was older than the threshold even before training started
                    valids = valids & (policy_version_before_train - mb.policy_version < self.cfg.max_policy_lag)

                    if self.cfg.with_vtrace:
                        with timing.add_time('vtrace'):
                            ratios_cpu = ratio.cpu()
                            values_cpu = values.cpu()
                            rewards_cpu = mb.rewards_cpu
                            dones_cpu = mb.dones_cpu

                            vtrace_rho = torch.min(rho_hat, ratios_cpu)
                            vtrace_c = torch.min(c_hat, ratios_cpu)

                            vs = torch.zeros((num_trajectories * recurrence))
                            adv = torch.zeros((num_trajectories * recurrence))

                            next_values = (values_cpu[recurrence - 1::recurrence] - rewards_cpu[recurrence - 1::recurrence]) / gamma
                            next_vs = next_values

                            for i in reversed(range(self.cfg.recurrence)):
                                rewards = rewards_cpu[i::recurrence]
                                dones = dones_cpu[i::recurrence]
                                not_done = 1.0 - dones
                                not_done_times_gamma = not_done * gamma

                                curr_values = values_cpu[i::recurrence]
                                curr_vtrace_rho = vtrace_rho[i::recurrence]
                                curr_vtrace_c = vtrace_c[i::recurrence]

                                delta_s = curr_vtrace_rho * (rewards + not_done_times_gamma * next_values - curr_values)
                                adv[i::recurrence] = curr_vtrace_rho * (rewards + not_done_times_gamma * next_vs - curr_values)
                                next_vs = curr_values + delta_s + not_done_times_gamma * curr_vtrace_c * (next_vs - next_values)
                                vs[i::recurrence] = next_vs

                                next_values = curr_values

                        targets = vs
                    else:
                        # using regular GAE
                        adv = mb.advantages
                        targets = mb.returns

                    # TODO: this should take validity masks into account!
                    adv_std, adv_mean = torch.std_mean(adv, dim=-1)

                    adv = (adv - adv_mean) / max(1e-7, adv_std.item())  # normalize advantage
                    adv = adv.to(self.device)  # TODO: is this redundant now?

                with timing.add_time('losses'):
                    policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids)
                    exploration_loss = self.exploration_loss_func(action_distribution, valids)
                    kl_old, kl_loss = self.kl_loss_func(self.actor_critic.action_space, mb.action_logits, action_distribution, valids)

                    actor_loss = policy_loss + exploration_loss + kl_loss
                    epoch_actor_losses.append(actor_loss.item())

                    targets = targets.to(self.device)
                    old_values = mb.values
                    value_loss = self._value_loss(values, old_values, targets, clip_value, valids)
                    critic_loss = value_loss

                    loss = actor_loss + critic_loss

                    high_loss = 30.0
                    if abs(to_scalar(policy_loss)) > high_loss or abs(to_scalar(value_loss)) > high_loss or abs(to_scalar(exploration_loss)) > high_loss or abs(to_scalar(kl_loss)) > high_loss:
                        log.warning(
                            'High loss value: l:%.4f pl:%.4f vl:%.4f exp_l:%.4f kl_l:%.4f (recommended to adjust the --reward_scale parameter)',
                            to_scalar(loss), to_scalar(policy_loss), to_scalar(value_loss), to_scalar(exploration_loss), to_scalar(kl_loss),
                        )

                        # perhaps something weird is happening, we definitely want summaries from this step
                        force_summaries = True

                with torch.no_grad(), timing.add_time('kl_divergence'):
                    # if kl_old is not None it is already calculated above
                    if kl_old is None:
                        # calculate KL-divergence with the behaviour policy action distribution
                        old_action_distribution = get_action_distribution(
                            self.actor_critic.action_space, mb.action_logits,
                        )
                        kl_old = action_distribution.kl_divergence(old_action_distribution)
                    kl_old_mean = kl_old.mean().item()
                    recent_kls.append(kl_old_mean)

                # update the weights
                with timing.add_time('update'):
                    # following advice from https://youtu.be/9mS1fIYj1So set grad to None instead of optimizer.zero_grad()
                    for p in self.actor_critic.parameters():
                        p.grad = None

                    loss.backward()

                    if self.cfg.max_grad_norm > 0.0:
                        with timing.add_time('clip'):
                            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)

                    curr_policy_version = self.train_step  # policy version before the weight update

                    with self.param_server.policy_lock:
                        self.optimizer.step()

                    num_sgd_steps += 1

                with torch.no_grad(), timing.add_time('after_optimizer'):
                    self._after_optimizer_step()

                    if self.lr_scheduler.invoke_after_each_minibatch():
                        self._update_lr(self.lr_scheduler.update(self._curr_lr(), recent_kls))

                    # collect and report summaries
                    should_record_summaries = with_summaries
                    should_record_summaries &= epoch == summaries_epoch and batch_num == summaries_batch
                    should_record_summaries |= force_summaries
                    if should_record_summaries:
                        stats_and_summaries = self._record_summaries(AttrDict(locals()))
                        force_summaries = False

            # end of an epoch
            if self.lr_scheduler.invoke_after_each_epoch():
                self._update_lr(self.lr_scheduler.update(self._curr_lr(), recent_kls))

            # this will force policy update on the inference worker (policy worker)
            self.buffer_mgr.policy_versions[self.policy_id] = self.train_step

            new_epoch_actor_loss = np.mean(epoch_actor_losses)
            loss_delta_abs = abs(prev_epoch_actor_loss - new_epoch_actor_loss)
            if loss_delta_abs < early_stopping_tolerance:
                early_stop = True
                log.debug(
                    'Early stopping after %d epochs (%d sgd steps), loss delta %.7f',
                    epoch + 1, num_sgd_steps, loss_delta_abs,
                )
                break

            prev_epoch_actor_loss = new_epoch_actor_loss
            epoch_actor_losses = []

        return stats_and_summaries

    # noinspection PyUnresolvedReferences
    def _record_summaries(self, train_loop_vars):
        var = train_loop_vars

        self.last_summary_time = time.time()
        stats = AttrDict()

        stats.lr = self._curr_lr()

        stats.update(self.actor_critic.summaries())

        stats.valids_fraction = var.valids.float().mean()
        stats.same_policy_fraction = (var.mb.policy_id == self.policy_id).float().mean()

        grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.actor_critic.parameters()
            if p.grad is not None
        ) ** 0.5
        stats.grad_norm = grad_norm
        stats.loss = var.loss
        stats.value = var.result['values'].mean()
        stats.entropy = var.action_distribution.entropy().mean()
        stats.policy_loss = var.policy_loss
        stats.kl_loss = var.kl_loss
        stats.value_loss = var.value_loss
        stats.exploration_loss = var.exploration_loss

        stats.adv_min = var.adv.min()
        stats.adv_max = var.adv.max()
        stats.adv_std = var.adv_std
        stats.max_abs_logprob = torch.abs(var.mb.action_logits).max()

        if hasattr(var.action_distribution, 'summaries'):
            stats.update(var.action_distribution.summaries())

        if var.epoch == self.cfg.num_epochs - 1 and var.batch_num == len(var.minibatches) - 1:
            # we collect these stats only for the last PPO batch, or every time if we're only doing one batch, IMPALA-style
            ratio_mean = torch.abs(1.0 - var.ratio).mean().detach()
            ratio_min = var.ratio.min().detach()
            ratio_max = var.ratio.max().detach()
            # log.debug('Learner %d ratio mean min max %.4f %.4f %.4f', self.policy_id, ratio_mean.cpu().item(), ratio_min.cpu().item(), ratio_max.cpu().item())

            value_delta = torch.abs(var.values - var.old_values)
            value_delta_avg, value_delta_max = value_delta.mean(), value_delta.max()

            stats.kl_divergence = var.kl_old_mean
            stats.kl_divergence_max = var.kl_old.max()
            stats.value_delta = value_delta_avg
            stats.value_delta_max = value_delta_max
            stats.fraction_clipped = ((var.ratio < var.clip_ratio_low).float() + (var.ratio > var.clip_ratio_high).float()).mean()
            stats.ratio_mean = ratio_mean
            stats.ratio_min = ratio_min
            stats.ratio_max = ratio_max
            stats.num_sgd_steps = var.num_sgd_steps

        # this caused numerical issues on some versions of PyTorch with second moment reaching infinity
        adam_max_second_moment = 0.0
        for key, tensor_state in self.optimizer.state.items():
            adam_max_second_moment = max(tensor_state['exp_avg_sq'].max().item(), adam_max_second_moment)
        stats.adam_max_second_moment = adam_max_second_moment

        version_diff = (var.curr_policy_version - var.mb.policy_version)[var.mb.policy_id == self.policy_id]
        stats.version_diff_avg = version_diff.mean()
        stats.version_diff_min = version_diff.min()
        stats.version_diff_max = version_diff.max()

        for key, value in stats.items():
            stats[key] = to_scalar(value)

        return stats

    def _prepare_batch(self, batch_idx: int):
        with torch.no_grad():
            # create a shallow copy so we can modify the dictionary
            # we still reference the same buffers though
            buff = copy.copy(self.batcher.training_batches[batch_idx])

            # TODO: how about device_and_type_for_input_tensor

            # calculate estimated value for the next step (T+1)
            self.actor_critic.eval()
            normalized_last_obs = self.actor_critic.normalizer(buff['obs'][:, -1])
            next_values = self.actor_critic(normalized_last_obs, buff['rnn_states'][:, -1], values_only=True)['values']

            # remove next step obs and rnn_states from the batch, we don't need them anymore
            buff['obs'] = buff['obs'][:, :-1]
            buff['rnn_states'] = buff['rnn_states'][:, :-1]

            buff['advantages'], buff['returns'] = gae_advantages_returns(
                buff['rewards'], buff['dones'], buff['values'], next_values, self.cfg.gamma, self.cfg.gae_lambda,
            )

            experience_size = self.cfg.batch_size * self.cfg.num_batches_per_epoch

            for d, k, v in iterate_recursively(buff):
                # collapse first two dimensions
                shape = (experience_size, ) + tuple(v.shape[2:])
                d[k] = v.reshape(shape)

            # TODO: inefficiency, do we need these on the GPU in the first place?
            buff['dones_cpu'] = buff['dones'].to('cpu', copy=True, dtype=torch.float, non_blocking=True)
            buff['rewards_cpu'] = buff['rewards'].to('cpu', copy=True, dtype=torch.float, non_blocking=True)

            # # will squeeze actions only in simple categorical case
            # for tensor_name in ['actions']:
            #     buff[tensor_name].squeeze_()

            # normalize obs and record data statistics (hence the "train" mode)
            self.actor_critic.train()
            with self.param_server.policy_lock:
                buff['normalized_obs'] = self.actor_critic.normalizer(buff['obs'])
            del buff['obs']  # we don't need the regular obs anymore

            return buff, experience_size

    def train(self, batch_idx: int, timing: Timing) -> Dict:
        with timing.add_time('prepare_batch'):
            buff, experience_size = self._prepare_batch(batch_idx)

        with timing.add_time('train'):
            train_stats = self._train(buff, self.cfg.batch_size, experience_size, timing)

        # TODO: don't use frameskip here
        self.env_steps += experience_size * self.env_info.frameskip

        stats = dict(learner_env_steps=self.env_steps, policy_id=self.policy_id)
        if train_stats is not None:
            stats['train'] = train_stats

            # TODO
            # if wait_stats is not None:
            #     wait_avg, wait_min, wait_max = wait_stats
            #     stats['train']['wait_avg'] = wait_avg
            #     stats['train']['wait_min'] = wait_min
            #     stats['train']['wait_max'] = wait_max

            stats['train']['discarded_rollouts'] = 0  # TODO
            stats['train']['discarding_rate'] = 0  # TODO

            stats['stats'] = memory_stats('learner', self.device)

        return stats

    def on_new_training_batch(self, batch_idx: int):
        stats = self.train(batch_idx, self.timing)
        self.training_batch_released.emit(batch_idx)
        self.report_msg.emit(stats)

        self.finished_training_iteration.emit()

    def on_stop(self, *_):
        self.save()
        log.debug(f'Stopping {self.object_id}...')

        if not self.cfg.serial_mode:
            self.join_batcher_thread()

        self.stop.emit(self.object_id, self.timing)

        if self.event_loop.owner is self:
            self.event_loop.stop()
            del self.actor_critic

        self.detach()  # remove from the current event loop
