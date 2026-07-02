from __future__ import annotations

import copy
from typing import Dict, Optional, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, POLICY_ID_KEY, STATS_KEY, TRAIN_STATS, memory_stats
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.replay_buffer import PrioritizedReplayBuffer, ReplayBuffer
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_dict import TensorDict, shallow_recursive_copy
from sample_factory.algo.utils.torch_utils import synchronize, to_scalar
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, InitModelData, PolicyID
from sample_factory.utils.utils import debug_log_every_n, log


class DQNLearner(Learner):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
        global_env_steps_tensor: Optional[Tensor] = None,
    ):
        super().__init__(cfg, env_info, policy_versions_tensor, policy_id, param_server)

        self.target_network = None
        self.replay_buffer: Optional[Union[ReplayBuffer, PrioritizedReplayBuffer]] = None

        # Shared tensor for global env steps to sync epsilon schedule
        self.global_env_steps_tensor = global_env_steps_tensor

        self.total_env_steps_for_training = 0
        self.last_target_update_step = 0
        self.last_train_env_steps = 0

        self.last_valid_ratio = 1.0
        self.last_valid_dropped = 0
        self.last_valid_total = 0

        self._last_q_mean = 0.0
        self._last_q_max = 0.0
        self._last_q_min = 0.0
        self._last_target_q_mean = 0.0
        self._last_td_error_mean = 0.0

        self.use_per = getattr(cfg, "per", False)
        self.per_beta_start = getattr(cfg, "per_beta_start", 0.4)
        self.per_beta_frames = getattr(cfg, "per_beta_frames", 100000)

    def init(self) -> InitModelData:
        init_data = super().init()

        if getattr(self.cfg, "use_rnn", False) or getattr(self.cfg, "recurrence", 1) > 1:
            log.error("DQN doesnt support RNN without sequence replay. Disable --use_rnn")
            raise RuntimeError("DQN doesnt support RNN")

        self.target_network = copy.deepcopy(self.actor_critic)
        self.target_network.eval()
        # No requires_grad=False for target network bcus torch.no_grad() always used when computing target Q-values

        if self.use_per:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=self.cfg.replay_buffer_size,
                obs_space=self.env_info.obs_space,
                action_space=self.env_info.action_space,
                omega=getattr(self.cfg, "per_omega", 0.6),
                beta_start=self.per_beta_start,
                device="cpu",
                share_memory=not self.cfg.serial_mode,
            )
            log.info(f"DQN using PER (omega={self.cfg.per_omega})")
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=self.cfg.replay_buffer_size,
                obs_space=self.env_info.obs_space,
                action_space=self.env_info.action_space,
                device="cpu",
                share_memory=not self.cfg.serial_mode,
            )
        log.info(f"DQN replay batch size: {self._dqn_batch_size()}")
        return init_data

    def _dqn_batch_size(self) -> int:
        dqn_batch_size = getattr(self.cfg, "dqn_batch_size", 0)
        return dqn_batch_size if dqn_batch_size > 0 else self.cfg.batch_size

    def _update_target_network(self, tau: float = 1.0) -> None:
        if self.target_network is None or self.actor_critic is None:
            return

        # Soft update every step, use Polyak avg-ing
        if tau < 1.0:
            with torch.no_grad():
                for target_param, param in zip(self.target_network.parameters(), self.actor_critic.parameters()):
                    # target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
                    target_param.data.mul_(1.0 - tau)
                    target_param.data.add_(param.data, alpha=tau)
            self.last_target_update_step = self.train_step
            return

        # Hard update
        if self.train_step - self.last_target_update_step >= self.cfg.target_update_interval:
            self.target_network.load_state_dict(self.actor_critic.state_dict())
            self.last_target_update_step = self.train_step
            log.debug(f"Hard Updated target network at step {self.train_step}")

    def _prepare_batch_for_buffer(self, batch: TensorDict) -> TensorDict:
        """
        Batch shape: [num_trajectories, rollout_length, ...]
        obs: [num_traj, rollout_len + 1, ...]
        rewards/actions/dones: [num_traj, rollout_len]
        So at t we get (s_t, a_t, r_t, done_t, s_{t+1})
        """
        with torch.no_grad():
            buff = shallow_recursive_copy(batch)

            num_traj = buff["rewards"].shape[0]
            rollout_len = buff["rewards"].shape[1]

            # obs[:, :-1] current states, obs[:, 1:] next states
            obs = buff["obs"]
            transitions = TensorDict()
            transitions["obs"] = TensorDict()
            for key, value in obs.items():
                current_obs = value[:, :-1]
                transitions["obs"][key] = current_obs.reshape((num_traj * rollout_len,) + current_obs.shape[2:])

            transitions["next_obs"] = TensorDict()
            for key, value in obs.items():
                next_obs = value[:, 1:]
                transitions["next_obs"][key] = next_obs.reshape((num_traj * rollout_len,) + next_obs.shape[2:])

            transitions["actions"] = buff["actions"].reshape(-1, *buff["actions"].shape[2:])
            transitions["rewards"] = buff["rewards"].reshape(-1)
            transitions["dones"] = buff["dones"].reshape(-1).float()
            if "time_outs" in buff:
                transitions["time_outs"] = buff["time_outs"].reshape(-1).float()

            # valids = buff["policy_id"] == self.policy_id
            # valids &= self.train_step - buff["policy_version"] < self.cfg.max_policy_lag
            # valids = valids.reshape(-1)
            # We dont filter policy lag, but keep this for PBT or many learners
            valids = (buff["policy_id"] == self.policy_id).reshape(-1)

            # This is to filter invalid transitions from
            num_total = valids.numel()
            num_valid = int(valids.sum().item())
            self.last_valid_total = num_total
            self.last_valid_dropped = num_total - num_valid
            self.last_valid_ratio = num_valid / max(1, num_total)
            if self.last_valid_dropped > 0:
                debug_log_every_n(
                    50,
                    f"DQN filtered {self.last_valid_dropped}/{num_total} invalid transitions ({(1 - self.last_valid_ratio) * 100:.2f}%)",
                )

            if not torch.all(valids).item():
                for key, value in transitions["obs"].items():
                    transitions["obs"][key] = value[valids]
                for key, value in transitions["next_obs"].items():
                    transitions["next_obs"][key] = value[valids]
                transitions["actions"] = transitions["actions"][valids]
                transitions["rewards"] = transitions["rewards"][valids]
                transitions["dones"] = transitions["dones"][valids]
                if "time_outs" in transitions:
                    transitions["time_outs"] = transitions["time_outs"][valids]

            return transitions

    def _calculate_dqn_loss(self, batch: TensorDict, weights: Optional[Tensor] = None) -> Tensor:
        """
        Can use Double DQN and PER
        """
        if self.actor_critic is None or self.target_network is None:
            raise RuntimeError("Networks not initialized")

        with self.param_server.policy_lock:
            normalized_obs = prepare_and_normalize_obs(self.actor_critic, batch["obs"])
            normalized_next_obs = prepare_and_normalize_obs(self.actor_critic, batch["next_obs"])

        actions = batch["actions"].long()
        rewards = batch["rewards"]
        dones = batch["dones"]
        if "time_outs" in batch:
            dones = dones * (1.0 - batch["time_outs"])

        # Reward already clipped at collection)
        dqn_reward_clip = getattr(self.cfg, "dqn_reward_clip", 0.0)
        if dqn_reward_clip > 0:
            rewards = rewards.clamp(-dqn_reward_clip, dqn_reward_clip)

        # Current q
        batch_size = actions.shape[0]
        # Note: Every transition is independent, adapt in QMIX
        rnn_states = torch.zeros(
            batch_size,
            self.actor_critic.core.get_out_size() if hasattr(self.actor_critic, "core") else 1,
            device=self.device,
        )
        result = self.actor_critic(normalized_obs, rnn_states, values_only=False, sample_actions=False)
        q_values = result["action_logits"]  # [batch, total_num_actions]

        action_space = self.env_info.action_space
        if hasattr(action_space, "spaces"):
            action_sizes = [s.n for s in action_space.spaces]
            q_splits = torch.split(q_values, action_sizes, dim=1)

            if actions.dim() == 1:
                actions = actions.unsqueeze(1)

            current_q_list = []
            for i, (q_head, a_head_size) in enumerate(zip(q_splits, action_sizes)):
                a_idx = actions[:, i : i + 1]  # [batch, 1]
                current_q_list.append(q_head.gather(1, a_idx).squeeze(1))

            current_q_heads = torch.stack(current_q_list, dim=1)

            with torch.no_grad():
                # Get next Q-values from target network
                target_result = self.target_network(
                    normalized_next_obs, rnn_states, values_only=False, sample_actions=False
                )
                next_q_target = target_result["action_logits"]
                next_q_target_splits = torch.split(next_q_target, action_sizes, dim=1)

                if self.cfg.double_dqn:
                    online_result = self.actor_critic(
                        normalized_next_obs, rnn_states, values_only=False, sample_actions=False
                    )
                    next_q_online = online_result["action_logits"]
                    next_q_online_splits = torch.split(next_q_online, action_sizes, dim=1)

                    next_q_list = []
                    for q_online, q_target in zip(next_q_online_splits, next_q_target_splits):
                        next_actions = q_online.argmax(dim=1, keepdim=True)
                        next_q_list.append(q_target.gather(1, next_actions).squeeze(1))
                    next_q = torch.stack(next_q_list, dim=1)
                else:
                    next_q_list = [q.max(dim=1)[0] for q in next_q_target_splits]
                    next_q = torch.stack(next_q_list, dim=1)

                # r + gamma * Q_target(s', a') * (1 - done)
                # https://stackoverflow.com/questions/58559415/setting-up-target-values-for-deep-q-learning
                target_q_heads = rewards.unsqueeze(1) + self.cfg.gamma * next_q * (1.0 - dones).unsqueeze(1)

            current_q = current_q_heads.mean(dim=1)
            target_q = target_q_heads.mean(dim=1)
            td_errors = torch.abs(current_q_heads - target_q_heads).mean(dim=1).detach()
            elementwise_loss = F.smooth_l1_loss(current_q_heads, target_q_heads, reduction="none").mean(dim=1)
        else:
            # Single action space, not composite_action_space
            if actions.dim() > 1:
                actions = actions.squeeze(-1)
            current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                target_result = self.target_network(
                    normalized_next_obs, rnn_states, values_only=False, sample_actions=False
                )
                next_q_target = target_result["action_logits"]

                if self.cfg.double_dqn:
                    online_result = self.actor_critic(
                        normalized_next_obs, rnn_states, values_only=False, sample_actions=False
                    )
                    next_q_online = online_result["action_logits"]
                    next_actions = next_q_online.argmax(dim=1, keepdim=True)
                    next_q = next_q_target.gather(1, next_actions).squeeze(1)
                else:
                    next_q = next_q_target.max(dim=1)[0]

                target_q = rewards + self.cfg.gamma * next_q * (1.0 - dones)

            # Element-wise TD errors
            td_errors = torch.abs(current_q - target_q).detach()

            # Huber loss (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py)
            elementwise_loss = F.smooth_l1_loss(current_q, target_q, reduction="none")

        self._last_q_mean = current_q.mean().item()
        self._last_q_max = current_q.max().item()
        self._last_q_min = current_q.min().item()
        self._last_target_q_mean = target_q.mean().item()
        self._last_td_error_mean = td_errors.mean().item()

        # Importance sampling weights
        if weights is not None:
            if weights.shape != elementwise_loss.shape:
                weights = weights.view_as(elementwise_loss)
            elementwise_loss *= weights

        loss = elementwise_loss.mean()

        return loss, td_errors

    def _train_on_batch(self, batch: TensorDict, weights: Optional[Tensor] = None, indices: Optional[Tensor] = None):
        if self.actor_critic is None or self.optimizer is None:
            return None, None

        self.actor_critic.train()

        loss, td_errors = self._calculate_dqn_loss(batch, weights)

        for p in self.actor_critic.parameters():
            p.grad = None

        loss.backward()

        if self.cfg.max_grad_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)

        with self.param_server.policy_lock:
            self.optimizer.step()

        self._after_optimizer_step()

        self._update_target_network(self.cfg.target_update_tau)

        # Sync weights
        synchronize(self.cfg, self.device)
        self.policy_versions_tensor[self.policy_id] = self.train_step

        stats = AttrDict()
        stats.loss = to_scalar(loss)
        stats.lr = self.curr_lr
        return stats, td_errors

    def train(self, batch: TensorDict) -> Optional[Dict]:
        with self.timing.add_time("misc"):
            self._maybe_update_cfg()
            self._maybe_load_policy()

        with self.timing.add_time("prepare_batch"):
            transitions = self._prepare_batch_for_buffer(batch)

            num_transitions = transitions["rewards"].shape[0]
            if self.cfg.summaries_use_frameskip:
                self.env_steps += num_transitions * self.env_info.frameskip
            else:
                self.env_steps += num_transitions

            if self.global_env_steps_tensor is not None:
                self.global_env_steps_tensor[self.policy_id] = self.env_steps

        with self.timing.add_time("add_to_buffer"):
            if self.replay_buffer is not None:
                self.replay_buffer.add(transitions)

        if self.replay_buffer is None or len(self.replay_buffer) < self.cfg.learning_starts:
            return {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}

        new_steps = transitions["rewards"].shape[0]
        steps_per_update = int(getattr(self.cfg, "train_frequency", 1))
        if steps_per_update <= 0:
            log.warning("DQN: train_frequency <= 0")
            steps_per_update = 1

        self.total_env_steps_for_training += new_steps

        # For async, 1 update every n steps is not possible, so we calculate the number of updates need to be made
        # minus full debt (num_updates * steps_per_update), then cap the actual work
        num_updates = self.total_env_steps_for_training // steps_per_update
        if num_updates == 0:
            return {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
        self.total_env_steps_for_training -= num_updates * steps_per_update
        max_updates = getattr(self.cfg, "dqn_max_updates_per_batch", 0)
        if max_updates > 0:
            num_updates = min(num_updates, max_updates)

        train_stats = None
        with self.timing.add_time("train"):
            for _ in range(num_updates):
                # Anneal PER beta linearly towards 1.0
                if self.use_per and isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                    if self.per_beta_frames <= 0:
                        beta = 1.0
                    else:
                        progress = min(1.0, float(self.env_steps) / float(self.per_beta_frames))
                        beta = self.per_beta_start + progress * (1.0 - self.per_beta_start)
                    self.replay_buffer.set_beta(beta)

                sampled = self.replay_buffer.sample(self._dqn_batch_size(), str(self.device))
                if sampled is not None:
                    if self.use_per and isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                        sampled_batch, weights, indices = sampled
                        train_stats, td_errors = self._train_on_batch(sampled_batch, weights=weights, indices=indices)
                        if td_errors is not None:
                            self.replay_buffer.update_priorities(indices, td_errors)
                    else:
                        sampled_batch = sampled
                        train_stats, _ = self._train_on_batch(sampled_batch)

        self.last_train_env_steps = self.env_steps

        stats = {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
        if train_stats is not None:
            train_stats.env_steps = self.env_steps

            train_stats.dqn_num_updates = num_updates
            train_stats.dqn_steps_per_update = steps_per_update
            train_stats.dqn_pending_steps = self.total_env_steps_for_training
            train_stats.policy_lag_valid_frac = self.last_valid_ratio
            train_stats.policy_lag_dropped = self.last_valid_dropped
            train_stats.policy_lag_total = self.last_valid_total

            train_stats.dqn_q_mean = self._last_q_mean
            train_stats.dqn_q_max = self._last_q_max
            train_stats.dqn_q_min = self._last_q_min
            train_stats.dqn_target_q_mean = self._last_target_q_mean
            train_stats.dqn_td_error_mean = self._last_td_error_mean

            grad_norm = (
                sum(p.grad.data.norm(2).item() ** 2 for p in self.actor_critic.parameters() if p.grad is not None)
                ** 0.5
            )
            train_stats.grad_norm = grad_norm

            stats[TRAIN_STATS] = train_stats
            stats[STATS_KEY] = memory_stats("learner", self.device)

        return stats

    def _get_checkpoint_dict(self):
        checkpoint = super()._get_checkpoint_dict()
        if self.target_network is not None:
            checkpoint["target_network"] = self.target_network.state_dict()
        if self.replay_buffer is not None:
            checkpoint["replay_buffer_size"] = len(self.replay_buffer)
        checkpoint["last_train_env_steps"] = self.last_train_env_steps
        checkpoint["last_target_update_step"] = self.last_target_update_step
        return checkpoint

    def _load_state(self, checkpoint_dict, load_progress=True):
        super()._load_state(checkpoint_dict, load_progress)
        if "target_network" in checkpoint_dict and self.target_network is not None:
            self.target_network.load_state_dict(checkpoint_dict["target_network"])
            log.info("Loaded target network from checkpoint")
        if load_progress and "last_train_env_steps" in checkpoint_dict:
            self.last_train_env_steps = checkpoint_dict["last_train_env_steps"]
        if load_progress and "last_target_update_step" in checkpoint_dict:
            self.last_target_update_step = checkpoint_dict["last_target_update_step"]
