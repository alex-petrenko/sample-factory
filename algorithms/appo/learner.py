import glob
import os
import random
import shutil
import signal
import threading
import time
from collections import OrderedDict, deque
from os.path import join
from queue import Empty, Queue
from threading import Thread

import numpy as np
import psutil
import torch
from torch.multiprocessing import Process, Event as MultiprocessingEvent

import fast_queue
from algorithms.appo.appo_utils import TaskType, list_of_dicts_to_dict_of_lists, memory_stats, cuda_envvars, \
    TensorBatcher, iter_dicts_recursively, copy_dict_structure, ObjectPool
from algorithms.appo.model import create_actor_critic
from algorithms.appo.population_based_training import PbtTask
from algorithms.utils.action_distributions import get_action_distribution
from algorithms.utils.algo_utils import calculate_gae
from algorithms.utils.multi_env import safe_get
from utils.decay import LinearDecay
from utils.timing import Timing
from utils.utils import log, AttrDict, experiment_dir, ensure_dir_exists, join_or_kill


class LearnerWorker:
    def __init__(
        self, worker_idx, policy_id, cfg, obs_space, action_space, report_queue, policy_worker_queues, traj_buffers,
        policy_lock,
    ):
        log.info('Initializing GPU learner %d for policy %d', worker_idx, policy_id)

        self.worker_idx = worker_idx
        self.policy_id = policy_id

        self.cfg = cfg

        # PBT-related stuff
        self.should_save_model = True  # set to true if we need to save the model to disk on the next training iteration
        self.load_policy_id = None  # non-None when we need to replace our parameters with another policy's parameters
        self.new_cfg = None  # non-None when we need to update the learning hyperparameters

        self.terminate = False

        self.obs_space = obs_space
        self.action_space = action_space

        self.rollout_tensors = traj_buffers.tensor_trajectories
        self.traj_tensors_available = traj_buffers.is_traj_tensor_available
        self.policy_versions = traj_buffers.policy_versions

        self.device = None
        self.actor_critic = None
        self.optimizer = None
        self.policy_lock = policy_lock

        self.task_queue = fast_queue.Queue()
        self.report_queue = report_queue

        self.initialized_event = MultiprocessingEvent()
        self.initialized_event.clear()

        self.model_saved_event = MultiprocessingEvent()
        self.model_saved_event.clear()

        # queues corresponding to policy workers using the same policy
        # we send weight updates via these queues
        self.policy_worker_queues = policy_worker_queues

        self.experience_buffer_queue = Queue()

        self.tensor_batch_pool = ObjectPool()
        self.tensor_batcher = TensorBatcher(self.tensor_batch_pool)

        self.with_training = True  # set to False for debugging no-training regime
        self.train_in_background = self.cfg.train_in_background_thread  # set to False for debugging

        self.training_thread = Thread(target=self._train_loop) if self.train_in_background else None
        self.train_thread_initialized = threading.Event()

        self.processing_experience_batch = threading.Condition()

        self.train_step = self.env_steps = 0

        # decay rate at which summaries are collected
        # save summaries every 30 seconds in the beginning, but decay to every 5 minutes in the limit, because we
        # do not need frequent summaries for longer experiments
        self.summary_rate_decay_seconds = LinearDecay([(0, 30), (100000, 120), (1000000, 300)])
        self.last_summary_time = time.time()

        self.last_saved_time = self.last_milestone_time = 0

        self.discarded_experience_over_time = deque([], maxlen=30)
        self.discarded_experience_timer = time.time()
        self.num_discarded_rollouts = 0

        self.process = Process(target=self._run, daemon=True)

    def start_process(self):
        self.process.start()

    def _init(self):
        log.info('Waiting for GPU learner to initialize...')
        self.train_thread_initialized.wait()
        log.info('GPU learner %d initialized', self.worker_idx)
        self.initialized_event.set()

    def _terminate(self):
        self.terminate = True

    def _broadcast_model_weights(self):
        state_dict = self.actor_critic.state_dict()
        policy_version = self.train_step
        model_state = (policy_version, state_dict)
        for q in self.policy_worker_queues:
            q.put((TaskType.INIT_MODEL, model_state))

    def _calculate_gae(self, buffer):
        """
        Calculate advantages using Generalized Advantage Estimation.
        This is leftover the from previous version of the algorithm.
        Perhaps should be re-implemented in PyTorch tensors, similar to V-trace for uniformity.
        """

        rewards = torch.stack(buffer.rewards).numpy().squeeze()  # [E, T]
        dones = torch.stack(buffer.dones).numpy().squeeze()  # [E, T]
        values_arr = torch.stack(buffer.values).numpy().squeeze()  # [E, T]

        # calculating fake values for the last step in the rollout
        # this will make sure that advantage of the very last action is always zero
        values = []
        for i in range(len(values_arr)):
            last_value, last_reward = values_arr[i][-1], rewards[i, -1]
            next_value = (last_value - last_reward) / self.cfg.gamma
            values.append(list(values_arr[i]))
            values[i].append(float(next_value))  # [T] -> [T+1]

        # calculating returns and GAE
        rewards = rewards.transpose((1, 0))  # [E, T] -> [T, E]
        dones = dones.transpose((1, 0))  # [E, T] -> [T, E]
        values = np.asarray(values).transpose((1, 0))  # [E, T+1] -> [T+1, E]

        advantages, returns = calculate_gae(rewards, dones, values, self.cfg.gamma, self.cfg.gae_lambda)

        # transpose tensors back to [E, T] before creating a single experience buffer
        buffer.advantages = advantages.transpose((1, 0))  # [T, E] -> [E, T]
        buffer.returns = returns.transpose((1, 0))  # [T, E] -> [E, T]
        buffer.returns = buffer.returns[:, :, np.newaxis]  # [E, T] -> [E, T, 1]

        buffer.advantages = [torch.tensor(buffer.advantages).reshape(-1)]
        buffer.returns = [torch.tensor(buffer.returns).reshape(-1)]

        return buffer

    def _mark_rollout_buffer_free(self, rollout):
        r = rollout
        self.traj_tensors_available[r.worker_idx, r.split_idx][r.env_idx, r.agent_idx, r.traj_buffer_idx] = 1

    def _prepare_train_buffer(self, rollouts, timing):
        trajectories = [AttrDict(r['t']) for r in rollouts]

        with timing.add_time('buffers'):
            buffer = AttrDict()

            # by the end of this loop the buffer is a dictionary containing lists of numpy arrays
            for i, t in enumerate(trajectories):
                for key, x in t.items():
                    if key not in buffer:
                        buffer[key] = []
                    buffer[key].append(x)

            # convert lists of dict observations to a single dictionary of lists
            for key, x in buffer.items():
                if isinstance(x[0], (dict, OrderedDict)):
                    buffer[key] = list_of_dicts_to_dict_of_lists(x)

        if not self.cfg.with_vtrace:
            with timing.add_time('calc_gae'):
                buffer = self._calculate_gae(buffer)

        with timing.add_time('batching'):
            buffer = self.tensor_batcher.cat(buffer, timing)

        with timing.add_time('buff_ready'):
            for r in rollouts:
                self._mark_rollout_buffer_free(r)

        with timing.add_time('tensors_gpu_float'):
            gpu_buffer = self._copy_train_data_to_gpu(buffer)

        with timing.add_time('squeeze'):
            # will squeeze actions only in simple categorical case
            tensors_to_squeeze = ['actions', 'log_prob_actions', 'policy_version', 'values', 'rewards', 'dones']
            for tensor_name in tensors_to_squeeze:
                gpu_buffer[tensor_name].squeeze_()

        self.tensor_batch_pool.put(buffer)
        return gpu_buffer

    def _process_macro_batch(self, rollouts, timing):
        assert self.cfg.macro_batch % self.cfg.rollout == 0
        assert self.cfg.rollout % self.cfg.recurrence == 0
        assert self.cfg.macro_batch % self.cfg.recurrence == 0

        samples = env_steps = 0
        for rollout in rollouts:
            samples += rollout['length']
            env_steps += rollout['env_steps']

        with timing.add_time('prepare'):
            buffer = self._prepare_train_buffer(rollouts, timing)
            self.experience_buffer_queue.put((buffer, samples, env_steps))

    def _process_rollouts(self, rollouts, timing):
        # log.info('Pending rollouts: %d (%d samples)', len(self.rollouts), len(self.rollouts) * self.cfg.rollout)
        rollouts_in_macro_batch = self.cfg.macro_batch // self.cfg.rollout

        if len(rollouts) < rollouts_in_macro_batch:
            return rollouts

        discard_rollouts = 0
        policy_version = self.train_step
        for r in rollouts:
            rollout_min_version = r['t']['policy_version'].min().item()
            if policy_version - rollout_min_version >= self.cfg.max_policy_lag:
                discard_rollouts += 1
                self._mark_rollout_buffer_free(r)
            else:
                break

        if discard_rollouts > 0:
            log.warning(
                'Discarding %d old rollouts, cut by policy lag threshold %d (learner %d)',
                discard_rollouts, self.cfg.max_policy_lag, self.policy_id,
            )
            rollouts = rollouts[discard_rollouts:]
            self.num_discarded_rollouts += discard_rollouts

        if len(rollouts) >= rollouts_in_macro_batch:
            # process newest rollouts
            rollouts_to_process = rollouts[:rollouts_in_macro_batch]
            rollouts = rollouts[rollouts_in_macro_batch:]

            self._process_macro_batch(rollouts_to_process, timing)
            # log.info('Unprocessed rollouts: %d (%d samples)', len(rollouts), len(rollouts) * self.cfg.rollout)

        return rollouts

    def _get_minibatches(self, experience_size):
        """Generating minibatches for training."""
        assert self.cfg.rollout % self.cfg.recurrence == 0
        assert experience_size % self.cfg.batch_size == 0

        if self.cfg.macro_batch == self.cfg.batch_size:
            return [None]  # single minibatch is actually the entire buffer, we don't need indices

        # indices that will start the mini-trajectories from the same episode (for bptt)
        indices = np.arange(0, experience_size, self.cfg.recurrence)
        indices = np.random.permutation(indices)

        # complete indices of mini trajectories, e.g. with recurrence==4: [4, 16] -> [4, 5, 6, 7, 16, 17, 18, 19]
        indices = [np.arange(i, i + self.cfg.recurrence) for i in indices]
        indices = np.concatenate(indices)

        assert len(indices) == experience_size

        num_minibatches = experience_size // self.cfg.batch_size
        minibatches = np.split(indices, num_minibatches)
        return minibatches

    @staticmethod
    def _get_minibatch(buffer, indices):
        if indices is None:
            # handle the case of a single batch, where the entire buffer is a minibatch
            return buffer

        mb = AttrDict()

        for item, x in buffer.items():
            if isinstance(x, (dict, OrderedDict)):
                mb[item] = AttrDict()
                for key, x_elem in x.items():
                    mb[item][key] = x_elem[indices]
            else:
                mb[item] = x[indices]

        return mb

    def _should_save_summaries(self):
        summaries_every_seconds = self.summary_rate_decay_seconds.at(self.train_step)
        if time.time() - self.last_summary_time < summaries_every_seconds:
            return False

        return True

    def _after_optimizer_step(self):
        """A hook to be called after each optimizer step."""
        self.train_step += 1
        self._maybe_save()

    def _maybe_save(self):
        if time.time() - self.last_saved_time >= self.cfg.save_every_sec or self.should_save_model:
            self._save()
            self.model_saved_event.set()
            self.should_save_model = False
            self.last_saved_time = time.time()

    @staticmethod
    def checkpoint_dir(cfg, policy_id):
        checkpoint_dir = join(experiment_dir(cfg=cfg), f'checkpoint_p{policy_id}')
        return ensure_dir_exists(checkpoint_dir)

    @staticmethod
    def get_checkpoints(checkpoints_dir):
        checkpoints = glob.glob(join(checkpoints_dir, 'checkpoint_*'))
        return sorted(checkpoints)

    def _get_checkpoint_dict(self):
        checkpoint = {
            'train_step': self.train_step,
            'env_steps': self.env_steps,
            'model': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        return checkpoint

    def _save(self):
        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None

        checkpoint_dir = self.checkpoint_dir(self.cfg, self.policy_id)
        tmp_filepath = join(checkpoint_dir, 'checkpoint_tmp')
        checkpoint_name = f'checkpoint_{self.train_step:09d}_{self.env_steps}.pth'
        filepath = join(checkpoint_dir, checkpoint_name)
        log.info('Saving %s...', tmp_filepath)
        torch.save(checkpoint, tmp_filepath)
        log.info('Renaming %s to %s', tmp_filepath, filepath)
        os.rename(tmp_filepath, filepath)

        while len(self.get_checkpoints(checkpoint_dir)) > self.cfg.keep_checkpoints:
            oldest_checkpoint = self.get_checkpoints(checkpoint_dir)[0]
            if os.path.isfile(oldest_checkpoint):
                log.debug('Removing %s', oldest_checkpoint)
                os.remove(oldest_checkpoint)

        if self.cfg.save_milestones_sec > 0:
            # milestones enabled
            if time.time() - self.last_milestone_time >= self.cfg.save_milestones_sec:
                milestones_dir = ensure_dir_exists(join(checkpoint_dir, 'milestones'))
                milestone_path = join(milestones_dir, f'{checkpoint_name}.milestone')
                log.debug('Saving a milestone %s', milestone_path)
                shutil.copy(filepath, milestone_path)
                self.last_milestone_time = time.time()

    @staticmethod
    def _policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high):
        clipped_ratio = torch.clamp(ratio, clip_ratio_low, clip_ratio_high)
        loss_unclipped = ratio * adv
        loss_clipped = clipped_ratio * adv
        loss = torch.min(loss_unclipped, loss_clipped)
        loss = -loss.mean()

        return loss

    def _value_loss(self, new_values, old_values, target, clip_value):
        value_clipped = old_values + torch.clamp(new_values - old_values, -clip_value, clip_value)
        value_original_loss = (new_values - target).pow(2)
        value_clipped_loss = (value_clipped - target).pow(2)
        value_loss = torch.max(value_original_loss, value_clipped_loss)
        value_loss = value_loss.mean()
        value_loss *= self.cfg.value_loss_coeff

        return value_loss

    def _prepare_observations(self, obs_tensors, gpu_buffer_obs):
        for d, gpu_d, k, v, _ in iter_dicts_recursively(obs_tensors, gpu_buffer_obs):
            device, dtype = self.actor_critic.device_and_type_for_input_tensor(k)
            tensor = v.detach().to(device, copy=True).type(dtype)
            gpu_d[k] = tensor

    def _copy_train_data_to_gpu(self, buffer):
        gpu_buffer = copy_dict_structure(buffer)

        for key, item in buffer.items():
            if key == 'obs':
                self._prepare_observations(item, gpu_buffer['obs'])
            else:
                gpu_tensor = item.detach().to(self.device, copy=True)
                gpu_buffer[key] = gpu_tensor.float()

        return gpu_buffer

    def _train(self, gpu_buffer, experience_size, timing):
        with torch.no_grad():
            rho_hat = c_hat = 1.0  # V-trace parameters
            # noinspection PyArgumentList
            rho_hat = torch.Tensor([rho_hat])
            # noinspection PyArgumentList
            c_hat = torch.Tensor([c_hat])

            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high

            clip_value = self.cfg.ppo_clip_value
            gamma = self.cfg.gamma
            recurrence = self.cfg.recurrence

            force_summaries = False
            num_sgd_steps = 0

            stats = None
            if not self.with_training:
                return stats

        for epoch in range(self.cfg.ppo_epochs):
            minibatches = self._get_minibatches(experience_size)

            for batch_num in range(len(minibatches)):
                indices = minibatches[batch_num]

                # current minibatch consisting of short trajectory segments with length == recurrence
                mb = self._get_minibatch(gpu_buffer, indices)

                # calculate policy head outside of recurrent loop
                with timing.add_time('forward_head'):
                    head_outputs = self.actor_critic.forward_head(mb.obs)

                # initial rnn states
                with timing.add_time('bptt_initial'):
                    rnn_states = mb.rnn_states[::recurrence]
                    is_same_episode = 1.0 - mb.dones.unsqueeze(dim=1)

                # calculate RNN outputs for each timestep in a loop
                with timing.add_time('bptt'):
                    core_outputs = []
                    for i in range(recurrence):
                        # indices of head outputs corresponding to the current timestep
                        step_head_outputs = head_outputs[i::recurrence]

                        with timing.add_time('bptt_forward_core'):
                            core_output, rnn_states = self.actor_critic.forward_core(step_head_outputs, rnn_states)
                            core_outputs.append(core_output)

                        if self.cfg.use_rnn:
                            # zero-out RNN states on the episode boundary
                            with timing.add_time('bptt_rnn_states'):
                                is_same_episode_step = is_same_episode[i::recurrence]
                                rnn_states.mul_(is_same_episode_step)

                with timing.add_time('tail'):
                    # transform core outputs from [T, Batch, D] to [Batch, T, D] and then to [Batch x T, D]
                    # which is the same shape as the minibatch
                    core_outputs = torch.stack(core_outputs)

                    num_timesteps, num_trajectories = core_outputs.shape[:2]
                    assert num_timesteps == recurrence
                    assert num_timesteps * num_trajectories == self.cfg.batch_size
                    core_outputs = core_outputs.transpose(0, 1).reshape(-1, *core_outputs.shape[2:])
                    assert core_outputs.shape[0] == head_outputs.shape[0]

                    # calculate policy tail outside of recurrent loop
                    result = self.actor_critic.forward_tail(core_outputs, with_action_distribution=True)

                    action_distribution = result.action_distribution
                    log_prob_actions = action_distribution.log_prob(mb.actions)
                    ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

                    values = result.values.squeeze()

                with torch.no_grad():  # these computations are not the part of the computation graph
                    if self.cfg.with_vtrace:
                        ratios_cpu = ratio.cpu()
                        values_cpu = values.cpu()
                        rewards_cpu = mb.rewards.cpu()  # we only need this on CPU, potential minor optimization
                        dones_cpu = mb.dones.cpu()

                        vtrace_rho = torch.min(rho_hat, ratios_cpu)
                        vtrace_c = torch.min(c_hat, ratios_cpu)

                        vs = torch.zeros((num_trajectories * recurrence))
                        adv = torch.zeros((num_trajectories * recurrence))

                        next_values = (values_cpu[recurrence - 1::recurrence] - rewards_cpu[recurrence - 1::recurrence]) / gamma
                        next_vs = next_values

                        with timing.add_time('vtrace'):
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

                    adv_mean = adv.mean()
                    adv_std = adv.std()
                    adv = (adv - adv_mean) / max(1e-2, adv_std)  # normalize advantage
                    adv = adv.to(self.device)

                with timing.add_time('losses'):
                    policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high)

                    targets = targets.to(self.device)
                    old_values = mb.values
                    value_loss = self._value_loss(values, old_values, targets, clip_value)

                    # entropy loss
                    entropy = action_distribution.entropy()
                    entropy_loss = -self.cfg.entropy_loss_coeff * entropy.mean()

                    loss = policy_loss + value_loss + entropy_loss

                    high_loss = 5.0
                    if abs(policy_loss.item()) > high_loss or abs(value_loss.item()) > high_loss or abs(entropy_loss.item()) > high_loss:
                        log.warning('High loss value: %.4f %.4f %.4f %.4f', loss.item(), policy_loss.item(), value_loss.item(), entropy_loss.item())
                        force_summaries = True

                with timing.add_time('update'):
                    # update the weights
                    self.optimizer.zero_grad()
                    loss.backward()

                    if self.cfg.max_grad_norm > 0.0:
                        with timing.add_time('clip'):
                            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)

                    curr_policy_version = self.train_step  # policy version before the weight update
                    with self.policy_lock:
                        self.optimizer.step()

                    num_sgd_steps += 1

                with torch.no_grad():
                    self._after_optimizer_step()

                    # collect and report summaries
                    with_summaries = self._should_save_summaries()
                    if with_summaries or force_summaries:
                        self.last_summary_time = time.time()
                        stats = AttrDict()

                        grad_norm = sum(
                            p.grad.data.norm(2).item() ** 2
                            for p in self.actor_critic.parameters()
                            if p.grad is not None
                        ) ** 0.5
                        stats.grad_norm = grad_norm
                        stats.loss = loss
                        stats.value = result.values.mean()
                        stats.entropy = action_distribution.entropy().mean()
                        stats.policy_loss = policy_loss
                        stats.value_loss = value_loss
                        stats.entropy_loss = entropy_loss
                        stats.adv_min = adv.min()
                        stats.adv_max = adv.max()
                        stats.max_abs_logprob = torch.abs(mb.action_logits).max()

                        if epoch == self.cfg.ppo_epochs - 1 and batch_num == len(minibatches) - 1:
                            # we collect these stats only for the last PPO batch, or every time if we're only doing
                            # one batch, IMPALA-style
                            ratio_mean = torch.abs(1.0 - ratio).mean().detach()
                            ratio_min = ratio.min().detach()
                            ratio_max = ratio.max().detach()
                            # log.debug('Learner %d ratio mean min max %.4f %.4f %.4f', self.policy_id, ratio_mean.cpu().item(), ratio_min.cpu().item(), ratio_max.cpu().item())

                            value_delta = torch.abs(values - old_values)
                            value_delta_avg, value_delta_max = value_delta.mean(), value_delta.max()

                            # calculate KL-divergence with the behaviour policy action distribution
                            old_action_distribution = get_action_distribution(
                                self.actor_critic.action_space, mb.action_logits,
                            )
                            kl_old = action_distribution.kl_divergence(old_action_distribution)
                            kl_old_mean = kl_old.mean()

                            stats.kl_divergence = kl_old_mean
                            stats.value_delta = value_delta_avg
                            stats.value_delta_max = value_delta_max
                            stats.fraction_clipped = ((ratio < clip_ratio_low).float() + (ratio > clip_ratio_high).float()).mean()
                            stats.ratio_mean = ratio_mean
                            stats.ratio_min = ratio_min
                            stats.ratio_max = ratio_max
                            stats.num_sgd_steps = num_sgd_steps

                        # this caused numerical issues on some versions of PyTorch with second moment getting to infinity
                        adam_max_second_moment = 0.0
                        for key, tensor_state in self.optimizer.state.items():
                            adam_max_second_moment = max(tensor_state['exp_avg_sq'].max().item(), adam_max_second_moment)
                        stats.adam_max_second_moment = adam_max_second_moment

                        version_diff = curr_policy_version - mb.policy_version
                        stats.version_diff_avg = version_diff.mean()
                        stats.version_diff_min = version_diff.min()
                        stats.version_diff_max = version_diff.max()

                        # we want this statistic for the last batch of the last epoch
                        for key, value in stats.items():
                            if isinstance(value, torch.Tensor):
                                stats[key] = value.detach().cpu()

        return stats

    def _update_pbt(self):
        """To be called from the training loop, same thread that updates the model!"""
        if self.load_policy_id is not None:
            assert self.cfg.with_pbt

            log.debug('Learner %d loads policy from %d', self.policy_id, self.load_policy_id)
            self.load_from_checkpoint(self.load_policy_id)
            self.load_policy_id = None

        if self.new_cfg is not None:
            for key, value in self.new_cfg.items():
                if self.cfg[key] != value:
                    log.debug('Learner %d replacing cfg parameter %r with new value %r', self.policy_id, key, value)
                    self.cfg[key] = value

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.cfg.learning_rate
                param_group['betas'] = (self.cfg.adam_beta1, self.cfg.adam_beta2)
                log.debug('Updated optimizer lr to value %.7f, betas: %r', param_group['lr'], param_group['betas'])

            self.new_cfg = None

    @staticmethod
    def load_checkpoint(checkpoints, device):
        if len(checkpoints) <= 0:
            log.warning('No checkpoints found')
            return None
        else:
            latest_checkpoint = checkpoints[-1]
            log.warning('Loading state from checkpoint %s...', latest_checkpoint)
            checkpoint_dict = torch.load(latest_checkpoint, map_location=device)
            return checkpoint_dict

    def _load_state(self, checkpoint_dict, load_progress=True):
        if load_progress:
            self.train_step = checkpoint_dict['train_step']
            self.env_steps = checkpoint_dict['env_steps']
        self.actor_critic.load_state_dict(checkpoint_dict['model'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        log.info('Loaded experiment state at training iteration %d, env step %d', self.train_step, self.env_steps)

    def init_model(self, timing):
        self.actor_critic = create_actor_critic(self.cfg, self.obs_space, self.action_space, timing)
        self.actor_critic.model_to_device(self.device)
        self.actor_critic.share_memory()

    def load_from_checkpoint(self, policy_id):
        checkpoints = self.get_checkpoints(self.checkpoint_dir(self.cfg, policy_id))
        checkpoint_dict = self.load_checkpoint(checkpoints, self.device)
        if checkpoint_dict is None:
            log.debug('Did not load from checkpoint, starting from scratch!')
        else:
            log.debug('Loading model from checkpoint')

            # if we're replacing our policy with another policy (under PBT), let's not reload the env_steps
            load_progress = policy_id == self.policy_id
            self._load_state(checkpoint_dict, load_progress=load_progress)

    def initialize(self, timing):
        with timing.timeit('init'):
            # initialize the Torch modules
            if self.cfg.seed is None:
                log.info('Starting seed is not provided')
            else:
                log.info('Setting fixed seed %d', self.cfg.seed)
                torch.manual_seed(self.cfg.seed)
                np.random.seed(self.cfg.seed)

            # this does not help with a single experiment
            # but seems to do better when we're running more than one experiment in parallel
            torch.set_num_threads(1)
            torch.backends.cudnn.benchmark = True

            # we should already see only one CUDA device, because of env vars
            assert torch.cuda.device_count() == 1
            self.device = torch.device('cuda', index=0)
            self.init_model(timing)

            self.optimizer = torch.optim.Adam(
                self.actor_critic.parameters(),
                self.cfg.learning_rate,
                betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
                eps=self.cfg.adam_eps,
            )

            self.load_from_checkpoint(self.policy_id)

            self._broadcast_model_weights()  # sync the very first version of the weights

        self.train_thread_initialized.set()

    def _process_training_data(self, data, timing, wait_stats=None):
        buffer, samples, env_steps = data
        self.env_steps += env_steps
        experience_size = buffer.rewards.shape[0]

        log.error('Learner env steps per batch: %d (Total: %d)', env_steps, self.env_steps)  # TODO: remove!
        stats = dict(learner_env_steps=self.env_steps, policy_id=self.policy_id)

        with timing.add_time('train'):
            discarding_rate = self._discarding_rate()

            self._update_pbt()

            # log.debug('Training policy %d on %d samples', self.policy_id, samples)
            train_stats = self._train(buffer, experience_size, timing)
            self.policy_versions[self.policy_id] = self.train_step

            if train_stats is not None:
                stats['train'] = train_stats

                if wait_stats is not None:
                    wait_avg, wait_min, wait_max = wait_stats
                    stats['train']['wait_avg'] = wait_avg
                    stats['train']['wait_min'] = wait_min
                    stats['train']['wait_max'] = wait_max

                stats['train']['discarded_rollouts'] = self.num_discarded_rollouts
                stats['train']['discarding_rate'] = discarding_rate

                stats['stats'] = memory_stats('learner', self.device)

        log.error('Learner reporting stats %r', stats)  # TODO: remove!
        self.report_queue.put(stats)

    def _train_loop(self):
        timing = Timing()
        self.initialize(timing)

        wait_times = deque([], maxlen=self.cfg.num_workers)
        last_cache_cleanup = time.time()
        num_batches_processed = 0

        while not self.terminate:
            with timing.timeit('train_wait'):
                data = safe_get(self.experience_buffer_queue)

            with self.processing_experience_batch:
                self.processing_experience_batch.notify_all()

            if self.terminate:
                break

            wait_stats = None
            wait_times.append(timing.train_wait)

            if len(wait_times) >= wait_times.maxlen:
                wait_times_arr = np.asarray(wait_times)
                wait_avg = np.mean(wait_times_arr)
                wait_min, wait_max = wait_times_arr.min(), wait_times_arr.max()
                # log.debug(
                #     'Training thread had to wait %.5f s for the new experience buffer (avg %.5f)',
                #     timing.train_wait, wait_avg,
                # )
                wait_stats = (wait_avg, wait_min, wait_max)

            self._process_training_data(data, timing, wait_stats)
            num_batches_processed += 1

            if time.time() - last_cache_cleanup > 300.0 or (not self.cfg.benchmark and num_batches_processed < 50):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                last_cache_cleanup = time.time()

        time.sleep(0.3)
        log.info('Train loop timing: %s', timing)
        del self.actor_critic
        del self.device

    def _experience_collection_rate_stats(self):
        now = time.time()
        if now - self.discarded_experience_timer > 1.0:
            self.discarded_experience_timer = now
            self.discarded_experience_over_time.append((now, self.num_discarded_rollouts))

    def _discarding_rate(self):
        if len(self.discarded_experience_over_time) <= 0:
            return 0

        first, last = self.discarded_experience_over_time[0], self.discarded_experience_over_time[-1]
        delta_rollouts = last[1] - first[1]
        delta_time = last[0] - first[0]
        discarding_rate = delta_rollouts / delta_time
        return discarding_rate

    def _extract_rollouts(self, data):
        data = AttrDict(data)
        worker_idx, split_idx, traj_buffer_idx = data.worker_idx, data.split_idx, data.traj_buffer_idx

        rollouts = []
        for rollout_data in data.rollouts:
            env_idx, agent_idx = rollout_data['env_idx'], rollout_data['agent_idx']
            tensors = self.rollout_tensors.index((worker_idx, split_idx, env_idx, agent_idx, traj_buffer_idx))

            rollout_data['t'] = tensors
            rollout_data['worker_idx'] = worker_idx
            rollout_data['split_idx'] = split_idx
            rollout_data['traj_buffer_idx'] = traj_buffer_idx
            rollouts.append(AttrDict(rollout_data))

        return rollouts

    def _process_pbt_task(self, pbt_task):
        task_type, data = pbt_task

        if task_type == PbtTask.SAVE_MODEL:
            policy_id = data
            assert policy_id == self.policy_id
            self.should_save_model = True
        elif task_type == PbtTask.LOAD_MODEL:
            policy_id, new_policy_id = data
            assert policy_id == self.policy_id
            assert new_policy_id is not None
            self.load_policy_id = new_policy_id
        elif task_type == PbtTask.UPDATE_CFG:
            policy_id, new_cfg = data
            assert policy_id == self.policy_id
            self.new_cfg = new_cfg

    def _run(self):
        # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        try:
            psutil.Process().nice(self.cfg.default_niceness)
        except psutil.AccessDenied:
            log.error('Low niceness requires sudo!!!')

        cuda_envvars(self.policy_id)
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.set_num_threads(1)

        timing = Timing()

        rollouts = []

        if self.train_in_background:
            self.training_thread.start()
        else:
            self.initialize(timing)
            log.error(
                'train_in_background set to False on learner %d! This is slow, use only for testing!', self.policy_id,
            )

        while not self.terminate:
            while True:
                try:
                    tasks = self.task_queue.get_many(timeout=0.005)

                    for task_type, data in tasks:
                        if task_type == TaskType.TRAIN:
                            with timing.add_time('extract'):
                                rollouts.extend(self._extract_rollouts(data))
                                # log.debug('Learner %d has %d rollouts', self.policy_id, len(rollouts))
                        elif task_type == TaskType.INIT:
                            self._init()
                        elif task_type == TaskType.TERMINATE:
                            time.sleep(0.3)
                            log.info('GPU learner timing: %s', timing)
                            self._terminate()
                            break
                        elif task_type == TaskType.PBT:
                            self._process_pbt_task(data)
                except Empty:
                    break

            while self.experience_buffer_queue.qsize() >= 1:
                with self.processing_experience_batch:
                    self.processing_experience_batch.wait(timeout=0.01)

            with torch.no_grad():
                rollouts = self._process_rollouts(rollouts, timing)

            if not self.train_in_background:
                while not self.experience_buffer_queue.empty():
                    training_data = self.experience_buffer_queue.get()
                    with self.processing_experience_batch:
                        self.processing_experience_batch.notify_all()
                    self._process_training_data(training_data, timing)

            self._experience_collection_rate_stats()

        if self.train_in_background:
            self.experience_buffer_queue.put(None)
            self.training_thread.join()

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.initialized_event.wait()

    def save_model(self, timeout=None):
        self.model_saved_event.clear()
        save_task = (PbtTask.SAVE_MODEL, self.policy_id)
        self.task_queue.put((TaskType.PBT, save_task))
        log.debug('Wait while learner %d saves the model...', self.policy_id)
        if self.model_saved_event.wait(timeout=timeout):
            log.debug('Learner %d saved the model!', self.policy_id)
        else:
            log.warning('Model saving request timed out!')
        self.model_saved_event.clear()

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        join_or_kill(self.process)


# WITHOUT TRAINING:
# [2019-11-27 22:06:02,056] Gpu learner timing: init: 3.1058, work: 0.0001
# [2019-11-27 22:06:02,059] Gpu worker timing: init: 2.7746, deserialize: 4.6964, to_device: 3.8011, forward: 14.2683, serialize: 8.4691, postprocess: 9.8058, policy_step: 32.8482, weight_update: 0.0005, gpu_waiting: 2.0623
# [2019-11-27 22:06:02,065] Gpu worker timing: init: 5.4169, deserialize: 3.6640, to_device: 3.1592, forward: 13.2836, serialize: 6.3964, postprocess: 7.6095, policy_step: 27.9706, weight_update: 0.0005, gpu_waiting: 1.8249
# [2019-11-27 22:06:02,067] Env runner 0: timing waiting: 0.8708, reset: 27.0515, save_policy_outputs: 0.0006, env_step: 26.0700, finalize: 0.3460, overhead: 1.1313, format_output: 0.3095, one_step: 0.0272, work: 36.5773
# [2019-11-27 22:06:02,079] Env runner 1: timing waiting: 0.8468, reset: 26.8022, save_policy_outputs: 0.0008, env_step: 26.1251, finalize: 0.3565, overhead: 1.1361, format_output: 0.3224, one_step: 0.0269, work: 36.6139

# WITH TRAINING 1 epoch:
# [2019-11-27 22:24:20,590] Gpu worker timing: init: 2.9078, deserialize: 5.5495, to_device: 5.6693, forward: 15.7285, serialize: 10.0113, postprocess: 13.4533, policy_step: 40.7373, weight_update: 0.0007, gpu_waiting: 2.0482
# [2019-11-27 22:24:20,596] Gpu worker timing: init: 4.8333, deserialize: 4.6056, to_device: 5.0975, forward: 14.8585, serialize: 8.0576, postprocess: 11.3531, policy_step: 36.2226, weight_update: 0.0006, gpu_waiting: 1.9836
# [2019-11-27 22:24:20,606] Env runner 1: timing waiting: 0.9328, reset: 27.9299, save_policy_outputs: 0.0005, env_step: 31.6222, finalize: 0.4432, overhead: 1.3904, format_output: 0.3692, one_step: 0.0309, work: 44.7151
# [2019-11-27 22:24:20,622] Env runner 0: timing waiting: 1.0276, reset: 27.5389, save_policy_outputs: 0.0009, env_step: 31.5377, finalize: 0.4614, overhead: 1.4103, format_output: 0.3564, one_step: 0.0269, work: 44.6398
# [2019-11-27 22:24:23,072] Gpu learner timing: init: 3.3635, last_values: 0.4506, gae: 3.5159, numpy: 0.6232, finalize: 4.6129, buffer: 6.4776, update: 16.3922, train: 26.0528, work: 37.2159
# [2019-11-27 22:24:52,618] Collected 1012576, FPS: 22177.3

# Env runner 0: timing waiting: 2.5731, reset: 5.0527, save_policy_outputs: 0.0007, env_step: 28.7689, overhead: 1.1565, format_inputs: 0.3170, one_step: 0.0276, work: 39.3452
# [2019-12-06 19:01:42,042] Env runner 1: timing waiting: 2.5900, reset: 4.9147, save_policy_outputs: 0.0004, env_step: 28.8585, overhead: 1.1266, format_inputs: 0.3087, one_step: 0.0254, work: 39.3333
# [2019-12-06 19:01:42,227] Gpu worker timing: init: 2.8738, weight_update: 0.0006, deserialize: 7.6602, to_device: 5.3244, forward: 8.1527, serialize: 14.3651, postprocess: 17.5523, policy_step: 38.8745, gpu_waiting: 0.5276
# [2019-12-06 19:01:42,232] Gpu learner timing: init: 3.3448, last_values: 0.2737, gae: 3.0682, numpy: 0.5308, finalize: 3.8888, buffer: 5.2451, forw_head: 0.2639, forw_core: 0.8289, forw_tail: 0.5334, clip: 4.5709, update: 12.0888, train: 19.6720, work: 28.8663
# [2019-12-06 19:01:42,723] Collected 1007616, FPS: 23975.2

# Last version using Plasma:
# [2020-01-07 00:24:27,690] Env runner 0: timing wait_actor: 0.0001, waiting: 2.2242, reset: 13.0768, save_policy_outputs: 0.0004, env_step: 27.5735, overhead: 1.0524, format_inputs: 0.2934, enqueue_policy_requests: 4.6075, complete_rollouts: 3.2226, one_step: 0.0250, work: 37.9023
# [2020-01-07 00:24:27,697] Env runner 1: timing wait_actor: 0.0042, waiting: 2.2486, reset: 13.3085, save_policy_outputs: 0.0005, env_step: 27.5389, overhead: 1.0206, format_inputs: 0.2921, enqueue_policy_requests: 4.5829, complete_rollouts: 3.3319, one_step: 0.0240, work: 37.8813
# [2020-01-07 00:24:27,890] Gpu worker timing: init: 3.0419, wait_policy: 0.0002, gpu_waiting: 0.4060, weight_update: 0.0007, deserialize: 0.0923, to_device: 4.7866, forward: 6.8820, serialize: 13.8782, postprocess: 16.9365, policy_step: 28.8341, one_step: 0.0000, work: 39.9577
# [2020-01-07 00:24:27,906] GPU learner timing: buffers: 0.0461, tensors: 8.7751, prepare: 8.8510
# [2020-01-07 00:24:27,907] Train loop timing: init: 3.0417, train_wait: 0.0969, bptt: 2.6350, vtrace: 5.7421, losses: 0.7799, clip: 4.6204, update: 9.1475, train: 21.3880
# [2020-01-07 00:24:28,213] Collected {0: 1015808}, FPS: 25279.4
# [2020-01-07 00:24:28,214] Timing: experience: 40.1832

# Version using Pytorch tensors with shared memory:
# [2020-01-07 01:08:05,569] Env runner 0: timing wait_actor: 0.0003, waiting: 0.6292, reset: 12.4041, save_policy_outputs: 0.4311, env_step: 30.1347, overhead: 4.3134, enqueue_policy_requests: 0.0677, complete_rollouts: 0.0274, one_step: 0.0261, work: 35.3962, wait_buffers: 0.0164
# [2020-01-07 01:08:05,596] Env runner 1: timing wait_actor: 0.0003, waiting: 0.7102, reset: 12.7194, save_policy_outputs: 0.4400, env_step: 30.1091, overhead: 4.2822, enqueue_policy_requests: 0.0630, complete_rollouts: 0.0234, one_step: 0.0270, work: 35.3405, wait_buffers: 0.0162
# [2020-01-07 01:08:05,762] Gpu worker timing: init: 2.8383, wait_policy: 0.0000, gpu_waiting: 2.3759, loop: 4.3098, weight_update: 0.0006, updates: 0.0008, deserialize: 0.8207, to_device: 6.8636, forward: 15.0019, postprocess: 2.4855, handle_policy_step: 29.5612, one_step: 0.0000, work: 33.9772
# [2020-01-07 01:08:05,896] Train loop timing: init: 2.9927, train_wait: 0.0001, bptt: 2.6755, vtrace: 6.3307, losses: 0.7319, update: 4.6164, train: 22.0022
# [2020-01-07 01:08:10,888] Collected {0: 1015808}, FPS: 28900.6
# [2020-01-07 01:08:10,888] Timing: experience: 35.1483

# Version V53, Torch 1.3.1
# [2020-01-09 20:33:23,540] Env runner 0: timing wait_actor: 0.0002, waiting: 0.7097, reset: 5.2281, save_policy_outputs: 0.3789, env_step: 29.3372, overhead: 4.2642, enqueue_policy_requests: 0.0660, complete_rollouts: 0.0313, one_step: 0.0244, work: 34.5037, wait_buffers: 0.0213
# [2020-01-09 20:33:23,556] Env runner 1: timing wait_actor: 0.0009, waiting: 0.6965, reset: 5.3100, save_policy_outputs: 0.3989, env_step: 29.3533, overhead: 4.2378, enqueue_policy_requests: 0.0685, complete_rollouts: 0.0290, one_step: 0.0261, work: 34.5326, wait_buffers: 0.0165
# [2020-01-09 20:33:23,711] Gpu worker timing: init: 1.3378, wait_policy: 0.0016, gpu_waiting: 2.3035, loop: 4.5320, weight_update: 0.0006, updates: 0.0008, deserialize: 0.8223, to_device: 6.4952, forward: 14.8064, postprocess: 2.4568, handle_policy_step: 28.7065, one_step: 0.0000, work: 33.3578
# [2020-01-09 20:33:23,816] GPU learner timing: extract: 0.0137, buffers: 0.0437, tensors: 6.6962, buff_ready: 0.1400, prepare: 6.9068
# [2020-01-09 20:33:23,892] Train loop timing: init: 1.3945, train_wait: 0.0000, bptt: 2.2262, vtrace: 5.5308, losses: 0.6580, update: 3.6261, train: 19.8292
# [2020-01-09 20:33:28,787] Collected {0: 1015808}, FPS: 29476.0
# [2020-01-09 20:33:28,787] Timing: experience: 34.4622

# Version V60
# [2020-01-19 03:25:14,014] Env runner 0: timing wait_actor: 0.0001, waiting: 9.7151, reset: 41.1152, save_policy_outputs: 0.5734, env_step: 39.1791, overhead: 6.5181, enqueue_policy_requests: 0.1089, complete_rollouts: 0.2901, one_step: 0.0163, work: 47.2741, wait_buffers: 0.2795
# [2020-01-19 03:25:14,015] Env runner 1: timing wait_actor: 0.0001, waiting: 10.1184, reset: 41.6788, save_policy_outputs: 0.5846, env_step: 39.1234, overhead: 6.4405, enqueue_policy_requests: 0.1021, complete_rollouts: 0.0304, one_step: 0.0154, work: 46.8807, wait_buffers: 0.0202
# [2020-01-19 03:25:14,178] Updated weights on worker 0, policy_version 251 (0.00032)
# [2020-01-19 03:25:14,201] Gpu worker timing: init: 1.3160, wait_policy: 0.0009, gpu_waiting: 9.5548, loop: 9.7118, weight_update: 0.0003, updates: 0.0005, deserialize: 1.5404, to_device: 12.7886, forward: 12.9712, postprocess: 4.9893, handle_policy_step: 37.9686, one_step: 0.0000, work: 47.9418
# [2020-01-19 03:25:14,221] GPU learner timing: extract: 0.0392, buffers: 0.0745, tensors: 11.0697, buff_ready: 0.4808, prepare: 11.7095
# [2020-01-19 03:25:14,321] Train loop timing: init: 1.4332, train_wait: 0.0451, tensors_gpu_float: 4.3031, bptt: 5.0880, vtrace: 2.4773, losses: 1.9113, update: 7.6270, train: 36.8291
# [2020-01-19 03:25:14,465] Collected {0: 2015232}, FPS: 35779.2
# [2020-01-19 03:25:14,465] Timing: experience: 56.3241

# Version V61, cudnn benchmark=True
# [2020-01-19 18:19:31,416] Env runner 0: timing wait_actor: 0.0002, waiting: 8.8857, reset: 41.9806, save_policy_outputs: 0.5918, env_step: 38.3737, overhead: 6.3290, enqueue_policy_requests: 0.1026, complete_rollouts: 0.0286, one_step: 0.0141, work: 46.0301, wait_buffers: 0.0181
# [2020-01-19 18:19:31,420] Env runner 1: timing wait_actor: 0.0002, waiting: 9.0225, reset: 42.5019, save_policy_outputs: 0.5540, env_step: 38.1044, overhead: 6.2374, enqueue_policy_requests: 0.1140, complete_rollouts: 0.2770, one_step: 0.0169, work: 45.8830, wait_buffers: 0.2664
# [2020-01-19 18:19:31,472] Updated weights on worker 0, policy_version 245 (0.00051)
# [2020-01-19 18:19:31,587] Updated weights on worker 0, policy_version 246 (0.00053)
# [2020-01-19 18:19:31,610] Gpu worker timing: init: 1.3633, wait_policy: 0.0037, gpu_waiting: 9.4391, loop: 9.6261, weight_update: 0.0005, updates: 0.0007, deserialize: 1.4722, to_device: 12.5683, forward: 12.8369, postprocess: 4.9932, handle_policy_step: 36.1579, one_step: 0.0000, work: 45.9985
# [2020-01-19 18:19:31,624] GPU learner timing: extract: 0.0376, buffers: 0.0769, tensors: 11.2689, buff_ready: 0.4423, prepare: 11.8845
# [2020-01-19 18:19:31,630] Train loop timing: init: 1.4804, train_wait: 0.0481, tensors_gpu_float: 4.1565, bptt: 5.2692, vtrace: 2.2177, losses: 1.7225, update: 7.5387, train: 31.5856
# [2020-01-19 18:19:31,797] Collected {0: 1966080}, FPS: 36238.5
# [2020-01-19 18:19:31,797] Timing: experience: 54.2540

# Version V64
# --env=doom_battle_hybrid --train_for_seconds=360000 --algo=APPO --env_frameskip=4 --use_rnn=True --reward_scale=0.5 --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --experiment=doom_battle_appo_v64_test --benchmark=True --res_w=128 --res_h=72 --wide_aspect_ratio=True --policy_workers_per_policy=1 --worker_num_splits=2 --init_workers_parallel=7 --max_grad_norm=0.0
# [2020-01-25 22:44:52,845] Env runner 1: timing wait_actor: 0.0068, waiting: 9.9934, reset: 16.0501, save_policy_outputs: 0.5885, env_step: 38.8288, overhead: 6.5314, enqueue_policy_requests: 0.1232, complete_rollouts: 0.0299, one_step: 0.0167, work: 46.7084, wait_buffers: 0.0195
# [2020-01-25 22:44:52,846] Env runner 0: timing wait_actor: 0.0002, waiting: 9.6433, reset: 14.8835, save_policy_outputs: 0.5988, env_step: 39.0076, overhead: 6.6748, enqueue_policy_requests: 0.1294, complete_rollouts: 0.0318, one_step: 0.0167, work: 47.0693, wait_buffers: 0.0211
# [2020-01-25 22:44:53,037] Gpu worker timing: init: 1.3123, wait_policy: 0.0024, gpu_waiting: 9.7236, loop: 10.5022, weight_update: 0.0005, updates: 0.0007, deserialize: 1.5961, to_device: 12.5846, forward: 13.2160, postprocess: 5.1388, handle_policy_step: 36.8111, one_step: 0.0000, work: 47.4999
# [2020-01-25 22:44:53,048] GPU learner timing: extract: 0.0329, buffers: 0.0760, tensors: 11.2344, buff_ready: 0.4467, prepare: 11.8263
# [2020-01-25 22:44:53,060] Train loop timing: init: 1.4357, train_wait: 0.1186, tensors_gpu_float: 4.1724, bptt: 5.2798, vtrace: 2.4177, losses: 1.8281, update: 7.7311, train: 32.4878
# [2020-01-25 22:44:53,219] Collected {0: 2015232}, FPS: 35969.4
# [2020-01-25 22:44:53,219] Timing: experience: 56.0263

# Version V66
# --env=doom_benchmark --train_for_seconds=360000 --algo=APPO --env_frameskip=4 --use_rnn=True  --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --experiment=doom_battle_appo_v66_test3 --benchmark=True --res_w=128 --res_h=72 --wide_aspect_ratio=True --policy_workers_per_policy=1 --worker_num_splits=2 --init_workers_parallel=7
# [2020-02-05 02:21:08,568][06063] Env runner 0, rollouts 780: timing wait_actor: 0.0002, waiting: 7.0481, reset: 9.3021, save_policy_outputs: 0.5583, env_step: 34.4028, overhead: 6.0476, complete_rollouts: 0.3592, enqueue_policy_requests: 0.1203, one_step: 0.0192, work: 42.1171, wait_buffers: 0.3469
# [2020-02-05 02:21:08,596][04810] Env runner 1, rollouts 770: timing wait_actor: 0.0001, waiting: 7.4001, reset: 23.0733, save_policy_outputs: 0.5752, env_step: 34.1180, overhead: 5.8824, complete_rollouts: 0.4621, enqueue_policy_requests: 0.1337, one_step: 0.0091, work: 41.799, wait_buffers: 0.4502
# [2020-02-05 02:21:08,764][04801] Policy worker timing: init: 1.4682, wait_policy: 0.0029, gpu_waiting: 101.1209, weight_update: 0.0003, updates: 0.0004, loop: 9.0389, handle_policy_step: 29.3456, one_step: 0.0000, work: 38.6256, deserialize: 1.2242, to_device: 10.7996, forward: 6.7556, postprocess: 7.2947
# [2020-02-05 02:21:08,780][04774] GPU learner timing: extract: 0.0555, buffers: 0.0706, tensors: 12.0109, buff_ready: 0.5418, prepare: 12.6917
# [2020-02-05 02:21:09,082][04715] Collected {0: 1974272}, FPS: 40185.7
# [2020-02-05 02:21:09,082][04715] Timing: experience: 49.1287

# Version V66
# python -m algorithms.appo.train_appo --env=doom_benchmark --algo=APPO --env_frameskip=4 --use_rnn=True  --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --experiment=doom_battle_appo_v66_test --benchmark=True --res_w=128 --res_h=72 --wide_aspect_ratio=True --policy_workers_per_policy=1 --worker_num_splits=2
# [2020-02-11 20:59:56,337][28791] Env runner 0, rollouts 800: timing wait_actor: 0.0002, waiting: 6.1882, reset: 15.3720, save_policy_outputs: 0.5901, env_step: 34.3711, overhead: 6.1186, complete_rollouts: 0.6102, enqueue_policy_requests: 0.1157, one_step: 0.0151, work: 42.4098, wait_buffers: 0.5976
# [2020-02-11 20:59:56,293][28793] Env runner 1, rollouts 790: timing wait_actor: 0.0022, waiting: 6.3542, reset: 14.9161, save_policy_outputs: 0.5772, env_step: 34.3059, overhead: 6.0034, complete_rollouts: 0.6385, enqueue_policy_requests: 0.1143, one_step: 0.0155, work: 42.2383, wait_buffers: 0.6263
# [2020-02-11 20:59:56,322][28790] Policy worker timing: init: 1.9307, wait_policy: 0.0000, gpu_waiting: 25.3407, weight_update: 0.0004, updates: 0.0007, loop: 8.7840, handle_policy_step: 29.2553, one_step: 0.0023, work: 38.2444, deserialize: 1.1606, to_device: 10.8993, forward: 6.6921, postprocess: 7.1333
# [2020-02-11 20:59:56,358][28767] GPU learner timing: extract: 0.0488, buffers: 0.0732, tensors: 12.1752, buff_ready: 0.4165, prepare: 12.7439
# [2020-02-11 20:59:56,391][28767] Train loop timing: init: 1.3657, train_wait: 0.0968, tensors_gpu_float: 4.1287, bptt: 5.5166, vtrace: 2.6644, losses: 0.6868, clip: 6.1476, update: 12.7627, train: 31.3387
# [2020-02-11 20:59:56,606][28705] Collected {0: 2015232}, FPS: 41630.1
# [2020-02-11 20:59:56,606][28705] Timing: experience: 48.4080

# Version V69 (numpy arrays with dtype=object to access shared memory, on all components)
# python -m algorithms.appo.train_appo --env=doom_benchmark --algo=APPO --env_frameskip=4 --use_rnn=True  --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --experiment=doom_battle_appo_v69_test --benchmark=True --res_w=128 --res_h=72 --wide_aspect_ratio=True --policy_workers_per_policy=1 --worker_num_splits=2
# [2020-03-13 02:01:33,655][11226] Env runner 0, rollouts 800: timing wait_actor: 0.0001, waiting: 6.8923, reset: 9.7055, save_policy_outputs: 1.1013, env_step: 34.8643, overhead: 2.6461, complete_rollouts: 0.0163, enqueue_policy_requests: 0.1268, one_step: 0.0149, work: 40.6362, wait_buffers: 0.0416
# [2020-03-13 02:01:33,717][11228] Env runner 1, rollouts 780: timing wait_actor: 0.0056, waiting: 7.2380, reset: 12.4794, save_policy_outputs: 1.1331, env_step: 34.3026, overhead: 2.6869, complete_rollouts: 0.0106, enqueue_policy_requests: 0.1320, one_step: 0.0149, work: 40.2842, wait_buffers: 0.2142
# [2020-03-13 02:01:33,701][11225] Policy worker timing: init: 1.7734, wait_policy: 0.0050, gpu_waiting: 23.6931, weight_update: 0.0005, updates: 0.0007, loop: 9.1436, handle_policy_step: 28.1835, one_step: 0.0000, work: 37.5288, deserialize: 1.2307, to_device: 11.0009, forward: 6.2592, postprocess: 6.3423
# [2020-03-13 02:01:33,711][11207] GPU learner timing: extract: 0.2552, buffers: 0.0665, tensors: 12.2804, buff_ready: 0.4470, prepare: 12.8576
# [2020-03-13 02:01:33,739][11207] Train loop timing: init: 1.3547, train_wait: 0.0001, tensors_gpu_float: 4.7140, bptt: 6.1209, vtrace: 2.3818, losses: 0.7224, clip: 6.4046, update: 13.7435, train: 33.4576
# [2020-03-13 02:01:33,937][11111] Collected {0: 2007040}, FPS: 42423.2
# [2020-03-13 02:01:33,937][11111] Timing: experience: 47.3100

# Version V70 (fast C++ queues)
# python -m algorithms.appo.train_appo --env=doom_benchmark --algo=APPO --env_frameskip=4 --use_rnn=True  --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --experiment=doom_battle_appo_v70_test --benchmark=True --res_w=128 --res_h=72 --wide_aspect_ratio=True --policy_workers_per_policy=1 --worker_num_splits=2
# [2020-03-13 05:41:33,867][27104] Env runner 0, rollouts 810: timing wait_actor: 0.0000, waiting: 4.8948, reset: 12.3130, save_policy_outputs: 1.1351, env_step: 34.4284, overhead: 2.6341, complete_rollouts: 0.0115, enqueue_policy_requests: 0.1613, one_step: 0.0143, work: 40.2222
# [2020-03-13 05:41:33,869][27106] Env runner 1, rollouts 790: timing wait_actor: 0.0000, waiting: 5.6380, reset: 10.6275, save_policy_outputs: 1.0901, env_step: 33.9549, overhead: 2.5067, complete_rollouts: 0.0110, enqueue_policy_requests: 0.1773, one_step: 0.0327, work: 39.5129
# [2020-03-13 05:41:33,829][27103] Policy worker timing: init: 1.7287, wait_policy_total: 16.8390, wait_policy: 0.0022, handle_policy_step: 41.4499, one_step: 0.0043, weight_update: 0.0004, updates: 0.0007, deserialize: 1.4689, to_device: 13.1793, forward: 11.4226, postprocess: 10.6741
# [2020-03-13 05:41:33,839][27085] GPU learner timing: extract: 0.2827, buffers: 0.0705, tensors: 11.6516, buff_ready: 0.5000, prepare: 12.3128
# [2020-03-13 05:41:33,853][27085] Train loop timing: init: 1.3051, train_wait: 0.0000, tensors_gpu_float: 4.5831, bptt: 6.0317, vtrace: 2.4454, losses: 0.7526, clip: 6.2237, update: 13.2196, train: 32.9858
# [2020-03-13 05:41:34,053][26983] Collected {0: 2015232}, FPS: 44822.8
# [2020-03-13 05:41:34,053][26983] Timing: experience: 44.9600

# Version V73 (process priority + to(device) in background thread on the learner)
# Policy #0 lag: (min: 1.0, avg: 4.4, max: 9.0)
# [2020-03-14 00:43:41,047][11371] Env runner 0, rollouts 800: timing wait_actor: 0.0005, waiting: 1.5608, reset: 15.0341, save_policy_outputs: 1.0898, env_step: 35.9769, overhead: 2.7699, complete_rollouts: 0.0159, enqueue_policy_requests: 0.1750, one_step: 0.0156, work: 41.9856
# [2020-03-14 00:43:41,071][11372] Env runner 1, rollouts 780: timing wait_actor: 0.0000, waiting: 1.6715, reset: 15.3323, save_policy_outputs: 1.1191, env_step: 35.8312, overhead: 2.7640, complete_rollouts: 0.0153, enqueue_policy_requests: 0.1592, one_step: 0.0149, work: 41.9102, wait_buffers: 0.1198
# [2020-03-14 00:43:41,310][11370] Policy worker avg. requests 4.36, timing: init: 1.7793, wait_policy_total: 16.5541, wait_policy: 0.0001, handle_policy_step: 41.6576, one_step: 0.0042, deserialize: 1.7135, obs_to_device: 5.2590, stack: 14.9778, forward: 13.9429, postprocess: 5.5726, weight_update: 0.0005
# [2020-03-14 00:43:41,481][11353] Train loop timing: init: 1.3224, train_wait: 0.0000, bptt: 11.2281, vtrace: 1.6551, losses: 0.8895, clip: 6.6278, update: 14.9070, train: 42.3450
# [2020-03-14 00:43:41,782][11353] GPU learner timing: extract: 0.1827, buffers: 0.0695, tensors: 8.7560, buff_ready: 0.3020, tensors_gpu_float: 5.9617, prepare: 15.1426
# [2020-03-14 00:43:41,916][11262] Collected {0: 2015232}, FPS: 46341.4
# [2020-03-14 00:43:41,916][11262] Timing: experience: 43.4867

# Version V76 (Pytorch 1.4, faster indexing, faster batching. Improvements mostly in the learner)
# python -m algorithms.appo.train_appo --env=doom_benchmark --algo=APPO --env_frameskip=4 --use_rnn=True  --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --experiment=doom_battle_appo_v76_test --benchmark=True --res_w=128 --res_h=72 --wide_aspect_ratio=True --policy_workers_per_policy=1 --worker_num_splits=2
# [2020-03-14 05:20:45,029][09278] Env runner 0, rollouts 810: timing wait_actor: 0.0000, waiting: 1.4923, reset: 11.3817, save_policy_outputs: 1.0137, env_step: 36.3330, overhead: 3.8381, complete_rollouts: 0.0150, enqueue_policy_requests: 0.1605, one_step: 0.0148, work: 43.4425
# [2020-03-14 05:20:45,041][09279] Env runner 1, rollouts 780: timing wait_actor: 0.0070, waiting: 1.7666, reset: 13.1651, save_policy_outputs: 1.0001, env_step: 36.2509, overhead: 3.8008, complete_rollouts: 0.0194, enqueue_policy_requests: 0.1582, one_step: 0.0160, work: 43.1787
# [2020-03-14 05:20:45,276][09277] Policy worker avg. requests 2.54, timing: init: 1.7812, wait_policy_total: 17.5261, wait_policy: 0.0022, handle_policy_step: 42.9551, one_step: 0.0022, deserialize: 1.7076, obs_to_device: 5.0701, stack: 14.8308, forward: 14.4700, postprocess: 5.5982, weight_update: 0.0004
# [2020-03-14 05:20:45,383][09251] GPU learner timing: extract: 0.1829, buffers: 0.0667, batching: 5.2437, buff_ready: 0.2342, tensors_gpu_float: 7.8419, squeeze: 0.0156, prepare: 13.4659, batcher_mem: 5.1972
# [2020-03-14 05:20:45,689][09251] Train loop timing: init: 1.3501, train_wait: 0.2603, forward_head: 10.0923, head_out_index: 0.0729, forward_core: 4.5256, bptt: 12.1633, tail: 0.7243, vtrace: 1.7064, losses: 0.4411, clip: 8.5073, update: 14.8810, train: 42.8166
# [2020-03-14 05:20:45,814][09212] Collected {0: 2015232}, FPS: 44886.8
# [2020-03-14 05:20:45,815][09212] Timing: experience: 44.8958

# Version V78 (bptt improvements)
# python -m algorithms.appo.train_appo --env=doom_benchmark --algo=APPO --env_frameskip=4 --use_rnn=True  --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --experiment=doom_battle_appo_v78_test --benchmark=True --res_w=128 --res_h=72 --wide_aspect_ratio=True --policy_workers_per_policy=1 --worker_num_splits=2
# [2020-03-14 21:07:21,403][11198] Env runner 0, rollouts 780: timing wait_actor: 0.0000, waiting: 1.5420, reset: 13.9346, save_policy_outputs: 0.9993, env_step: 35.5315, overhead: 3.6310, complete_rollouts: 0.0140, enqueue_policy_requests: 0.1555, one_step: 0.0157, work: 42.2544
# [2020-03-14 21:07:21,406][11199] Env runner 1, rollouts 780: timing wait_actor: 0.0000, waiting: 1.6902, reset: 16.2373, save_policy_outputs: 0.9954, env_step: 35.4816, overhead: 3.5863, complete_rollouts: 0.0145, enqueue_policy_requests: 0.1549, one_step: 0.0148, work: 42.1175
# [2020-03-14 21:07:21,653][11197] Policy worker avg. requests 3.72, timing: init: 1.7326, wait_policy_total: 16.2322, wait_policy: 0.0051, handle_policy_step: 41.8431, one_step: 0.0000, deserialize: 1.6981, obs_to_device: 5.2822, stack: 14.7678, forward: 14.0202, postprocess: 5.5750, weight_update: 0.0004
# [2020-03-14 21:07:21,760][11180] GPU learner timing: extract: 0.1839, buffers: 0.0659, batching: 5.1729, buff_ready: 0.2354, tensors_gpu_float: 7.1679, squeeze: 0.0112, prepare: 12.7145, batcher_mem: 5.1284
# [2020-03-14 21:07:22,066][11180] Train loop timing: init: 1.3305, train_wait: 0.2645, forward_head: 9.1401, bptt_initial: 0.7022, forward_core: 5.8878, bptt_rnn_states: 3.6712, bptt: 9.7196, tail: 0.5777, vtrace: 2.0181, losses: 0.4139, clip: 9.0324, update: 15.5247, train: 40.9327
# [2020-03-14 21:07:22,205][11140] Collected {0: 2015232}, FPS: 46035.7
# [2020-03-14 21:07:22,205][11140] Timing: experience: 43.7754

# Version V81 (after DMLab-related refactoring and adding Dummy Sampler)
# python -m algorithms.appo.train_appo --env=doom_benchmark --algo=APPO --env_frameskip=4 --use_rnn=True  --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --experiment=doom_battle_appo_v81_test --benchmark=True --res_w=128 --res_h=72 --wide_aspect_ratio=True --policy_workers_per_policy=1 --worker_num_splits=2
# [2020-04-03 02:39:43,305][20114] Env runner 0, rollouts 820: timing wait_actor: 0.0000, waiting: 0.3761, reset: 9.6688, save_policy_outputs: 0.9744, env_step: 35.8396, overhead: 3.8425, complete_rollouts: 0.0153, enqueue_policy_requests: 0.1639, one_step: 0.0147, work: 42.8321
# [2020-04-03 02:39:43,316][20115] Env runner 1, rollouts 800: timing wait_actor: 0.0000, waiting: 0.4349, reset: 12.9288, save_policy_outputs: 0.9748, env_step: 36.0682, overhead: 3.6751, complete_rollouts: 0.0156, enqueue_policy_requests: 0.1574, one_step: 0.0167, work: 42.7851
# [2020-04-03 02:39:43,557][20113] Policy worker avg. requests 2.18, timing: init: 1.7894, wait_policy_total: 14.0706, wait_policy: 0.0001, handle_policy_step: 41.2697, one_step: 0.0025, deserialize: 1.4548, obs_to_device: 4.5620, stack: 13.0836, forward: 15.5851, postprocess: 4.8575, weight_update: 0.0005
# [2020-04-03 02:39:43,664][20096] GPU learner timing: extract: 0.1808, buffers: 0.0643, batching: 5.1004, buff_ready: 0.2323, tensors_gpu_float: 7.9368, squeeze: 0.0091, prepare: 13.4034, batcher_mem: 5.0559
# [2020-04-03 02:39:43,971][20096] Train loop timing: init: 1.3558, train_wait: 0.3017, forward_head: 9.8845, bptt_initial: 0.7820, bptt_forward_core: 7.4162, bptt_rnn_states: 4.5037, bptt: 12.0998, tail: 0.5640, vtrace: 0.9889, losses: 0.3733, clip: 8.1355, update: 14.2952, train: 41.5756
# [2020-04-03 02:39:44,107][20058] Collected {0: 2015232}, FPS: 46778.4
# [2020-04-03 02:39:44,107][20058] Timing: experience: 43.0804

# Version V83 (Mostly refactoring)
# python -m algorithms.appo.train_appo --env=doom_benchmark --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=20 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --experiment=doom_battle_appo_v83_test --benchmark=True --res_w=128 --res_h=72 --wide_aspect_ratio=True --policy_workers_per_policy=1 --worker_num_splits=2
# [2020-04-04 02:03:59,045][02465] Env runner 1, rollouts 800: timing wait_actor: 0.0000, waiting: 0.4651, reset: 10.2102, save_policy_outputs: 0.9537, env_step: 36.2177, overhead: 3.7336, complete_rollouts: 0.0158, enqueue_policy_requests: 0.1606, one_step: 0.0152, work: 43.0656
# [2020-04-04 02:03:59,060][02464] Env runner 0, rollouts 800: timing wait_actor: 0.0000, waiting: 0.5348, reset: 14.2725, save_policy_outputs: 0.9687, env_step: 36.1914, overhead: 3.6750, complete_rollouts: 0.0151, enqueue_policy_requests: 0.1582, one_step: 0.0157, work: 42.9876
# [2020-04-04 02:03:59,275][02463] Policy worker avg. requests 2.98, timing: init: 1.8110, wait_policy_total: 14.7359, wait_policy: 0.0051, handle_policy_step: 41.7147, one_step: 0.0000, deserialize: 1.3855, obs_to_device: 5.1736, stack: 13.7928, forward_encoder: 7.4016, forward: 15.6072, postprocess: 4.7609, weight_update: 0.0005
# [2020-04-04 02:03:59,385][02449] GPU learner timing: extract: 0.1784, buffers: 0.0654, batching: 5.0967, buff_ready: 0.2283, tensors_gpu_float: 8.2659, squeeze: 0.0121, prepare: 13.7269, batcher_mem: 5.0500
# [2020-04-04 02:03:59,691][02449] Train loop timing: init: 1.3370, train_wait: 0.2528, forward_encoder: 11.0017, forward_head: 11.0051, bptt_initial: 1.0097, bptt_forward_core: 6.6520, bptt_rnn_states: 4.2696, bptt: 11.0963, tail: 0.6196, vtrace: 1.0200, losses: 0.4056, clip: 8.2484, update: 14.1095, train: 42.0111
# [2020-04-04 02:03:59,829][02416] Collected {0: 2015232}, FPS: 46444.4
# [2020-04-04 02:03:59,829][02416] Timing: experience: 43.3902
