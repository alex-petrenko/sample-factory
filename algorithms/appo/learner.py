import glob
import os
import threading
import time
from collections import OrderedDict, deque
from os.path import join
from queue import Empty, Queue
from threading import Thread

import numpy as np
import torch
from torch.multiprocessing import Process, Queue as TorchQueue

from algorithms.appo.appo_utils import TaskType, list_of_dicts_to_dict_of_lists, iterate_recursively, device_for_policy
from algorithms.appo.model import ActorCritic
from algorithms.utils.action_distributions import get_action_distribution
from algorithms.utils.algo_utils import calculate_gae
from algorithms.utils.multi_env import safe_get
from utils.decay import LinearDecay
from utils.timing import Timing
from utils.utils import log, AttrDict, experiment_dir, ensure_dir_exists, memory_consumption_mb


class LearnerWorker:
    def __init__(
        self, worker_idx, policy_id, cfg, obs_space, action_space, report_queue, weight_queues,
    ):
        log.info('Initializing GPU learner %d for policy %d', worker_idx, policy_id)

        self.worker_idx = worker_idx
        self.policy_id = policy_id
        self.cfg = cfg

        self.terminate = False

        self.obs_space = obs_space
        self.action_space = action_space

        self.device = None
        self.actor_critic = None
        self.optimizer = None

        self.task_queue = TorchQueue()
        self.report_queue = report_queue
        self.weight_queues = weight_queues

        self.rollout_tensors = dict()
        self.traj_buffer_ready = dict()

        self.experience_buffer_queue = Queue()

        self.with_training = True  # TODO: debug, remove
        self.train_in_background = True  # TODO!!!! Debug!!! should always train in separate thread
        self.training_thread = Thread(target=self._train_loop) if self.train_in_background else None
        self.train_thread_initialized = threading.Event()
        self.processing_experience_batch = threading.Event()

        self.train_step = self.env_steps = 0

        self.summary_rate_decay = LinearDecay([(0, 100), (1000000, 2000), (10000000, 10000)])
        self.last_summary_written = -1e9
        self.save_rate_decay = LinearDecay([(0, self.cfg.initial_save_rate), (1000000, 5000)], staircase=100)

        # some stats we measure in the end of the last training epoch
        self.last_batch_stats = AttrDict()

        self.discarded_experience_over_time = deque([], maxlen=30)
        self.discarded_experience_timer = time.time()
        self.num_discarded_rollouts = 0

        self.kl_coeff = self.cfg.initial_kl_coeff

        self.process = Process(target=self._run, daemon=True)

    def start_process(self):
        self.process.start()

    def _init(self):
        log.info('Waiting for GPU learner to initialize...')
        self.train_thread_initialized.wait()
        log.info('GPU learner %d initialized', self.worker_idx)

    def _terminate(self):
        self.terminate = True

    def _broadcast_weights(self, discarding_rate):
        state_dict = self.actor_critic.state_dict()
        policy_version = self.train_step
        weight_update = (policy_version, state_dict, discarding_rate)
        for q in self.weight_queues:
            q.put((TaskType.UPDATE_WEIGHTS, weight_update))

    def _calculate_gae(self, buffer):
        rewards = np.asarray(buffer.rewards)  # [E, T]
        dones = np.asarray(buffer.dones)  # [E, T]
        values_arr = np.array(buffer.values).squeeze()  # [E, T]

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

        return buffer

    def _prepare_train_buffer(self, rollouts, timing):
        trajectories = [AttrDict(r['t']) for r in rollouts]

        # if self.cfg.benchmark:
        #     log.info('%r', trajectories[0].policy_version)
        #     log.info('%r', trajectories[-1].policy_version)

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

            # normalize advantages if needed
            if self.cfg.normalize_advantage:
                adv_mean = buffer.advantages.mean()
                adv_std = buffer.advantages.std()
                # adv_max, adv_min = buffer.advantages.max(), buffer.advantages.min()
                # adv_max_abs = max(adv_max, abs(adv_min))
                # log.info(
                #     'Adv mean %.3f std %.3f, min %.3f, max %.3f, max abs %.3f',
                #     adv_mean, adv_std, adv_min, adv_max, adv_max_abs,
                # )
                buffer.advantages = (buffer.advantages - adv_mean) / max(1e-2, adv_std)

        with timing.add_time('tensors'):
            for d, key, value in iterate_recursively(buffer):
                d[key] = torch.cat(value, dim=0).to(self.device).float()

            # will squeeze actions only in simple categorical case
            tensors_to_squeeze = ['actions', 'log_prob_actions', 'policy_version', 'values']
            for tensor_name in tensors_to_squeeze:
                buffer[tensor_name].squeeze_()

        with timing.add_time('buff_ready'):
            for r in rollouts:
                r = AttrDict(r)

                # we copied the data from the shared buffer, now we can mark the buffers as free
                traj_buffer_ready = self.traj_buffer_ready[(r.worker_idx, r.split_idx)]
                traj_buffer_ready[r.env_idx, r.agent_idx, r.traj_buffer_idx] = 1

        return buffer

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
        work_done = False

        discard_rollouts = 0
        policy_version = self.train_step
        for r in rollouts:
            rollout_min_version = r['t']['policy_version'].min().item()
            if policy_version - rollout_min_version >= self.cfg.max_policy_lag:
                discard_rollouts += 1
            else:
                break

        if discard_rollouts > 0:
            log.warning(
                'Discarding %d old rollouts (learner is not fast enough to process experience)',
                discard_rollouts,
            )
            rollouts = rollouts[discard_rollouts:]
            self.num_discarded_rollouts += discard_rollouts

        if len(rollouts) >= rollouts_in_macro_batch:
            # process newest rollouts
            rollouts_to_process = rollouts[:rollouts_in_macro_batch]
            rollouts = rollouts[rollouts_in_macro_batch:]

            self._process_macro_batch(rollouts_to_process, timing)
            # log.info('Unprocessed rollouts: %d (%d samples)', len(rollouts), len(rollouts) * self.cfg.rollout)

            work_done = True

        return rollouts, work_done

    def _get_minibatches(self, experience_size):
        """Generating minibatches for training."""
        assert self.cfg.rollout % self.cfg.recurrence == 0
        assert experience_size % self.cfg.batch_size == 0

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
        summaries_every = self.summary_rate_decay.at(self.train_step)
        return self.train_step - self.last_summary_written > summaries_every

    def _after_optimizer_step(self):
        """A hook to be called after each optimizer step."""
        self.train_step += 1
        self._maybe_save()

    def _maybe_save(self):
        save_every = self.save_rate_decay.at(self.train_step)
        if (self.train_step + 1) % save_every == 0 or self.train_step <= 1:
            self._save()

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
            'kl_coeff': self.kl_coeff,
            'model': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        return checkpoint

    def _save(self):
        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None

        checkpoint_dir = self.checkpoint_dir(self.cfg, self.policy_id)
        filepath = join(checkpoint_dir, f'checkpoint_{self.train_step:09d}_{self.env_steps}.pth')
        log.info('Saving %s...', filepath)
        torch.save(checkpoint, filepath)

        while len(self.get_checkpoints(checkpoint_dir)) > self.cfg.keep_checkpoints:
            oldest_checkpoint = self.get_checkpoints(checkpoint_dir)[0]
            if os.path.isfile(oldest_checkpoint):
                log.debug('Removing %s', oldest_checkpoint)
                os.remove(oldest_checkpoint)

    @staticmethod
    def _policy_loss(ratio, adv, clip_ratio):
        # TODO: get rid of leaky PPO, simplify this function

        positive_clip_ratio = clip_ratio
        negative_clip_ratio = 1.0 / clip_ratio

        is_adv_positive = (adv > 0.0).float()
        is_ratio_too_big = (ratio > positive_clip_ratio).float() * is_adv_positive

        is_adv_negative = (adv < 0.0).float()
        is_ratio_too_small = (ratio < negative_clip_ratio).float() * is_adv_negative

        clipping = is_adv_positive * positive_clip_ratio + is_adv_negative * negative_clip_ratio

        is_ratio_clipped = is_ratio_too_big + is_ratio_too_small
        is_ratio_not_clipped = 1.0 - is_ratio_clipped

        # total_non_clipped = torch.sum(is_ratio_not_clipped).float()
        fraction_clipped = is_ratio_clipped.mean()

        objective = ratio * adv
        leak = 0.0  # currently not used
        objective_clipped = -leak * ratio * adv + clipping * adv * (1.0 + leak)

        policy_loss = -(objective * is_ratio_not_clipped + objective_clipped * is_ratio_clipped)
        policy_loss = policy_loss.mean()

        return policy_loss, fraction_clipped

    def _value_loss(self, new_values, old_values, target, clip_value):
        value_clipped = old_values + torch.clamp(new_values - old_values, -clip_value, clip_value)
        value_original_loss = (new_values - target).pow(2)
        value_clipped_loss = (value_clipped - target).pow(2)
        value_loss = torch.max(value_original_loss, value_clipped_loss)
        value_loss = value_loss.mean()
        value_loss *= self.cfg.value_loss_coeff
        value_delta = torch.abs(new_values - old_values)

        return value_loss, value_delta

    def _train(self, buffer, experience_size, timing):
        stats = None

        rho_hat = c_hat = 1.0  # V-trace parameters
        # noinspection PyArgumentList
        rho_hat = torch.Tensor([rho_hat]).to(self.device)
        # noinspection PyArgumentList
        c_hat = torch.Tensor([c_hat]).to(self.device)

        clip_ratio = self.cfg.ppo_clip_ratio
        clip_value = self.cfg.ppo_clip_value

        kl_old_mean = kl_old_max = 0.0
        value_delta_avg = value_delta_max = 0.0
        fraction_clipped = 0.0
        rnn_dist = 0.0
        ratio_mean = ratio_min = ratio_max = 0.0

        early_stopping = False
        num_sgd_steps = 0

        for epoch in range(self.cfg.ppo_epochs):
            if early_stopping:
                break

            minibatches = self._get_minibatches(experience_size)

            for batch_num in range(len(minibatches)):
                indices = minibatches[batch_num]

                # current minibatch consisting of short trajectory segments with length == recurrence
                mb = self._get_minibatch(buffer, indices)

                # calculate policy head outside of recurrent loop
                head_outputs = self.actor_critic.forward_head(mb.obs)

                # indices corresponding to 1st frames of trajectory segments
                traj_indices = indices[::self.cfg.recurrence]

                # initial rnn states
                rnn_states = buffer.rnn_states[traj_indices]

                # calculate RNN outputs for each timestep in a loop
                with timing.add_time('bptt'):
                    core_outputs = []
                    for i in range(self.cfg.recurrence):
                        # indices of head outputs corresponding to the current timestep
                        timestep = np.arange(i, self.cfg.batch_size, self.cfg.recurrence)
                        step_head_outputs = head_outputs[timestep]

                        dones = mb.dones[timestep].unsqueeze(dim=1)
                        rnn_states = (1.0 - dones) * rnn_states + dones * mb.rnn_states[timestep]

                        core_output, rnn_states = self.actor_critic.forward_core(step_head_outputs, rnn_states)

                        core_outputs.append(core_output)

                # transform core outputs from [T, Batch, D] to [Batch, T, D] and then to [Batch x T, D]
                # which is the same shape as the minibatch
                core_outputs = torch.stack(core_outputs)
                num_timesteps, num_trajectories = core_outputs.shape[:2]
                assert num_timesteps == self.cfg.recurrence
                assert num_timesteps * num_trajectories == self.cfg.batch_size
                core_outputs = core_outputs.transpose(0, 1).reshape(-1, *core_outputs.shape[2:])
                assert core_outputs.shape[0] == head_outputs.shape[0]

                # calculate policy tail outside of recurrent loop
                result = self.actor_critic.forward_tail(core_outputs, with_action_distribution=True)

                action_distribution = result.action_distribution
                log_prob_actions = action_distribution.log_prob(mb.actions)
                ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

                vtrace_c = torch.min(c_hat, ratio.detach())
                vtrace_rho = torch.min(rho_hat, ratio.detach())

                values = result.values.squeeze()

                last_timestep = np.arange(self.cfg.recurrence - 1, self.cfg.batch_size, self.cfg.recurrence)
                next_values = (values[last_timestep].detach() - mb.rewards[last_timestep]) / self.cfg.gamma
                next_vs = next_values

                gamma = self.cfg.gamma

                vs = torch.zeros((num_trajectories * self.cfg.recurrence)).to(self.device)
                adv = torch.zeros((num_trajectories * self.cfg.recurrence)).to(self.device)

                with timing.add_time('vtrace'):
                    for i in reversed(range(self.cfg.recurrence)):
                        timestep = np.arange(i, self.cfg.batch_size, self.cfg.recurrence)

                        rewards = mb.rewards[timestep]
                        dones = mb.dones[timestep]
                        not_done = 1.0 - dones

                        delta_s = (vtrace_rho[timestep] * (rewards + not_done * gamma * next_values - values[timestep])).detach()
                        adv[timestep] = (vtrace_rho[timestep] * (rewards + not_done * gamma * next_vs - values[timestep])).detach()

                        vs[timestep] = (values[timestep] + delta_s + not_done * gamma * vtrace_c[timestep] * (next_vs - next_values)).detach()

                        next_values = values[timestep].detach()
                        next_vs = vs[timestep].detach()

                adv = adv.detach()
                adv_mean = adv.mean()
                adv_std = adv.std()
                adv = (adv - adv_mean) / max(1e-2, adv_std)

                with timing.add_time('losses'):
                    policy_loss, fraction_clipped = self._policy_loss(ratio, adv, clip_ratio)
                    ratio_mean = torch.abs(1.0 - ratio).mean()
                    ratio_min = ratio.min()
                    ratio_max = ratio.max()

                    targets = vs.detach()
                    old_values = mb.values
                    value_loss, value_delta = self._value_loss(values, old_values, targets, clip_value)
                    value_delta_avg, value_delta_max = value_delta.mean(), value_delta.max()

                    # entropy loss
                    kl_prior = action_distribution.kl_prior()
                    kl_prior = kl_prior.mean()
                    prior_loss = self.cfg.prior_loss_coeff * kl_prior

                    old_action_distribution = get_action_distribution(self.actor_critic.action_space, mb.action_logits)

                    # small KL penalty for being different from the behavior policy
                    kl_old = action_distribution.kl_divergence(old_action_distribution)
                    kl_old_max = kl_old.max()
                    kl_old_mean = kl_old.mean()
                    kl_penalty = self.kl_coeff * kl_old_mean

                    loss = policy_loss + value_loss + prior_loss + kl_penalty

                with timing.add_time('update'):
                    # update the weights
                    self.optimizer.zero_grad()
                    loss.backward()

                    if self.cfg.max_grad_norm > 0.0:
                        with timing.add_time('clip'):
                            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)

                    self.optimizer.step()
                    num_sgd_steps += 1

                self._after_optimizer_step()

                # collect and report summaries
                with_summaries = self._should_save_summaries()
                if with_summaries:
                    self.last_summary_written = self.train_step

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
                    stats.kl_prior = kl_prior
                    stats.value_loss = value_loss
                    stats.prior_loss = prior_loss
                    stats.kl_coeff = self.kl_coeff
                    stats.kl_penalty = kl_penalty
                    stats.adv_min = adv.min()
                    stats.adv_max = adv.max()
                    stats.max_abs_logprob = torch.abs(mb.action_logits).max()

                    curr_policy_version = self.train_step
                    version_diff = curr_policy_version - mb.policy_version
                    stats.version_diff_avg = version_diff.mean()
                    stats.version_diff_min = version_diff.min()
                    stats.version_diff_max = version_diff.max()

                    # we want this statistic for the last batch of the last epoch
                    for key, value in self.last_batch_stats.items():
                        stats[key] = value

                    for key, value in stats.items():
                        if isinstance(value, torch.Tensor):
                            stats[key] = value.detach()

        # adjust KL-penalty coefficient if KL divergence at the end of training is high
        if kl_old_mean > self.cfg.target_kl:
            self.kl_coeff *= 1.5
        elif kl_old_mean < self.cfg.target_kl / 2:
            self.kl_coeff /= 1.5
        self.kl_coeff = max(self.kl_coeff, 1e-6)

        self.last_batch_stats.kl_divergence = kl_old_mean
        self.last_batch_stats.kl_max = kl_old_max
        self.last_batch_stats.value_delta = value_delta_avg
        self.last_batch_stats.value_delta_max = value_delta_max
        self.last_batch_stats.fraction_clipped = fraction_clipped
        self.last_batch_stats.rnn_dist = rnn_dist
        self.last_batch_stats.ratio_mean = ratio_mean
        self.last_batch_stats.ratio_min = ratio_min
        self.last_batch_stats.ratio_max = ratio_max
        self.last_batch_stats.num_sgd_steps = num_sgd_steps

        return stats

    @staticmethod
    def load_checkpoint(checkpoints, policy_id):
        if len(checkpoints) <= 0:
            log.warning('No checkpoints found')
            return None
        else:
            latest_checkpoint = checkpoints[-1]
            log.warning('Loading state from checkpoint %s...', latest_checkpoint)

            device = device_for_policy(policy_id)
            checkpoint_dict = torch.load(latest_checkpoint, map_location=device)
            return checkpoint_dict

    def _load_state(self, checkpoint_dict):
        self.train_step = checkpoint_dict['train_step']
        self.env_steps = checkpoint_dict['env_steps']
        self.kl_coeff = checkpoint_dict['kl_coeff']
        self.actor_critic.load_state_dict(checkpoint_dict['model'])
        self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        log.info('Loaded experiment state at training iteration %d, env step %d', self.train_step, self.env_steps)

    def init_model(self):
        self.actor_critic = ActorCritic(self.obs_space, self.action_space, self.cfg)
        self.actor_critic.to(self.device)
        self.actor_critic.share_memory()

    def load_from_checkpoint(self):
        checkpoints = self.get_checkpoints(self.checkpoint_dir(self.cfg, self.policy_id))
        checkpoint_dict = self.load_checkpoint(checkpoints, self.policy_id)
        if checkpoint_dict is None:
            log.debug('Did not load from checkpoint, starting from scratch!')
        else:
            log.debug('Loading model from checkpoint')
            self._load_state(checkpoint_dict)

    def initialize(self, timing):
        with timing.timeit('init'):
            # initialize the Torch modules
            if self.cfg.seed is not None:
                log.info('Setting fixed seed %d', self.cfg.seed)
                torch.manual_seed(self.cfg.seed)
                np.random.seed(self.cfg.seed)

            torch.set_num_threads(1)  # TODO: experimental

            self.device = device_for_policy(self.policy_id)
            self.init_model()

            self.optimizer = torch.optim.Adam(
                self.actor_critic.parameters(),
                self.cfg.learning_rate,
                betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
                eps=self.cfg.adam_eps,
            )

            self.load_from_checkpoint()

            self._broadcast_weights(self._discarding_rate())  # sync the very first version of the weights

        self.train_thread_initialized.set()

    def _process_training_data(self, data, timing, wait_stats=None):
        buffer, samples, env_steps = data
        self.env_steps += env_steps
        experience_size = buffer.rewards.shape[0]

        stats = dict(env_steps=self.env_steps, policy_id=self.policy_id)

        with timing.add_time('train'):
            discarding_rate = self._discarding_rate()

            if self.with_training:
                # log.debug('Training policy %d on %d samples', self.policy_id, samples)
                train_stats = self._train(buffer, experience_size, timing)

                if train_stats is not None:
                    stats['train'] = train_stats

                    if wait_stats is not None:
                        wait_avg, wait_min, wait_max = wait_stats
                        stats['train']['wait_avg'] = wait_avg
                        stats['train']['wait_min'] = wait_min
                        stats['train']['wait_max'] = wait_max

                    stats['train']['discarded_rollouts'] = self.num_discarded_rollouts
                    stats['train']['discarding_rate'] = discarding_rate

                    memory_mb = memory_consumption_mb()
                    stats['stats'] = dict(memory_learner=memory_mb)

                self._broadcast_weights(discarding_rate)

        self.report_queue.put(stats)

    def _train_loop(self):
        timing = Timing()
        self.initialize(timing)

        wait_times = deque([], maxlen=self.cfg.num_workers)

        while not self.terminate:
            with timing.timeit('train_wait'):
                data = safe_get(self.experience_buffer_queue)

            self.processing_experience_batch.set()

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

    def _init_rollout_tensors(self, data):
        data = AttrDict(data)
        assert self.policy_id == data.policy_id

        worker_idx, split_idx, traj_buffer_idx = data.worker_idx, data.split_idx, data.traj_buffer_idx
        for env_agent, rollout_data in data.tensors.items():
            env_idx, agent_idx = env_agent
            tensor_dict_key = (worker_idx, split_idx, env_idx, agent_idx, traj_buffer_idx)
            assert tensor_dict_key not in self.rollout_tensors

            self.rollout_tensors[tensor_dict_key] = rollout_data

        self.traj_buffer_ready[(worker_idx, split_idx)] = data.is_ready_tensor

    def _extract_rollouts(self, data):
        data = AttrDict(data)
        worker_idx, split_idx, traj_buffer_idx = data.worker_idx, data.split_idx, data.traj_buffer_idx

        rollouts = []
        for rollout_data in data.rollouts:
            env_idx, agent_idx = rollout_data['env_idx'], rollout_data['agent_idx']
            tensor_dict_key = (worker_idx, split_idx, env_idx, agent_idx, traj_buffer_idx)
            tensors = self.rollout_tensors[tensor_dict_key]

            rollout_data['t'] = tensors
            rollout_data['worker_idx'] = worker_idx
            rollout_data['split_idx'] = split_idx
            rollout_data['traj_buffer_idx'] = traj_buffer_idx
            rollouts.append(rollout_data)

        if not self.with_training:
            return []

        return rollouts

    def _run(self):
        torch.multiprocessing.set_sharing_strategy('file_system')

        timing = Timing()

        rollouts = []

        if self.train_in_background:
            self.training_thread.start()
        else:
            self.initialize(timing)

        while not self.terminate:
            while True:
                try:
                    task_type, data = self.task_queue.get_nowait()

                    if task_type == TaskType.INIT:
                        self._init()
                    elif task_type == TaskType.TERMINATE:
                        log.info('GPU learner timing: %s', timing)
                        self._terminate()
                        break
                    elif task_type == TaskType.INIT_TENSORS:
                        self._init_rollout_tensors(data)
                    elif task_type == TaskType.TRAIN:
                        with timing.add_time('extract'):
                            rollouts.extend(self._extract_rollouts(data))
                except Empty:
                    break

            while self.experience_buffer_queue.qsize() > 1:
                self.processing_experience_batch.clear()
                self.processing_experience_batch.wait()

            rollouts, work_done = self._process_rollouts(rollouts, timing)

            if not self.train_in_background:
                while not self.experience_buffer_queue.empty():
                    training_data = self.experience_buffer_queue.get()
                    self.processing_experience_batch.set()
                    self._process_training_data(training_data, timing)

            self._experience_collection_rate_stats()

            if not work_done:
                # if we didn't do anything let's sleep to prevent wasting CPU time
                time.sleep(0.005)

        if self.train_in_background:
            self.experience_buffer_queue.put(None)
            self.training_thread.join()

    def init(self):
        self.task_queue.put((TaskType.INIT, None))
        self.task_queue.put((TaskType.EMPTY, None))

        # wait until we finished initializing
        while self.task_queue.qsize() > 0:
            time.sleep(0.01)

    def close(self):
        self.task_queue.put((TaskType.TERMINATE, None))

    def join(self):
        self.process.join(timeout=5)


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
