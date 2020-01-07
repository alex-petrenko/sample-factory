import json
import math
import time
from collections import deque
from os.path import join
from queue import Empty

import numpy as np
from tensorboardX import SummaryWriter
from torch.multiprocessing import Event, Queue as TorchQueue

from algorithms.appo.actor_worker import make_env_func, ActorWorker
from algorithms.appo.policy_worker import PolicyWorker
from algorithms.appo.learner import LearnerWorker
from utils.timing import Timing
from utils.utils import summaries_dir, experiment_dir, log, str2bool, memory_consumption_mb, cfg_file, ensure_dir_exists


class Algorithm:
    @classmethod
    def add_cli_args(cls, parser):
        p = parser

        p.add_argument('--seed', default=42, type=int, help='Set a fixed seed value')

        p.add_argument('--initial_save_rate', default=1000, type=int,
                       help='Save model every N train steps in the beginning of training')
        p.add_argument('--keep_checkpoints', default=2, type=int, help='Number of model checkpoints to keep')

        p.add_argument('--stats_episodes', default=100, type=int, help='How many episodes to average to measure performance (avg. reward etc)')

        p.add_argument('--learning_rate', default=1e-4, type=float, help='LR')

        p.add_argument('--train_for_env_steps', default=int(1e10), type=int, help='Stop after all policies are trained for this many env steps')
        p.add_argument('--train_for_seconds', default=int(1e10), type=int, help='Stop training after this many seconds')

        # observation preprocessing
        p.add_argument('--obs_subtract_mean', default=0.0, type=float, help='Observation preprocessing, mean value to subtract from observation (e.g. 128.0 for 8-bit RGB)')
        p.add_argument('--obs_scale', default=1.0, type=float, help='Observation preprocessing, divide observation tensors by this scalar (e.g. 128.0 for 8-bit RGB)')

        # RL
        p.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
        p.add_argument(
            '--reward_scale', default=1.0, type=float,
            help=('Multiply all rewards but this factor before feeding into RL algorithm.'
                  'Sometimes the overall scale of rewards is too high which makes value estimation a harder regression task.'
                  'Loss values become too high which requires a smaller learning rate, etc.'),
        )
        p.add_argument('--reward_clip', default=10.0, type=float, help='Clip rewards between [-c, c]. Default [-10, 10] virtually means no clipping for most envs')

        # policy size and configuration
        p.add_argument('--encoder', default='convnet_simple', type=str, help='Type of the policy head (e.g. convolutional encoder)')
        p.add_argument('--hidden_size', default=512, type=int, help='Size of hidden layer in the model, or the size of RNN hidden state in recurrent model (e.g. GRU)')

    def __init__(self, cfg):
        self.cfg = cfg


class APPO(Algorithm):
    """Async PPO."""

    @classmethod
    def add_cli_args(cls, parser):
        p = parser
        super().add_cli_args(p)

        p.add_argument('--adam_eps', default=1e-6, type=float, help='Adam epsilon parameter (1e-8 to 1e-5 seem to reliably work okay, 1e-3 and up does not work)')
        p.add_argument('--adam_beta1', default=0.9, type=float, help='Adam momentum decay coefficient')
        p.add_argument('--adam_beta2', default=0.999, type=float, help='Adam second momentum decay coefficient')

        p.add_argument('--gae_lambda', default=0.95, type=float, help='Generalized Advantage Estimation discounting')

        p.add_argument('--rollout', default=64, type=int, help='Length of the rollout from each environment in timesteps. Size of the training batch is rollout X num_envs')

        p.add_argument('--num_workers', default=16, type=int, help='Number of parallel environment workers. Should be less than num_envs and should divide num_envs')

        p.add_argument('--recurrence', default=32, type=int, help='Trajectory length for backpropagation through time. If recurrence=1 there is no backpropagation through time, and experience is shuffled completely randomly')
        p.add_argument('--use_rnn', default=True, type=str2bool, help='Whether to use RNN core in a policy or not')

        p.add_argument('--ppo_clip_ratio', default=1.1, type=float, help='We use unbiased clip(x, e, 1/e) instead of clip(x, 1+e, 1-e) in the paper')
        p.add_argument('--ppo_clip_value', default=0.2, type=float, help='Maximum absolute change in value estimate until it is clipped. Sensitive to value magnitude')
        p.add_argument('--batch_size', default=1024, type=int, help='PPO minibatch size')
        p.add_argument('--ppo_epochs', default=4, type=int, help='Number of training epochs before a new batch of experience is collected')
        p.add_argument('--target_kl', default=0.02, type=float, help='Target distance from behavior policy at the end of training on each experience batch')

        p.add_argument('--normalize_advantage', default=True, type=str2bool, help='Whether to normalize advantages or not (subtract mean and divide by standard deviation)')

        p.add_argument('--max_grad_norm', default=2.0, type=float, help='Max L2 norm of the gradient vector')

        # components of the loss function
        p.add_argument(
            '--prior_loss_coeff', default=0.001, type=float,
            help=('Coefficient for the exploration component of the loss function. Typically this is entropy maximization, but here we use KL-divergence between our policy and a prior.'
                  'By default prior is a uniform distribution, and this is numerically equivalent to maximizing entropy.'
                  'Alternatively we can use custom prior distributions, e.g. to encode domain knowledge'),
        )
        p.add_argument('--initial_kl_coeff', default=0.0001, type=float, help='Initial value of KL-penalty coefficient. This is adjusted during the training such that policy change stays close to target_kl')
        p.add_argument('--kl_coeff_large', default=0.0, type=float, help='Loss coefficient for the quadratic KL term')
        p.add_argument('--value_loss_coeff', default=0.5, type=float, help='Coefficient for the critic loss')

        # APPO-specific
        p.add_argument('--num_envs_per_worker', default=2, type=int, help='Number of envs on a single CPU actor')
        p.add_argument('--worker_num_splits', default=2, type=int, help='Typically we split a vector of envs into two parts for "double buffered" experience collection')
        p.add_argument('--num_policies', default=1, type=int, help='Number of policies to train jointly')
        p.add_argument('--policy_workers_per_policy', default=1, type=int, help='Number of GPU workers that compute policy forward pass (per policy)')
        p.add_argument('--macro_batch', default=6144, type=int, help='Amount of experience to collect per policy before passing experience to the learner')
        p.add_argument('--max_policy_lag', default=25, type=int, help='Max policy lag in policy versions. Discard all experience that is older than this.')

        p.add_argument('--sync_mode', default=False, type=str2bool, help='Fully synchronous mode to compare against the standard PPO implementation')

        p.add_argument('--with_vtrace', default=True, type=str2bool, help='Enables V-trace off-policy correction')

        # debugging options
        p.add_argument('--benchmark', default=False, type=str2bool, help='Benchmark mode')
        p.add_argument('--sampler_only', default=False, type=str2bool, help='Do not send experience to the learner, measuring sampling throughput')

    def __init__(self, cfg):
        super().__init__(cfg)

        self.obs_space = self.action_space = None

        self.actor_workers = None

        self.report_queue = TorchQueue()
        self.policy_workers = dict()
        self.policy_queues = dict()

        self.learner_workers = dict()

        self.workers_by_handle = None

        self.trajectories = dict()
        self.currently_training = set()

        self.policy_inputs = [[] for _ in range(self.cfg.num_policies)]
        self.policy_outputs = dict()
        for worker_idx in range(self.cfg.num_workers):
            for split_idx in range(self.cfg.worker_num_splits):
                self.policy_outputs[(worker_idx, split_idx)] = dict()

        self.episode_rewards = [deque(maxlen=100) for _ in range(self.cfg.num_policies)]
        self.episode_lengths = [deque(maxlen=100) for _ in range(self.cfg.num_policies)]

        self.last_timing = dict()
        self.env_steps = dict()
        self.samples_collected = [0 for _ in range(self.cfg.num_policies)]
        self.total_env_steps_since_resume = 0

        self.total_train_seconds = 0  # TODO: save and load from checkpoint??

        self.last_report = time.time()
        self.report_interval = 3.0  # sec

        self.fps_stats = deque([], maxlen=10)
        self.throughput_stats = [deque([], maxlen=10) for _ in range(self.cfg.num_policies)]
        self.avg_stats = dict()

        self.writers = dict()
        writer_keys = list(range(self.cfg.num_policies))
        for key in writer_keys:
            summary_dir = join(summaries_dir(experiment_dir(cfg=self.cfg)), str(key))
            summary_dir = ensure_dir_exists(summary_dir)
            self.writers[key] = SummaryWriter(summary_dir, flush_secs=20)

    def _cfg_dict(self):
        if isinstance(self.cfg, dict):
            return self.cfg
        else:
            return vars(self.cfg)

    def _save_cfg(self):
        cfg_dict = self._cfg_dict()
        with open(cfg_file(self.cfg), 'w') as json_file:
            json.dump(cfg_dict, json_file, indent=2)

    def initialize(self):
        tmp_env = make_env_func(self.cfg, env_config=None)
        self.obs_space = tmp_env.observation_space
        self.action_space = tmp_env.action_space
        tmp_env.close()

        self._save_cfg()

    def finalize(self):
        pass

    def create_actor_worker(self, idx, actor_queue, traj_buffer_events):
        learner_queues = {p: w.task_queue for p, w in self.learner_workers.items()}

        return ActorWorker(
            self.cfg, self.obs_space, self.action_space, idx, task_queue=actor_queue,
            policy_queues=self.policy_queues, report_queue=self.report_queue, learner_queues=learner_queues,
            traj_buffer_events=traj_buffer_events,
        )

    # noinspection PyProtectedMember
    def init_subset(self, indices, actor_queues, traj_buffer_events):
        workers = dict()
        started_reset = dict()
        for i in indices:
            w = self.create_actor_worker(i, actor_queues[i], traj_buffer_events[i])
            w.init()
            time.sleep(0.1)  # just in case
            w.request_reset()
            workers[i] = w
            started_reset[i] = time.time()

        fastest_reset_time = None
        workers_finished = set()

        while len(workers_finished) < len(workers):
            for w in workers.values():
                if w.task_queue.qsize() > 0:
                    time.sleep(0.05)
                    continue

                if len(workers_finished) <= 0:
                    fastest_reset_time = time.time() - started_reset[w.worker_idx]
                    log.debug('Fastest reset in %.3f seconds', fastest_reset_time)

                if not w.critical_error.is_set():
                    workers_finished.add(w.worker_idx)

            if workers_finished:
                log.warning('Workers finished: %r', workers_finished)

            for worker_idx, w in workers.items():
                if worker_idx in workers_finished:
                    continue

                time_passed = time.time() - started_reset[w.worker_idx]
                if fastest_reset_time is None:
                    timeout = False
                else:
                    timeout = time_passed > max(fastest_reset_time * 1.5, fastest_reset_time + 10)

                is_process_alive = w.process.is_alive() and not w.critical_error.is_set()

                if timeout or not is_process_alive:
                    # if it takes more than 1.5x the usual time to reset, this worker is probably stuck
                    log.error('Worker %d seems to be stuck (%.3f). Reset!', w.worker_idx, time_passed)
                    log.debug('Status: %r %r', is_process_alive, w.critical_error.is_set())
                    stuck_worker = w
                    stuck_worker.process.kill()

                    new_worker = self.create_actor_worker(worker_idx, actor_queues[worker_idx], traj_buffer_events)
                    new_worker.init()
                    new_worker.request_reset()
                    started_reset[worker_idx] = time.time()

                    workers[worker_idx] = new_worker
                    del stuck_worker

        return workers.values()

    # noinspection PyUnresolvedReferences
    def init_workers(self):
        actor_queues = [TorchQueue() for _ in range(self.cfg.num_workers)]

        # two buffers - one for learner to process while another is being filled by the actor
        num_trajectory_buffers = 2

        # events are used to signal when learner is done with the shared trajectory buffer, so it can be reused
        # on the actor. The alternative is to send the buffers back from learner to the actor, but this introduces
        # another set of queues. This seems like a good compromise.
        shape = [
            self.cfg.num_workers,
            self.cfg.worker_num_splits,
            num_trajectory_buffers,
        ]
        traj_buffer_events = np.ndarray(shape, dtype=object)
        for worker_idx in range(self.cfg.num_workers):
            for split_idx in range(self.cfg.worker_num_splits):
                for buffer_idx in range(num_trajectory_buffers):
                    e = Event()
                    e.set()  # initially all buffers are available
                    traj_buffer_events[worker_idx, split_idx, buffer_idx] = e

        weight_queues = dict()
        for policy_id in range(self.cfg.num_policies):
            weight_queues[policy_id] = []
            for i in range(self.cfg.policy_workers_per_policy):
                weight_queues[policy_id].append(TorchQueue())

        log.info('Initializing GPU learners...')
        learner_idx = 0
        for policy_id in range(self.cfg.num_policies):
            learner_worker = LearnerWorker(
                learner_idx, policy_id, self.cfg, self.obs_space, self.action_space,
                self.report_queue, weight_queues[policy_id], traj_buffer_events,
            )
            learner_worker.start_process()
            learner_worker.init()

            self.learner_workers[policy_id] = learner_worker
            learner_idx += 1

        log.info('Initializing GPU workers...')
        policy_worker_idx = 0
        for policy_id in range(self.cfg.num_policies):
            self.policy_workers[policy_id] = []

            policy_queue = TorchQueue()
            self.policy_queues[policy_id] = policy_queue

            for i in range(self.cfg.policy_workers_per_policy):
                policy_worker = PolicyWorker(
                    policy_worker_idx, policy_id, self.cfg, self.obs_space, self.action_space,
                    policy_queue, actor_queues, weight_queues[policy_id][i], self.report_queue,
                )
                self.policy_workers[policy_id].append(policy_worker)
                policy_worker_idx += 1

        log.info('Initializing actors...')

        self.actor_workers = []
        max_parallel_init = 5
        worker_indices = list(range(self.cfg.num_workers))
        for i in range(0, self.cfg.num_workers, max_parallel_init):
            workers = self.init_subset(worker_indices[i:i + max_parallel_init], actor_queues, traj_buffer_events)
            self.actor_workers.extend(workers)

        # wait for GPU workers to finish initializing
        for policy_id, workers in self.policy_workers.items():
            for w in workers:
                w.start_process()
                w.init()

    def process_report(self, report):
        if 'policy_id' in report:
            policy_id = report['policy_id']

            if 'env_steps' in report:
                if policy_id in self.env_steps:
                    delta = report['env_steps'] - self.env_steps[policy_id]
                    self.total_env_steps_since_resume += delta
                self.env_steps[policy_id] = report['env_steps']

            if 'episodic' in report:
                s = report['episodic']
                self.episode_rewards[policy_id].append(s['reward'])
                self.episode_lengths[policy_id].append(s['len'])

            if 'train' in report:
                self.report_train_summaries(report['train'], policy_id)

            if 'samples' in report:
                self.samples_collected[policy_id] += report['samples']

        if 'timing' in report:
            for k, v in report['timing'].items():
                if k not in self.avg_stats:
                    self.avg_stats[k] = deque([], maxlen=50)
                self.avg_stats[k].append(v)

    def report(self):
        now = time.time()

        total_env_steps = sum(self.env_steps.values())
        self.fps_stats.append((now, total_env_steps))
        if len(self.fps_stats) <= 1:
            return

        past_moment, past_frames = self.fps_stats[0]
        fps = (total_env_steps - past_frames) / (now - past_moment)

        avg_reward, avg_length, sample_throughput = dict(), dict(), dict()
        for policy_id in range(self.cfg.num_policies):
            if len(self.episode_rewards[policy_id]) >= self.episode_rewards[policy_id].maxlen:
                avg_reward[policy_id] = np.mean(self.episode_rewards[policy_id])
            else:
                avg_reward[policy_id] = math.nan

            if len(self.episode_lengths[policy_id]) >= self.episode_lengths[policy_id].maxlen:
                avg_length[policy_id] = np.mean(self.episode_lengths[policy_id])
            else:
                avg_length[policy_id] = math.nan

            self.throughput_stats[policy_id].append((now, self.samples_collected[policy_id]))
            if len(self.throughput_stats[policy_id]) > 1:
                past_moment, past_samples = self.throughput_stats[policy_id][0]
                sample_throughput[policy_id] = (self.samples_collected[policy_id] - past_samples) / (now - past_moment)
            else:
                sample_throughput[policy_id] = math.nan

        self.print_stats(fps, sample_throughput, avg_reward, total_env_steps)
        self.report_basic_summaries(fps, sample_throughput, avg_reward, avg_length)

    def print_stats(self, fps, sample_throughput, avg_reward, total_env_steps):
        samples_per_policy = ', '.join([f'{p}: {s:.1f}' for p, s in sample_throughput.items()])
        log.debug(
            'Fps is %.1f. Total num frames: %d. Throughput: %s. Samples: %d',
            fps, total_env_steps, samples_per_policy, sum(self.samples_collected),
        )
        log.debug('Avg episode reward: %r', [f'{p}: {r:.3f}' for p, r in avg_reward.items()])

    def report_train_summaries(self, stats, policy_id):
        for key, scalar in stats.items():
            self.writers[policy_id].add_scalar(f'train/{key}', scalar, self.env_steps[policy_id])

    def report_basic_summaries(self, fps, sample_throughput, avg_reward, avg_length):
        memory_mb = memory_consumption_mb()

        default_policy = 0
        for policy_id, env_steps in self.env_steps.items():
            if policy_id == default_policy:
                self.writers[policy_id].add_scalar('0_aux/fps', fps, env_steps)
                self.writers[policy_id].add_scalar('0_aux/master_process_memory_mb', float(memory_mb), env_steps)
                for key, value in self.avg_stats.items():
                    if len(value) >= value.maxlen:
                        self.writers[policy_id].add_scalar(f'stats/{key}', np.mean(value), env_steps)

            if math.isnan(avg_reward[policy_id]) or math.isnan(avg_length[policy_id]):
                # not enough data to report yet
                continue

            self.writers[policy_id].add_scalar('0_aux/sample_throughput', sample_throughput[policy_id], env_steps)
            self.writers[policy_id].add_scalar('0_aux/avg_reward', float(avg_reward[policy_id]), env_steps)
            self.writers[policy_id].add_scalar('0_aux/avg_length', float(avg_length[policy_id]), env_steps)

    def _should_end_training(self):
        end = len(self.env_steps) > 0 and all(s > self.cfg.train_for_env_steps for s in self.env_steps.values())
        end |= self.total_train_seconds > self.cfg.train_for_seconds

        if self.cfg.benchmark:
            end |= self.total_env_steps_since_resume >= int(1e6)
            end |= sum(self.samples_collected) >= int(4e5)

        return end

    def learn(self):
        self.init_workers()

        log.info('Collecting experience...')

        timing = Timing()
        with timing.timeit('experience'):

            while not self._should_end_training():
                for w in self.learner_workers.values():
                    while True:
                        try:
                            report = w.report_queue.get(timeout=0.01)
                            self.process_report(report)
                        except Empty:
                            break

                if time.time() - self.last_report > self.report_interval:
                    self.report()

                    now = time.time()
                    self.total_train_seconds += now - self.last_report
                    self.last_report = now

        all_workers = self.actor_workers
        for workers in self.policy_workers.values():
            all_workers.extend(workers)
        all_workers.extend(self.learner_workers.values())

        time.sleep(1.0)
        for w in all_workers:
            w.close()
            time.sleep(0.01)
        for w in all_workers:
            w.join()

        fps = sum(self.env_steps.values()) / timing.experience
        log.info('Collected %r, FPS: %.1f', self.env_steps, fps)
        log.info('Timing: %s', timing)

        time.sleep(0.1)
        log.info('Done!')


# No training
# W20 V20 S2 G2: 26591FPS
# [2019-11-20 19:32:22,965] Gpu worker timing: init: 3.7416, gpu_waiting: 5.6309, deserialize: 5.3061, obs_dict: 0.0868, to_device: 3.9529, forward: 14.0111, serialize: 5.3650, postprocess: 6.7834, policy_step: 31.0437, work: 31.8166
# [2019-11-20 19:32:22,993] Env runner 0: timing waiting: 0.5965, reset: 20.5919, parse_policy_outputs: 0.0004, env_step: 26.4536, finalize: 3.9813, overhead: 4.7497, format_output: 4.6372, one_step: 0.0234, work: 36.8783

# W20 V20 S1 G2: 24996FPS
# [2019-11-20 19:49:01,397] Gpu worker timing: init: 3.6439, gpu_waiting: 9.9744, deserialize: 3.5391, obs_dict: 0.0786, to_device: 4.1121, forward: 16.6075, serialize: 2.7663, postprocess: 4.0433, policy_step: 29.2388, work: 29.9234
# [2019-11-20 19:49:01,404] Env runner 1: timing waiting: 6.4043, reset: 21.3081, parse_policy_outputs: 0.0006, env_step: 24.1964, finalize: 3.8485, overhead: 4.5882, format_output: 4.0478, one_step: 0.0533, work: 33.5044

# W32 V20 S2 (2 GPU workers): 30370FPS
# [2019-11-20 19:17:19,969] Gpu worker timing: init: 3.7086, gpu_waiting: 3.6520, work: 29.1827
# [2019-11-20 19:17:19,970] Env runner 1: timing waiting: 4.4399, reset: 21.1310, parse_policy_outputs: 0.0007, env_step: 19.1307, finalize: 3.5949, overhead: 4.1450, format_output: 3.8386, one_step: 0.0311, work: 28.1974

# W32 V40 S2 (2 GPU workers): 30701FPS
# [2019-11-20 19:24:17,261] Env runner 0: timing waiting: 1.4417, reset: 42.2417, parse_policy_outputs: 0.0015, env_step: 21.1332, finalize: 3.9994, overhead: 4.6047, format_output: 4.0152, one_step: 0.0813, work: 30.7172
# [2019-11-20 19:24:17,339] Env runner 1: timing waiting: 1.3387, reset: 39.7958, parse_policy_outputs: 0.0026, env_step: 21.2498, finalize: 3.7511, overhead: 4.4223, format_output: 4.2317, one_step: 0.0676, work: 30.8883

# W32 V40 S1 G2: 30529FPS
# [2019-11-20 19:56:44,631] Gpu worker timing: init: 3.5720, gpu_waiting: 8.4949, deserialize: 4.6235, obs_dict: 0.0809, to_device: 4.2894, forward: 9.3965, serialize: 3.6527, postprocess: 4.4345, policy_step: 23.5292, work: 24.2091
# [2019-11-20 19:56:44,669] Env runner 0: timing waiting: 4.6958, reset: 44.1553, parse_policy_outputs: 0.0010, env_step: 19.5480, finalize: 3.8980, overhead: 4.5100, format_output: 3.4880, one_step: 0.1341, work: 28.0031
