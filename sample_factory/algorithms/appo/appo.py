"""
Algorithm entry poing.
Methods of the APPO class initiate all other components (rollout & policy workers and learners) in the main thread,
and then fork their separate processes.
All data structures that are shared between processes are also created during the construction of APPO.

This class contains the algorithm main loop. All the actual work is done in separate worker processes, so
the only task of the main loop is to collect summaries and stats from the workers and log/save them to disk.

Hyperparameters specific to policy gradient algorithms are defined in this file. See also algorithm.py.

"""

import json
import math
import multiprocessing
import os
import time
from collections import deque
from os.path import join
from queue import Empty

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.multiprocessing import JoinableQueue as TorchJoinableQueue

from sample_factory.algorithms.algorithm import ReinforcementLearningAlgorithm
from sample_factory.algorithms.appo.actor_worker import ActorWorker
from sample_factory.algorithms.appo.appo_utils import make_env_func, iterate_recursively, set_global_cuda_envvars
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.policy_worker import PolicyWorker
from sample_factory.algorithms.appo.population_based_training import PopulationBasedTraining
from sample_factory.algorithms.appo.shared_buffers import SharedBuffers
from sample_factory.algorithms.utils.algo_utils import EXTRA_PER_POLICY_SUMMARIES, EXTRA_EPISODIC_STATS_PROCESSING, \
    ExperimentStatus
from sample_factory.envs.env_utils import get_default_reward_shaping
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import summaries_dir, experiment_dir, log, str2bool, memory_consumption_mb, cfg_file, \
    ensure_dir_exists, list_child_processes, kill_processes, AttrDict, done_filename, save_git_diff, init_file_logger

if os.name == 'nt':
    from sample_factory.utils import Queue as MpQueue
else:
    from faster_fifo import Queue as MpQueue
    # noinspection PyUnresolvedReferences
    import faster_fifo_reduction


torch.multiprocessing.set_sharing_strategy('file_system')


class APPO(ReinforcementLearningAlgorithm):
    """Async PPO."""

    @classmethod
    def add_cli_args(cls, parser):
        p = parser
        super().add_cli_args(p)
        p.add_argument('--experiment_summaries_interval', default=20, type=int, help='How often in seconds we write avg. statistics about the experiment (reward, episode length, extra stats...)')

        p.add_argument('--adam_eps', default=1e-6, type=float, help='Adam epsilon parameter (1e-8 to 1e-5 seem to reliably work okay, 1e-3 and up does not work)')
        p.add_argument('--adam_beta1', default=0.9, type=float, help='Adam momentum decay coefficient')
        p.add_argument('--adam_beta2', default=0.999, type=float, help='Adam second momentum decay coefficient')

        p.add_argument('--gae_lambda', default=0.95, type=float, help='Generalized Advantage Estimation discounting (only used when V-trace is False')

        p.add_argument(
            '--rollout', default=32, type=int,
            help='Length of the rollout from each environment in timesteps.'
                 'Once we collect this many timesteps on actor worker, we send this trajectory to the learner.'
                 'The length of the rollout will determine how many timesteps are used to calculate bootstrapped'
                 'Monte-Carlo estimates of discounted rewards, advantages, GAE, or V-trace targets. Shorter rollouts'
                 'reduce variance, but the estimates are less precise (bias vs variance tradeoff).'
                 'For RNN policies, this should be a multiple of --recurrence, so every rollout will be split'
                 'into (n = rollout / recurrence) segments for backpropagation. V-trace algorithm currently requires that'
                 'rollout == recurrence, which what you want most of the time anyway.'
                 'Rollout length is independent from the episode length. Episode length can be both shorter or longer than'
                 'rollout, although for PBT training it is currently recommended that rollout << episode_len'
                 '(see function finalize_trajectory in actor_worker.py)',
        )

        p.add_argument('--num_workers', default=multiprocessing.cpu_count(), type=int, help='Number of parallel environment workers. Should be less than num_envs and should divide num_envs')

        p.add_argument(
            '--recurrence', default=32, type=int,
            help='Trajectory length for backpropagation through time. If recurrence=1 there is no backpropagation through time, and experience is shuffled completely randomly'
                 'For V-trace recurrence should be equal to rollout length.',
        )

        p.add_argument('--use_rnn', default=True, type=str2bool, help='Whether to use RNN core in a policy or not')
        p.add_argument('--rnn_type', default='gru', choices=['gru', 'lstm'], type=str, help='Type of RNN cell to use if use_rnn is True')
        p.add_argument('--rnn_num_layers', default=1, type=int, help='Number of RNN layers to use if use_rnn is True')

        p.add_argument('--ppo_clip_ratio', default=0.1, type=float, help='We use unbiased clip(x, 1+e, 1/(1+e)) instead of clip(x, 1+e, 1-e) in the paper')
        p.add_argument('--ppo_clip_value', default=1.0, type=float, help='Maximum absolute change in value estimate until it is clipped. Sensitive to value magnitude')
        p.add_argument('--batch_size', default=1024, type=int, help='Minibatch size for SGD')
        p.add_argument(
            '--num_batches_per_iteration', default=1, type=int,
            help='How many minibatches we collect before training on the collected experience. It is generally recommended to set this to 1 for most experiments, because any higher value will increase the policy lag.'
                 'But in some specific circumstances it can be beneficial to have a larger macro-batch in order to shuffle and decorrelate the minibatches.'
                 'Here and throughout the codebase: macro batch is the portion of experience that learner processes per iteration (consisting of 1 or several minibatches)',
        )
        p.add_argument('--ppo_epochs', default=1, type=int, help='Number of training epochs before a new batch of experience is collected')

        p.add_argument(
            '--num_minibatches_to_accumulate', default=-1, type=int,
            help='This parameter governs the maximum number of minibatches the learner can accumulate before further experience collection is stopped.'
                 'The default value (-1) will set this to 2 * num_batches_per_iteration, so if the experience collection is faster than the training,'
                 'the learner will accumulate enough minibatches for 2 iterations of training (but no more). This is a good balance between policy-lag and throughput.'
                 'When the limit is reached, the learner will notify the actor workers that they ought to stop the experience collection until accumulated minibatches'
                 'are processed. Set this parameter to 1 * num_batches_per_iteration to further reduce policy-lag.' 
                 'If the experience collection is very non-uniform, increasing this parameter can increase overall throughput, at the cost of increased policy-lag.'
                 'A value of 0 is treated specially. This means the experience accumulation is turned off, and all experience collection will be halted during training.'
                 'This is the regime with potentially lowest policy-lag.'
                 'When this parameter is 0 and num_workers * num_envs_per_worker * rollout == num_batches_per_iteration * batch_size, the algorithm is similar to'
                 'regular synchronous PPO.',
        )

        p.add_argument('--max_grad_norm', default=4.0, type=float, help='Max L2 norm of the gradient vector')

        # components of the loss function
        p.add_argument('--exploration_loss_coeff', default=0.003, type=float,
                       help='Coefficient for the exploration component of the loss function.')
        p.add_argument('--value_loss_coeff', default=0.5, type=float, help='Coefficient for the critic loss')
        p.add_argument('--kl_loss_coeff', default=0.0, type=float,
                       help='Coefficient for fixed KL loss (as used by Schulman et al. in https://arxiv.org/pdf/1707.06347.pdf). '
                            'Highly recommended for environments with continuous action spaces.',
                       )
        p.add_argument('--exploration_loss', default='entropy', type=str, choices=['entropy', 'symmetric_kl'],
                       help='Usually the exploration loss is based on maximizing the entropy of the probability'
                            ' distribution. Note that mathematically maximizing entropy of the categorical probability '
                            'distribution is exactly the same as minimizing the (regular) KL-divergence between'
                            ' this distribution and a uniform prior. The downside of using the entropy term '
                            '(or regular asymmetric KL-divergence) is the fact that penalty does not increase as '
                            'probabilities of some actions approach zero. I.e. numerically, there is almost '
                            'no difference between an action distribution with a probability epsilon > 0 for '
                            'some action and an action distribution with a probability = zero for this action.'
                            ' For many tasks the first (epsilon) distribution is preferrable because we keep some '
                            '(albeit small) amount of exploration, while the second distribution will never explore '
                            'this action ever again.'
                            'Unlike the entropy term, symmetric KL divergence between the action distribution '
                            'and a uniform prior approaches infinity when entropy of the distribution approaches zero,'
                            ' so it can prevent the pathological situations where the agent stops exploring. '
                            'Empirically, symmetric KL-divergence yielded slightly better results on some problems.',
                       )

        # APPO-specific
        p.add_argument(
            '--num_envs_per_worker', default=2, type=int,
            help='Number of envs on a single CPU actor, in high-throughput configurations this should be in 10-30 range for Atari/VizDoom'
                 'Must be even for double-buffered sampling!',
        )
        p.add_argument(
            '--worker_num_splits', default=2, type=int,
            help='Typically we split a vector of envs into two parts for "double buffered" experience collection'
                 'Set this to 1 to disable double buffering. Set this to 3 for triple buffering!',
        )

        p.add_argument('--num_policies', default=1, type=int, help='Number of policies to train jointly')
        p.add_argument('--policy_workers_per_policy', default=1, type=int, help='Number of policy workers that compute forward pass (per policy)')
        p.add_argument(
            '--max_policy_lag', default=10000, type=int,
            help='Max policy lag in policy versions. Discard all experience that is older than this. This should be increased for configurations with multiple epochs of SGD because naturally'
                 'policy-lag may exceed this value.',
        )
        p.add_argument(
            '--traj_buffers_excess_ratio', default=1.3, type=float,
            help='Increase this value to make sure the system always has enough free trajectory buffers (can be useful when i.e. a lot of inactive agents in multi-agent envs)'
                 'Decrease this to 1.0 to save as much RAM as possible.',
        )
        p.add_argument(
            '--decorrelate_experience_max_seconds', default=10, type=int,
            help='Decorrelating experience serves two benefits. First: this is better for learning because samples from workers come from random moments in the episode, becoming more "i.i.d".'
                 'Second, and more important one: this is good for environments with highly non-uniform one-step times, including long and expensive episode resets. If experience is not decorrelated'
                 'then training batches will come in bursts e.g. after a bunch of environments finished resets and many iterations on the learner might be required,'
                 'which will increase the policy-lag of the new experience collected. The performance of the Sample Factory is best when experience is generated as more-or-less'
                 'uniform stream. Try increasing this to 100-200 seconds to smoothen the experience distribution in time right from the beginning (it will eventually spread out and settle anyway)',
        )
        p.add_argument(
            '--decorrelate_envs_on_one_worker', default=True, type=str2bool,
            help='In addition to temporal decorrelation of worker processes, also decorrelate envs within one worker process'
                 'For environments with a fixed episode length it can prevent the reset from happening in the same rollout for all envs simultaneously, which makes experience collection more uniform.',
        )

        p.add_argument('--with_vtrace', default=True, type=str2bool, help='Enables V-trace off-policy correction. If this is True, then GAE is not used')
        p.add_argument('--vtrace_rho', default=1.0, type=float, help='rho_hat clipping parameter of the V-trace algorithm (importance sampling truncation)')
        p.add_argument('--vtrace_c', default=1.0, type=float, help='c_hat clipping parameter of the V-trace algorithm. Low values for c_hat can reduce variance of the advantage estimates (similar to GAE lambda < 1)')

        p.add_argument(
            '--set_workers_cpu_affinity', default=True, type=str2bool,
            help='Whether to assign workers to specific CPU cores or not. The logic is beneficial for most workloads because prevents a lot of context switching.'
                 'However for some environments it can be better to disable it, to allow one worker to use all cores some of the time. This can be the case for some DMLab environments with very expensive episode reset'
                 'that can use parallel CPU cores for level generation.',
        )
        p.add_argument(
            '--force_envs_single_thread', default=True, type=str2bool,
            help='Some environments may themselves use parallel libraries such as OpenMP or MKL. Since we parallelize environments on the level of workers, there is no need to keep this parallel semantic.'
                 'This flag uses threadpoolctl to force libraries such as OpenMP and MKL to use only a single thread within the environment.'
                 'Default value (True) is recommended unless you are running fewer workers than CPU cores.',
        )
        p.add_argument('--reset_timeout_seconds', default=120, type=int, help='Fail worker on initialization if not a single environment was reset in this time (worker probably got stuck)')

        p.add_argument('--default_niceness', default=0, type=int, help='Niceness of the highest priority process (the learner). Values below zero require elevated privileges.')

        p.add_argument(
            '--train_in_background_thread', default=True, type=str2bool,
            help='Using background thread for training is faster and allows preparing the next batch while training is in progress.'
                 'Unfortunately debugging can become very tricky in this case. So there is an option to use only a single thread on the learner to simplify the debugging.',
        )
        p.add_argument('--learner_main_loop_num_cores', default=1, type=int, help='When batching on the learner is the bottleneck, increasing the number of cores PyTorch uses can improve the performance')
        p.add_argument('--actor_worker_gpus', default=[], type=int, nargs='*', help='By default, actor workers only use CPUs. Changes this if e.g. you need GPU-based rendering on the actors')

        # PBT stuff
        p.add_argument('--with_pbt', default=False, type=str2bool, help='Enables population-based training basic features')
        p.add_argument('--pbt_mix_policies_in_one_env', default=True, type=str2bool, help='For multi-agent envs, whether we mix different policies in one env.')
        p.add_argument('--pbt_period_env_steps', default=int(5e6), type=int, help='Periodically replace the worst policies with the best ones and perturb the hyperparameters')
        p.add_argument('--pbt_start_mutation', default=int(2e7), type=int, help='Allow initial diversification, start PBT after this many env steps')
        p.add_argument('--pbt_replace_fraction', default=0.3, type=float, help='A portion of policies performing worst to be replace by better policies (rounded up)')
        p.add_argument('--pbt_mutation_rate', default=0.15, type=float, help='Probability that a parameter mutates')
        p.add_argument('--pbt_replace_reward_gap', default=0.1, type=float, help='Relative gap in true reward when replacing weights of the policy with a better performing one')
        p.add_argument('--pbt_replace_reward_gap_absolute', default=1e-6, type=float, help='Absolute gap in true reward when replacing weights of the policy with a better performing one')
        p.add_argument('--pbt_optimize_batch_size', default=False, type=str2bool, help='Whether to optimize batch size or not (experimental)')
        p.add_argument(
            '--pbt_target_objective', default='true_reward', type=str,
            help='Policy stat to optimize with PBT. true_reward (default) is equal to raw env reward if not specified, but can also be any other per-policy stat.'
                 'For DMlab-30 use value "dmlab_target_objective" (which is capped human normalized score)',
        )

        # CPC|A options
        p.add_argument('--use_cpc', default=False, type=str2bool, help='Use CPC|A as an auxiliary loss durning learning')
        p.add_argument('--cpc_forward_steps', default=8, type=int, help='Number of forward prediction steps for CPC')
        p.add_argument('--cpc_time_subsample', default=6, type=int, help='Number of timesteps to sample from each batch. This should be less than recurrence to decorrelate experience.')
        p.add_argument('--cpc_forward_subsample', default=2, type=int, help='Number of forward steps to sample for loss computation. This should be less than cpc_forward_steps to decorrelate gradients.')

        # debugging options
        p.add_argument('--benchmark', default=False, type=str2bool, help='Benchmark mode')
        p.add_argument('--sampler_only', default=False, type=str2bool, help='Do not send experience to the learner, measuring sampling throughput')

    def __init__(self, cfg):
        super().__init__(cfg)

        # we should not use CUDA in the main thread, only on the workers
        set_global_cuda_envvars(cfg)

        tmp_env = make_env_func(self.cfg, env_config=None)
        self.obs_space = tmp_env.observation_space
        self.action_space = tmp_env.action_space
        self.num_agents = tmp_env.num_agents

        self.reward_shaping_scheme = None
        if self.cfg.with_pbt:
            self.reward_shaping_scheme = get_default_reward_shaping(tmp_env)

        tmp_env.close()

        # shared memory allocation
        self.traj_buffers = SharedBuffers(self.cfg, self.num_agents, self.obs_space, self.action_space)

        self.actor_workers = None

        self.report_queue = MpQueue(40 * 1000 * 1000)
        self.policy_workers = dict()
        self.policy_queues = dict()

        self.learner_workers = dict()

        self.workers_by_handle = None

        self.policy_inputs = [[] for _ in range(self.cfg.num_policies)]
        self.policy_outputs = dict()
        for worker_idx in range(self.cfg.num_workers):
            for split_idx in range(self.cfg.worker_num_splits):
                self.policy_outputs[(worker_idx, split_idx)] = dict()

        self.policy_avg_stats = dict()
        self.policy_lag = [dict() for _ in range(self.cfg.num_policies)]

        self.last_timing = dict()
        self.env_steps = dict()
        self.samples_collected = [0 for _ in range(self.cfg.num_policies)]
        self.total_env_steps_since_resume = 0

        # currently this applies only to the current run, not experiment as a whole
        # to change this behavior we'd need to save the state of the main loop to a filesystem
        self.total_train_seconds = 0

        self.last_report = time.time()
        self.last_experiment_summaries = 0

        self.report_interval = 5.0  # sec
        self.experiment_summaries_interval = self.cfg.experiment_summaries_interval  # sec

        self.avg_stats_intervals = (2, 12, 60)  # 10 seconds, 1 minute, 5 minutes

        self.fps_stats = deque([], maxlen=max(self.avg_stats_intervals))
        self.throughput_stats = [deque([], maxlen=5) for _ in range(self.cfg.num_policies)]
        self.avg_stats = dict()
        self.stats = dict()  # regular (non-averaged) stats

        self.writers = dict()
        writer_keys = list(range(self.cfg.num_policies))
        for key in writer_keys:
            summary_dir = join(summaries_dir(experiment_dir(cfg=self.cfg)), str(key))
            summary_dir = ensure_dir_exists(summary_dir)
            self.writers[key] = SummaryWriter(summary_dir, flush_secs=20)

        self.pbt = PopulationBasedTraining(self.cfg, self.reward_shaping_scheme, self.writers)

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
        self._save_cfg()
        save_git_diff(experiment_dir(cfg=self.cfg))
        init_file_logger(experiment_dir(self.cfg))

    def finalize(self):
        pass

    def create_actor_worker(self, idx, actor_queue):
        learner_queues = {p: w.task_queue for p, w in self.learner_workers.items()}

        return ActorWorker(
            self.cfg, self.obs_space, self.action_space, self.num_agents, idx, self.traj_buffers,
            task_queue=actor_queue, policy_queues=self.policy_queues,
            report_queue=self.report_queue, learner_queues=learner_queues,
        )

    # noinspection PyProtectedMember
    def init_subset(self, indices, actor_queues):
        """
        Initialize a subset of actor workers (rollout workers) and wait until the first reset() is completed for all
        envs on these workers.

        This function will retry if the worker process crashes during the initial reset.

        :param indices: indices of actor workers to initialize
        :param actor_queues: task queues corresponding to these workers
        :return: initialized workers
        """

        reset_timelimit_seconds = self.cfg.reset_timeout_seconds  # fail worker if not a single env was reset in that time

        workers = dict()
        last_env_initialized = dict()
        for i in indices:
            w = self.create_actor_worker(i, actor_queues[i])
            w.init()
            w.request_reset()
            workers[i] = w
            last_env_initialized[i] = time.time()

        total_num_envs = self.cfg.num_workers * self.cfg.num_envs_per_worker
        envs_initialized = [0] * self.cfg.num_workers
        workers_finished = set()

        while len(workers_finished) < len(workers):
            failed_worker = -1

            try:
                report = self.report_queue.get(timeout=1.0)

                if 'initialized_env' in report:
                    worker_idx, split_idx, env_i = report['initialized_env']
                    last_env_initialized[worker_idx] = time.time()
                    envs_initialized[worker_idx] += 1

                    log.debug(
                        'Progress for %d workers: %d/%d envs initialized...',
                        len(indices), sum(envs_initialized), total_num_envs,
                    )
                elif 'finished_reset' in report:
                    workers_finished.add(report['finished_reset'])
                elif 'critical_error' in report:
                    failed_worker = report['critical_error']
            except Empty:
                pass

            for worker_idx, w in workers.items():
                if worker_idx in workers_finished:
                    continue

                time_passed = time.time() - last_env_initialized[worker_idx]
                timeout = time_passed > reset_timelimit_seconds

                if timeout or failed_worker == worker_idx or not w.process.is_alive():
                    envs_initialized[worker_idx] = 0

                    log.error('Worker %d is stuck or failed (%.3f). Reset!', w.worker_idx, time_passed)
                    log.debug('Status: %r', w.process.is_alive())
                    stuck_worker = w
                    stuck_worker.process.kill()

                    new_worker = self.create_actor_worker(worker_idx, actor_queues[worker_idx])
                    new_worker.init()
                    new_worker.request_reset()

                    last_env_initialized[worker_idx] = time.time()
                    workers[worker_idx] = new_worker
                    del stuck_worker

        return workers.values()

    # noinspection PyUnresolvedReferences
    def init_workers(self):
        """
        Initialize all types of workers and start their worker processes.
        """

        actor_queues = [MpQueue(2 * 1000 * 1000) for _ in range(self.cfg.num_workers)]

        policy_worker_queues = dict()
        for policy_id in range(self.cfg.num_policies):
            policy_worker_queues[policy_id] = []
            for i in range(self.cfg.policy_workers_per_policy):
                policy_worker_queues[policy_id].append(TorchJoinableQueue())

        log.info('Initializing learners...')
        policy_locks = [multiprocessing.Lock() for _ in range(self.cfg.num_policies)]
        resume_experience_collection_cv = [multiprocessing.Condition() for _ in range(self.cfg.num_policies)]

        learner_idx = 0
        for policy_id in range(self.cfg.num_policies):
            learner_worker = LearnerWorker(
                learner_idx, policy_id, self.cfg, self.obs_space, self.action_space,
                self.report_queue, policy_worker_queues[policy_id], self.traj_buffers,
                policy_locks[policy_id], resume_experience_collection_cv[policy_id],
            )
            learner_worker.start_process()
            learner_worker.init()

            self.learner_workers[policy_id] = learner_worker
            learner_idx += 1

        log.info('Initializing policy workers...')
        for policy_id in range(self.cfg.num_policies):
            self.policy_workers[policy_id] = []

            policy_queue = MpQueue()
            self.policy_queues[policy_id] = policy_queue

            for i in range(self.cfg.policy_workers_per_policy):
                policy_worker = PolicyWorker(
                    i, policy_id, self.cfg, self.obs_space, self.action_space, self.traj_buffers,
                    policy_queue, actor_queues, self.report_queue, policy_worker_queues[policy_id][i],
                    policy_locks[policy_id], resume_experience_collection_cv[policy_id],
                )
                self.policy_workers[policy_id].append(policy_worker)
                policy_worker.start_process()

        log.info('Initializing actors...')

        # We support actor worker initialization in groups, which can be useful for some envs that
        # e.g. crash when too many environments are being initialized in parallel.
        # Currently the limit is not used since it is not required for any envs supported out of the box,
        # so we parallelize initialization as hard as we can.
        # If this is required for your environment, perhaps a better solution would be to use global locks,
        # like FileLock (see doom_gym.py)
        self.actor_workers = []
        max_parallel_init = int(1e9)  # might be useful to limit this for some envs
        worker_indices = list(range(self.cfg.num_workers))
        for i in range(0, self.cfg.num_workers, max_parallel_init):
            workers = self.init_subset(worker_indices[i:i + max_parallel_init], actor_queues)
            self.actor_workers.extend(workers)

    def init_pbt(self):
        if self.cfg.with_pbt:
            self.pbt.init(self.learner_workers, self.actor_workers)

    def finish_initialization(self):
        """Wait until policy workers are fully initialized."""
        for policy_id, workers in self.policy_workers.items():
            for w in workers:
                log.debug('Waiting for policy worker %d-%d to finish initialization...', policy_id, w.worker_idx)
                w.init()
                log.debug('Policy worker %d-%d initialized!', policy_id, w.worker_idx)

    def update_env_steps_actor(self):
        for w in self.actor_workers:
            w.update_env_steps(self.env_steps)

    def process_report(self, report):
        """Process stats from various types of workers."""

        if 'policy_id' in report:
            policy_id = report['policy_id']

            if 'learner_env_steps' in report:
                if policy_id in self.env_steps:
                    delta = report['learner_env_steps'] - self.env_steps[policy_id]
                    self.total_env_steps_since_resume += delta
                self.env_steps[policy_id] = report['learner_env_steps']

            if 'episodic' in report:
                s = report['episodic']
                for _, key, value in iterate_recursively(s):
                    if key not in self.policy_avg_stats:
                        self.policy_avg_stats[key] = [deque(maxlen=self.cfg.stats_avg) for _ in range(self.cfg.num_policies)]

                    self.policy_avg_stats[key][policy_id].append(value)

                    for extra_stat_func in EXTRA_EPISODIC_STATS_PROCESSING:
                        extra_stat_func(policy_id, key, value, self.cfg)

            if 'train' in report:
                self.report_train_summaries(report['train'], policy_id)

            if 'samples' in report:
                self.samples_collected[policy_id] += report['samples']

        if 'timing' in report:
            for k, v in report['timing'].items():
                if k not in self.avg_stats:
                    self.avg_stats[k] = deque([], maxlen=50)
                self.avg_stats[k].append(v)

        if 'stats' in report:
            self.stats.update(report['stats'])

    def report(self):
        """
        Called periodically (every X seconds, see report_interval).
        Print experiment stats (FPS, avg rewards) to console and dump TF summaries collected from workers to disk.
        """

        if len(self.env_steps) < self.cfg.num_policies:
            return

        now = time.time()
        self.fps_stats.append((now, self.total_env_steps_since_resume))
        if len(self.fps_stats) <= 1:
            return

        fps = []
        for avg_interval in self.avg_stats_intervals:
            past_moment, past_frames = self.fps_stats[max(0, len(self.fps_stats) - 1 - avg_interval)]
            fps.append((self.total_env_steps_since_resume - past_frames) / (now - past_moment))

        sample_throughput = dict()
        for policy_id in range(self.cfg.num_policies):
            self.throughput_stats[policy_id].append((now, self.samples_collected[policy_id]))
            if len(self.throughput_stats[policy_id]) > 1:
                past_moment, past_samples = self.throughput_stats[policy_id][0]
                sample_throughput[policy_id] = (self.samples_collected[policy_id] - past_samples) / (now - past_moment)
            else:
                sample_throughput[policy_id] = math.nan

        total_env_steps = sum(self.env_steps.values())
        self.print_stats(fps, sample_throughput, total_env_steps)

        if time.time() - self.last_experiment_summaries > self.experiment_summaries_interval:
            self.report_experiment_summaries(fps[0], sample_throughput)
            self.last_experiment_summaries = time.time()

    def print_stats(self, fps, sample_throughput, total_env_steps):
        fps_str = []
        for interval, fps_value in zip(self.avg_stats_intervals, fps):
            fps_str.append(f'{int(interval * self.report_interval)} sec: {fps_value:.1f}')
        fps_str = f'({", ".join(fps_str)})'

        samples_per_policy = ', '.join([f'{p}: {s:.1f}' for p, s in sample_throughput.items()])

        lag_stats = self.policy_lag[0]
        lag = AttrDict()
        for key in ['min', 'avg', 'max']:
            lag[key] = lag_stats.get(f'version_diff_{key}', -1)
        policy_lag_str = f'min: {lag.min:.1f}, avg: {lag.avg:.1f}, max: {lag.max:.1f}'

        log.debug(
            'Fps is %s. Total num frames: %d. Throughput: %s. Samples: %d. Policy #0 lag: (%s)',
            fps_str, total_env_steps, samples_per_policy, sum(self.samples_collected), policy_lag_str,
        )

        if 'reward' in self.policy_avg_stats:
            policy_reward_stats = []
            for policy_id in range(self.cfg.num_policies):
                reward_stats = self.policy_avg_stats['reward'][policy_id]
                if len(reward_stats) > 0:
                    policy_reward_stats.append((policy_id, f'{np.mean(reward_stats):.3f}'))
            log.debug('Avg episode reward: %r', policy_reward_stats)

    def report_train_summaries(self, stats, policy_id):
        for key, scalar in stats.items():
            self.writers[policy_id].add_scalar(f'train/{key}', scalar, self.env_steps[policy_id])
            if 'version_diff' in key:
                self.policy_lag[policy_id][key] = scalar

    def report_experiment_summaries(self, fps, sample_throughput):
        memory_mb = memory_consumption_mb()

        default_policy = 0
        for policy_id, env_steps in self.env_steps.items():
            if policy_id == default_policy:
                self.writers[policy_id].add_scalar('0_aux/_fps', fps, env_steps)
                self.writers[policy_id].add_scalar('0_aux/master_process_memory_mb', float(memory_mb), env_steps)
                for key, value in self.avg_stats.items():
                    if len(value) >= value.maxlen or (len(value) > 10 and self.total_train_seconds > 300):
                        self.writers[policy_id].add_scalar(f'stats/{key}', np.mean(value), env_steps)

                for key, value in self.stats.items():
                    self.writers[policy_id].add_scalar(f'stats/{key}', value, env_steps)

            if not math.isnan(sample_throughput[policy_id]):
                self.writers[policy_id].add_scalar('0_aux/_sample_throughput', sample_throughput[policy_id], env_steps)

            for key, stat in self.policy_avg_stats.items():
                if len(stat[policy_id]) >= stat[policy_id].maxlen or (len(stat[policy_id]) > 10 and self.total_train_seconds > 300):
                    stat_value = np.mean(stat[policy_id])
                    writer = self.writers[policy_id]

                    # custom summaries have their own sections in tensorboard
                    if '/' in key:
                        avg_tag = key
                        min_tag = f'{key}_min'
                        max_tag = f'{key}_max'
                    else:
                        avg_tag = f'0_aux/avg_{key}'
                        min_tag = f'0_aux/avg_{key}_min'
                        max_tag = f'0_aux/avg_{key}_max'

                    writer.add_scalar(avg_tag, float(stat_value), env_steps)

                    # for key stats report min/max as well
                    if key in ('reward', 'true_reward', 'len'):
                        writer.add_scalar(min_tag, float(min(stat[policy_id])), env_steps)
                        writer.add_scalar(max_tag, float(max(stat[policy_id])), env_steps)

            for extra_summaries_func in EXTRA_PER_POLICY_SUMMARIES:
                extra_summaries_func(policy_id, self.policy_avg_stats, env_steps, self.writers[policy_id], self.cfg)

    def _should_end_training(self):
        end = len(self.env_steps) > 0 and all(s > self.cfg.train_for_env_steps for s in self.env_steps.values())
        end |= self.total_train_seconds > self.cfg.train_for_seconds

        if self.cfg.benchmark:
            end |= self.total_env_steps_since_resume >= int(2e6)
            end |= sum(self.samples_collected) >= int(1e6)

        return end

    def run(self):
        """
        This function contains the main loop of the algorithm, as well as initialization/cleanup code.

        :return: ExperimentStatus (SUCCESS, FAILURE, INTERRUPTED). Useful in testing.
        """

        status = ExperimentStatus.SUCCESS

        if os.path.isfile(done_filename(self.cfg)):
            log.warning('Training already finished! Remove "done" file to continue training')
            return status

        self.init_workers()
        self.init_pbt()
        self.finish_initialization()

        log.info('Collecting experience...')

        timing = Timing()
        with timing.timeit('experience'):
            # noinspection PyBroadException
            try:
                while not self._should_end_training():
                    try:
                        reports = self.report_queue.get_many(timeout=0.1)
                        for report in reports:
                            self.process_report(report)
                    except Empty:
                        pass

                    if time.time() - self.last_report > self.report_interval:
                        self.report()

                        now = time.time()
                        self.total_train_seconds += now - self.last_report
                        self.last_report = now

                        self.update_env_steps_actor()

                    self.pbt.update(self.env_steps, self.policy_avg_stats)

            except Exception:
                log.exception('Exception in driver loop')
                status = ExperimentStatus.FAILURE
            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected in driver loop, exiting...')
                status = ExperimentStatus.INTERRUPTED

        for learner in self.learner_workers.values():
            # timeout is needed here because some environments may crash on KeyboardInterrupt (e.g. VizDoom)
            # Therefore the learner train loop will never do another iteration and will never save the model.
            # This is not an issue with normal exit, e.g. due to desired number of frames reached.
            learner.save_model(timeout=5.0)

        all_workers = self.actor_workers
        for workers in self.policy_workers.values():
            all_workers.extend(workers)
        all_workers.extend(self.learner_workers.values())

        child_processes = list_child_processes()

        time.sleep(0.1)
        log.debug('Closing workers...')
        for i, w in enumerate(all_workers):
            w.close()
            time.sleep(0.01)
        for i, w in enumerate(all_workers):
            w.join()
        log.debug('Workers joined!')

        # VizDoom processes often refuse to die for an unidentified reason, so we're force killing them with a hack
        kill_processes(child_processes)

        fps = self.total_env_steps_since_resume / timing.experience
        log.info('Collected %r, FPS: %.1f', self.env_steps, fps)
        log.info('Timing: %s', timing)

        if self._should_end_training():
            with open(done_filename(self.cfg), 'w') as fobj:
                fobj.write(f'{self.env_steps}')

        time.sleep(0.5)
        log.info('Done!')

        return status
