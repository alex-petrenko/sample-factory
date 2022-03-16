import json
import math
import os
import time
from collections import deque, OrderedDict
from os.path import join
from typing import Dict

import numpy as np
from tensorboardX import SummaryWriter

from sample_factory.algorithms.appo.appo_utils import iterate_recursively
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus, EXTRA_EPISODIC_STATS_PROCESSING, \
    EXTRA_PER_POLICY_SUMMARIES
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import AttrDict, done_filename, ensure_dir_exists, experiment_dir, log, \
    memory_consumption_mb, summaries_dir, save_git_diff, init_file_logger, cfg_file
from sample_factory.utils.wandb_utils import init_wandb


class Callback:
    def __init__(self, func):
        self.func = func

    def conditions_met(self, runner):
        raise NotImplementedError()

    def __call__(self, runner):
        self.func(runner)


class PeriodicTimerCallback(Callback):
    def __init__(self, func, period_sec):
        super().__init__(func)
        self.period_sec = period_sec
        self.last_call = time.time() - period_sec  # so we call this callback as soon as the app starts

    def conditions_met(self, runner):
        return time.time() - self.last_call > self.period_sec

    def __call__(self, runner):
        self.last_call = time.time()
        super().__call__(runner)


class PeriodicCallbackEnvSteps(Callback):
    def __init__(self, func, period_env_steps):
        super().__init__(func)
        self.period_env_steps = period_env_steps
        self.last_call = 0

    def conditions_met(self, runner):
        env_steps_since_last_call = runner.total_env_steps_since_resume - self.last_call
        return env_steps_since_last_call >= self.period_env_steps

    def __call__(self, runner):
        self.last_call = runner.total_env_steps_since_resume
        super().__call__(runner)


class PeriodicCallbackEnvStepsPerPolicy(PeriodicCallbackEnvSteps):
    def __init__(self, func, period_env_steps, policy_id):
        super().__init__(func, period_env_steps)
        self.policy_id = policy_id

    def conditions_met(self, runner):
        env_steps_since_last_call = runner.env_steps[self.policy_id] - self.last_call
        return env_steps_since_last_call >= self.period_env_steps

    def __call__(self, runner):
        self.last_call = runner.env_steps[self.policy_id]
        self.func(runner, self.policy_id)


class Runner(Configurable):
    def __init__(self, cfg, comm_broker, sampler, batcher, learner, timing=None):
        super().__init__(cfg)

        self.timing = Timing() if timing is None else timing

        # component used for communication between components (i.e. sampler, learner) and the runner
        # this way we can keep a consistent interface between sync and async runners
        self.comm_broker = comm_broker

        self.sampler = sampler
        self.batcher = batcher
        self.learner = learner

        # env_steps counts total number of simulation steps per policy (including frameskipped)
        self.env_steps = dict()

        # samples_collected counts the total number of observations processed by the algorithm
        self.samples_collected = [0 for _ in range(self.cfg.num_policies)]

        self.total_env_steps_since_resume = 0

        # currently, this applies only to the current run, not experiment as a whole
        # to change this behavior we'd need to save the state of the main loop to a filesystem
        self.total_train_seconds = 0

        self.last_report = time.time()

        self.report_interval_sec = 5.0
        self.avg_stats_intervals = (2, 12, 60)  # by default: 10 seconds, 60 seconds, 5 minutes
        self.summaries_interval_sec = self.cfg.experiment_summaries_interval  # sec

        self.fps_stats = deque([], maxlen=max(self.avg_stats_intervals))
        self.throughput_stats = [deque([], maxlen=5) for _ in range(self.cfg.num_policies)]

        self.stats = dict()  # regular (non-averaged) stats
        self.avg_stats = dict()

        self.policy_avg_stats = dict()
        self.policy_lag = [dict() for _ in range(self.cfg.num_policies)]

        init_wandb(self.cfg)  # should be done before writers are initialized

        self.writers: Dict[int, SummaryWriter] = dict()
        for policy_id in range(self.cfg.num_policies):
            summary_dir = join(summaries_dir(experiment_dir(cfg=self.cfg)), str(policy_id))
            summary_dir = ensure_dir_exists(summary_dir)
            self.writers[policy_id] = SummaryWriter(summary_dir, flush_secs=20)

        # global msg handlers for messages from algo components
        self.msg_handlers = dict(
            timing=[self._timing_msg_handler],
            stats=[self._stats_msg_handler],
        )

        # handlers for policy-specific messages
        self.policy_msg_handlers = dict(
            learner_env_steps=[self._learner_steps_handler],
            episodic=[self._episodic_stats_handler],
            train=[self._train_stats_handler],
            samples_collected=[self._samples_stats_handler],
        )

        self.periodic_callbacks = []
        self.register_timer_callback(self._update_stats_and_print_report, period_sec=self.report_interval_sec)
        self.register_timer_callback(self._report_experiment_summaries, period_sec=self.summaries_interval_sec)
        self.register_timer_callback(self._propagate_training_info, period_sec=5)
        # TODO: save model callback

    def _should_end_training(self):
        end = len(self.env_steps) > 0 and all(s > self.cfg.train_for_env_steps for s in self.env_steps.values())
        end |= self.total_train_seconds > self.cfg.train_for_seconds

        if self.cfg.benchmark:
            end |= self.total_env_steps_since_resume >= int(2e6)
            end |= sum(self.samples_collected) >= int(1e6)

        return end

    def _process_msgs(self, msgs):
        for msg in msgs:
            if not isinstance(msg, (dict, OrderedDict)):
                log.error('While parsing a message: expected a dictionary, found %r', msg)
                continue

            # some messages are policy-specific
            policy_id = msg.get('policy_id', None)

            for key in msg:
                for handler in self.msg_handlers.get(key, []):
                    handler(self, msg)
                for handler in self.policy_msg_handlers.get(key, []):
                    handler(self, msg, policy_id)

    @staticmethod
    def _timing_msg_handler(runner, msg):
        for k, v in msg['timing'].items():
            if k not in runner.avg_stats:
                runner.avg_stats[k] = deque([], maxlen=50)
            runner.avg_stats[k].append(v)

    @staticmethod
    def _stats_msg_handler(runner, msg):
        runner.stats.update(msg['stats'])

    @staticmethod
    def _learner_steps_handler(runner, msg, policy_id):
        if policy_id in runner.env_steps:
            delta = msg['learner_env_steps'] - runner.env_steps[policy_id]
            runner.total_env_steps_since_resume += delta
        runner.env_steps[policy_id] = msg['learner_env_steps']

    @staticmethod
    def _episodic_stats_handler(runner, msg, policy_id):
        s = msg['episodic']
        for _, key, value in iterate_recursively(s):
            if key not in runner.policy_avg_stats:
                runner.policy_avg_stats[key] = [
                    deque(maxlen=runner.cfg.stats_avg) for _ in range(runner.cfg.num_policies)
                ]

            runner.policy_avg_stats[key][policy_id].append(value)

            for extra_stat_func in EXTRA_EPISODIC_STATS_PROCESSING:  # TODO: replace this with an extra handler
                extra_stat_func(policy_id, key, value, runner.cfg)

    @staticmethod
    def _train_stats_handler(runner, msg, policy_id):
        """We write the train summaries to disk right away instead of accumulating them."""
        for key, scalar in msg['train'].items():
            runner.writers[policy_id].add_scalar(f'train/{key}', scalar, runner.env_steps[policy_id])

        for key in ['version_diff_min', 'version_diff_max', 'version_diff_avg']:
            if key in msg['train']:
                runner.policy_lag[policy_id][key] = msg['train'][key]

    @staticmethod
    def _samples_stats_handler(runner, msg, policy_id):
        runner.samples_collected[policy_id] += msg['samples_collected']

    @staticmethod
    def _register_msg_handler(handlers_dict, key, func):
        handlers_dict[key] = func

    def _periodic_callbacks(self):
        for callback in self.periodic_callbacks:
            if callback.conditions_met(self):
                callback(self)

    def _get_perf_stats(self):
        # total env steps simulated across all policies
        fps_stats = []
        for avg_interval in self.avg_stats_intervals:
            fps_for_interval = math.nan
            if len(self.fps_stats) > 1:
                t1, x1 = self.fps_stats[max(0, len(self.fps_stats) - 1 - avg_interval)]
                t2, x2 = self.fps_stats[-1]
                fps_for_interval = (x2 - x1) / (t2 - t1)

            fps_stats.append(fps_for_interval)

        # learning throughput per policy (in observations per sec)
        sample_throughput = dict()
        for policy_id in range(self.cfg.num_policies):
            sample_throughput[policy_id] = math.nan
            if len(self.throughput_stats[policy_id]) > 1:
                t1, x1 = self.throughput_stats[policy_id][0]
                t2, x2 = self.throughput_stats[policy_id][-1]
                sample_throughput[policy_id] = (x2 - x1) / (t2 - t1)

        return fps_stats, sample_throughput

    def print_stats(self, fps, sample_throughput, total_env_steps):
        fps_str = []
        for interval, fps_value in zip(self.avg_stats_intervals, fps):
            fps_str.append(f'{int(interval * self.report_interval_sec)} sec: {fps_value:.1f}')
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

    # noinspection PyProtectedMember
    @staticmethod
    def _update_stats_and_print_report(runner):
        """
        Called periodically (every self.report_interval_sec seconds).
        Print experiment stats (FPS, avg rewards) to console and dump TF summaries collected from workers to disk.
        """

        # don't have enough statistic from the learners yet
        if len(runner.env_steps) < runner.cfg.num_policies:
            return

        now = time.time()
        runner.fps_stats.append((now, runner.total_env_steps_since_resume))

        for policy_id in range(runner.cfg.num_policies):
            runner.throughput_stats[policy_id].append((now, runner.samples_collected[policy_id]))

        fps_stats, sample_throughput = runner._get_perf_stats()
        total_env_steps = sum(runner.env_steps.values())
        runner.print_stats(fps_stats, sample_throughput, total_env_steps)

    # noinspection PyProtectedMember
    @staticmethod
    def _report_experiment_summaries(runner):
        memory_mb = memory_consumption_mb()

        fps_stats, sample_throughput = runner._get_perf_stats()
        fps = fps_stats[0]

        default_policy = 0
        for policy_id, env_steps in runner.env_steps.items():
            if policy_id == default_policy:
                if not math.isnan(fps):
                    runner.writers[policy_id].add_scalar('perf/_fps', fps, env_steps)

                runner.writers[policy_id].add_scalar('stats/master_process_memory_mb', float(memory_mb), env_steps)
                for key, value in runner.avg_stats.items():
                    if len(value) >= value.maxlen or (len(value) > 10 and runner.total_train_seconds > 300):
                        runner.writers[policy_id].add_scalar(f'stats/{key}', np.mean(value), env_steps)

                for key, value in runner.stats.items():
                    runner.writers[policy_id].add_scalar(f'stats/{key}', value, env_steps)

            if not math.isnan(sample_throughput[policy_id]):
                runner.writers[policy_id].add_scalar('perf/_sample_throughput', sample_throughput[policy_id], env_steps)

            for key, stat in runner.policy_avg_stats.items():
                if len(stat[policy_id]) >= stat[policy_id].maxlen or (len(stat[policy_id]) > 10 and runner.total_train_seconds > 300):
                    stat_value = np.mean(stat[policy_id])
                    writer = runner.writers[policy_id]

                    # custom summaries have their own sections in tensorboard
                    if '/' in key:
                        avg_tag = key
                        min_tag = f'{key}_min'
                        max_tag = f'{key}_max'
                    else:
                        avg_tag = f'policy/avg_{key}'
                        min_tag = f'policy/avg_{key}_min'
                        max_tag = f'policy/avg_{key}_max'

                    writer.add_scalar(avg_tag, float(stat_value), env_steps)

                    # for key stats report min/max as well
                    if key in ('reward', 'true_reward', 'len'):
                        writer.add_scalar(min_tag, float(min(stat[policy_id])), env_steps)
                        writer.add_scalar(max_tag, float(max(stat[policy_id])), env_steps)

            for extra_summaries_func in EXTRA_PER_POLICY_SUMMARIES:  # TODO: replace with extra callbacks/handlers
                extra_summaries_func(
                    policy_id, runner.policy_avg_stats, env_steps, runner.writers[policy_id], runner.cfg,
                )

        # flush
        for w in runner.writers.values():
            w.flush()

    @staticmethod
    def _propagate_training_info(runner):
        """
        Send the training stats (such as the number of processed env steps) to the sampler.
        This can be used later by the envs to configure curriculums and so on.
        """
        runner.sampler.update_training_info(runner.env_steps, runner.stats, runner.avg_stats, runner.policy_avg_stats)

        # TODO!
        # for w in self.actor_workers:
        #     w.update_env_steps(self.env_steps)

    def register_msg_handler(self, key, func):
        self._register_msg_handler(self.msg_handlers, key, func)

    def register_policy_msg_handler(self, key, func):
        self._register_msg_handler(self.policy_msg_handlers, key, func)

    def register_periodic_callback(self, callback):
        self.periodic_callbacks.append(callback)

    def register_timer_callback(self, func, period_sec):
        callback = PeriodicTimerCallback(func, period_sec)
        self.register_periodic_callback(callback)

    def register_periodic_callback_env_steps(self, func, period_env_steps):
        callback = PeriodicCallbackEnvSteps(func, period_env_steps)
        self.register_periodic_callback(callback)

    def register_periodic_callback_env_steps_per_policy(self, func, period_env_steps, policy_id):
        callback = PeriodicCallbackEnvStepsPerPolicy(func, period_env_steps, policy_id)
        self.register_periodic_callback(callback)

    def _cfg_dict(self):
        if isinstance(self.cfg, dict):
            return self.cfg
        else:
            return vars(self.cfg)

    def _save_cfg(self):
        cfg_dict = self._cfg_dict()
        with open(cfg_file(self.cfg), 'w') as json_file:
            json.dump(cfg_dict, json_file, indent=2)

    def init(self):
        self._save_cfg()
        save_git_diff(experiment_dir(cfg=self.cfg))
        init_file_logger(experiment_dir(self.cfg))

    def run(self):
        status = ExperimentStatus.SUCCESS

        if os.path.isfile(done_filename(self.cfg)):
            log.warning(
                'Existence of the "done" file in the experiment folder indicates that this training session '
                'is finished! Remove "done" file if you wish to continue training'
            )
            return status

        with self.timing.timeit('main_loop'):
            # noinspection PyBroadException
            try:
                while not self._should_end_training():
                    self.algo_step(self.timing)
                    self.upkeep()

                    # TODO: add a periodic callback for this!
                    # self.pbt.update(self.env_steps, self.policy_avg_stats)

            except Exception:
                log.exception('Exception in the runner main loop')
                status = ExperimentStatus.FAILURE
            except KeyboardInterrupt:
                log.warning('Keyboard interrupt detected in the main loop, exiting...')
                status = ExperimentStatus.INTERRUPTED

        fps = self.total_env_steps_since_resume / self.timing.main_loop
        log.info('Collected %r, FPS: %.1f', self.env_steps, fps)
        log.info(self.timing)

        return status

    def upkeep(self):
        msgs = self.comm_broker.get_msgs(block=False, timeout=None)
        self._process_msgs(msgs)
        self._periodic_callbacks()

    def algo_step(self, timing):
        """Algorithm-specific step function called from the main loop of the runner."""
        raise NotImplementedError()
