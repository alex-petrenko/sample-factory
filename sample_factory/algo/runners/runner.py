import json
import math
import multiprocessing
import time
from collections import deque, OrderedDict
from os.path import join
from typing import Dict, Tuple, List, Optional

import numpy as np
from tensorboardX import SummaryWriter

from sample_factory.algo.inference.inference_worker import InferenceWorker
from sample_factory.algo.learning.batcher import Batcher
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.rollout_worker import RolloutWorker
from sample_factory.algo.utils.env_info import obtain_env_info_in_a_separate_process, EnvInfo
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.multiprocessing_utils import get_queue
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.cfg.configurable import Configurable
from sample_factory.signal_slot.signal_slot import EventLoopObject, EventLoop, EventLoopStatus, Timer, signal
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.utils.gpu_utils import set_global_cuda_envvars
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import PolicyID, MpQueue
from sample_factory.utils.utils import AttrDict, ensure_dir_exists, experiment_dir, log, \
    memory_consumption_mb, summaries_dir, save_git_diff, init_file_logger, cfg_file
from sample_factory.utils.wandb_utils import init_wandb

StatusCode = int


# class Callback:
#     def __init__(self, func):
#         self.func = func
#
#     def conditions_met(self, runner):
#         raise NotImplementedError()
#
#     def __call__(self, runner):
#         self.func(runner)
#
#
# class PeriodicTimerCallback(Callback):
#     def __init__(self, func, period_sec):
#         super().__init__(func)
#         self.period_sec = period_sec
#         self.last_call = time.time() - period_sec  # so we call this callback as soon as the app starts
#
#     def conditions_met(self, runner):
#         return time.time() - self.last_call > self.period_sec
#
#     def __call__(self, runner):
#         self.last_call = time.time()
#         super().__call__(runner)
#
#
# class PeriodicCallbackEnvSteps(Callback):
#     def __init__(self, func, period_env_steps):
#         super().__init__(func)
#         self.period_env_steps = period_env_steps
#         self.last_call = 0
#
#     def conditions_met(self, runner):
#         env_steps_since_last_call = runner.total_env_steps_since_resume - self.last_call
#         return env_steps_since_last_call >= self.period_env_steps
#
#     def __call__(self, runner):
#         self.last_call = runner.total_env_steps_since_resume
#         super().__call__(runner)
#
#
# class PeriodicCallbackEnvStepsPerPolicy(PeriodicCallbackEnvSteps):
#     def __init__(self, func, period_env_steps, policy_id):
#         super().__init__(func, period_env_steps)
#         self.policy_id = policy_id
#
#     def conditions_met(self, runner):
#         env_steps_since_last_call = runner.env_steps[self.policy_id] - self.last_call
#         return env_steps_since_last_call >= self.period_env_steps
#
#     def __call__(self, runner):
#         self.last_call = runner.env_steps[self.policy_id]
#         self.func(runner, self.policy_id)


class Runner(EventLoopObject, Configurable):
    def __init__(self, cfg, unique_name=None):
        Configurable.__init__(self, cfg)

        unique_name = Runner.__name__ if unique_name is None else unique_name
        self.event_loop: EventLoop = EventLoop(unique_loop_name=f'{unique_name}_EvtLoop', serial_mode=cfg.serial_mode)
        self.event_loop.owner = self
        EventLoopObject.__init__(self, self.event_loop, object_id=unique_name)

        self.stopped = False

        self.env_info: Optional[EnvInfo] = None
        self.buffer_mgr = None

        self.mp_ctx = self.multiprocessing_context()

        self.inference_queues: Dict[PolicyID, MpQueue] = {p: get_queue(cfg.serial_mode) for p in range(self.cfg.num_policies)}

        self.learners: Dict[PolicyID, Learner] = dict()
        self.batchers: Dict[PolicyID, Batcher] = dict()
        self.inference_workers: Dict[PolicyID, List[InferenceWorker]] = dict()
        self.rollout_workers: List[RolloutWorker] = []

        self._save_cfg()
        save_git_diff(experiment_dir(cfg=self.cfg))
        init_file_logger(experiment_dir(self.cfg))

        self.timing = Timing('Runner profile')

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
        self.throughput_stats = [deque([], maxlen=10) for _ in range(self.cfg.num_policies)]

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

        def periodic(period, cb):
            return Timer(self.event_loop, period).timeout.connect(cb)

        periodic(self.report_interval_sec, self._update_stats_and_print_report)
        periodic(self.summaries_interval_sec, self._report_experiment_summaries)

        periodic(self.cfg.save_every_sec, self._save_policy)
        periodic(self.cfg.save_best_every_sec, self._save_best_policy)

        periodic(5, self._propagate_training_info)

        self.components_to_stop: List[EventLoopObject] = []
        self.component_profiles: List[Tuple[str, Timing]] = []

    # singals emitted by the runner
    @signal
    def inference_workers_initialized(self): pass

    @signal
    def save_periodic(self): pass

    @signal
    def save_best(self): pass

    """Emitted when we're about to stop the experiment."""
    @signal
    def stop(self): pass

    @signal
    def all_components_stopped(self): pass

    def multiprocessing_context(self) -> Optional[multiprocessing.context.BaseContext]:
        raise NotImplementedError()

    def _process_msg(self, msgs):
        if isinstance(msgs, (dict, OrderedDict)):
            msgs = (msgs, )

        if not (isinstance(msgs, (List, Tuple)) and isinstance(msgs[0], (dict, OrderedDict))):
            log.error('While parsing a message: expected a dictionary or list/tuple of dictionaries, found %r', msgs)
            return

        for msg in msgs:
            # some messages are policy-specific
            policy_id = msg.get('policy_id', None)

            for key in msg:
                for handler in self.msg_handlers.get(key, []):
                    handler(self, msg)
                if policy_id is not None:
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

            if isinstance(value, np.ndarray) and value.ndim > 0:
                if len(value) > runner.policy_avg_stats[key][policy_id].maxlen:
                    # increase maxlen to make sure we never ignore any stats from the environments
                    runner.policy_avg_stats[key][policy_id] = deque(maxlen=len(value))

                runner.policy_avg_stats[key][policy_id].extend(value)
            else:
                runner.policy_avg_stats[key][policy_id].append(value)

            # for extra_stat_func in EXTRA_EPISODIC_STATS_PROCESSING:  # TODO: replace this with an extra handler
            #     extra_stat_func(policy_id, key, value, runner.cfg)

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

    def _update_stats_and_print_report(self):
        """
        Called periodically (every self.report_interval_sec seconds).
        Print experiment stats (FPS, avg rewards) to console and dump TF summaries collected from workers to disk.
        """

        # don't have enough statistic from the learners yet
        if len(self.env_steps) < self.cfg.num_policies:
            return

        now = time.time()
        self.fps_stats.append((now, self.total_env_steps_since_resume))

        for policy_id in range(self.cfg.num_policies):
            self.throughput_stats[policy_id].append((now, self.samples_collected[policy_id]))

        fps_stats, sample_throughput = self._get_perf_stats()
        total_env_steps = sum(self.env_steps.values())
        self.print_stats(fps_stats, sample_throughput, total_env_steps)

    def _report_experiment_summaries(self):
        memory_mb = memory_consumption_mb()

        fps_stats, sample_throughput = self._get_perf_stats()
        fps = fps_stats[0]

        default_policy = 0
        for policy_id, env_steps in self.env_steps.items():
            if policy_id == default_policy:
                if not math.isnan(fps):
                    self.writers[policy_id].add_scalar('perf/_fps', fps, env_steps)

                self.writers[policy_id].add_scalar('stats/master_process_memory_mb', float(memory_mb), env_steps)
                for key, value in self.avg_stats.items():
                    if len(value) >= value.maxlen or (len(value) > 10 and self.total_train_seconds > 300):
                        self.writers[policy_id].add_scalar(f'stats/{key}', np.mean(value), env_steps)

                for key, value in self.stats.items():
                    self.writers[policy_id].add_scalar(f'stats/{key}', value, env_steps)

            if not math.isnan(sample_throughput[policy_id]):
                self.writers[policy_id].add_scalar('perf/_sample_throughput', sample_throughput[policy_id], env_steps)

            for key, stat in self.policy_avg_stats.items():
                if len(stat[policy_id]) >= stat[policy_id].maxlen or (len(stat[policy_id]) > 10 and self.total_train_seconds > 300):
                    stat_value = np.mean(stat[policy_id])
                    writer = self.writers[policy_id]

                    if '/' in key:
                        # custom summaries have their own sections in tensorboard
                        avg_tag = key
                        min_tag = f'{key}_min'
                        max_tag = f'{key}_max'
                    elif key in ('reward', 'len'):
                        # reward and length get special treatment
                        avg_tag = f'{key}/{key}'
                        min_tag = f'{key}/{key}_min'
                        max_tag = f'{key}/{key}_max'
                    else:
                        avg_tag = f'policy_stats/avg_{key}'
                        min_tag = f'policy_stats/avg_{key}_min'
                        max_tag = f'policy_stats/avg_{key}_max'

                    writer.add_scalar(avg_tag, float(stat_value), env_steps)

                    # for key stats report min/max as well
                    if key in ('reward', 'true_objective', 'len'):
                        writer.add_scalar(min_tag, float(min(stat[policy_id])), env_steps)
                        writer.add_scalar(max_tag, float(max(stat[policy_id])), env_steps)

            # for extra_summaries_func in EXTRA_PER_POLICY_SUMMARIES:  # TODO: replace with extra callbacks/handlers
            #     extra_summaries_func(
            #         policy_id, self.policy_avg_stats, env_steps, self.writers[policy_id], self.cfg,
            #     )

        for w in self.writers.values():
            w.flush()

    def _propagate_training_info(self):
        """
        Send the training stats (such as the number of processed env steps) to the sampler.
        This can be used later by the envs to configure curriculums and so on.
        """
        # TODO: send this to rollout workers
        # self.sampler.update_training_info(self.env_steps, self.stats, self.avg_stats, self.policy_avg_stats)

        # TODO!
        # for w in self.actor_workers:
        #     w.update_env_steps(self.env_steps)
        pass

    def _save_policy(self):
        self.save_periodic.emit()

    def _save_best_policy(self):
        # don't have enough statistic from the learners yet
        if len(self.env_steps) < self.cfg.num_policies:
            return

        metric = self.cfg.save_best_metric
        if metric in self.policy_avg_stats:
            for policy_id in range(self.cfg.num_policies):
                # check if number of samples collected is greater than cfg.save_best_after
                env_steps = self.env_steps[policy_id]
                if env_steps < self.cfg.save_best_after:
                    continue

                stats = self.policy_avg_stats[metric][policy_id]
                if len(stats) > 0:
                    avg_metric = np.mean(stats)
                    self.save_best.emit(policy_id, metric, avg_metric)

    def register_msg_handler(self, key, func):
        self._register_msg_handler(self.msg_handlers, key, func)

    def register_policy_msg_handler(self, key, func):
        self._register_msg_handler(self.policy_msg_handlers, key, func)

    def _cfg_dict(self):
        if isinstance(self.cfg, dict):
            return self.cfg
        else:
            return vars(self.cfg)

    def _save_cfg(self):
        cfg_dict = self._cfg_dict()
        with open(cfg_file(self.cfg), 'w') as json_file:
            json.dump(cfg_dict, json_file, indent=2)

    def _make_batcher(self, event_loop, policy_id: PolicyID):
        return Batcher(event_loop, policy_id, self.buffer_mgr, self.cfg, self.env_info)

    def _make_learner(self, event_loop, policy_id: PolicyID, batcher: Batcher):
        return Learner(event_loop, self.cfg, self.env_info, self.buffer_mgr, batcher, policy_id=policy_id, mp_ctx=self.mp_ctx)

    def _make_inference_worker(self, event_loop, policy_id: PolicyID, worker_idx: int, param_server: ParameterServer):
        return InferenceWorker(
            event_loop, policy_id, worker_idx, self.buffer_mgr, param_server, self.inference_queues[policy_id], self.cfg, self.env_info,
        )

    def _make_rollout_worker(self, event_loop, worker_idx: int):
        return RolloutWorker(event_loop, worker_idx, self.buffer_mgr, self.inference_queues, self.cfg, self.env_info)

    def init(self):
        set_global_cuda_envvars(self.cfg)
        self.env_info = obtain_env_info_in_a_separate_process(self.cfg)
        self.buffer_mgr = BufferMgr(self.cfg, self.env_info)

    def _on_start(self):
        pass

    def _setup_component_heartbeat(self):
        # TODO!!!
        pass

    # noinspection PyUnresolvedReferences
    def _setup_component_termination(self, stop_trigger_obj: EventLoopObject, component_to_stop: EventLoopObject):
        stop_trigger_obj.stop.connect(component_to_stop.on_stop)
        self.components_to_stop.append(component_to_stop)
        component_to_stop.stop.connect(self._component_stopped)

    def connect_components(self):
        # runner initialization
        self.event_loop.start.connect(self._on_start)

        for policy_id in range(self.cfg.num_policies):
            # when runner is ready we initialize the learner first
            learner = self.learners[policy_id]
            batcher = self.batchers[policy_id]
            self.event_loop.start.connect(learner.init)  # TODO: base class with init()
            learner.initialized.connect(batcher.init)

            for i in range(self.cfg.policy_workers_per_policy):
                inference_worker = self.inference_workers[policy_id][i]

                # once learner is initialized and the model is ready, we initialize inference workers
                learner.model_initialized.connect(inference_worker.init)

                # inference workers signal when their initialization is complete
                inference_worker.initialized.connect(self.inference_worker_ready)

                # stop and start experience collection to throttle sampling speed if it is faster than learning
                batcher.stop_experience_collection.connect(inference_worker.should_stop_experience_collection)
                batcher.resume_experience_collection.connect(inference_worker.should_resume_experience_collection)

            # batcher gives learner batches of trajectories ready for learning
            batcher.training_batches_available.connect(learner.on_new_training_batch)
            # once learner is done with a training batch, it is given back to the batcher
            learner.training_batch_released.connect(batcher.on_training_batch_released)

            # auxiliary connections, such as summary reporting and checkpointing
            for i in range(self.cfg.policy_workers_per_policy):
                self.inference_workers[policy_id][i].report_msg.connect(self._process_msg)
            learner.finished_training_iteration.connect(self._after_training_iteration)
            learner.report_msg.connect(self._process_msg)
            self.save_periodic.connect(learner.save)
            self.save_best.connect(learner.save_best)

            # stop components when needed
            self._setup_component_termination(self, batcher)
            self._setup_component_termination(batcher, learner)
            for i in range(self.cfg.policy_workers_per_policy):
                self._setup_component_termination(learner, self.inference_workers[policy_id][i])

        for rollout_worker_idx in range(self.cfg.num_workers):
            # once all learners and inference workers are initialized, we can initialize rollout workers
            rollout_worker = self.rollout_workers[rollout_worker_idx]
            self.inference_workers_initialized.connect(rollout_worker.init)

            # inference worker signals to advance rollouts when actions are ready
            for policy_id in range(self.cfg.num_policies):
                for inference_worker_idx in range(self.cfg.policy_workers_per_policy):
                    self.inference_workers[policy_id][inference_worker_idx].connect(f'advance{rollout_worker_idx}', rollout_worker.advance_rollouts)

                # rollout workers send new trajectories to batchers
                rollout_worker.connect(f'p{policy_id}_trajectories', self.batchers[policy_id].on_new_trajectories)
                # wake up rollout workers when trajectory buffers are released
                self.batchers[policy_id].trajectory_buffers_available.connect(rollout_worker.on_trajectory_buffers_available)

            rollout_worker.report_msg.connect(self._process_msg)

            # stop rollout workers
            self._setup_component_termination(self, rollout_worker)

        # final cleanup
        self.all_components_stopped.connect(self._on_everything_stopped)

    def inference_worker_ready(self, policy_id: PolicyID, worker_idx: int):
        assert not self.inference_workers[policy_id][worker_idx].is_ready
        log.info(f'Inference worker {policy_id}-{worker_idx} is ready!')
        self.inference_workers[policy_id][worker_idx].is_ready = True

        # check if all workers for all policies are ready
        all_ready = True
        for policy_id in range(self.cfg.num_policies):
            all_ready &= all(w.is_ready for w in self.inference_workers[policy_id])

        if all_ready:
            log.info('All inference workers are ready! Signal rollout workers to start!')
            self.inference_workers_initialized.emit()

    def _should_end_training(self):
        end = len(self.env_steps) > 0 and all(s > self.cfg.train_for_env_steps for s in self.env_steps.values())
        end |= self.total_train_seconds > self.cfg.train_for_seconds
        return end

    def _after_training_iteration(self):
        if self._should_end_training() and not self.stopped:
            self._save_policy()
            self._save_best_policy()

            self.stop.emit(self.object_id)
            self.stopped = True

    def _component_stopped(self, component_obj_id, component_profile: Timing):
        log.debug(f'Component {component_obj_id} stopped!')

        for i, component in enumerate(self.components_to_stop):
            if component.object_id == component_obj_id:
                del self.components_to_stop[i]
                if self.components_to_stop:
                    log.debug(f'Waiting for {[c.object_id for c in self.components_to_stop]} to stop...')
                break

        if component_profile is not None:
            self.component_profiles.append((component_obj_id, component_profile))

        if not self.components_to_stop:
            self.all_components_stopped.emit()

    def _on_everything_stopped(self):
        # sort profiles by first element in the tuple
        self.component_profiles = sorted(self.component_profiles, key=lambda x: x[0])
        for component, profile in self.component_profiles:
            log.info(profile)

        assert self.event_loop.owner is self
        self.event_loop.stop()

    # noinspection PyBroadException
    def run(self) -> StatusCode:
        status = ExperimentStatus.SUCCESS

        with self.timing.timeit('main_loop'):
            try:
                evt_loop_status = self.event_loop.exec()
                status = ExperimentStatus.INTERRUPTED if evt_loop_status == EventLoopStatus.INTERRUPTED else status
                self.stop.emit(self.object_id)
            except Exception:
                log.exception(f'Uncaught exception in {self.object_id} evt loop')
                status = ExperimentStatus.FAILURE

        log.info(self.timing)
        fps = self.total_env_steps_since_resume / self.timing.main_loop
        log.info('Collected %r, FPS: %.1f', self.env_steps, fps)
        return status
