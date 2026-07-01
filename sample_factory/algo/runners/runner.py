from __future__ import annotations

import json
import math
import shutil
import time
from collections import OrderedDict, deque
from os.path import isdir, join
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import numpy as np
from signal_slot.signal_slot import EventLoop, EventLoopObject, EventLoopStatus, Timer, process_name, signal
from tensorboardX import SummaryWriter

from sample_factory.algo.learning.batcher import Batcher
from sample_factory.algo.learning.learner_worker import LearnerWorker
from sample_factory.algo.sampling.sampler import AbstractSampler
from sample_factory.algo.sampling.stats import samples_stats_handler, stats_msg_handler, timing_msg_handler
from sample_factory.algo.utils.env_info import EnvInfo, obtain_env_info_in_a_separate_process
from sample_factory.algo.utils.heartbeat import HeartbeatStoppableEventLoopObject
from sample_factory.algo.utils.misc import (
    EPISODIC,
    LEARNER_ENV_STEPS,
    SAMPLES_COLLECTED,
    STATS_KEY,
    TIMING_STATS,
    TRAIN_STATS,
    ExperimentStatus,
)
from sample_factory.algo.utils.shared_buffers import BufferMgr
from sample_factory.cfg.arguments import cfg_dict, cfg_str, preprocess_cfg
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.dicts import iterate_recursively
from sample_factory.utils.gpu_utils import set_global_cuda_envvars
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import PolicyID, StatusCode
from sample_factory.utils.utils import (
    cfg_file,
    debug_log_every_n,
    ensure_dir_exists,
    experiment_dir,
    init_file_logger,
    log,
    memory_consumption_mb,
    save_git_diff,
    summaries_dir,
)
from sample_factory.utils.wandb_utils import init_wandb
from sample_factory.utils.state_proxy import StateProxy


class AlgoObserver:
    def on_init(self, runner: Runner) -> None:
        """Called after ctor, but before signal-slots are connected or any processes are started."""
        pass

    def on_connect_components(self, runner: Runner) -> None:
        """Connect additional signal-slot pairs in the observers if needed."""
        pass

    def on_start(self, runner: Runner) -> None:
        """Called right after sampling/learning processes are started."""
        pass

    def on_training_step(self, runner: Runner, training_iteration_since_resume: int) -> None:
        """Called after each training step."""
        pass

    def extra_summaries(self, runner: Runner, policy_id: PolicyID, env_steps: int, writer: SummaryWriter) -> None:
        pass

    def on_stop(self, runner: Runner) -> None:
        pass


MsgHandler = Callable[[Any, dict], None]
PolicyMsgHandler = Callable[[Any, dict, PolicyID], None]


class Runner(StateProxy, EventLoopObject, Configurable):
    _STATE_ATTRS = frozenset(('avg_stats', 'avg_stats_intervals', 'batchers', 'buffer_mgr', 'component_profiles', 'components_to_stop', 'env_info', 'env_steps', 'event_loop', 'fps_stats', 'heartbeat_dict', 'heartbeat_report_sec', 'last_report', 'learners', 'msg_handlers', 'observers', 'policy_avg_stats', 'policy_lag', 'policy_msg_handlers', 'queue_size_dict', 'report_interval_sec', 'reward_shaping', 'sampler', 'samples_collected', 'start_time', 'stats', 'status', 'stopped', 'summaries_interval_sec', 'throughput_stats', 'timers', 'timing', 'total_env_steps_since_resume', 'total_train_seconds', 'update_training_info_every_sec', 'writers'))

    def __init__(self, cfg, unique_name=None):
        self._init_state_proxy()
        Configurable.__init__(self, cfg)

        unique_name = Runner.__name__ if unique_name is None else unique_name
        self._init_event_loop(cfg, unique_name)
        self._init_state()
        self._init_stats()

        self._handle_restart()

        init_wandb(self.cfg)  # should be done before writers are initialized

        self._init_writers(cfg)
        self._init_msg_handlers()
        self._state.observers: List[AlgoObserver] = []
        self._init_timers()

        self._state.heartbeat_dict = {}

    def _init_event_loop(self, cfg, unique_name: str) -> None:
        self._state.event_loop: EventLoop = EventLoop(unique_loop_name=f"{unique_name}_EvtLoop", serial_mode=cfg.serial_mode)
        self._state.event_loop.owner = self
        EventLoopObject.__init__(self, self._state.event_loop, object_id=unique_name)

    def _init_state(self) -> None:
        self._state.status: StatusCode = ExperimentStatus.SUCCESS
        self._state.stopped: bool = False
        self._state.env_info: Optional[EnvInfo] = None
        self._state.reward_shaping: List[Optional[Dict]] = [None for _ in range(self.cfg.num_policies)]
        self._state.buffer_mgr = None

        self._state.learners: Dict[PolicyID, LearnerWorker] = dict()
        self._state.batchers: Dict[PolicyID, Batcher] = dict()
        self._state.sampler: Optional[AbstractSampler] = None

        self._state.timing = Timing("Runner profile")
        self._state.env_steps: Dict[PolicyID, int] = dict()
        self._state.samples_collected = [0 for _ in range(self.cfg.num_policies)]
        self._state.total_env_steps_since_resume: Optional[int] = None
        self._state.start_time: float = time.time()
        self._state.total_train_seconds = 0
        self._state.last_report = time.time()

        self._state.report_interval_sec = 5.0
        self._state.avg_stats_intervals = (2, 12, 60)  # by default: 10 seconds, 60 seconds, 5 minutes
        self._state.summaries_interval_sec = self.cfg.experiment_summaries_interval  # sec
        self._state.heartbeat_report_sec = self.cfg.heartbeat_reporting_interval
        self._state.update_training_info_every_sec = 5.0

    def _init_stats(self) -> None:
        self._state.fps_stats = deque([], maxlen=max(self._state.avg_stats_intervals))
        self._state.throughput_stats = [deque([], maxlen=10) for _ in range(self.cfg.num_policies)]

        self._state.stats = dict()  # regular (non-averaged) stats
        self._state.avg_stats = dict()

        self._state.policy_avg_stats: Dict[str, List[Deque]] = dict()
        self._state.policy_lag = [dict() for _ in range(self.cfg.num_policies)]

    def _init_writers(self, cfg) -> None:
        self._state.writers: Dict[int, SummaryWriter] = dict()
        for policy_id in range(self.cfg.num_policies):
            summary_dir = join(summaries_dir(experiment_dir(cfg=self.cfg)), str(policy_id))
            summary_dir = ensure_dir_exists(summary_dir)
            self._state.writers[policy_id] = SummaryWriter(summary_dir, flush_secs=cfg.flush_summaries_interval)

    def _init_msg_handlers(self) -> None:
        self._state.msg_handlers: Dict[str, List[MsgHandler]] = {
            TIMING_STATS: [timing_msg_handler],
            STATS_KEY: [stats_msg_handler],
        }

        self._state.policy_msg_handlers: Dict[str, List[PolicyMsgHandler]] = {
            LEARNER_ENV_STEPS: [self._learner_steps_handler],
            EPISODIC: [self._episodic_stats_handler],
            TRAIN_STATS: [self._train_stats_handler],
            SAMPLES_COLLECTED: [samples_stats_handler],
        }

    def _init_timers(self) -> None:
        self._state.timers: List[Timer] = []

        def periodic(period, cb):
            t = Timer(self._state.event_loop, period)
            t.timeout.connect(cb)
            self._state.timers.append(t)

        periodic(self._state.report_interval_sec, self._update_stats_and_print_report)
        periodic(self._state.summaries_interval_sec, self._report_experiment_summaries)

        periodic(self.cfg.save_every_sec, self._save_policy)
        periodic(self.cfg.save_best_every_sec, self._save_best_policy)

        periodic(self._state.update_training_info_every_sec, self._propagate_training_info)

        if self.cfg.save_milestones_sec > 0:
            periodic(self.cfg.save_milestones_sec, self._save_milestone_policy)

        periodic(self._state.heartbeat_report_sec, self._check_heartbeat)
        self._state.queue_size_dict = {}

        self._state.components_to_stop: List[EventLoopObject] = []
        self._state.component_profiles: Dict[str, Timing] = dict()

    # signals emitted by the runner
    @signal
    def save_periodic(self): ...

    @signal
    def save_best(self): ...

    @signal
    def update_training_info(self): ...

    @signal
    def save_milestone(self): ...

    @signal
    def stop(self):
        """Emitted when we're about to stop the experiment."""
        ...

    @signal
    def all_components_stopped(self): ...

    def _handle_restart(self):
        exp_dir = experiment_dir(self.cfg, mkdir=False)
        if isdir(exp_dir):
            log.debug(f"Experiment dir {exp_dir} already exists!")
            if self.cfg.restart_behavior == "resume":
                log.debug(f"Resuming existing experiment from {exp_dir}...")
            else:
                if self.cfg.restart_behavior == "restart":
                    attempt = 0
                    old_exp_dir = exp_dir
                    while isdir(old_exp_dir):
                        attempt += 1
                        old_exp_dir = f"{exp_dir}_old{attempt:04d}"

                    # move the existing experiment dir to a new one with a suffix
                    log.debug(f"Moving the existing experiment dir to {old_exp_dir}...")
                    shutil.move(exp_dir, old_exp_dir)
                elif self.cfg.restart_behavior == "overwrite":
                    log.debug(f"Overwriting the existing experiment dir {exp_dir}...")
                    shutil.rmtree(exp_dir)
                else:
                    raise ValueError(f"Unknown restart behavior {self.cfg.restart_behavior}")

                log.debug(f"Starting training in {exp_dir}...")

    def _process_msg(self, msgs):
        if isinstance(msgs, (dict, OrderedDict)):
            msgs = (msgs,)

        if not (isinstance(msgs, (List, Tuple)) and isinstance(msgs[0], (dict, OrderedDict))):
            log.error("While parsing a message: expected a dictionary or list/tuple of dictionaries, found %r", msgs)
            return

        for msg in msgs:
            # some messages are policy-specific
            policy_id = msg.get("policy_id", None)

            for key in msg:
                for handler in self._state.msg_handlers.get(key, ()):
                    handler(self, msg)
                if policy_id is not None:
                    for handler in self._state.policy_msg_handlers.get(key, ()):
                        handler(self, msg, policy_id)

    @staticmethod
    def _learner_steps_handler(runner: Runner, msg: Dict, policy_id: PolicyID) -> None:
        env_steps: int = msg[LEARNER_ENV_STEPS]
        if policy_id in runner.env_steps:
            delta = env_steps - runner.env_steps[policy_id]
            runner.total_env_steps_since_resume += delta
        elif runner.total_env_steps_since_resume is None:
            runner.total_env_steps_since_resume = 0

        runner.env_steps[policy_id] = env_steps

    @staticmethod
    def _episodic_stats_handler(runner: Runner, msg: Dict, policy_id: PolicyID) -> None:
        s = msg[EPISODIC]
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

    @staticmethod
    def _train_stats_handler(runner: Runner, msg: Dict, policy_id: PolicyID) -> None:
        """We write the train summaries to disk right away instead of accumulating them."""
        train_stats = msg[TRAIN_STATS]
        for key, scalar in train_stats.items():
            runner.writers[policy_id].add_scalar(f"train/{key}", scalar, runner.env_steps[policy_id])

        for key in ["version_diff_min", "version_diff_max", "version_diff_avg"]:
            if key in train_stats:
                runner.policy_lag[policy_id][key] = train_stats[key]

    def _get_perf_stats(self):
        # total env steps simulated across all policies
        fps_stats = []
        for avg_interval in self._state.avg_stats_intervals:
            fps_for_interval = math.nan
            if len(self._state.fps_stats) > 1:
                t1, x1 = self._state.fps_stats[max(0, len(self._state.fps_stats) - 1 - avg_interval)]
                t2, x2 = self._state.fps_stats[-1]
                fps_for_interval = (x2 - x1) / (t2 - t1)

            fps_stats.append(fps_for_interval)

        # learning throughput per policy (in observations per sec)
        sample_throughput = dict()
        for policy_id in range(self.cfg.num_policies):
            sample_throughput[policy_id] = math.nan
            if len(self._state.throughput_stats[policy_id]) > 1:
                t1, x1 = self._state.throughput_stats[policy_id][0]
                t2, x2 = self._state.throughput_stats[policy_id][-1]
                sample_throughput[policy_id] = (x2 - x1) / (t2 - t1)

        return fps_stats, sample_throughput

    def print_stats(self, fps, sample_throughput, total_env_steps):
        fps_str = []
        for interval, fps_value in zip(self._state.avg_stats_intervals, fps):
            fps_str.append(f"{int(interval * self._state.report_interval_sec)} sec: {fps_value:.1f}")
        fps_str = f'({", ".join(fps_str)})'

        samples_per_policy = ", ".join([f"{p}: {s:.1f}" for p, s in sample_throughput.items()])

        lag_stats = self._state.policy_lag[0]
        lag = AttrDict()
        for key in ["min", "avg", "max"]:
            lag[key] = lag_stats.get(f"version_diff_{key}", -1)
        policy_lag_str = f"min: {lag.min:.1f}, avg: {lag.avg:.1f}, max: {lag.max:.1f}"

        log.debug(
            "Fps is %s. Total num frames: %d. Throughput: %s. Samples: %d. Policy #0 lag: (%s)",
            fps_str,
            total_env_steps,
            samples_per_policy,
            sum(self._state.samples_collected),
            policy_lag_str,
        )

        if "reward" in self._state.policy_avg_stats:
            policy_reward_stats = []
            for policy_id in range(self.cfg.num_policies):
                reward_stats = self._state.policy_avg_stats["reward"][policy_id]
                if len(reward_stats) > 0:
                    policy_reward_stats.append((policy_id, f"{np.mean(reward_stats):.3f}"))
            log.debug("Avg episode reward: %r", policy_reward_stats)

    def _update_stats_and_print_report(self):
        """
        Called periodically (every self._state.report_interval_sec seconds).
        Print experiment stats (FPS, avg rewards) to console and dump TF summaries collected from workers to disk.
        """

        # don't have enough statistic from the learners yet
        if len(self._state.env_steps) < self.cfg.num_policies:
            return

        if self._state.total_env_steps_since_resume is None:
            return

        now = time.time()
        self._state.fps_stats.append((now, self._state.total_env_steps_since_resume))

        for policy_id in range(self.cfg.num_policies):
            self._state.throughput_stats[policy_id].append((now, self._state.samples_collected[policy_id]))

        fps_stats, sample_throughput = self._get_perf_stats()
        total_env_steps = sum(self._state.env_steps.values())
        self.print_stats(fps_stats, sample_throughput, total_env_steps)

    def _report_experiment_summaries(self):
        memory_mb = memory_consumption_mb()

        fps_stats, sample_throughput = self._get_perf_stats()
        fps = fps_stats[0]

        default_policy = 0
        for policy_id, env_steps in self._state.env_steps.items():
            writer = self._state.writers[policy_id]
            if policy_id == default_policy:
                if not math.isnan(fps):
                    writer.add_scalar("perf/_fps", fps, env_steps)

                writer.add_scalar("stats/master_process_memory_mb", float(memory_mb), env_steps)
                for key, value in self._state.avg_stats.items():
                    if len(value) >= value.maxlen or (len(value) > 10 and self._state.total_train_seconds > 300):
                        writer.add_scalar(f"stats/{key}", np.mean(value), env_steps)

                for key, value in self._state.stats.items():
                    writer.add_scalar(f"stats/{key}", value, env_steps)

            if not math.isnan(sample_throughput[policy_id]):
                writer.add_scalar("perf/_sample_throughput", sample_throughput[policy_id], env_steps)

            for key, stat in self._state.policy_avg_stats.items():
                if len(stat[policy_id]) >= stat[policy_id].maxlen or (
                    len(stat[policy_id]) > 10 and self._state.total_train_seconds > 300
                ):
                    stat_value = np.mean(stat[policy_id])

                    if "/" in key:
                        # custom summaries have their own sections in tensorboard
                        avg_tag = key
                        min_tag = f"{key}_min"
                        max_tag = f"{key}_max"
                    elif key in ("reward", "len"):
                        # reward and length get special treatment
                        avg_tag = f"{key}/{key}"
                        min_tag = f"{key}/{key}_min"
                        max_tag = f"{key}/{key}_max"
                    else:
                        avg_tag = f"policy_stats/avg_{key}"
                        min_tag = f"policy_stats/avg_{key}_min"
                        max_tag = f"policy_stats/avg_{key}_max"

                    writer.add_scalar(avg_tag, float(stat_value), env_steps)

                    # for key stats report min/max as well
                    if key in ("reward", "true_objective", "len"):
                        writer.add_scalar(min_tag, float(min(stat[policy_id])), env_steps)
                        writer.add_scalar(max_tag, float(max(stat[policy_id])), env_steps)

            self._observers_call(AlgoObserver.extra_summaries, self, policy_id, writer, env_steps)

        for w in self._state.writers.values():
            w.flush()

    def _propagate_training_info(self):
        """
        Send the training stats (such as the number of processed env steps) to the sampler.
        This can be used by the envs to configure curriculums and so on.
        """

        training_info: Dict[PolicyID, Dict[str, Any]] = dict()
        for policy_id in range(self.cfg.num_policies):
            training_info[policy_id] = dict(
                policy_id=policy_id,
                # "approx" here because it will lag behind a little bit due to the async nature of the system
                approx_total_training_steps=self._state.env_steps.get(policy_id, 0),
                reward_shaping=self._state.reward_shaping[policy_id],
                # add more stats if needed (commented by default for efficiency)
                # stats=self._state.stats,
                # avg_stats=self._state.avg_stats,
                # policy_avg_stats=self._state.policy_avg_stats,
            )

        self.update_training_info.emit(training_info)

    def update_reward_shaping(self, policy_id: PolicyID, reward_shaping: Dict[str, Any]) -> None:
        self._state.reward_shaping[policy_id] = reward_shaping

        # send the updated data to other components (e.g. the sampler)
        # this allows us to change reward shaping on the fly, PBT can take advantage of this
        self._propagate_training_info()

    def _save_policy(self):
        self.save_periodic.emit()

    def _save_milestone_policy(self):
        self.save_milestone.emit()

    def _save_best_policy(self):
        # don't have enough statistic from the learners yet
        if len(self._state.env_steps) < self.cfg.num_policies:
            return

        metric = self.cfg.save_best_metric
        if metric in self._state.policy_avg_stats:
            for policy_id in range(self.cfg.num_policies):
                # check if number of samples collected is greater than cfg.save_best_after
                env_steps = self._state.env_steps[policy_id]
                if env_steps < self.cfg.save_best_after:
                    continue

                stats = self._state.policy_avg_stats[metric][policy_id]
                if len(stats) > 0:
                    avg_metric = np.mean(stats)
                    self.save_best.emit(policy_id, metric, avg_metric)

    @staticmethod
    def _register_msg_handler(handlers_dict, key, func):
        handlers_dict[key] = func

    def register_msg_handler(self, key, func):
        self._register_msg_handler(self._state.msg_handlers, key, func)

    def register_policy_msg_handler(self, key, func):
        self._register_msg_handler(self._state.policy_msg_handlers, key, func)

    def register_episodic_stats_handler(self, func: PolicyMsgHandler):
        self._state.policy_msg_handlers[EPISODIC].append(func)

    def register_observer(self, observer: AlgoObserver) -> None:
        self._state.observers.append(observer)

    def _observers_call(self, func, *args, **kwargs) -> None:
        for observer in self._state.observers:
            getattr(observer, func.__name__)(*args, **kwargs)

    def _save_cfg(self):
        fname = cfg_file(self.cfg)
        with open(fname, "w") as json_file:
            log.debug(f"Saving configuration to {fname}...")
            json.dump(cfg_dict(self.cfg), json_file, indent=2)

    def _make_batcher(self, event_loop, policy_id: PolicyID):
        return Batcher(event_loop, policy_id, self._state.buffer_mgr, self.cfg, self._state.env_info)

    def _make_learner(self, event_loop, policy_id: PolicyID, batcher: Batcher):
        return LearnerWorker(
            event_loop,
            self.cfg,
            self._state.env_info,
            self._state.buffer_mgr,
            batcher,
            policy_id=policy_id,
        )

    def _make_sampler(self, sampler_cls: type, event_loop: EventLoop):
        assert len(self._state.learners) == self.cfg.num_policies, "Learners not created yet"
        param_servers = {policy: self._state.learners[policy].param_server for policy in self._state.learners}
        return sampler_cls(event_loop, self._state.buffer_mgr, param_servers, self.cfg, self._state.env_info)

    def init(self) -> StatusCode:
        set_global_cuda_envvars(self.cfg)
        self._state.env_info = obtain_env_info_in_a_separate_process(self.cfg)

        for policy_id in range(self.cfg.num_policies):
            self._state.reward_shaping[policy_id] = self._state.env_info.reward_shaping_scheme

        # check for any incompatible arguments
        if not preprocess_cfg(self.cfg, self._state.env_info):
            return ExperimentStatus.FAILURE

        log.debug(f"Starting experiment with the following configuration:\n{cfg_str(self.cfg)}")

        init_file_logger(self.cfg)
        self._save_cfg()
        save_git_diff(experiment_dir(self.cfg))

        self._state.buffer_mgr = BufferMgr(self.cfg, self._state.env_info)

        self._observers_call(AlgoObserver.on_init, self)

        return ExperimentStatus.SUCCESS

    def _on_start(self):
        """Override this in a subclass to do something right when the experiment is started."""
        self._state.sampler.init()
        self._propagate_training_info()
        self._observers_call(AlgoObserver.on_start, self)

    def _setup_component_heartbeat(self, component: HeartbeatStoppableEventLoopObject):
        """
        Groups components with heartbeat mechanism by type and records starting time.
        When all components of the same type do not respond in the reporting timeframe, stops the run
        """
        component_type = type(component)
        if component_type not in self._state.heartbeat_dict:
            self._state.heartbeat_dict[component_type] = {}
        type_dict = self._state.heartbeat_dict[component_type]
        type_dict[component.object_id] = None

        # setup up queue_size report with heartbeat, grouped by event_loop_process_name
        p_name = process_name(component.event_loop.process)
        if p_name not in self._state.queue_size_dict:
            self._state.queue_size_dict[p_name] = 0

        component.heartbeat.connect(self._receive_heartbeat)

    def _receive_heartbeat(self, component_type: type, component_id: str, p_name: str, qsize: int):
        """
        Record the time the most recent heartbeat was received
        """
        curr_time = time.time()
        heartbeat_time = self._state.heartbeat_dict[component_type][component_id]
        if heartbeat_time is None:
            log.info(f"Heartbeat connected on {component_id}")
        elif curr_time - heartbeat_time > self._state.heartbeat_report_sec:
            log.info(f"Heartbeat reconnected after {int(curr_time - heartbeat_time)} seconds from {component_id}")
        self._state.heartbeat_dict[component_type][component_id] = curr_time
        self._state.queue_size_dict[p_name] = qsize

    def _check_heartbeat(self):
        """
        Reports components whose last heartbeat signal is longer than self._state.heartbeat_report_sec.
        If all components of the same type fail, stop the run
        """
        curr_time = time.time()
        comp_list = []
        type_list = []
        none_list = []
        for component_type, heartbeat_dict in self._state.heartbeat_dict.items():
            num_components = len(heartbeat_dict)
            num_stopped = 0
            for component_id, heartbeat_time in heartbeat_dict.items():
                if heartbeat_time is None:
                    none_list.append(component_id)
                    continue
                if curr_time - heartbeat_time > self._state.heartbeat_report_sec:
                    comp_list.append(f"{component_id} ({(int(curr_time - heartbeat_time))} seconds)")
                    num_stopped += 1
            if num_stopped == num_components:
                type_list.append(str(component_type))

        if len(none_list) > 0:
            wait_time = time.time() - self._state.start_time
            log.debug(f"Components not started: {', '.join(none_list)}, {wait_time=:.1f} seconds")
            if wait_time > 3 * self._state.heartbeat_report_sec:
                log.error(f"Components take too long to start: {', '.join(none_list)}. Aborting the experiment!\n\n\n")
                self._stop_training(failed=True)

        if len(comp_list) > 0:
            log.error(f"No heartbeat for components: {', '.join(comp_list)}")

        if len(type_list) > 0:
            log.error(f"Stopping training due to lack of heartbeats from {', '.join(type_list)}")
            self._stop_training(failed=True)

        for p_name, qsize in self._state.queue_size_dict.items():
            if qsize > 5:
                debug_log_every_n(1000, f"Process: {p_name} has queue size: {qsize}")

    def _setup_component_termination(self, stop_signal: signal, component_to_stop: HeartbeatStoppableEventLoopObject):
        stop_signal.connect(component_to_stop.on_stop)
        self._state.components_to_stop.append(component_to_stop)
        component_to_stop.stop.connect(self._component_stopped)

    def connect_components(self):
        self._state.event_loop.start.connect(self._on_start)

        sampler = self._state.sampler
        for policy_id in range(self.cfg.num_policies):
            # when runner is ready we initialize the learner first and then all other components in a chain
            learner_worker = self._state.learners[policy_id]
            batcher = self._state.batchers[policy_id]
            self._state.event_loop.start.connect(learner_worker.init)
            learner_worker.initialized.connect(batcher.init)
            sampler.connect_model_initialized(policy_id, learner_worker.model_initialized)

            # key connections - sampler and batcher exchanging connections back and forth
            sampler.connect_on_new_trajectories(policy_id, batcher.on_new_trajectories)
            sampler.connect_trajectory_buffers_available(batcher.trajectory_buffers_available)

            # batcher gives learner batches of trajectories ready for learning
            batcher.training_batches_available.connect(learner_worker.on_new_training_batch)
            # once learner is done with a training batch, it is given back to the batcher
            learner_worker.training_batch_released.connect(batcher.on_training_batch_released)

            # signals that allow us to throttle the sampler if the learner can't keep up
            sampler.connect_stop_experience_collection(batcher.stop_experience_collection)
            sampler.connect_resume_experience_collection(batcher.resume_experience_collection)

            # auxiliary connections, such as summary reporting and checkpointing
            learner_worker.finished_training_iteration.connect(self._after_training_iteration)
            learner_worker.report_msg.connect(self._process_msg)
            sampler.connect_report_msg(self._process_msg)
            sampler.connect_update_training_info(self.update_training_info)
            self.save_periodic.connect(learner_worker.save)
            self.save_best.connect(learner_worker.save_best)
            self.save_milestone.connect(learner_worker.save_milestone)

            # stop components when needed
            self._setup_component_termination(self.stop, batcher)
            self._setup_component_termination(batcher.stop, learner_worker)

            # Heartbeats
            self._setup_component_heartbeat(batcher)
            self._setup_component_heartbeat(learner_worker)

        for sampler_component in sampler.stoppable_components():
            self._setup_component_termination(self.stop, sampler_component)

        for sampler_component in sampler.heartbeat_components():
            self._setup_component_heartbeat(sampler_component)

        # final cleanup
        self.all_components_stopped.connect(self._on_everything_stopped)

        # connect additional signal-slot pairs in the observers if needed
        self._observers_call(AlgoObserver.on_connect_components, self)

    def _should_end_training(self):
        self._state.total_train_seconds = time.time() - self._state.start_time
        end = len(self._state.env_steps) > 0 and all(s > self.cfg.train_for_env_steps for s in self._state.env_steps.values())
        end |= self._state.total_train_seconds > self.cfg.train_for_seconds
        return end

    def _after_training_iteration(self, training_iteration_since_resume: int):
        self._observers_call(AlgoObserver.on_training_step, self, training_iteration_since_resume)

        if self._should_end_training():
            self._stop_training()

    def _stop_training(self, failed: bool = False) -> None:
        if not self._state.stopped:
            self._propagate_training_info()

            self._observers_call(AlgoObserver.on_stop, self)

            self._save_policy()
            self._save_best_policy()

            for timer in self._state.timers:
                timer.stop()
            self.stop.emit(self.object_id)

            if failed:
                self._state.status = ExperimentStatus.FAILURE

            self._state.stopped = True

    def _component_stopped(self, component_obj_id, component_profiles: Dict[str, Timing]):
        remaining = []
        for i, component in enumerate(self._state.components_to_stop):
            if component.object_id == component_obj_id:
                log.debug(f"Component {component_obj_id} stopped!")
                continue

            try:
                if not component.event_loop.process.is_alive():
                    log.warning(f"Component {component.object_id} process died already! Don't wait for it.")
                    continue
            except AttributeError:
                # in serial mode there's no process, plus event_loop can be None for stopped components
                pass

            remaining.append(component)

        self._state.components_to_stop = remaining
        if self._state.components_to_stop and self._state.status == ExperimentStatus.FAILURE:
            log.debug(f"Waiting for {[c.object_id for c in self._state.components_to_stop]} to stop...")

        self._state.component_profiles.update(component_profiles)

        if not self._state.components_to_stop:
            self.all_components_stopped.emit()

    def _on_everything_stopped(self):
        # sort profiles by name
        self._state.component_profiles = sorted(list(self._state.component_profiles.items()), key=lambda x: x[0])
        for component, profile in self._state.component_profiles:
            log.info(profile)

        for w in self._state.writers.values():
            w.flush()

        assert self._state.event_loop.owner is self
        self._state.event_loop.stop()

    # noinspection PyBroadException
    def run(self) -> StatusCode:
        with self._state.timing.timeit("main_loop"):
            try:
                evt_loop_status = self._state.event_loop.exec()
                self._state.status = (
                    ExperimentStatus.INTERRUPTED if evt_loop_status == EventLoopStatus.INTERRUPTED else self._state.status
                )
                self.stop.emit(self.object_id)
            except Exception:
                log.exception(f"Uncaught exception in {self.object_id} evt loop")
                self._state.status = ExperimentStatus.FAILURE

        log.info(self._state.timing)
        if self._state.total_env_steps_since_resume is None:
            self._state.total_env_steps_since_resume = 0
        fps = self._state.total_env_steps_since_resume / self._state.timing.main_loop
        log.info("Collected %r, FPS: %.1f", self._state.env_steps, fps)
        return self._state.status
