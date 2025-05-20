from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import psutil
import torch
from signal_slot.signal_slot import signal

from sample_factory.algo.sampling.batched_sampling import BatchedVectorEnvRunner
from sample_factory.algo.sampling.non_batched_sampling import NonBatchedVectorEnvRunner
from sample_factory.algo.sampling.sampling_utils import VectorEnvRunner, rollout_worker_device
from sample_factory.algo.utils.context import SampleFactoryContext, set_global_context
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.heartbeat import HeartbeatStoppableEventLoopObject
from sample_factory.algo.utils.misc import advance_rollouts_signal, new_trajectories_signal
from sample_factory.algo.utils.rl_utils import total_num_agents, trajectories_per_training_iteration
from sample_factory.algo.utils.torch_utils import inference_context
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.gpu_utils import set_gpus_for_process
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import MpQueue, PolicyID
from sample_factory.utils.utils import (
    cores_for_worker_process,
    debug_log_every_n,
    init_file_logger,
    log,
    set_process_cpu_affinity,
)


def init_rollout_worker_process(sf_context: SampleFactoryContext, worker: RolloutWorker):
    log.debug(f"Rollout worker {worker.worker_idx} starting...")

    set_global_context(sf_context)
    log.info(f"ROLLOUT worker {worker.worker_idx}\tpid {os.getpid()}\tparent {os.getppid()}")

    # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
    import signal as os_signal

    os_signal.signal(os_signal.SIGINT, os_signal.SIG_IGN)

    cfg = worker.cfg
    init_file_logger(cfg)

    # on MacOS, psutil.Process() has no method 'cpu_affinity'
    if hasattr(psutil.Process(), "cpu_affinity"):
        curr_process = psutil.Process()
        available_cores = curr_process.cpu_affinity()
        desired_cores = cores_for_worker_process(worker.worker_idx, cfg.num_workers, len(available_cores))
    else:
        desired_cores = cores_for_worker_process(worker.worker_idx, cfg.num_workers, psutil.cpu_count(logical=True))

    if desired_cores is not None and len(desired_cores) == 1 and cfg.force_envs_single_thread:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=1, user_api=None)

    if cfg.set_workers_cpu_affinity:
        set_process_cpu_affinity(worker.worker_idx, cfg.num_workers)

    if cfg.num_workers > 1:
        psutil.Process().nice(min(cfg.default_niceness + 10, 20))
        torch.set_num_threads(1)

    if cfg.actor_worker_gpus:
        worker_gpus = set_gpus_for_process(
            worker.worker_idx,
            num_gpus_per_process=1,
            process_type="actor",
            gpu_mask=cfg.actor_worker_gpus,
        )
        assert len(worker_gpus) == 1

    torch.multiprocessing.set_sharing_strategy("file_system")


class RolloutWorker(HeartbeatStoppableEventLoopObject, Configurable):
    def __init__(
        self, event_loop, worker_idx: int, buffer_mgr, inference_queues: Dict[PolicyID, MpQueue], cfg, env_info: EnvInfo
    ):
        Configurable.__init__(self, cfg)
        unique_name = f"{RolloutWorker.__name__}_w{worker_idx}"
        HeartbeatStoppableEventLoopObject.__init__(self, event_loop, unique_name, cfg.heartbeat_interval)

        self.timing = Timing(name=f"{self.object_id} profile")

        self.buffer_mgr = buffer_mgr
        self.inference_queues = inference_queues

        self.env_info = env_info
        self.worker_idx = worker_idx
        self.sampling_device = str(rollout_worker_device(self.worker_idx, self.cfg, self.env_info))

        self.vector_size = cfg.num_envs_per_worker
        self.num_splits = cfg.worker_num_splits
        assert self.vector_size >= self.num_splits
        assert self.vector_size % self.num_splits == 0, "Vector size should be divisible by num_splits"

        self.env_runners: List[VectorEnvRunner] = []

        # training status updated by the runner
        self.training_info: List[Optional[Dict[str, Any]]] = [None for _ in range(self.cfg.num_policies)]

        self.training_iteration: List[int] = [0] * self.cfg.num_policies

        if cfg.async_rl:
            rollouts_per_iteration = int(1e10)
        else:
            # In sync mode we use this mechanism to stop experience collection when the total number of rollouts
            # across all workers is sufficient to saturate the learner for one iteration.
            # This guarantees that all experience will be on policy when the learner starts training on it.
            # In async mode we don't care so initialize this to a large number.
            trajectories_training_iteration = trajectories_per_training_iteration(cfg) * cfg.num_policies
            sampling_trajectories = total_num_agents(cfg, env_info)

            # we do similar checks in verify_cfg() but let's throw in a check here just to be sure
            assert trajectories_training_iteration >= sampling_trajectories
            assert trajectories_training_iteration % sampling_trajectories == 0
            rollouts_per_iteration = trajectories_training_iteration // sampling_trajectories
            assert rollouts_per_iteration > 0

        # log.debug(f"Rollout worker {worker_idx} rollouts per iteration: {rollouts_per_iteration}")
        self.rollouts_per_iteration: int = rollouts_per_iteration
        self.remaining_rollouts: List[int] = [self.rollouts_per_iteration for _ in range(self.num_splits)]

        self.experience_decorrelated: bool = False
        self.is_initialized: bool = False

    @signal
    def report_msg(self):
        ...

    def init(self):
        for split_idx in range(self.num_splits):
            env_runner_cls = BatchedVectorEnvRunner if self.cfg.batched_sampling else NonBatchedVectorEnvRunner

            env_runner = env_runner_cls(
                self.cfg,
                self.env_info,
                self.vector_size // self.num_splits,
                self.worker_idx,
                split_idx,
                self.buffer_mgr,
                self.sampling_device,
                self.training_info,
            )

            env_runner.init(self.timing)

            # send signal to the inference worker to start processing new observations
            self.env_runners.append(env_runner)

        for r in self.env_runners:
            # This should kickstart experience collection. We will send a policy request to inference worker and
            # will get an "advance_rollout" signal back, and continue this loop of
            # advance_rollout->inference->advance_rollout until we collect the full rollout.
            # On rare occasions we might not be able to get a free buffer here (i.e. if all buffers are
            # taken by other workers). In that case, we will just enter an event loop and be woken up when
            # a buffer is freed (see on_trajectory_buffers_available()).
            self._maybe_send_policy_request(r)

        self.is_initialized = True

    def _decorrelate_experience(self):
        delay = (float(self.worker_idx) / self.cfg.num_workers) * self.cfg.decorrelate_experience_max_seconds
        if delay > 0.0:
            log.info(
                "Worker %d, sleep for %.3f sec to decorrelate experience collection",
                self.worker_idx,
                delay,
            )
            time.sleep(delay)
            log.info("Worker %d awakens!", self.worker_idx)

    def _maybe_send_policy_request(self, runner: VectorEnvRunner):
        if self.remaining_rollouts[runner.split_idx] <= 0:
            # This should only happen in sync mode -- means we completed a sufficient number of rollouts
            # to saturate the learner for one iteration. We will wait for the next iteration to start before
            # we can continue sampling.
            # log.debug(f"Ran out of remaining rollouts on {runner.worker_idx}-{runner.split_idx}: {self.remaining_rollouts}")
            return

        if not runner.update_trajectory_buffers(self.timing):
            # could not get a buffer, wait for one to be freed
            return

        with self.timing.add_time("enqueue_policy_requests"):
            policy_request = runner.generate_policy_request()

            # make sure all writes to shared device buffers are completed
            runner.synchronize_devices()

        with self.timing.add_time("enqueue_policy_requests"):
            if policy_request is not None:
                self._enqueue_policy_request(runner.split_idx, policy_request)

    def _enqueue_policy_request(self, split_idx, policy_inputs):
        """Distribute action requests to their corresponding queues."""

        for policy_id, requests in policy_inputs.items():
            policy_request = (self.worker_idx, split_idx, requests, self.sampling_device)
            self.inference_queues[policy_id].put(policy_request)

        if not policy_inputs:
            # This can happen if all agents on this worker were deactivated (is_active=False)
            debug_log_every_n(
                100,
                f"Worker {self.worker_idx}-{split_idx} has no active agents... We immediately continue to the next iteration without notifying the inference worker",
            )
            fake_policy_id = -1
            # it's easier to self ourselves a signal than call advance_rollouts() directly because
            # this way we don't have to worry about getting stuck in an infinite loop or processing things like
            # stopping signal
            self.emit(advance_rollouts_signal(self.worker_idx), split_idx, fake_policy_id)

    def _enqueue_complete_rollouts(self, complete_rollouts: List[Dict]):
        """Emit complete rollouts."""
        rollouts_per_policy = dict()
        for rollout in complete_rollouts:
            policy_id = rollout["policy_id"]
            if policy_id not in rollouts_per_policy:
                rollouts_per_policy[policy_id] = []
            rollouts_per_policy[policy_id].append(rollout)

        for policy_id, rollouts in rollouts_per_policy.items():
            self.emit(new_trajectories_signal(policy_id), rollouts, self.sampling_device)

    def advance_rollouts(self, split_idx: int, policy_id: PolicyID) -> None:
        # TODO: update comment
        """
        Process incoming request from policy worker. Use the data (policy outputs, actions) to advance the simulation
        by one step on the corresponding VectorEnvRunner.

        If we successfully managed to advance the simulation, send requests to policy workers to get actions for the
        next step. If we completed the entire rollout, also send request to the learner!
        """
        with inference_context(self.cfg.serial_mode):
            runner = self.env_runners[split_idx]
            complete_rollouts, episodic_stats = runner.advance_rollouts(policy_id, self.timing)

            with self.timing.add_time("complete_rollouts"):
                if complete_rollouts:
                    self._enqueue_complete_rollouts(complete_rollouts)
                    if not self.experience_decorrelated and not self.cfg.benchmark:
                        # we just finished our first complete rollouts, perfect time to wait for experience derorrelation
                        # this guarantees that there won't be any obsolete trajectories when we awaken
                        self._decorrelate_experience()
                        self.experience_decorrelated = True

                    self.remaining_rollouts[split_idx] -= 1

            if episodic_stats:
                self.report_msg.emit(episodic_stats)

            # We finished one step of environment simulation.
            # If we also have the trajectory buffer to share the new data with the inference worker then
            # we are ready to enqueue inference request
            self._maybe_send_policy_request(runner)

    def on_trajectory_buffers_available(self, policy_id: PolicyID, training_iteration: int):
        """
        Used to wake up rollout workers waiting for trajectory buffers to be freed.
        In addition to that, we also send information about training iteration per policy. This is useful for
        sync mode where we make progress in lock-step: the next batch of experience is collected only when we
        finished training on the previous batch.

        This becomes tricky in multi-policy case, because agent-policy mapping
        may not necessarily guarantee the same number of trajectories per policy per iteration (i.e. if more agents
        collect experience for one policy than another). Multi-policy sync mode is thus an experimental feature,
        most of the time you should prefer using cfg.async_rl=True for multi-policy training.
        """
        if not self.cfg.async_rl:
            # in sync mode we progress one iteration at a time
            assert training_iteration - self.training_iteration[policy_id] in (0, 1)

        prev_iteration = min(self.training_iteration)
        self.training_iteration[policy_id] = training_iteration
        curr_iteration = min(self.training_iteration)
        if curr_iteration > prev_iteration:
            # allow runners to collect the next portion of rollouts
            self.remaining_rollouts = [self.rollouts_per_iteration for _ in range(self.num_splits)]

        # we can receive this signal during batcher initialization, before the worker is initialized
        # this is fine, we just ignore it and get the trajectory buffers from the queue later after we do env.reset()
        if not self.is_initialized:
            return

        # it is possible that we finished the simulation step, but were unable to send data to inference worker
        # because we ran out of trajectory buffers. The purpose of this signal handler is to wake up the worker,
        # request a new trajectory (since they're now available), and finally send observations to the inference worker
        for split_idx in range(self.num_splits):
            self._maybe_send_policy_request(self.env_runners[split_idx])

    def on_update_training_info(self, training_info: Dict[PolicyID, Dict[str, Any]]) -> None:
        """Update training info, this will be propagated to environments using TrainingInfoInterface and RewardShapingInterface."""
        for policy_id, info in training_info.items():
            self.training_info[policy_id] = info

    def on_stop(self, *args):
        for env_runner in self.env_runners:
            env_runner.close()

        timings = dict()
        if self.worker_idx in [0, self.cfg.num_workers - 1]:
            timings[self.object_id] = self.timing
        self.stop.emit(self.object_id, timings)
        super().on_stop(*args)
