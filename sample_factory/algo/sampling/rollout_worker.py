from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import psutil
import torch
from signal_slot.signal_slot import EventLoopObject, signal

from sample_factory.algo.sampling.batched_sampling import BatchedVectorEnvRunner
from sample_factory.algo.sampling.non_batched_sampling import NonBatchedVectorEnvRunner
from sample_factory.algo.sampling.sampling_utils import VectorEnvRunner
from sample_factory.algo.utils.context import SampleFactoryContext, set_global_context
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import new_trajectories_signal
from sample_factory.algo.utils.stoppable import StoppableEventLoopObject
from sample_factory.algo.utils.torch_utils import inference_context
from sample_factory.cfg.configurable import Configurable
from sample_factory.utils.gpu_utils import gpus_for_process, set_gpus_for_process
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import MpQueue, PolicyID
from sample_factory.utils.utils import AttrDict, cores_for_worker_process, log, set_process_cpu_affinity


def init_rollout_worker_process(sf_context: SampleFactoryContext, worker: RolloutWorker):
    log.debug(f"Rollout worker {worker.worker_idx} starting...")

    set_global_context(sf_context)
    log.info(f"ROLLOUT worker {worker.worker_idx}\tpid {os.getpid()}\tparent {os.getppid()}")

    # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
    import signal as os_signal

    os_signal.signal(os_signal.SIGINT, os_signal.SIG_IGN)

    cfg = worker.cfg

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


def rollout_worker_device(worker_idx, cfg: AttrDict) -> torch.device:
    # TODO: this should correspond to whichever device we have observations on, not just whether we use this device at all
    # TODO: test with Megaverse on a multi-GPU system
    # TODO: actions on a GPU device? Convert to CPU for some envs?

    gpus_to_use = gpus_for_process(worker_idx, num_gpus_per_process=1, gpu_mask=cfg.actor_worker_gpus)
    assert len(gpus_to_use) <= 1
    sampling_device = torch.device("cuda", index=gpus_to_use[0]) if gpus_to_use else torch.device("cpu")
    return sampling_device


class RolloutWorker(StoppableEventLoopObject, Configurable):
    def __init__(
        self, event_loop, worker_idx: int, buffer_mgr, inference_queues: Dict[PolicyID, MpQueue], cfg, env_info: EnvInfo
    ):
        Configurable.__init__(self, cfg)
        unique_name = f"{RolloutWorker.__name__}_w{worker_idx}"
        EventLoopObject.__init__(self, event_loop, unique_name)

        self.timing = Timing(name=f"{self.object_id} profile")

        self.buffer_mgr = buffer_mgr
        self.inference_queues = inference_queues

        self.env_info = env_info
        self.worker_idx = worker_idx
        self.sampling_device = str(rollout_worker_device(self.worker_idx, self.cfg))

        self.vector_size = cfg.num_envs_per_worker
        self.num_splits = cfg.worker_num_splits
        assert self.vector_size >= self.num_splits
        assert self.vector_size % self.num_splits == 0, "Vector size should be divisible by num_splits"

        self.env_runners: List[VectorEnvRunner] = []

        self.reward_shaping = [None for _ in range(self.cfg.num_policies)]

        self.num_complete_rollouts = 0

        self.is_initialized = False

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
                self.reward_shaping,
            )

            policy_request = env_runner.init(self.timing)

            # send signal to the inference worker to start processing new observations
            self._enqueue_policy_request(split_idx, policy_request)
            self.env_runners.append(env_runner)

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
        if runner.update_trajectory_buffers(self.timing):
            policy_request = runner.generate_policy_request(self.timing)
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
            # log.warning('No policy requests on worker %d-%d', self.worker_idx, split_idx)
            # log.warning('Send fake signal to our own queue to wake up the worker on the next iteration')
            # advance_rollout_request = dict(split_idx=split_idx, policy_id=-1)
            # TODO: sent the same type of signal inference worker sends to us
            # TODO: connect to this signal
            # TODO: or maybe just proceed to the next iteration right away?
            # self.task_queue.put((TaskType.ROLLOUT_STEP, advance_rollout_request))
            pass

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

    def advance_rollouts(self, data: Tuple):
        # TODO: update comment
        """
        Process incoming request from policy worker. Use the data (policy outputs, actions) to advance the simulation
        by one step on the corresponding VectorEnvRunner.

        If we successfully managed to advance the simulation, send requests to policy workers to get actions for the
        next step. If we completed the entire rollout, also send request to the learner!

        :param data: request from the policy worker, containing actions and other policy outputs
        """
        with inference_context(self.cfg.serial_mode):
            split_idx, policy_id = data
            runner = self.env_runners[split_idx]
            complete_rollouts, episodic_stats = runner.advance_rollouts(policy_id, self.timing)

            with self.timing.add_time("complete_rollouts"):
                if complete_rollouts:
                    self._enqueue_complete_rollouts(complete_rollouts)
                    if self.num_complete_rollouts == 0 and not self.cfg.benchmark:
                        # we just finished our first complete rollouts, perfect time to wait for experience derorrelation
                        # this guarantees that there won't be any obsolete trajectories when we awaken
                        self._decorrelate_experience()
                    self.num_complete_rollouts += len(complete_rollouts)

            if episodic_stats:
                self.report_msg.emit(episodic_stats)

            # We finished one step of environment simulation.
            # If we also have the trajectory buffer to share the new data with the inference worker then
            # we are ready to enqueue inference request
            self._maybe_send_policy_request(runner)

    def on_trajectory_buffers_available(self):
        # we can receive this signal during batcher initialization, before the worker is initialized
        # this is fine, we just ignore it and get the trajectory buffers from the queue later after we do env.reset()
        if not self.is_initialized:
            return

        # it is possible that we finished the simulation step, but were unable to send data to inference worker
        # because we ran out of trajectory buffers. The purpose of this signal handler is to wake up the worker,
        # request a new trajectory (since they're now available), and finally send observations to the inference worker
        for split_idx in range(self.num_splits):
            self._maybe_send_policy_request(self.env_runners[split_idx])

    def on_stop(self, *args):
        for env_runner in self.env_runners:
            env_runner.close()

        timings = dict()
        if self.worker_idx in [0, self.cfg.num_workers - 1]:
            timings[self.object_id] = self.timing
        self.stop.emit(self.object_id, timings)
        super().on_stop(*args)
