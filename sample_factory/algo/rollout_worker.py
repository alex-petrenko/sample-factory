from __future__ import annotations

import os
import time
from queue import Empty
from typing import Dict, Optional, Any, List, Tuple

import psutil
import torch
from torch import Tensor

from sample_factory.algo.utils.context import SampleFactoryContext, set_global_context
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algorithms.appo.appo_utils import make_env_func, SequentialVectorizeWrapper, set_gpus_for_process, \
    gpus_for_process
from sample_factory.algorithms.appo.policy_manager import PolicyManager
from sample_factory.cfg.configurable import Configurable
from sample_factory.signal_slot.signal_slot import EventLoopObject, signal
from sample_factory.utils.timing import Timing
from sample_factory.utils.typing import PolicyID, MpQueue
from sample_factory.utils.utils import log, set_process_cpu_affinity, AttrDict


def init_rollout_worker_process(sf_context: SampleFactoryContext, worker: RolloutWorker):
    set_global_context(sf_context)
    log.info(f'ROLLOUT worker {worker.worker_idx}\tpid {os.getpid()}\tparent {os.getppid()}')

    # workers should ignore Ctrl+C because the termination is handled in the event loop by a special msg
    import signal as os_signal
    os_signal.signal(os_signal.SIGINT, os_signal.SIG_IGN)

    cfg = worker.cfg

    if cfg.force_envs_single_thread:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1, user_api=None)  # TODO: this should be set to 1 only if this worker uses 1 core

    if cfg.set_workers_cpu_affinity:
        set_process_cpu_affinity(worker.worker_idx, cfg.num_workers)

    # TODO: for envs like IsaacGym we probably don't want to decrease the process priority
    psutil.Process().nice(min(cfg.default_niceness + 10, 20))

    # TODO: don't do this!
    if cfg.actor_worker_gpus:
        worker_gpus = set_gpus_for_process(
            worker.worker_idx,
            num_gpus_per_process=1, process_type='actor', gpu_mask=cfg.actor_worker_gpus,
        )
        assert len(worker_gpus) == 1

    torch.multiprocessing.set_sharing_strategy('file_system')


# TODO: remove code duplication (actor_worker.py)
def preprocess_actions(env_info: EnvInfo, actions: Tensor):
    if env_info.integer_actions:
        actions = actions.to(torch.int32)  # is it faster to do on GPU or CPU?

    if not env_info.gpu_actions:
        actions = actions.cpu().numpy()

    # TODO: do we need this? actions are a tensor of size [batch_size, action_shape] (or just [batch_size] if it is a single action per env)
    # if len(actions) == 1:
    #     actions = actions.item()

    return actions


def rollout_worker_device(worker_idx, cfg: AttrDict) -> torch.device:
    gpus_to_use = gpus_for_process(worker_idx, num_gpus_per_process=1, gpu_mask=cfg.actor_worker_gpus)
    assert len(gpus_to_use) <= 1
    sampling_device = torch.device('cuda', index=gpus_to_use[0]) if gpus_to_use else torch.device('cpu')
    return sampling_device


class BatchedVectorEnvRunner:
    # TODO: comment
    """
    A collection of environments simulated sequentially.
    With double buffering each actor worker holds two vector runners and switches between them.
    Without single buffering we only use a single VectorEnvRunner per actor worker.

    All envs on a single VectorEnvRunner run in unison, e.g. they all do one step at a time together.
    This also means they all finish their rollouts together. This allows us to minimize the amount of messages
    passed around.

    Individual envs (or agents in these envs in case of multi-agent) can potentially be controlled by different
    policies when we're doing PBT. We only start simulating the next step in the environment when
    all actions from all envs and all policies are collected. This leaves optimization potential: we can start
    simulating some envs right away as actions for them arrive. But usually double-buffered sampling masks
    this type of inefficiency anyway. The worker is probably still rendering a previous vector of envs when
    the actions arrive.
    """

    def __init__(
            self, cfg, env_info, num_envs, worker_idx, split_idx, policy_id: PolicyID,
            buffer_mgr, sampling_device: str, pbt_reward_shaping,  #TODO pbt reward
    ):
        # TODO: comment
        """
        Ctor.

        :param cfg: global system config (all CLI params)
        :param num_envs: number of envs to run in this vector runner
        :param worker_idx: idx of the parent worker
        :param split_idx: index of the environment group in double-buffered sampling (either 0 or 1). Always 0 when
        double-buffered sampling is disabled.
        :param num_agents: number of agents in each env (1 for single-agent envs)
        :param buffer_mgr: a collection of all shared data structures used by the algorithm. Most importantly,
        the trajectory buffers in shared memory.
        :param pbt_reward_shaping: initial reward shaping dictionary, for configuration where PBT optimizes
        reward coefficients in environments.
        """

        self.cfg = cfg
        self.env_info = env_info

        self.num_envs = num_envs
        self.worker_idx = worker_idx
        self.split_idx = split_idx
        self.policy_id = policy_id

        self.vec_env = None
        self.last_obs = None
        self.last_rnn_state = None
        self.policy_id_buffer = None

        self.buffer_mgr = buffer_mgr

        self.traj_tensors = buffer_mgr.traj_tensors[sampling_device]
        self.policy_output_tensors = buffer_mgr.policy_output_tensors[sampling_device][self.worker_idx, self.split_idx]

        self.traj_buffer_queue = buffer_mgr.traj_buffer_queues[sampling_device]

        self.curr_traj: Optional[TensorDict] = None
        self.curr_step: Optional[TensorDict] = None
        self.curr_traj_slice: Optional[slice] = None

        self.rollout_step: int = 0  # current position in the rollout across all envs in vec_env
        self.env_step_ready = False

        self.curr_episode_reward = self.curr_episode_len = None

        self.pbt_reward_shaping = pbt_reward_shaping  # TODO

    def init(self, timing) -> Dict:
        """
        Actually instantiate the env instances.
        Also creates ActorState objects that hold the state of individual actors in (potentially) multi-agent envs.
        """
        envs = []
        for env_i in range(self.num_envs):
            vector_idx = self.split_idx * self.num_envs + env_i

            # global env id within the entire system
            env_id = self.worker_idx * self.cfg.num_envs_per_worker + vector_idx

            env_config = AttrDict(
                worker_index=self.worker_idx, vector_index=vector_idx, env_id=env_id,
            )

            # log.info('Creating env %r... %d-%d-%d', env_config, self.worker_idx, self.split_idx, env_i)
            # a vectorized environment - we assume that it always provides a dict of vectors of obs, rewards, dones, infos
            env = make_env_func(self.cfg, env_config=env_config)
            env.seed(env_id)
            envs.append(env)

        if len(envs) == 1:
            self.vec_env = envs[0]
        else:
            self.vec_env = SequentialVectorizeWrapper(envs)

        self.update_trajectory_buffers(timing)
        assert self.curr_traj is not None and self.curr_traj_slice is not None

        self.last_obs = self.vec_env.reset()
        self.last_rnn_state = self.traj_tensors['rnn_states'][0:self.vec_env.num_agents, 0].clone().fill_(0.0)
        self.env_step_ready = True
        assert self.rollout_step == 0

        self.policy_id_buffer = self.traj_tensors['policy_id'][0:self.vec_env.num_agents, 0].clone().fill_(self.policy_id)

        self.curr_episode_reward = torch.zeros(self.vec_env.num_agents)
        self.curr_episode_len = torch.zeros(self.vec_env.num_agents, dtype=torch.int32)

        policy_request = self.generate_policy_request(timing)
        assert policy_request is not None
        return policy_request

    def _process_rewards(self, rewards_orig: Tensor, infos: Dict[Any, Any], values: Tensor):
        rewards = rewards_orig * self.cfg.reward_scale
        rewards.clamp_(-self.cfg.reward_clip, self.cfg.reward_clip)

        if self.cfg.value_bootstrap and 'time_outs' in infos:
            # What we really want here is v(t+1) which we don't have, using v(t) is an approximation that
            # requires that rew(t) can be generally ignored.
            # TODO: if gamma is modified by PBT it should be updated here too?!
            rewards.add_(self.cfg.gamma * values * infos['time_outs'].float())

        return rewards

    def _process_env_step(self, rewards_orig, dones_orig, infos):
        rewards = rewards_orig.cpu()
        dones = dones_orig.cpu()

        self.curr_episode_reward += rewards
        self.curr_episode_len += 1

        finished_episodes = dones.nonzero(as_tuple=True)[0]

        # TODO: get rid of the loop (we can do it vectorized)
        # TODO: remove code duplication
        reports = []
        for i in finished_episodes:
            agent_i = i.item()

            last_episode_reward = self.curr_episode_reward[agent_i].item()
            last_episode_duration = self.curr_episode_len[agent_i].item()

            last_episode_true_objective = last_episode_reward
            last_episode_extra_stats = None

            # TODO: we somehow need to deal with two cases: when infos is a dict of tensors and when it is a list of dicts
            # this only handles the latter.
            if isinstance(infos, (list, tuple)):
                last_episode_true_objective = infos[agent_i].get('true_objective', last_episode_reward)
                last_episode_extra_stats = infos[agent_i].get('episode_extra_stats', None)

            stats = dict(reward=last_episode_reward, len=last_episode_duration, true_objective=last_episode_true_objective)
            if last_episode_extra_stats:
                stats['episode_extra_stats'] = last_episode_extra_stats

            report = dict(episodic=stats, policy_id=self.policy_id)
            reports.append(report)

        self.curr_episode_reward[finished_episodes] = 0
        self.curr_episode_len[finished_episodes] = 0
        return reports

    def _finalize_trajectories(self) -> List[Dict]:
        # Saving obs and hidden states for the step AFTER the last step in the current rollout.
        # We're going to need them later when we calculate next step value estimates.
        self.curr_traj['obs'][:, self.cfg.rollout] = self.last_obs
        self.curr_traj['rnn_states'][:, self.cfg.rollout] = self.last_rnn_state

        # TODO: make this consistent with the CPU sampler
        traj_dict = dict(policy_id=self.policy_id, traj_buffer_idx=self.curr_traj_slice)
        return [traj_dict]

    def advance_rollouts(self, data, timing) -> Tuple[List[Dict], List[Dict]]:
        # TODO: comment
        """
        Main function in VectorEnvRunner. Does one step of simulation (if all actions for all actors are available).

        :param data: incoming data from policy workers (policy outputs), including new actions
        :param timing: this is just for profiling
        :return: same as reset(), return a set of requests for policy workers, asking them to generate actions for
        the next env step.
        """
        with timing.add_time('process_policy_outputs'):
            # save actions/logits/values etc. for the current rollout step
            self.curr_step[:] = self.policy_output_tensors  # TODO: output tensors should contain the policy version
            actions = preprocess_actions(self.env_info, self.policy_output_tensors['actions'])

        complete_rollouts, episodic_stats = [], []  # TODO

        with timing.add_time('env_step'):
            self.last_obs, rewards, dones, infos = self.vec_env.step(actions)

        with timing.add_time('post_env_step'):
            self.policy_id_buffer.fill_(self.policy_id)

            # TODO: for vectorized envs we either have a dictionary of tensors (isaacgym_examples), or a list of dictionaries (i.e. swarm_rl quadrotors)
            # Need an adapter class so it's consistent, i.e. always a dict of tensors.
            # this should yield indices of inactive agents
            #
            # if infos:
            #     inactive_agents = [i for i, info in enumerate(infos) if not info.get('is_active', True)]
            #     self.policy_id_buffer[inactive_agents] = -1
            # TODO: batcher runner probably won't have inactive agent support for now.

            # record the results from the env step
            processed_rewards = self._process_rewards(rewards, infos, self.policy_output_tensors['values'])
            self.curr_step[:] = dict(rewards=processed_rewards, dones=dones, policy_id=self.policy_id_buffer)

            # reset next-step hidden states to zero if we encountered an episode boundary
            # not sure if this is the best practice, but this is what everybody seems to be doing
            not_done = (1.0 - self.curr_step['dones'].float()).unsqueeze(-1)
            self.last_rnn_state = self.policy_output_tensors['new_rnn_states'] * not_done

            stats = self._process_env_step(rewards, dones, infos)
            episodic_stats.extend(stats)

        self.rollout_step += 1

        with timing.add_time('finalize_trajectories'):
            if self.rollout_step == self.cfg.rollout:
                # finalize and serialize the trajectory if we have a complete rollout
                complete_rollouts = self._finalize_trajectories()
                self.rollout_step = 0
                # we will need to request a new trajectory buffer!
                self.curr_traj_slice = self.curr_traj = None

        self.env_step_ready = True
        return complete_rollouts, episodic_stats

    def update_trajectory_buffers(self, timing):
        if self.curr_traj_slice is not None and self.curr_traj is not None:
            # don't need to do anything - we have a trajectory buffer already
            return

        with timing.add_time('wait_for_trajectories'):
            # TODO some fancy logic with attempts and retries to make it clear when we don't have enough trajectories
            try:
                # TODO: different function for sync vs async!
                buffers = self.traj_buffer_queue.get(block=False)
            except Empty:
                return

            self.curr_traj_slice = buffers
            self.curr_traj = self.traj_tensors[self.curr_traj_slice]

    def generate_policy_request(self, timing) -> Optional[Dict]:
        if not self.env_step_ready:
            # we haven't actually simulated the environment yet
            log.debug('Cannot generate policy request because we have not finished the env simulation step yet!')
            return None

        if self.curr_traj is None:
            # we don't have a shared buffer to store data in - still waiting for one to become available
            # log.debug('Cannot generate policy request because we don not have the trajectory buffer')
            return None

        with timing.add_time('prepare_next_step'):
            self.curr_step = self.curr_traj[:, self.rollout_step]
            # save observations and RNN states in a trajectory
            self.curr_step[:] = dict(obs=self.last_obs, rnn_states=self.last_rnn_state)
            policy_request = {self.policy_id: (self.curr_traj_slice, self.rollout_step)}
            self.env_step_ready = False

        return policy_request


class RolloutWorker(EventLoopObject, Configurable):
    def __init__(self, event_loop, worker_idx: int, buffer_mgr, inference_queues: Dict[PolicyID, MpQueue], cfg, env_info: EnvInfo):
        Configurable.__init__(self, cfg)
        unique_name = f'{RolloutWorker.__name__}_w{worker_idx}'
        EventLoopObject.__init__(self, event_loop, unique_name)

        self.timing = Timing(name=f'{self.object_id} profile')

        self.buffer_mgr = buffer_mgr
        self.inference_queues = inference_queues

        self.env_info = env_info
        self.worker_idx = worker_idx
        self.sampling_device = str(rollout_worker_device(self.worker_idx, self.cfg))

        self.vector_size = cfg.num_envs_per_worker
        self.num_splits = cfg.worker_num_splits
        assert self.vector_size >= self.num_splits
        assert self.vector_size % self.num_splits == 0, 'Vector size should be divisible by num_splits'

        self.env_runners = []

        self.reward_shaping = [None for _ in range(self.cfg.num_policies)]

        self.num_complete_rollouts = 0

        self.is_initialized = False

    @signal
    def report_msg(self): pass

    @signal
    def stop(self): pass

    def init(self):
        policy_id = self.worker_idx % self.cfg.num_policies
        log.debug(f'Worker {self.object_id} uses policy {policy_id}')  # don't print this for non-batched runners

        for split_idx in range(self.num_splits):
            # TODO: logic for batched vs non-batched runner
            env_runner = BatchedVectorEnvRunner(
                self.cfg, self.env_info, self.vector_size // self.num_splits,
                self.worker_idx, split_idx, policy_id, self.buffer_mgr, self.sampling_device, self.reward_shaping,
            )

            policy_request = env_runner.init(self.timing)

            # send signal to the inference worker to start processing new observations
            self._enqueue_policy_request(split_idx, policy_request)
            self.env_runners.append(env_runner)

        self.is_initialized = True

    def _decorrelate_experience(self):
        delay = (float(self.worker_idx) / self.cfg.num_workers) * self.cfg.decorrelate_experience_max_seconds
        log.info(
            'Worker %d, sleep for %.3f sec to decorrelate experience collection',
            self.worker_idx, delay,
        )
        time.sleep(delay)
        log.info('Worker %d awakens!', self.worker_idx)

    def _maybe_send_policy_request(self, runner):
        runner.update_trajectory_buffers(self.timing)
        policy_request = runner.generate_policy_request(self.timing)
        with self.timing.add_time('enqueue_policy_requests'):
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
            advance_rollout_request = dict(split_idx=split_idx, policy_id=-1)
            # TODO: sent the same type of signal inference worker sends to us
            # TODO: connect to this signal
            # TODO: or maybe just proceed to the next iteration right away?
            # self.task_queue.put((TaskType.ROLLOUT_STEP, advance_rollout_request))

    def _enqueue_complete_rollouts(self, complete_rollouts: List[Dict]):
        """Emit complete rollouts."""
        rollouts_per_policy = dict()
        for rollout in complete_rollouts:
            policy_id = rollout['policy_id']
            if policy_id not in rollouts_per_policy:
                rollouts_per_policy[policy_id] = []
            rollouts_per_policy[policy_id].append(rollout)

        for policy_id, rollouts in rollouts_per_policy.items():
            self.emit(f'p{policy_id}_trajectories', rollouts, self.sampling_device)

    # TODO: this should be connected to a signal from the inference worker
    def advance_rollouts(self, data: Tuple):
        # TODO: comment
        """
        Process incoming request from policy worker. Use the data (policy outputs, actions) to advance the simulation
        by one step on the corresponding VectorEnvRunner.

        If we successfully managed to advance the simulation, send requests to policy workers to get actions for the
        next step. If we completed the entire rollout, also send request to the learner!

        :param data: request from the policy worker, containing actions and other policy outputs
        """
        split_idx, policy_id = data
        runner = self.env_runners[split_idx]
        complete_rollouts, episodic_stats = runner.advance_rollouts(data, self.timing)

        with self.timing.add_time('complete_rollouts'):
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

    # TODO: base class for this to avoid repeating this code everywhere
    def on_stop(self, emitter_id):
        log.debug(f'Stopping {self.object_id}...')

        self.stop.emit(self.object_id)

        if self.event_loop.owner is self:
            self.event_loop.stop()

        self.detach()  # remove from the current event loop
        log.info(self.timing)
