from typing import Dict, Any

import torch
from torch import Tensor

from sample_factory.algorithms.appo.appo_utils import make_env_func_v2
from sample_factory.cfg.configurable import Configurable
from sample_factory.signal_slot.signal_slot import signal, EventLoopObject
from sample_factory.utils.utils import AttrDict


# TODO: remove code duplication (actor_worker.py)
def preprocess_actions(env_info, actions):
    if env_info.integer_actions:
        actions = actions.to(torch.int32)  # is it faster to do on GPU or CPU?

    if not env_info.gpu_actions:
        actions = actions.cpu().numpy()

    # TODO: do we need this? actions are a tensor of size [batch_size, action_shape] (or just [batch_size] if it is a single action per env)
    # if len(actions) == 1:
    #     actions = actions.item()

    return actions


class Sampler(Configurable):
    def __init__(self, cfg, env_info):
        super().__init__(cfg)
        self.env_info = env_info


class SyncSampler(EventLoopObject, Sampler):
    def __init__(self, evt_loop, cfg, env_info, actor_critic, device, buffer_mgr, timing):
        Sampler.__init__(self, cfg, env_info)

        self.curr_policy_id = 0  # sync sampler does not support multi-policy learning as of now
        unique_name = f'{SyncSampler.__name__}_{self.curr_policy_id}'
        EventLoopObject.__init__(self, evt_loop, unique_name)

        self.timing = timing

        self.actor_critic = actor_critic
        self.device = device

        self.traj_tensors = None

        self.vec_env = None
        self.last_obs = None
        self.last_rnn_state = None
        self.policy_id_buffer = None

        self.buffer_mgr = buffer_mgr
        self.traj_tensors = buffer_mgr.traj_tensors

        self.traj_start = 0

        self.curr_episode_reward = self.curr_episode_len = None

    @signal
    def report_msg(self): pass

    @signal
    def new_trajectories(self): pass

    def init(self):
        # with sync sampler there aren't any workers, hence 0/0/0 should suffice
        env_config = AttrDict(worker_index=0, vector_index=0, env_id=0)

        # a vectorized environment - we assume that it always provides a dict of vectors of obs, rewards, dones, infos
        self.vec_env = make_env_func_v2(self.cfg, env_config=env_config)

        self.last_obs = self.vec_env.reset()
        self.last_rnn_state = self.traj_tensors['rnn_states'][0:self.env_info.num_agents, 0].clone().fill_(0.0)
        self.policy_id_buffer = self.traj_tensors['policy_id'][0:self.env_info.num_agents, 0].clone()

        self.curr_episode_reward = torch.zeros(self.env_info.num_agents)
        self.curr_episode_len = torch.zeros(self.env_info.num_agents, dtype=torch.int32)

    def process_rewards(self, rewards_orig: Tensor, infos: Dict[Any, Any], values: Tensor):
        rewards = rewards_orig * self.cfg.reward_scale
        rewards.clamp_(-self.cfg.reward_clip, self.cfg.reward_clip)

        if self.cfg.value_bootstrap and 'time_outs' in infos:
            # What we really want here is v(t+1) which we don't have, using v(t) is an approximation that
            # requires that rew(t) can be generally ignored.
            # TODO: if gamma is modified by PBT it should be updated here too?!
            rewards.add_(self.cfg.gamma * values * infos['time_outs'].float())

        return rewards

    def process_env_step(self, rewards_orig, dones_orig, infos):
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

            report = dict(episodic=stats, policy_id=self.curr_policy_id)
            reports.append(report)

        self.curr_episode_reward[finished_episodes] = 0
        self.curr_episode_len[finished_episodes] = 0
        return reports

    def collect_trajectories(self):
        with torch.no_grad(), self.timing.add_time('sampling'):
            self.actor_critic.eval()

            num_agents = self.env_info.num_agents
            if self.traj_start + num_agents > self.buffer_mgr.total_num_trajectories:  # TODO: need mechanism to actually allocate and clean up trajectories
                self.traj_start = 0

            # subset of trajectory buffers we're going to populate in the current iteration
            curr_traj = self.traj_tensors[self.traj_start:self.traj_start + num_agents]

            episodic_stats = []
            for step in range(self.cfg.rollout):
                curr_step = curr_traj[:, step]

                # save observations and RNN states in a trajectory
                curr_step[:] = dict(obs=self.last_obs, rnn_states=self.last_rnn_state)

                with self.timing.add_time('norm'):
                    normalized_obs = self.actor_critic.normalizer(curr_step['obs'])

                # obs and rnn_states obtained from the trajectory buffers should be on the same device as the model
                with self.timing.add_time('inference'):
                    policy_outputs = self.actor_critic(normalized_obs, curr_step['rnn_states'])

                with self.timing.add_time('post_inference'):
                    new_rnn_state = policy_outputs['rnn_states']

                    # copy all policy outputs to corresponding trajectory buffers - except for rnn_states!
                    # they should be saved to the next step
                    del policy_outputs['rnn_states']

                    for key, value in policy_outputs.items():
                        curr_step[key][:] = value

                    curr_step[:] = policy_outputs
                    curr_step['policy_version'].fill_(self.buffer_mgr.policy_versions[self.curr_policy_id])

                    actions = preprocess_actions(self.env_info, policy_outputs['actions'])

                with self.timing.add_time('env_step'):
                    self.last_obs, rewards, dones, infos = self.vec_env.step(actions)

                with self.timing.add_time('post_env_step'):
                    self.policy_id_buffer.fill_(self.curr_policy_id)

                    # TODO: for vectorized envs we either have a dictionary of tensors (isaacgym), or a list of dictionaries (i.e. swarm_rl quadrotors)
                    # Need an adapter class so it's consistent, i.e. always a dict of tensors.
                    # this should yield indices of inactive agents
                    #
                    # if infos:
                    #     inactive_agents = [i for i, info in enumerate(infos) if not info.get('is_active', True)]
                    #     self.policy_id_buffer[inactive_agents] = -1

                    # record the results from the env step
                    processed_rewards = self.process_rewards(rewards, infos, policy_outputs['values'])
                    curr_step[:] = dict(rewards=processed_rewards, dones=dones, policy_id=self.policy_id_buffer)

                    # reset next-step hidden states to zero if we encountered an episode boundary
                    # not sure if this is the best practice, but this is what everybody seems to be doing
                    not_done = (1.0 - curr_step['dones'].float()).unsqueeze(-1)
                    self.last_rnn_state = new_rnn_state * not_done

                    stats = self.process_env_step(rewards, dones, infos)
                    episodic_stats.extend(stats)

            # Saving obs and hidden states for the step AFTER the last step in the current rollout.
            # We're going to need them later when we calculate next step value estimates.
            curr_traj['obs'][:, self.cfg.rollout] = self.last_obs
            curr_traj['rnn_states'][:, self.cfg.rollout] = self.last_rnn_state

            # returning the slice of the trajectory buffer we managed to populate
            traj_slice = slice(self.traj_start, self.traj_start + num_agents)
            self.traj_start += num_agents
            self.new_trajectories.emit([traj_slice])

            samples_since_last_report = num_agents * self.cfg.rollout
            report = [dict(samples_collected=samples_since_last_report, policy_id=self.curr_policy_id)]
            report.extend(episodic_stats)
            self.report_msg.emit(report)

    def update_training_info(self, env_steps, stats, avg_stats, policy_avg_stats):
        """
        TODO: we should propagate the training info to the environment instances, similar to "set_training_info()"
        call in actor_worker.py
        """
        pass
