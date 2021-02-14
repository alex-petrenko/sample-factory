import sys
import gym
import random
import numpy as np

from copy import deepcopy
from envs.env_utils import RewardShapingInterface
from utils.utils import log

DEFAULT_QUAD_REWARD_SHAPING = dict(
    quad_rewards=dict(
        pos=1.0, effort=0.05, spin=0.1, vel=0.0, crash=1.0, orient=1.0, yaw=0.0,
        quadcol_bin=0.0, quadsettle=0.0, quadcol_bin_obst=0.0, quad_spacing_coeff=0.0
    ),
)


class ReplayBuffer(object):
    def __init__(self, control_frequency, cp_step_size=0.5, buffer_size=1e6):
        self.control_frequency = control_frequency
        self.cp_step_size_sec = cp_step_size  # how often (seconds) a checkpoint is saved
        self.cp_step_size_freq = self.cp_step_size_sec * self.control_frequency
        self.buffer_size = buffer_size
        self.buffer_idx = 0
        self.buffer = []
        self.checkpoint_history = []
        self._cp_history_size = int(3.0 / cp_step_size)  # keep only checkpoints from the last 3 seconds

    def save_checkpoint(self, cp):
        '''
        Save a checkpoint every X steps so that we may load it later if a collision was found. This is NOT the same as the buffer
        Checkpoints are added to the buffer only if we find a collision and want to replay that event later on
        :param cp: A tuple of (env, actions)
        '''
        self.checkpoint_history.append(cp)
        self.checkpoint_history = self.checkpoint_history[:self._cp_history_size]

    def clear_checkpoints(self):
        self.checkpoint_history = []

    def write_cp_to_buffer(self, seconds_ago=1.5):
        '''
        A collision was found and we want to load the corresponding checkpoint from X seconds ago into the buffer to be sampled later on
        '''
        steps_ago = int(seconds_ago / self.cp_step_size_sec)
        try:
            env, actions = self.checkpoint_history[-steps_ago]
        except IndexError:
            log.info("tried to get checkpoint out of bounds of checkpoint_history in the replay buffer")
            sys.exit(1)

        env.saved_in_replay_buffer = True

        if len(self.buffer) < self.buffer_size:
            self.buffer.append((env, actions))
        else:
            self.buffer[self.buffer_idx] = (env, actions)  # override existing event
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size

    def sample_event(self):
        '''
        Sample an event to replay
        '''
        return random.choice(self.buffer)

    def __len__(self):
        return len(self.buffer)


class QuadsRewardShapingWrapper(gym.Wrapper, RewardShapingInterface):
    def __init__(self, env, reward_shaping_scheme=None):
        gym.Wrapper.__init__(self, env)
        RewardShapingInterface.__init__(self)
        self.reset_wrapper(env, reward_shaping_scheme=reward_shaping_scheme)
        # replay events with collisions
        self.replay_buffer = ReplayBuffer(env.envs[0].control_freq) if env.use_replay_buffer == True else None
        self.epsilon = 0.99  # TODO: Make configurable??

    def reset_wrapper(self, env, reward_shaping_scheme=None):
        '''
        reset self excluding the replay buffer
        '''
        gym.Wrapper.__init__(self, env)
        RewardShapingInterface.__init__(self)

        self.reward_shaping_scheme = reward_shaping_scheme
        self.cumulative_rewards = [dict() for _ in range(self.num_agents)]
        self.episode_actions = []

        self.num_agents = env.num_agents if hasattr(env, 'num_agents') else 1

        # save a reference to this wrapper in the actual env class, for other wrappers and for outside access
        self.env.unwrapped.reward_shaping_interface = self

    def get_default_reward_shaping(self):
        return self.reward_shaping_scheme

    def get_current_reward_shaping(self, agent_idx: int):
        return self.reward_shaping_scheme

    def set_reward_shaping(self, reward_shaping, unused_agent_idx):
        self.reward_shaping_scheme = reward_shaping

    def reset_env(self):
        '''
        Wrapper for env.reset(). Need this to sample from the replay buffer
        '''
        if self.replay_buffer:
            self.replay_buffer.clear_checkpoints()  # empty the checkpoints from the previous rollout
        if np.random.uniform(0, 1) < self.epsilon and self.replay_buffer and self.env.activate_replay_buffer\
                and len(self.replay_buffer) > 0:
            env, actions = self.replay_buffer.sample_event()
            env, actions = deepcopy(env), deepcopy(actions)
            self.reset_wrapper(env, self.reward_shaping_scheme)
            #  reset attributes that were deleted when first pickled
            self.env.reset_scene()
            self.env.reset_thrust_noise()
            obs, _, _, _ = self.env.step(actions)  # replay an event with a collision
        else:
            obs = self.env.reset()  # use the current env and proceed as normal
            self.env.saved_in_replay_buffer = False
        return obs

    def reset(self):
        obs = self.env.reset()
        self.cumulative_rewards = [dict() for _ in range(self.num_agents)]
        self.episode_actions = []
        return obs

    def step(self, action):
        self.episode_actions.append(action)

        # set the (potentially updated) reward shaping scheme
        env_reward_shaping = self.env.unwrapped.rew_coeff
        for key, weight in self.reward_shaping_scheme['quad_rewards'].items():
            env_reward_shaping[key] = weight

        obs, rewards, dones, infos = self.env.step(action)
        if self.env.is_multiagent:
            infos_multi, dones_multi = infos, dones
        else:
            infos_multi, dones_multi = [infos], [dones]

        for i, info in enumerate(infos_multi):
            rew_dict = info['rewards']

            for key, value in rew_dict.items():
                if key.startswith('rew'):
                    if key not in self.cumulative_rewards[i]:
                        self.cumulative_rewards[i][key] = 0
                    self.cumulative_rewards[i][key] += value

            if dones_multi[i]:
                true_reward = self.cumulative_rewards[i]['rewraw_main']
                true_reward_consider_collisions = True
                if true_reward_consider_collisions:
                    # we ideally want zero collisions, so collisions between quads are given very high weight
                    true_reward += 10000 * self.cumulative_rewards[i]['rewraw_quadcol']

                info['true_reward'] = true_reward
                if 'episode_extra_stats' not in info:
                    info['episode_extra_stats'] = dict()
                info['episode_extra_stats'].update(self.cumulative_rewards[i])

                episode_actions = np.array(self.episode_actions)
                episode_actions = episode_actions.transpose()
                for action_idx in range(episode_actions.shape[0]):
                    mean_action = np.mean(episode_actions[action_idx])
                    std_action = np.std(episode_actions[action_idx])
                    info['episode_extra_stats'][f'z_action{action_idx}_mean'] = mean_action
                    info['episode_extra_stats'][f'z_action{action_idx}_std'] = std_action

                self.cumulative_rewards[i] = dict()

        if any(dones_multi):
            self.episode_actions = []

        return obs, rewards, dones, infos

    def close(self):
        # remove the reference to avoid dependency cycles
        self.env.unwrapped.reward_shaping_interface = None
        return self.env.close()
