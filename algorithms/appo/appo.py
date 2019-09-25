import copy
import math
import time
from collections import OrderedDict

import ray
import torch
import numpy as np
from tensorboardX import SummaryWriter

from algorithms.ppo.agent_ppo import ActorCritic
from algorithms.utils.algo_utils import EPS, num_env_steps
from algorithms.utils.multi_agent import MultiAgentWrapper
from envs.create_env import create_env
from utils.timing import Timing
from utils.utils import summaries_dir, experiment_dir, AttrDict, log, str2bool


class Algorithm:
    @classmethod
    def add_cli_args(cls, parser):
        p = parser

        p.add_argument('--seed', default=42, type=int, help='Set a fixed seed value')

        p.add_argument('--initial_save_rate', default=1000, type=int,
                       help='Save model every N steps in the beginning of training')
        p.add_argument('--keep_checkpoints', default=4, type=int, help='Number of model checkpoints to keep')

        p.add_argument('--stats_episodes', default=100, type=int, help='How many episodes to average to measure performance (avg. reward etc)')

        p.add_argument('--learning_rate', default=1e-4, type=float, help='LR')

        p.add_argument('--train_for_steps', default=int(1e10), type=int, help='Stop training after this many SGD steps')
        p.add_argument('--train_for_env_steps', default=int(1e10), type=int, help='Stop training after this many environment steps')
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

        # TODO:
        # if self.cfg.seed is not None:
        #     log.info('Settings fixed seed %d', self.cfg.seed)
        #     torch.manual_seed(self.cfg.seed)
        #     np.random.seed(self.cfg.seed)

        self.train_step = self.env_steps = 0

        self.total_train_seconds = 0
        self.last_training_step = time.time()

        self.best_avg_reward = math.nan

        summary_dir = summaries_dir(experiment_dir(cfg=self.cfg))
        self.writer = SummaryWriter(summary_dir, flush_secs=10)


class Agent:
    """Entity responsible for acting in the environment, e.g. collecting rollouts."""

    def __init__(self, cfg):
        self.cfg = cfg


class Learner:
    """Entity responsible for learning from experience."""

    def __init__(self, cfg):
        self.cfg = cfg


class AgentAPPO(Agent):
    def __init__(self, cfg, env):
        super().__init__(cfg)

        torch.set_num_threads(1)
        self.device = 'cpu'

        # initialize the Torch module
        self.actor_critic = ActorCritic(env.observation_space, env.action_space, cfg)
        self.actor_critic.to(self.device)

        self.observations = None
        self.rnn_states = None
        self.policy_output = None

        self.trajectories = None

    def preprocess_observations(self, observations):
        if len(observations) <= 0:
            return observations

        obs_dict = AttrDict()
        if isinstance(observations[0], (dict, OrderedDict)):
            for key in observations[0].keys():
                if not isinstance(observations[0][key], str):
                    obs_dict[key] = [o[key] for o in observations]
        else:
            # handle flat observations also as dict
            obs_dict.obs = observations

        mean = self.cfg.obs_subtract_mean
        scale = self.cfg.obs_scale

        # TODO: convert to numpy?!
        # TODO: do the preprocessing later to avoid sharing float buffers!
        if abs(mean) > EPS and abs(scale - 1.0) > EPS:
            obs_dict.obs = (obs_dict.obs - mean) * (1.0 / scale)  # convert rgb observations to [-1, 1]

        self.observations = obs_dict

    def reset_rnn_states(self):
        self.rnn_states = torch.zeros(self.observations.shape[0], self.cfg.hidden_size).to(self.device)

    def step(self):
        if self.rnn_states is None:
            self.reset_rnn_states()

        self.policy_output = self.actor_critic(self.observations, self.rnn_states)
        return self.policy_output

    def _trajectory_add_args(self, args):
        for arg_name, arg_value in args.items():
            if arg_value is None:
                continue

            for i, x in enumerate(arg_value.items()):
                if arg_name not in self.trajectories[i]:
                    self.trajectories[i][arg_name] = [x]
                else:
                    self.trajectories[i][arg_name].append(x)

    def _trajectory_add_attributes(
            self, obs, rnn_states, actions, action_logits, log_prob_actions, values, rewards, dones,
    ):
        args = copy.copy(locals())
        self._trajectory_add_args(args)

    def update_trajectories(self, rewards, dones):
        if self.trajectories is None:
            self.trajectories = [dict() for _ in range(len(rewards))]

        res = self.policy_output

        self._trajectory_add_attributes(
            self.observations, self.rnn_states,
            res.actions, res.action_logits, res.log_prob_actions, res.values,
            rewards, dones,
        )

    def calculate_next_values(self):
        next_values = self.actor_critic(self.observations, self.rnn_states).values
        for t, v in zip(self.trajectories, next_values):
            t['values'].append(v)


@ray.remote
class ActorWorker:
    def __init__(self, worker_index, cfg):
        self.cfg = cfg

        def make_env_func(env_config_):
            return create_env(cfg.env, cfg=cfg, env_config=env_config_)

        self.worker_index = worker_index
        env_config = AttrDict({'worker_index': worker_index, 'vector_index': 0})

        self.env = make_env_func(env_config)

        if not hasattr(self.env, 'num_agents'):
            self.env = MultiAgentWrapper(self.env)

        torch.set_num_threads(1)

        self.policies = dict()
        self.policy_actor_map = dict()
        for policy_id in range(self.cfg.num_policies):
            policy_name = f'policy_{policy_id}'
            self.policies[policy_name] = AgentAPPO(self.cfg, self.env)
            self.policy_actor_map[policy_name] = []

        # number of simultaneous agents in the environment
        self.num_agents = self.env.num_agents

        # default actor-policy map (can be changed during training)
        self.actor_policy_map = []
        default_policy_id = list(self.policies.keys())[0]
        for agent_id in range(self.num_agents):
            self.actor_policy_map.append(default_policy_id)
            self.policy_actor_map[default_policy_id].append(agent_id)

        initial_observations = self.env.reset()
        self.preprocess_observations(initial_observations)

        self.trajectory_counter = 0

    def update_parameters(self, state_dict):
        log.info('Loading new parameters on worker %d', self.worker_index)
        self.agent.actor_critic.load_state_dict(state_dict['model'])

    def preprocess_observations(self, obs):
        """Obs is a list with the size = num_agents."""
        for policy_id, actors in self.policy_actor_map.items():
            if len(actors) <= 0:
                # this policy does not participate in the current rollout
                continue

            policy_obs = []
            for actor_id in actors:
                policy_obs.append(obs[actor_id])

            self.policies[policy_id].preprocess_observations(policy_obs)

    def update_rnn_states(self, dones):
        for policy_id, actors in self.policy_actor_map.items():
            if len(actors) > 0:
                policy_dones = np.asarray(dones)[actors]
                if all(policy_dones):
                    self.policies[policy_id].reset_rnn_states()

    def policy_step(self):
        actions = [None] * self.num_agents

        for policy_id, actors in self.policy_actor_map.items():
            if len(actors) <= 0:
                # this policy does not participate in the current rollout
                continue

            policy_outputs = self.policies[policy_id].step()

            for i, agent_index in enumerate(actors):
                policy_action = policy_outputs[i].actions.cpu().numpy()
                actions[agent_index] = policy_action

        return actions

    def update_trajectories(self, rewards, dones):
        for policy_id, actors in self.policy_actor_map.items():
            if len(actors) <= 0:
                continue

            policy_rewards = np.asarray(rewards)[actors]
            policy_dones = np.asarray(dones)[actors]
            self.policies[policy_id].update_trajectories(policy_rewards, policy_dones)

    def calculate_next_values(self):
        for policy_id, actors in self.policy_actor_map.items():
            if len(actors) > 0:
                self.policies[policy_id].calculate_next_values()

    def finalize_trajectories(self):
        trajectories = dict()

        for policy_id, policy in self.policies.values():
            if policy.trajectories is not None and len(policy.trajectories) > 0:
                trajectories[policy_id] = []

                for t in policy.trajectories:
                    t_id = f'{self.worker_index}_{self.trajectory_counter}'
                    trajectories[policy_id].append((t_id, t))

                policy.trajectories = None

        # TODO: calculate GAE?
        return trajectories

    def generate_rollout(self):
        timing = Timing()
        num_steps = 0

        with torch.no_grad():
            for rollout_step in range(self.cfg.rollout):
                actions = self.policy_step()

                # wait for all the workers to complete an environment step
                with timing.add_time('env_step'):
                    new_obs, rewards, dones, infos = self.env.step(actions)

                rewards = np.asarray(rewards, dtype=np.float32)
                rewards = np.clip(rewards, -self.cfg.reward_clip, self.cfg.reward_clip)
                rewards = rewards * self.cfg.reward_scale

                self.update_trajectories(rewards, dones)

                self.preprocess_observations(new_obs)
                self.update_rnn_states(dones)

                num_steps += num_env_steps(infos)
                if all(self.dones):
                    break

            self.calculate_next_values()

            trajectories = self.finalize_trajectories()
            return trajectories

    def close(self):
        self.env.close()


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

        p.add_argument('--num_envs', default=96, type=int, help='Number of environments to collect experience from. Size of the training batch is rollout X num_envs')
        p.add_argument('--num_workers', default=16, type=int, help='Number of parallel environment workers. Should be less than num_envs and should divide num_envs')

        p.add_argument('--recurrence', default=32, type=int, help='Trajectory length for backpropagation through time. If recurrence=1 there is no backpropagation through time, and experience is shuffled completely randomly')
        p.add_argument('--use_rnn', default=True, type=str2bool, help='Whether to use RNN core in a policy or not')

        p.add_argument('--ppo_clip_ratio', default=1.1, type=float, help='We use unbiased clip(x, e, 1/e) instead of clip(x, 1+e, 1-e) in the paper')
        p.add_argument('--ppo_clip_value', default=0.2, type=float, help='Maximum absolute change in value estimate until it is clipped. Sensitive to value magnitude')
        p.add_argument('--batch_size', default=1024, type=int, help='PPO minibatch size')
        p.add_argument('--ppo_epochs', default=4, type=int, help='Number of training epochs before a new batch of experience is collected')
        p.add_argument('--target_kl', default=0.02, type=float, help='Target distance from behavior policy at the end of training on each experience batch')
        p.add_argument('--early_stopping', default=False, type=str2bool, help='Early stop training on the experience batch when KL-divergence is too high')

        p.add_argument('--normalize_advantage', default=True, type=str2bool, help='Whether to normalize advantages or not (subtract mean and divide by standard deviation)')

        p.add_argument('--max_grad_norm', default=2.0, type=float, help='Max L2 norm of the gradient vector')

        # components of the loss function
        p.add_argument(
            '--prior_loss_coeff', default=0.0005, type=float,
            help=('Coefficient for the exploration component of the loss function. Typically this is entropy maximization, but here we use KL-divergence between our policy and a prior.'
                  'By default prior is a uniform distribution, and this is numerically equivalent to maximizing entropy.'
                  'Alternatively we can use custom prior distributions, e.g. to encode domain knowledge'),
        )
        p.add_argument('--initial_kl_coeff', default=0.0001, type=float, help='Initial value of KL-penalty coefficient. This is adjusted during the training such that policy change stays close to target_kl')
        p.add_argument('--kl_coeff_large', default=0.0, type=float, help='Loss coefficient for the quadratic KL term')
        p.add_argument('--value_loss_coeff', default=0.5, type=float, help='Coefficient for the critic loss')

        # APPO-specific
        p.add_argument('--num_policies', default=8, type=int, help='Number of policies to train jointly')

        # EXPERIMENTAL: external memory
        p.add_argument('--mem_size', default=0, type=int, help='Number of external memory cells')
        p.add_argument('--mem_feature', default=64, type=int, help='Size of the memory cell (dimensionality)')
        p.add_argument('--obs_mem', default=False, type=str2bool, help='Observation-based memory')

    def __init__(self, cfg):
        super().__init__(cfg)

        self.workers = None

    def initialize(self):
        if not ray.is_initialized():
            ray.init(local_mode=True)

    def finalize(self):
        ray.shutdown()

    def learn(self):
        self.workers = [
            ActorWorker.remote(i, self.cfg) for i in range(self.cfg.num_workers)
        ]

        rollouts = [w.generate_rollout.remote() for w in self.workers]
        ready_rollouts = []

        while len(rollouts) > 0:
            ready_rollouts, rollouts = ray.wait(rollouts, num_returns=1, timeout=0.01)

        log.info('Ready rollouts: %d', len(ready_rollouts))
