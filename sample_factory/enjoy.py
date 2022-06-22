import sys
import time
from collections import deque

import numpy as np
import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.model.model import create_actor_critic
from sample_factory.model.model_utils import get_hidden_size
from sample_factory.algo.utils.action_distributions import ContinuousActionDistribution
from sample_factory.cfg.arguments import parse_args, load_from_checkpoint
from sample_factory.algo.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.utils.utils import log, AttrDict


def enjoy(cfg, max_num_frames=1e9):
    cfg = load_from_checkpoint(cfg)

    render_action_repeat = cfg.render_action_repeat if cfg.render_action_repeat is not None else cfg.env_frameskip
    if render_action_repeat is None:
        log.warning('Not using action repeat!')
        render_action_repeat = 1
    log.debug('Using action repeat %d during evaluation', render_action_repeat)

    cfg.env_frameskip = 1  # for evaluation
    cfg.num_envs = 1

    env = make_env_func_batched(cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0))
    # env.seed(0)  # TODO: make a parameter for this?
    env_info = extract_env_info(env, cfg)

    # TODO: use the same code as in the sampler
    is_multiagent = is_multiagent_env(env)
    if not is_multiagent:
        env = MultiAgentWrapper(env)

    if hasattr(env.unwrapped, 'reset_on_init'):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)

    # TODO: move this to a separate IO module
    policy_id = cfg.policy_index
    name_prefix = dict(latest='checkpoint', best='best')[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f'{name_prefix}_*')
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    obs = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_hidden_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None
    finished_episode = [False] * env.num_agents

    with torch.inference_mode():
        while not max_frames_reached(num_frames):
            normalized_obs = actor_critic.normalizer(obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)

            # sample actions from the distribution by default
            actions = policy_outputs['actions']

            action_distribution = actor_critic.action_distribution()
            if isinstance(action_distribution, ContinuousActionDistribution):
                if not cfg.continuous_actions_sample:  # TODO: add similar option for discrete actions
                    actions = action_distribution.means

            actions = preprocess_actions(env_info, actions)  # TODO: move this to some utils module

            rnn_states = policy_outputs['new_rnn_states']

            for _ in range(render_action_repeat):
                if not cfg.no_render:
                    target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
                    current_delay = time.time() - last_render_start
                    time_wait = target_delay - current_delay

                    if time_wait > 0:
                        # log.info('Wait time %.3f', time_wait)
                        time.sleep(time_wait)

                    last_render_start = time.time()
                    env.render()

                obs, rew, dones, infos = env.step(actions)
                infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

                if episode_reward is None:
                    episode_reward = rew.clone()
                else:
                    episode_reward += rew

                num_frames += 1

                dones = dones.cpu().numpy()
                for agent_i, done_flag in enumerate(dones):
                    if done_flag:
                        finished_episode[agent_i] = True
                        rew = episode_reward[agent_i].item()
                        episode_rewards[agent_i].append(rew)

                        true_objective = rew
                        if isinstance(infos, (list, tuple)):
                            true_objective = infos[agent_i].get('true_objective', rew)
                        true_objectives[agent_i].append(true_objective)

                        log.info('Episode finished for agent %d at %d frames. Reward: %.3f, true_objective: %.3f', agent_i, num_frames, episode_reward[agent_i], true_objectives[agent_i][-1])
                        rnn_states[agent_i] = torch.zeros([get_hidden_size(cfg)], dtype=torch.float32, device=device)
                        episode_reward[agent_i] = 0

                # if episode terminated synchronously for all agents, pause a bit before starting a new one
                if all(dones):
                    if not cfg.no_render:
                        env.render()
                    time.sleep(0.05)

                if all(finished_episode):
                    finished_episode = [False] * env.num_agents
                    avg_episode_rewards_str, avg_true_objective_str = '', ''
                    for agent_i in range(env.num_agents):
                        avg_rew = np.mean(episode_rewards[agent_i])
                        avg_true_obj = np.mean(true_objectives[agent_i])

                        if not np.isnan(avg_rew):
                            if avg_episode_rewards_str:
                                avg_episode_rewards_str += ', '
                            avg_episode_rewards_str += f'#{agent_i}: {avg_rew:.3f}'
                        if not np.isnan(avg_true_obj):
                            if avg_true_objective_str:
                                avg_true_objective_str += ', '
                            avg_true_objective_str += f'#{agent_i}: {avg_true_obj:.3f}'

                    log.info('Avg episode rewards: %s, true rewards: %s', avg_episode_rewards_str, avg_true_objective_str)
                    log.info('Avg episode reward: %.3f, avg true_objective: %.3f', np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]), np.mean([np.mean(true_objectives[i]) for i in range(env.num_agents)]))

                # VizDoom multiplayer stuff
                # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
                #     key = f'PLAYER{player}_FRAGCOUNT'
                #     if key in infos[0]:
                #         log.debug('Score for player %d: %r', player, infos[0][key])

    env.close()

    return ExperimentStatus.SUCCESS, np.mean(episode_rewards)


def main():
    """Script entry point."""
    cfg = parse_args(evaluation=True)
    status, avg_reward = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
