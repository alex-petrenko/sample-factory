"""
Evaluate different policies against one another in a multiplayer VizDoom match.

This is an old/unsupported script, update for SF2 API would be needed.
"""

# import os
# import sys
# import time
# from multiprocessing import Process
# from os.path import join
#
# import numpy as np
# import torch
#
# from sample_factory.algo.utils.make_env import NonBatchedMultiAgentWrapper, is_multiagent_env
# from sample_factory.cfg.arguments import load_from_checkpoint
# from sample_factory.envs.create_env import create_env
# from sample_factory.utils.utils import AttrDict, log
#
#
# class Rival:
#     def __init__(self, name, args):
#         self.name = name
#         self.args = args
#         self.policy_index = None
#         self.parsed_config = parse_args(args, evaluation=True)
#         self.cfg = load_from_checkpoint(self.parsed_config)
#
#         self.actor_critic = None
#         self.rnn_state = None
#
#
# RIVALS = [
#     Rival(
#         name="duel",
#         args=[
#             "--env=doom_duel",
#             "--algo=APPO",
#             "--experiment=00_bots_ssl2_fs2_ppo_1",
#             "--experiments_root=paper_doom_duel_v65_fs2/bots_ssl2_fs2",
#         ],
#     ),
#     Rival(
#         name="duel_bots",
#         args=[
#             "--env=doom_duel_bots",
#             "--algo=APPO",
#             "--experiment=00_bots_ssl2_fs2_ppo_1",
#             "--experiments_root=paper_doom_duel_bots_v65_fs2/bots_ssl2_fs2",
#         ],
#     ),
# ]
#
# ENV_NAME = "doom_duel"
# NO_RENDER = True
# FPS = 10000
#
#
# def multi_agent_match(policy_indices, max_num_episodes=int(1e9), max_num_frames=1e10):
#     log.debug("Starting eval process with policies %r", policy_indices)
#     for i, rival in enumerate(RIVALS):
#         rival.policy_index = policy_indices[i]
#
#     curr_dir = os.path.dirname(os.path.abspath(__file__))
#     evaluation_filename = join(curr_dir, f'eval_{"vs".join([str(pi) for pi in policy_indices])}.txt')
#     with open(evaluation_filename, "w") as fobj:
#         fobj.write("start\n")
#
#     common_config = RIVALS[0].cfg
#
#     render_action_repeat = (
#         common_config.render_action_repeat
#         if common_config.render_action_repeat is not None
#         else common_config.env_frameskip
#     )
#     if render_action_repeat is None:
#         log.warning("Not using action repeat!")
#         render_action_repeat = 1
#     log.debug("Using action repeat %d during evaluation", render_action_repeat)
#
#     common_config.env_frameskip = 1  # for evaluation
#     common_config.num_envs = 1
#     common_config.timelimit = 4.0  # for faster evaluation
#
#     def make_env_func(env_config):
#         return create_env(ENV_NAME, cfg=common_config, env_config=env_config)
#
#     env = make_env_func(AttrDict({"worker_index": 0, "vector_index": 0}))
#     env.seed(0)
#
#     is_multiagent = is_multiagent_env(env)
#     if not is_multiagent:
#         env = NonBatchedMultiAgentWrapper(env)
#     else:
#         assert env.num_agents == len(RIVALS)
#
#     device = torch.device("cuda")
#     for rival in RIVALS:
#         rival.actor_critic = create_actor_critic(rival.cfg, env.observation_space, env.action_space)
#         rival.actor_critic.model_to_device(device)
#
#         policy_id = rival.policy_index
#         checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(rival.cfg, policy_id))
#         checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
#         rival.actor_critic.load_state_dict(checkpoint_dict["model"])
#
#     episode_rewards = []
#     num_frames = 0
#
#     last_render_start = time.time()
#
#     def max_frames_reached(frames):
#         return max_num_frames is not None and frames > max_num_frames
#
#     wins = [0 for _ in RIVALS]
#     ties = 0
#     frag_differences = []
#
#     with torch.no_grad():
#         for _ in range(max_num_episodes):
#             obs = env.reset()
#             obs_dict_torch = dict()
#
#             done = [False] * len(obs)
#             for rival in RIVALS:
#                 rival.rnn_states = torch.zeros([1, rival.cfg.rnn_size], dtype=torch.float32, device=device)
#
#             episode_reward = 0
#             prev_frame = time.time()
#
#             while True:
#                 actions = []
#                 for i, obs_dict in enumerate(obs):
#                     for key, x in obs_dict.items():
#                         obs_dict_torch[key] = torch.from_numpy(x).to(device).float().view(1, *x.shape)
#
#                     rival = RIVALS[i]
#                     policy_outputs = rival.actor_critic(obs_dict_torch, rival.rnn_states)
#                     rival.rnn_states = policy_outputs.rnn_states
#                     actions.append(policy_outputs.actions[0].cpu().numpy())
#
#                 for _ in range(render_action_repeat):
#                     if not NO_RENDER:
#                         target_delay = 1.0 / FPS if FPS > 0 else 0
#                         current_delay = time.time() - last_render_start
#                         time_wait = target_delay - current_delay
#
#                         if time_wait > 0:
#                             # log.info('Wait time %.3f', time_wait)
#                             time.sleep(time_wait)
#
#                         last_render_start = time.time()
#                         env.render()
#
#                     obs, rew, done, infos = env.step(actions)
#                     if all(done):
#                         log.debug("Finished episode!")
#
#                         frag_diff = infos[0]["PLAYER1_FRAGCOUNT"] - infos[0]["PLAYER2_FRAGCOUNT"]
#                         if frag_diff > 0:
#                             wins[0] += 1
#                         elif frag_diff < 0:
#                             wins[1] += 1
#                         else:
#                             ties += 1
#
#                         frag_differences.append(frag_diff)
#                         avg_frag_diff = np.mean(frag_differences)
#
#                         report = f"wins: {wins}, ties: {ties}, avg_frag_diff: {avg_frag_diff}"
#                         with open(evaluation_filename, "a") as fobj:
#                             fobj.write(report + "\n")
#
#                     # log.info('%d:%d', infos[0]['PLAYER1_FRAGCOUNT'], infos[0]['PLAYER2_FRAGCOUNT'])
#
#                     episode_reward += np.mean(rew)
#                     num_frames += 1
#
#                     if num_frames % 100 == 0:
#                         log.debug("%.1f", render_action_repeat / (time.time() - prev_frame))
#                     prev_frame = time.time()
#
#                     if all(done):
#                         log.info("Episode finished at %d frames", num_frames)
#                         break
#
#                 if all(done) or max_frames_reached(num_frames):
#                     break
#
#             if not NO_RENDER:
#                 env.render()
#             time.sleep(0.01)
#
#             episode_rewards.append(episode_reward)
#             last_episodes = episode_rewards[-100:]
#             avg_reward = sum(last_episodes) / len(last_episodes)
#             log.info(
#                 "Episode reward: %f, avg reward for %d episodes: %f",
#                 episode_reward,
#                 len(last_episodes),
#                 avg_reward,
#             )
#
#             if max_frames_reached(num_frames):
#                 break
#
#     env.close()
#
#
# def main():
#     """Script entry point."""
#     multi_process = True
#     if multi_process:
#         num_policies = 8
#         processes = []
#         for p_id in range(num_policies):
#             process = Process(target=multi_agent_match, args=[(p_id, 0)])
#             process.start()
#             processes.append(process)
#             time.sleep(5.0)
#
#         for process in processes:
#             process.join()
#     else:
#         multi_agent_match((0, 0))
#
#     return 0
#
#
# if __name__ == "__main__":
#     sys.exit(main())
