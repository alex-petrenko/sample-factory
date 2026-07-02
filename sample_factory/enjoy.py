import time
from collections import deque
from typing import Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import BatchedVecEnv, make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.huggingface.huggingface_utils import generate_model_card, generate_replay_video, push_to_hf
from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import debug_log_every_n, experiment_dir, log


def visualize_policy_inputs(normalized_obs: Dict[str, Tensor]) -> None:
    """
    Display actual policy inputs after all wrappers and normalizations using OpenCV imshow.
    """
    import cv2

    if "obs" not in normalized_obs.keys():
        return

    obs = normalized_obs["obs"]
    # visualize obs only for the 1st agent
    obs = obs[0]
    if obs.dim() != 3:
        # this function is only for RGB images
        return

    # convert to HWC
    obs = obs.permute(1, 2, 0)
    # convert to numpy
    obs = obs.cpu().numpy()
    # convert to uint8
    obs = cv2.normalize(
        obs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1
    )  # this will be different frame-by-frame but probably good enough to give us an idea?
    scale = 5
    obs = cv2.resize(obs, (obs.shape[1] * scale, obs.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("policy inputs", obs)
    cv2.waitKey(delay=1)


def render_frame(cfg, env, video_frames, num_episodes, last_render_start) -> float:
    render_start = time.time()

    if cfg.save_video:
        need_video_frame = len(video_frames) < cfg.video_frames or cfg.video_frames < 0 and num_episodes == 0
        if need_video_frame:
            frame = env.render()
            if frame is not None:
                video_frames.append(frame.copy())
    else:
        if not cfg.no_render:
            target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
            current_delay = render_start - last_render_start
            time_wait = target_delay - current_delay

            if time_wait > 0:
                # log.info("Wait time %.3f", time_wait)
                time.sleep(time_wait)

            try:
                env.render()
            except (gym.error.Error, TypeError) as ex:
                debug_log_every_n(1000, f"Exception when calling env.render() {str(ex)}")

    return render_start


def make_env(cfg: Config, render_mode: Optional[str] = None) -> BatchedVecEnv:
    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
    )
    return env


def load_state_dict(cfg: Config, actor_critic: ActorCritic, device: torch.device) -> None:
    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    if checkpoint_dict:
        actor_critic.load_state_dict(checkpoint_dict["model"])
    else:
        raise RuntimeError("Could not load checkpoint")


def _prepare_eval_config(cfg: Config):
    cfg = load_from_checkpoint(cfg)

    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

    cfg.num_envs = 1
    render_mode = "human"
    if cfg.save_video:
        render_mode = "rgb_array"
    elif cfg.no_render:
        render_mode = None

    return cfg, render_action_repeat, render_mode


def _init_eval_actor_critic(cfg: Config, env: BatchedVecEnv) -> Tuple[ActorCritic, torch.device]:
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)
    load_state_dict(cfg, actor_critic, device)
    return actor_critic, device


def _init_enjoy_state(cfg: Config, env: BatchedVecEnv, device: torch.device) -> AttrDict:
    obs, infos = env.reset()
    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None

    return AttrDict(
        obs=obs,
        infos=infos,
        action_mask=action_mask,
        rnn_states=torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device),
        episode_rewards=[deque([], maxlen=100) for _ in range(env.num_agents)],
        true_objectives=[deque([], maxlen=100) for _ in range(env.num_agents)],
        episode_reward=None,
        finished_episode=[False for _ in range(env.num_agents)],
        video_frames=[],
        reward_list=[],
        num_frames=0,
        num_episodes=0,
        last_render_start=time.time(),
    )


def _max_frames_reached(cfg: Config, frames: int) -> bool:
    return cfg.max_num_frames is not None and frames > cfg.max_num_frames


def _select_actions(cfg: Config, env_info, actor_critic: ActorCritic, state: AttrDict) -> Tensor:
    normalized_obs = prepare_and_normalize_obs(actor_critic, state.obs)

    if not cfg.no_render:
        visualize_policy_inputs(normalized_obs)
    policy_outputs = actor_critic(normalized_obs, state.rnn_states, action_mask=state.action_mask)

    actions = policy_outputs["actions"]
    if cfg.eval_deterministic:
        action_distribution = actor_critic.action_distribution()
        actions = argmax_actions(action_distribution)

    if actions.ndim == 1:
        actions = unsqueeze_tensor(actions, dim=-1)

    state.rnn_states = policy_outputs["new_rnn_states"]
    return preprocess_actions(env_info, actions)


def _update_episode_reward(state: AttrDict, rew: Tensor) -> None:
    if state.episode_reward is None:
        state.episode_reward = rew.float().clone()
    else:
        state.episode_reward += rew.float()


def _record_done_agent(cfg: Config, env: BatchedVecEnv, state: AttrDict, infos, agent_i: int, verbose: bool) -> None:
    state.finished_episode[agent_i] = True
    rew = state.episode_reward[agent_i].item()
    state.episode_rewards[agent_i].append(rew)

    true_objective = rew
    if isinstance(infos, (list, tuple)):
        true_objective = infos[agent_i].get("true_objective", rew)
    state.true_objectives[agent_i].append(true_objective)

    if verbose:
        log.info(
            "Episode finished for agent %d at %d frames. Reward: %.3f, true_objective: %.3f",
            agent_i,
            state.num_frames,
            state.episode_reward[agent_i],
            state.true_objectives[agent_i][-1],
        )

    device = state.rnn_states.device
    state.rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
    state.episode_reward[agent_i] = 0

    if cfg.use_record_episode_statistics:
        if "episode" in infos[agent_i].keys():
            state.num_episodes += 1
            state.reward_list.append(infos[agent_i]["episode"]["r"])
    else:
        state.num_episodes += 1
        state.reward_list.append(true_objective)


def _log_completed_episodes(env: BatchedVecEnv, state: AttrDict) -> None:
    state.finished_episode = [False] * env.num_agents
    avg_episode_rewards, avg_true_objectives = [], []

    for agent_i in range(env.num_agents):
        avg_rew = np.mean(state.episode_rewards[agent_i])
        avg_true_obj = np.mean(state.true_objectives[agent_i])

        if not np.isnan(avg_rew):
            avg_episode_rewards.append(f"#{agent_i}: {avg_rew:.3f}")
        if not np.isnan(avg_true_obj):
            avg_true_objectives.append(f"#{agent_i}: {avg_true_obj:.3f}")

    log.info(
        "Avg episode rewards: %s, true rewards: %s",
        ", ".join(avg_episode_rewards),
        ", ".join(avg_true_objectives),
    )
    log.info(
        "Avg episode reward: %.3f, avg true_objective: %.3f",
        np.mean([np.mean(state.episode_rewards[i]) for i in range(env.num_agents)]),
        np.mean([np.mean(state.true_objectives[i]) for i in range(env.num_agents)]),
    )


def _advance_env_once(
    cfg: Config, env: BatchedVecEnv, env_info, device: torch.device, state: AttrDict, actions
) -> None:
    state.last_render_start = render_frame(cfg, env, state.video_frames, state.num_episodes, state.last_render_start)

    state.obs, rew, terminated, truncated, infos = env.step(actions)
    state.action_mask = state.obs.pop("action_mask").to(device) if "action_mask" in state.obs else None
    dones = make_dones(terminated, truncated)
    infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos
    state.infos = infos

    _update_episode_reward(state, rew)

    state.num_frames += 1
    if state.num_frames % 100 == 0:
        log.debug(f"Num frames {state.num_frames}...")

    state.dones = dones.cpu().numpy()


def _handle_env_step_end(cfg: Config, env: BatchedVecEnv, state: AttrDict, verbose: bool) -> None:
    for agent_i, done_flag in enumerate(state.dones):
        if done_flag:
            _record_done_agent(cfg, env, state, state.infos, agent_i, verbose)

    if all(state.dones):
        render_frame(cfg, env, state.video_frames, state.num_episodes, state.last_render_start)
        time.sleep(0.05)

    if all(state.finished_episode):
        _log_completed_episodes(env, state)


def _generate_enjoy_outputs(cfg: Config, state: AttrDict) -> None:
    if cfg.save_video:
        fps = cfg.fps if cfg.fps > 0 else 30
        generate_replay_video(experiment_dir(cfg=cfg), state.video_frames, fps, cfg)

    if cfg.push_to_hub:
        generate_model_card(
            experiment_dir(cfg=cfg),
            cfg.algo,
            cfg.env,
            cfg.hf_repository,
            state.reward_list,
            cfg.enjoy_script,
            cfg.train_script,
        )
        push_to_hf(experiment_dir(cfg=cfg), cfg.hf_repository)


def _mean_episode_reward(env: BatchedVecEnv, state: AttrDict) -> float:
    return sum([sum(state.episode_rewards[i]) for i in range(env.num_agents)]) / sum(
        [len(state.episode_rewards[i]) for i in range(env.num_agents)]
    )


def enjoy(cfg: Config) -> Tuple[StatusCode, float]:
    verbose = False
    cfg, render_action_repeat, render_mode = _prepare_eval_config(cfg)

    env = make_env(cfg, render_mode=render_mode)
    env_info = extract_env_info(env, cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic, device = _init_eval_actor_critic(cfg, env)
    state = _init_enjoy_state(cfg, env, device)

    with torch.no_grad():
        while not _max_frames_reached(cfg, state.num_frames):
            actions = _select_actions(cfg, env_info, actor_critic, state)
            for _ in range(render_action_repeat):
                _advance_env_once(cfg, env, env_info, device, state, actions)
                _handle_env_step_end(cfg, env, state, verbose)

                # VizDoom multiplayer stuff
                # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
                #     key = f'PLAYER{player}_FRAGCOUNT'
                #     if key in infos[0]:
                #         log.debug('Score for player %d: %r', player, infos[0][key])

            if state.num_episodes >= cfg.max_num_episodes:
                break

    env.close()
    _generate_enjoy_outputs(cfg, state)
    return ExperimentStatus.SUCCESS, _mean_episode_reward(env, state)
