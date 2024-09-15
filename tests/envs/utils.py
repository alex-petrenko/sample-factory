import time

from sample_factory.envs.env_utils import num_env_steps
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log


def eval_env_performance(make_env, env_type, verbose=False, eval_frames=10_000):
    t = Timing()
    with t.timeit("init"):
        env = make_env(AttrDict({"worker_index": 0, "vector_index": 0}))
        total_num_frames, frames = eval_frames, 0
        num_agents = env.num_agents if hasattr(env, "num_agents") else 1
        is_multiagent = env.is_multiagent if hasattr(env, "is_multiagent") else num_agents > 1

    with t.timeit("first_reset"):
        env.reset()

    t.reset = t.step = 1e-9
    num_resets = 0
    with t.timeit("experience"):
        while frames < total_num_frames:
            done = False

            start_reset = time.time()
            obs, info = env.reset()

            t.reset += time.time() - start_reset
            num_resets += 1

            while not done and frames < total_num_frames:
                start_step = time.time()
                if verbose:
                    env.render()
                    time.sleep(1.0 / 40)

                if is_multiagent:
                    action_mask = [o.get("action_mask") if isinstance(o, dict) else None for o in obs]
                    action = [env.action_space.sample(m) for m in action_mask]
                else:
                    action_mask = obs.get("action_mask") if isinstance(obs, dict) else None
                    action = env.action_space.sample(action_mask)

                obs, rew, terminated, truncated, info = env.step(action)

                if is_multiagent:
                    done = all(a | b for a, b in zip(terminated, truncated))
                else:
                    done = terminated | truncated
                    info = [info]
                    if verbose:
                        log.info("Received reward %.3f", rew)

                t.step += time.time() - start_step
                frames += num_env_steps(info)

    fps = total_num_frames / t.experience
    log.debug("%s performance:", env_type)
    log.debug("Took %.3f sec to collect %d frames on one CPU, %.1f FPS", t.experience, total_num_frames, fps)
    log.debug("Avg. reset time %.3f s", t.reset / num_resets)
    log.debug("Timing: %s", t)
    env.close()
