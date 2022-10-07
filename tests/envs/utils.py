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

    with t.timeit("first_reset"):
        env.reset()

    t.reset = t.step = 1e-9
    num_resets = 0
    with t.timeit("experience"):
        while frames < total_num_frames:
            done = False

            start_reset = time.time()
            env.reset()

            t.reset += time.time() - start_reset
            num_resets += 1

            while not done and frames < total_num_frames:
                start_step = time.time()
                if verbose:
                    env.render()
                    time.sleep(1.0 / 40)

                obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
                done = terminated | truncated
                if verbose:
                    log.info("Received reward %.3f", rew)

                t.step += time.time() - start_step
                frames += num_env_steps([info])

    fps = total_num_frames / t.experience
    log.debug("%s performance:", env_type)
    log.debug("Took %.3f sec to collect %d frames on one CPU, %.1f FPS", t.experience, total_num_frames, fps)
    log.debug("Avg. reset time %.3f s", t.reset / num_resets)
    log.debug("Timing: %s", t)
    env.close()
