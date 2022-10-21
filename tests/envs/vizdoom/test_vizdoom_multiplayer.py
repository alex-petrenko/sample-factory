import time
from multiprocessing import Process

import pytest

from sample_factory.algo.utils.context import reset_global_context
from sample_factory.algo.utils.rl_utils import make_dones
from sample_factory.envs.env_utils import vizdoom_available
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.utils import log


@pytest.mark.skipif(not vizdoom_available(), reason="Please install VizDoom to run a full test suite")
class TestDoom:
    @pytest.fixture(scope="class", autouse=True)
    def register_doom_fixture(self):
        from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components

        register_vizdoom_components()
        yield
        reset_global_context()

    @staticmethod
    def make_standard_dm(env_config, render_mode):
        from sf_examples.vizdoom.doom.doom_params import default_doom_cfg
        from sf_examples.vizdoom.doom.doom_utils import make_doom_env

        cfg = default_doom_cfg()
        cfg.env_frameskip = 2
        env = make_doom_env("doom_deathmatch_full", cfg=cfg, env_config=env_config, render_mode=render_mode)
        env.skip_frames = cfg.env_frameskip
        return env

    @staticmethod
    def doom_multiagent(make_multi_env, worker_index, num_steps=1000):
        env_config = AttrDict(worker_index=worker_index, vector_index=0, safe_init=False)
        visualize = False
        render_mode = "human" if visualize else None
        multi_env = make_multi_env(env_config, render_mode)
        obs, infos = multi_env.reset()

        start = time.time()

        for i in range(num_steps):
            actions = [multi_env.action_space.sample()] * len(obs)
            obs, rew, terminated, truncated, infos = multi_env.step(actions)
            dones = make_dones(terminated, truncated)

            if visualize:
                multi_env.render()

            if i % 100 == 0 or any(dones):
                log.info("Rew %r done %r info %r", rew, dones, infos)

            if all(dones):
                multi_env.reset()

        took = time.time() - start
        log.info("Took %.3f seconds for %d steps", took, num_steps)
        log.info("Server steps per second: %.1f", num_steps / took)
        log.info("Observations fps: %.1f", num_steps * multi_env.num_agents / took)
        log.info("Environment fps: %.1f", num_steps * multi_env.num_agents * multi_env.skip_frames / took)

        multi_env.close()

    def test_doom_multiagent(self):
        self.doom_multiagent(self.make_standard_dm, worker_index=0)

    def test_doom_multiagent_parallel(self):
        num_workers = 16
        workers = []

        for i in range(num_workers):
            log.info("Starting worker #%d", i)
            worker = Process(target=self.doom_multiagent, args=(self.make_standard_dm, i, 200))
            worker.start()
            workers.append(worker)
            time.sleep(0.01)

        for i in range(num_workers):
            workers[i].join()
