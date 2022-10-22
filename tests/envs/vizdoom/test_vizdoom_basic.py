import os

import pytest

from sample_factory.algo.utils.context import reset_global_context
from sample_factory.envs.env_utils import vizdoom_available
from tests.envs.utils import eval_env_performance


@pytest.mark.skipif(not vizdoom_available(), reason="Please install VizDoom to run a full test suite")
class TestDoom:
    @pytest.fixture(scope="class", autouse=True)
    def register_doom_fixture(self):
        from sf_examples.vizdoom.train_vizdoom import register_vizdoom_components

        register_vizdoom_components()
        yield  # this is where the actual test happens
        reset_global_context()

    # noinspection PyUnusedLocal
    @staticmethod
    def make_env_singleplayer(env_config, **kwargs):
        from sf_examples.vizdoom.doom.doom_params import default_doom_cfg
        from sf_examples.vizdoom.doom.doom_utils import make_doom_env

        return make_doom_env("doom_benchmark", cfg=default_doom_cfg(), env_config=env_config, **kwargs)

    @staticmethod
    def make_env_bots_hybrid_actions(env_config, **kwargs):
        from sf_examples.vizdoom.doom.doom_params import default_doom_cfg
        from sf_examples.vizdoom.doom.doom_utils import make_doom_env

        return make_doom_env("doom_deathmatch_bots", cfg=default_doom_cfg(), env_config=env_config, **kwargs)

    def test_doom_env(self):
        assert self.make_env_singleplayer(None) is not None

    def test_doom_performance(self):
        eval_env_performance(self.make_env_singleplayer, "doom")

    def test_doom_performance_bots_hybrid_actions(self):
        eval_env_performance(self.make_env_bots_hybrid_actions, "doom")

    def test_doom_two_color(self):
        from sf_examples.vizdoom.doom.doom_params import default_doom_cfg
        from sf_examples.vizdoom.doom.doom_utils import make_doom_env

        eval_env_performance(
            lambda env_config: make_doom_env("doom_two_colors_easy", cfg=default_doom_cfg(), env_config=None),
            "doom",
            verbose=False,
        )

    def test_recording(self):
        from sf_examples.vizdoom.doom.doom_params import default_doom_cfg
        from sf_examples.vizdoom.doom.doom_utils import make_doom_env

        rec_dir = "/tmp/"
        cfg = default_doom_cfg()
        cfg.record_to = rec_dir
        env = make_doom_env("doom_benchmark", cfg=cfg, env_config=None)
        env.reset()
        for i in range(20):
            env.step(env.action_space.sample())

        env.reset()
        env.close()

        demo_dir = env.unwrapped.curr_demo_dir
        from sf_examples.vizdoom.doom.doom_gym import VizdoomEnv

        demo_path = VizdoomEnv.demo_path(0, demo_dir)

        env = make_doom_env("doom_benchmark", cfg=cfg, env_config=None, custom_resolution="1920x1080")

        VizdoomEnv.replay(env, demo_path)

        assert os.path.isfile(demo_path)
        os.remove(demo_path)
        assert not os.path.isfile(demo_path)
