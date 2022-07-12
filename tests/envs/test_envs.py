import pytest

from sample_factory.cfg.arguments import default_cfg
from sample_factory.envs.dmlab.dmlab_utils import string_to_hash_bucket
from sample_factory.envs.env_utils import dmlab_available
from tests.envs.utils import eval_env_performance


class TestAtari:
    @pytest.fixture(scope="class", autouse=True)
    def register_atari_fixture(self):
        from sf_examples.atari_examples.train_atari import register_atari_components

        register_atari_components()

    # noinspection PyUnusedLocal
    @staticmethod
    def make_env(env_config):
        from sf_examples.atari_examples.atari_utils import make_atari_env

        return make_atari_env("atari_breakout", cfg=default_cfg(env="atari_breakout"), env_config=None)

    def test_atari_performance(self):
        eval_env_performance(self.make_env, "atari")


class TestDmlab:
    """DMLab tests fail too often just randomly (EGL errors), so we're skipping them for now."""

    # noinspection PyUnusedLocal
    @staticmethod
    def make_env(env_config):
        from sample_factory.envs.dmlab.dmlab_env import make_dmlab_env

        return make_dmlab_env("dmlab_nonmatch", cfg=default_cfg(env="dmlab_nonmatch"), env_config=None)

    @pytest.mark.skipif(not dmlab_available(), reason="Dmlab package not installed")
    def test_dmlab_performance(self):
        eval_env_performance(self.make_env, "dmlab")

    def test_hash_bucket(self):
        vocab_size = 42
        data = {
            "a cupful of liquid that was almost, but not quite, entirely unlike tea": 37,
            "aaabbbccc": 28,
            "12313123132": 28,
            "RL": 24,
            "dmlab": 35,
        }
        for s, h in data.items():
            assert string_to_hash_bucket(s, vocab_size) == h
