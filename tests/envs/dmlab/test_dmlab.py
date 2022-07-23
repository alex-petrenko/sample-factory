import pytest

from sf_examples.dmlab_examples.dmlab_utils import dmlab_available, string_to_hash_bucket
from tests.envs.utils import eval_env_performance


class TestDmlab:
    """DMLab tests fail too often just randomly (EGL errors), so we're skipping them for now."""

    @pytest.mark.skipif(not dmlab_available(), reason="Dmlab package not installed")
    @pytest.fixture(scope="class", autouse=True)
    def register_dmlab_fixture(self):
        from sf_examples.dmlab_examples.train_dmlab import register_dmlab_components

        register_dmlab_components()

    @staticmethod
    def make_env(_env_config):
        from sf_examples.dmlab_examples.dmlab_env import make_dmlab_env
        from sf_examples.dmlab_examples.train_dmlab import parse_dmlab_args

        cfg = parse_dmlab_args(argv=["--algo=APPO", "--env=dmlab_nonmatch", "--experiment=test_dmlab"])
        return make_dmlab_env("dmlab_nonmatch", cfg=cfg, env_config=None)

    @pytest.mark.skipif(not dmlab_available(), reason="Dmlab package not installed")
    def test_dmlab_performance(self):
        eval_env_performance(self.make_env, "dmlab", eval_frames=1000)

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
