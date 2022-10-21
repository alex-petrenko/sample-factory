import pytest

from sample_factory.algo.utils.context import reset_global_context
from sf_examples.dmlab.dmlab_utils import dmlab_available, string_to_hash_bucket
from tests.envs.utils import eval_env_performance


class TestDmlab:
    @staticmethod
    def make_env(_env_config):
        from sf_examples.dmlab.dmlab_env import make_dmlab_env
        from sf_examples.dmlab.train_dmlab import parse_dmlab_args

        cfg = parse_dmlab_args(argv=["--algo=APPO", "--env=dmlab_nonmatch", "--experiment=test_dmlab"])
        return make_dmlab_env("dmlab_nonmatch", cfg=cfg, env_config=None)

    @pytest.mark.skipif(not dmlab_available(), reason="Dmlab package not installed")
    def test_dmlab_performance(self):
        from sf_examples.dmlab.train_dmlab import register_dmlab_components

        register_dmlab_components()
        eval_env_performance(self.make_env, "dmlab", eval_frames=1000)
        reset_global_context()

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
