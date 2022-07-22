import pytest

from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.train import run_rl
from sample_factory.utils.utils import log
from sf_examples.atari_examples.train_atari import parse_atari_args


class TestAtariEnv:
    @pytest.fixture(scope="class", autouse=True)
    def register_atari_fixture(self):
        from sf_examples.atari_examples.train_atari import register_atari_components

        register_atari_components()

    @staticmethod
    def _run_test_env(
        env: str = "atari_breakout",
        num_workers: int = 8,
        train_steps: int = 128,
        batched_sampling: bool = False,
        serial_mode: bool = True,
        async_rl: bool = False,
        batch_size: int = 64,
    ):
        log.debug(f"Testing with parameters {locals()}...")
        assert train_steps > batch_size, "We need sufficient number of steps to accumulate at least one batch"

        experiment_name = "test_" + env

        cfg = parse_atari_args(argv=["--algo=APPO", f"--env={env}", f"--experiment={experiment_name}"])
        cfg.serial_mode = serial_mode
        cfg.async_rl = async_rl
        cfg.batched_sampling = batched_sampling
        cfg.num_workers = num_workers
        cfg.num_envs_per_worker = 2
        cfg.train_for_env_steps = train_steps
        cfg.batch_size = batch_size
        cfg.seed = 0
        cfg.device = "cpu"

        status = run_rl(cfg)
        assert status == ExperimentStatus.SUCCESS

    @pytest.mark.parametrize(
        "env_name",
        [
            "atari_montezuma",
            "atari_pong",
            "atari_breakout",
            # probably no reason to test on all of them, as they are kind of the same
            # "atari_qbert",
            # "atari_spaceinvaders",
            # "atari_asteroids",
            # "atari_gravitar",
            # "atari_mspacman",
            # "atari_seaquest",
        ],
    )
    @pytest.mark.parametrize("batched_sampling", [False, True])
    def test_basic_envs(self, env_name, batched_sampling):
        self._run_test_env(env=env_name, num_workers=1, batched_sampling=batched_sampling)
