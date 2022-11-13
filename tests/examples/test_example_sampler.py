import pytest

from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sf_examples.mujoco.mujoco_utils import mujoco_available
from sf_examples.mujoco.train_mujoco import parse_mujoco_cfg, register_mujoco_components
from sf_examples.sampler.use_simplified_sampling_api import generate_trajectories


class TestSampler:
    @pytest.mark.skipif(not mujoco_available(), reason="mujoco not installed")
    @pytest.mark.parametrize("batched_sampling", [False, True])
    def test_sampler(self, batched_sampling: bool):
        # test on Mujoco because why not
        register_mujoco_components()
        cfg = parse_mujoco_cfg(
            argv=["--env=mujoco_halfcheetah", "--decorrelate_envs_on_one_worker=False", "--device=cpu"]
        )
        cfg.batched_sampling = batched_sampling

        tmp_env = make_env_func_batched(cfg, env_config=None)
        env_info = extract_env_info(tmp_env, cfg)

        status = generate_trajectories(cfg, env_info, sample_env_steps=20000)
        assert status == ExperimentStatus.SUCCESS
