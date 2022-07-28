import pytest

from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sf_examples.mujoco_examples.mujoco.mujoco_utils import mujoco_available
from sf_examples.mujoco_examples.train_mujoco import parse_mujoco_cfg, register_mujoco_components
from sf_examples.sampler.generate_trajectories import sample


class TestSampler:
    @pytest.mark.skipif(not mujoco_available(), reason="mujoco not installed")
    def test_sampler(self):
        # test on Mujoco because why not
        register_mujoco_components()
        cfg = parse_mujoco_cfg(
            argv=["--env=mujoco_halfcheetah", "--decorrelate_envs_on_one_worker=False", "--device=cpu"]
        )

        tmp_env = make_env_func_batched(cfg, env_config=None)
        env_info = extract_env_info(tmp_env, cfg)

        status = sample(cfg, env_info, sample_env_steps=20000)
        assert status == ExperimentStatus.SUCCESS
