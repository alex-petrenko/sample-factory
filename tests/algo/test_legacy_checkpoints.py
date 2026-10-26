"""Checkpoints saved before the 2022 model refactors should still load (issue #310).

Two renames changed the parameter names stored in checkpoints:
  * 521b7edc ("V75"): ConvEncoderImpl.fc_layers -> mlp_layers
  * #198 (MultiInputEncoder): encoder.<x> -> encoder.encoders.obs.<x>
Pretrained models linked from the docs (e.g. the VizDoom and MuJoCo ones on the HF Hub) still use the old names.
"""

from os.path import join

import gymnasium as gym
import numpy as np
import pytest
import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.enjoy import load_state_dict
from sample_factory.model.actor_critic import create_actor_critic
from sf_examples.mujoco.mujoco_utils import mujoco_available
from sf_examples.mujoco.train_mujoco import parse_mujoco_cfg, register_mujoco_components

IMAGE_OBS = gym.spaces.Box(0, 255, (3, 64, 64), np.uint8)
VECTOR_OBS = gym.spaces.Box(-1.0, 1.0, (8,), np.float32)


def _make_eval_cfg(tmp_path):
    argv = ["--algo=APPO", "--env=dummy", "--experiment=legacy_checkpoint", f"--train_dir={tmp_path}"]
    parser, _ = parse_sf_args(argv, evaluation=True)
    return parse_full_cfg(parser, argv)


def _make_model(cfg, obs_space, seed):
    torch.manual_seed(seed)
    return create_actor_critic(cfg, gym.spaces.Dict(obs=obs_space), gym.spaces.Discrete(4))


def _save_checkpoint(cfg, model_state_dict):
    checkpoint_path = join(Learner.checkpoint_dir(cfg, policy_id=0), "checkpoint_000000001_1.pth")
    torch.save(dict(model=model_state_dict), checkpoint_path)


def _pre_v75_key(key):
    """Name a conv-encoder parameter had before 521b7edc."""
    return key.replace(".enc.mlp_layers.", ".enc.fc_layers.")


def _pre_multi_input_key(key):
    """Name an encoder parameter had before MultiInputEncoder (#198)."""
    return key.replace("encoder.encoders.obs.", "encoder.", 1)


def _assert_same_weights(model, reference):
    expected = reference.state_dict()
    actual = model.state_dict()
    assert actual.keys() == expected.keys()
    for key, tensor in expected.items():
        assert torch.equal(actual[key], tensor), f"{key} was not loaded from the checkpoint"


@pytest.mark.parametrize(
    "obs_space, legacy_renames",
    [
        pytest.param(IMAGE_OBS, [_pre_v75_key], id="conv_fc_layers"),  # e.g. sample-factory-2-doom-battle
        pytest.param(VECTOR_OBS, [_pre_multi_input_key], id="mlp_single_encoder"),  # e.g. sample-factory-2-mujoco-ant
        pytest.param(IMAGE_OBS, [_pre_multi_input_key, _pre_v75_key], id="conv_both_renames"),
    ],
)
def test_legacy_checkpoint_loads(tmp_path, obs_space, legacy_renames):
    cfg = _make_eval_cfg(tmp_path)
    trained = _make_model(cfg, obs_space, seed=0)

    legacy_state_dict = {}
    for key, tensor in trained.state_dict().items():
        for rename in legacy_renames:
            key = rename(key)
        legacy_state_dict[key] = tensor
    assert legacy_state_dict.keys() != trained.state_dict().keys(), "test setup should produce legacy key names"
    _save_checkpoint(cfg, legacy_state_dict)

    fresh = _make_model(cfg, obs_space, seed=1)  # different initial weights, so we can tell they were loaded
    load_state_dict(cfg, fresh, torch.device("cpu"))

    _assert_same_weights(fresh, trained)


@pytest.mark.parametrize("obs_space", [IMAGE_OBS, VECTOR_OBS], ids=["conv", "mlp"])
def test_current_checkpoint_still_loads(tmp_path, obs_space):
    cfg = _make_eval_cfg(tmp_path)
    trained = _make_model(cfg, obs_space, seed=0)
    _save_checkpoint(cfg, trained.state_dict())

    fresh = _make_model(cfg, obs_space, seed=1)
    load_state_dict(cfg, fresh, torch.device("cpu"))

    _assert_same_weights(fresh, trained)


@pytest.mark.skipif(not mujoco_available(), reason="mujoco not installed or not available on this machine")
def test_learner_resumes_training_from_legacy_checkpoint(tmp_path):
    register_mujoco_components()
    cfg = parse_mujoco_cfg(argv=["--env=mujoco_ant", "--experiment=legacy_resume", f"--train_dir={tmp_path}"])
    cfg.device = "cpu"
    cfg.serial_mode = True
    env_info = extract_env_info(make_env_func_batched(cfg, env_config=None), cfg)

    def init_learner(seed):
        torch.manual_seed(seed)
        policy_versions = torch.zeros([cfg.num_policies], dtype=torch.int32)
        param_server = ParameterServer(0, policy_versions, cfg.serial_mode)
        learner = Learner(cfg, env_info, policy_versions, 0, param_server)
        learner.init()  # loads the latest checkpoint in the experiment dir, if there is one
        return learner

    trained = init_learner(seed=0)
    checkpoint = trained._get_checkpoint_dict()
    checkpoint["model"] = {_pre_multi_input_key(key): value for key, value in checkpoint["model"].items()}
    checkpoint["train_step"] = 123
    torch.save(checkpoint, join(Learner.checkpoint_dir(cfg, policy_id=0), "checkpoint_000000123_0.pth"))

    resumed = init_learner(seed=1)

    assert resumed.train_step == 123
    _assert_same_weights(resumed.actor_critic, trained.actor_critic)


def test_unrelated_mismatch_is_still_an_error(tmp_path):
    """Upgrading legacy names must not hide real mismatches, e.g. a checkpoint from a different architecture."""
    cfg = _make_eval_cfg(tmp_path)
    trained = _make_model(cfg, VECTOR_OBS, seed=0)
    state_dict = trained.state_dict()
    state_dict["encoder.encoders.obs.fc_layers.0.weight"] = torch.zeros(1)  # not a legacy name of anything
    _save_checkpoint(cfg, state_dict)

    fresh = _make_model(cfg, VECTOR_OBS, seed=1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        load_state_dict(cfg, fresh, torch.device("cpu"))
