import copy
import random

import pytest
import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.simplified_sampling_api import SyncSamplingAPI
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.rl_utils import samples_per_trajectory
from sample_factory.algo.utils.tensor_dict import cat_tensordicts
from sample_factory.cfg.arguments import verify_cfg
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.dicts import iterate_recursively
from sf_examples.mujoco.train_mujoco import parse_mujoco_cfg, register_mujoco_components


def _learner_losses_res(learner: Learner, dataset: AttrDict, num_invalids: int) -> AttrDict:
    # noinspection PyProtectedMember
    (
        action_distribution,
        policy_loss,
        exploration_loss,
        kl_old,
        kl_loss,
        value_loss,
        loss_locals,
    ) = learner._calculate_losses(dataset, num_invalids)

    return AttrDict(
        policy_loss=policy_loss,
        exploration_loss=exploration_loss,
        kl_old=kl_old,
        kl_loss=kl_loss,
        value_loss=value_loss,
    )


class TestValidMasks:
    @pytest.fixture(scope="class", autouse=True)
    def register_mujoco_fixture(self):
        register_mujoco_components()

    @pytest.mark.parametrize("use_rnn", [False, True])
    def test_losses_match(self, use_rnn: bool):
        cfg = parse_mujoco_cfg(argv=["--env=mujoco_humanoid", "--experiment=test_learner"])
        # this matches what we used when data was collected
        cfg.num_workers = 2
        cfg.rollout = 8
        cfg.batch_size = 32
        cfg.device = "cpu"
        cfg.serial_mode = True
        cfg.decorrelate_envs_on_one_worker = False
        cfg.normalize_returns = False
        cfg.normalize_input = False
        cfg.use_rnn = use_rnn
        cfg.recurrence = cfg.rollout if cfg.use_rnn else 1

        # enable all losses
        cfg.exploration_loss_coeff = 0.001

        tmp_env = make_env_func_batched(cfg, env_config=None)
        env_info = extract_env_info(tmp_env, cfg)

        assert verify_cfg(cfg, env_info)

        # here we're setting up the sampler to collect some trajectory batches as test data
        policy_id = 0
        policy_versions = torch.zeros([cfg.num_policies], dtype=torch.int32)
        param_server = ParameterServer(policy_id, policy_versions, cfg.serial_mode)
        sampler = SyncSamplingAPI(cfg, env_info, param_servers={policy_id: param_server})

        learner: Learner = Learner(cfg, env_info, policy_versions, policy_id, param_server)
        init_model_data = learner.init()
        assert learner.actor_critic is not None
        assert init_model_data is not None
        assert init_model_data[0] == policy_id
        sampler.start({policy_id: init_model_data})

        trajectories = []
        sampled = 0
        while sampled < cfg.batch_size:
            traj = sampler.get_trajectories_sync()
            assert traj is not None
            sampled += samples_per_trajectory(traj)
            trajectories.append(traj)

        sampler.stop()

        og_batch = cat_tensordicts(copy.deepcopy(trajectories))
        dataset, experience_size, invalids = learner._prepare_batch(og_batch)
        assert invalids == 0
        dataset = AttrDict(dataset)

        # sanity check: make sure we get the same losses on the same batch
        n_iter = 3
        res = prev_res = None
        for _ in range(n_iter):
            res = _learner_losses_res(learner, dataset, invalids)
            if prev_res is not None:
                for k in res.keys():
                    assert torch.allclose(res[k], prev_res[k])
            prev_res = res

        # Now we add some invalid transitions to the batch to simulate a situation where part of
        # the trajectory is collected by another policy or corresponds to an inactive agent state.
        # The easiest way to do this seems to be to increase the rollout length by inserting invalid data into
        # each trajectory.

        traj_invalid_data = copy.deepcopy(trajectories)
        for traj in traj_invalid_data:
            # place to insert invalid data
            j = random.randint(0, cfg.rollout // 2)

            # each trajectory is a tensordict itself
            for d, k, v in iterate_recursively(traj):
                sh = v.shape
                invalid_data_shape = torch.Size((sh[0], cfg.rollout) + sh[2:])
                new_data_shape = torch.Size((sh[0], sh[1] + cfg.rollout) + sh[2:])

                if v.dtype == torch.bool:
                    invalid_data = torch.randint(0, 2, invalid_data_shape)
                else:
                    invalid_data = torch.randint(-1, 1, invalid_data_shape) * 4242
                invalid_data = invalid_data.to(v.device).type(v.dtype)

                # splice the invalid data into the trajectory
                new_data = torch.empty(new_data_shape, dtype=v.dtype, device=v.device)
                new_data[:, :j] = v[:, :j]
                new_data[:, j : j + cfg.rollout] = invalid_data
                new_data[:, j + cfg.rollout :] = v[:, j:]
                d[k] = new_data

            # to indicate explicitly that this part of the trajectory is invalid (this one way how the learner can
            # detect it, i.e. inactive agents will be marked as having policy_id = -1)
            traj["policy_id"][:, j : j + cfg.rollout] = -1

            # we deliberately do not ignore dones that happen in inactive agent states, so we need to make sure
            # that in this test data all dones are set to False
            traj["dones"][:, j : j + cfg.rollout] = False

            # same for the timeouts - let's assume there are none in the invalid data
            traj["time_outs"][:, j : j + cfg.rollout] = False

        invalid_dataset, invalid_experience_size, invalids = learner._prepare_batch(cat_tensordicts(traj_invalid_data))
        assert invalid_experience_size == experience_size * 2
        assert invalids == experience_size
        invalid_dataset = AttrDict(invalid_dataset)

        if dataset["valids"][0]:
            # it is hard to come up with code that would yield completely the same returns as the original data,
            # because of some boundary conditions on the edge of the invalid data (i.e. we can use one value but
            # not the other in the GAE TD-1 formula).
            # But since we're initializing the invalid data with some huge values, we can at least check that
            # the returns are not too far from the original data.
            for key in ["returns", "advantages"]:
                assert torch.allclose(dataset[key][0], invalid_dataset[key][0], atol=0.1, rtol=0.1)

        invalid_res = _learner_losses_res(learner, invalid_dataset, invalids)
        atol, rtol = 0.02, 0.02
        assert torch.allclose(res.policy_loss, invalid_res.policy_loss, atol=atol, rtol=rtol)
        assert torch.allclose(res.exploration_loss, invalid_res.exploration_loss, atol=atol, rtol=rtol)
        assert torch.allclose(res.kl_loss, invalid_res.kl_loss, atol=atol, rtol=rtol)
        assert torch.allclose(res.value_loss, invalid_res.value_loss, atol=atol, rtol=rtol)
