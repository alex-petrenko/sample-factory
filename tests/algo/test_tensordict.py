import numpy as np
import pytest
import torch

from sample_factory.algo.utils.tensor_dict import TensorDict, cat_tensordicts


class TestParams:
    # setting up tensor dict for a typical RL task
    @pytest.mark.parametrize("n_agents", [16])
    @pytest.mark.parametrize("n_obs", [5])
    @pytest.mark.parametrize("rollout", [8])
    def test_tensordict_simple(self, n_agents, n_obs, rollout):
        # dictionary observations
        obs = TensorDict()
        obs["pos"] = torch.rand((n_agents, rollout, n_obs))
        obs["vel"] = torch.rand((n_agents, rollout, n_obs))

        # main tensor dict with obs, rewards, dones, maybe something else
        d = TensorDict()
        d["observations"] = obs
        d["rewards"] = torch.ones((n_agents, rollout))
        d["dones"] = torch.zeros((n_agents, rollout))

        # imagine we got some data from the environment
        curr_rollout_step = 3
        curr_obs = dict(pos=torch.ones((n_agents, n_obs)), vel=torch.zeros((n_agents, n_obs)))
        curr_step_data = dict(observations=curr_obs, rewards=torch.zeros((n_agents,)), dones=torch.ones((n_agents,)))

        # save the current step data in the buffer of trajectories with one line of code
        d[:, curr_rollout_step] = curr_step_data

        # Verify that the data is set. First, slice the current step data:
        step = d[:, curr_rollout_step]  # this will create a slice of the entire recursive TensorDict

        assert step["dones"][0].item() == 1.0
        assert step["rewards"][5].item() == 0.0
        assert step["observations"]["vel"][7, 0].item() == 0.0

        # get a subset of agents
        odd_agents = d[1::2]  # this will slice the TensorDict to only contain odd-numbered agents
        assert odd_agents["observations"]["pos"].shape == (n_agents // 2, rollout, n_obs)

        # we can also assign numpy arrays, not only tensors
        odd_agents[:] = dict(rewards=np.arange((n_agents // 2) * rollout).reshape((n_agents // 2, rollout)))
        assert d[1]["rewards"].equal(odd_agents[0]["rewards"])
        assert d[3]["rewards"].equal(odd_agents[1]["rewards"])

        # and we can do many other things...

    def test_cat_tensordicts(self):
        # noinspection DuplicatedCode
        d1 = TensorDict(dict(a=torch.zeros((2, 3)), b=torch.ones((2, 3))))
        d2 = TensorDict(dict(a=torch.ones((2, 3)), b=torch.zeros((2, 3))))

        d_cat = cat_tensordicts([d1, d2])
        assert d_cat["a"].equal(torch.cat([d1["a"], d2["a"]]))
        assert d_cat["b"].equal(torch.cat([d1["b"], d2["b"]]))

        # now same test with numpy instead of torch
        # noinspection DuplicatedCode
        d1 = TensorDict(dict(a=np.zeros((2, 3)), b=np.ones((2, 3))))
        d2 = TensorDict(dict(a=np.ones((2, 3)), b=np.zeros((2, 3))))

        d_cat = cat_tensordicts([d1, d2])
        assert np.array_equal(d_cat["a"], np.concatenate([d1["a"], d2["a"]]))
        assert np.array_equal(d_cat["b"], np.concatenate([d1["b"], d2["b"]]))
