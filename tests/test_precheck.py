import pytest
import torch
import torch.multiprocessing as mp

from tests.examples.test_example_multi import default_multi_cfg, run_test_env_multi


def subp(q, test_idx_: int):
    t = q.get()
    t[0][0][0] = test_idx_


class TestPreCheck:
    @pytest.mark.parametrize("test_idx", list(range(10)))
    def test_torch_tensor_share(self, test_idx: int):
        ctx = mp.get_context("spawn")
        t = torch.rand((300, 200, test_idx * 10 + 1))
        m = torch.clone(t)
        m[0][0][0] = test_idx
        t.share_memory_()
        q = ctx.Queue()
        q.put(t)
        p = ctx.Process(target=subp, args=(q, test_idx))
        p.start()
        p.join()

        assert t.is_shared()
        assert torch.equal(t, m)

    def test_multi_policies(self):
        cfg, eval_cfg = default_multi_cfg()
        cfg.async_rl = True
        cfg.serial_mode = False
        cfg.train_for_env_steps = 80000
        cfg.num_workers = 8
        cfg.batch_size = 512

        cfg.custom_env_num_actions = eval_cfg.custom_env_num_actions = 100
        cfg.num_policies = 3
        cfg.with_pbt = True
        cfg.pbt_period_env_steps = 20000
        cfg.pbt_start_mutation = 20000
        cfg.pbt_mutation_rate = 0.9
        cfg.pbt_optimize_gamma = True

        run_test_env_multi(
            cfg,
            eval_cfg,
            expected_reward_at_least=-0.1,  # 0 is the best we can do, random policy gets ~ -5
        )
