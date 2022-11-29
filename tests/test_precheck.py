import pytest
import torch
import torch.multiprocessing as mp


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
