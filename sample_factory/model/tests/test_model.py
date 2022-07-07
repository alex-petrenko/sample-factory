import pytest
import torch

from sample_factory.cfg.arguments import default_cfg
from sample_factory.envs.create_env import create_env
from sample_factory.model.model import create_actor_critic
from sample_factory.model.model_utils import get_hidden_size
from sample_factory.utils.timing import Timing
from sample_factory.utils.utils import log


class TestModel:
    @pytest.fixture(scope="class", autouse=True)
    def register_atari_fixture(self):
        from sample_factory_examples.atari_examples.train_atari import register_atari_components

        register_atari_components()

    @staticmethod
    def forward_pass(device_type):
        env_name = "atari_breakout"
        cfg = default_cfg(algo="APPO", env=env_name)
        cfg.actor_critic_share_weights = True
        cfg.hidden_size = 128
        cfg.use_rnn = True
        cfg.env_framestack = 4

        env = create_env(env_name, cfg=cfg)

        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = True

        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        device = torch.device(device_type)
        actor_critic.to(device)

        timing = Timing()
        with timing.timeit("all"):
            batch = 128
            with timing.add_time("input"):
                # better avoid hardcoding here...
                observations = dict(obs=torch.rand([batch, 4, 84, 84]).to(device))
                rnn_states = torch.rand([batch, get_hidden_size(cfg)]).to(device)

            n = 200
            for i in range(n):
                with timing.add_time("forward"):
                    _ = actor_critic(observations, rnn_states)

                log.debug("Progress %d/%d", i, n)

        log.debug("Timing: %s", timing)

    def test_forward_pass_cpu(self):
        self.forward_pass("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="This test requires a GPU")
    def test_forward_pass_gpu(self):
        self.forward_pass("cuda")
