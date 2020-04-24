from unittest import TestCase

import torch

from algorithms.appo.model import create_actor_critic
from algorithms.appo.model_utils import get_hidden_size
from algorithms.utils.arguments import default_cfg
from envs.create_env import create_env
from utils.timing import Timing
from utils.utils import log


class TestModel(TestCase):
    def test_forward_pass(self):
        env_name = 'quadrotor_single'
        cfg = default_cfg(algo='APPO', env=env_name)
        cfg.actor_critic_share_weights = True
        cfg.hidden_size = 128
        cfg.use_rnn = False
        env = create_env(env_name, cfg=cfg)

        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = True

        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        device = torch.device('cpu')
        actor_critic.to(device)

        timing = Timing()
        with timing.timeit('all'):
            batch = 128
            with timing.add_time('input'):
                # observations = dict(obs=torch.rand([batch, 3, 72, 128]).to(device))
                observations = dict(obs=torch.rand([batch, 18]).to(device))
                rnn_states = torch.rand([batch, get_hidden_size(cfg)]).to(device)

            output = actor_critic(observations, rnn_states)

            # with timing.add_time('trace'):
            #     actor_critic_traced = torch.jit.trace(actor_critic, (observations, rnn_states), check_trace=False)

            n = 10000
            for i in range(n):
                with timing.add_time('forward'):
                    # output = actor_critic_traced(observations, rnn_states)
                    output = actor_critic(observations, rnn_states)

                if i % 100 == 0:
                    log.debug('Progress %d/%d', i, n)

        log.debug('Timing: %s', timing)
