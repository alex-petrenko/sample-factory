import numpy as np

from unittest import TestCase

from algorithms.appo.appo_utils import make_env_func
from algorithms.appo.shared_buffers import SharedBuffers
from algorithms.utils.arguments import default_cfg
from utils.timing import Timing
from utils.utils import log


class TestAppoUtils(TestCase):
    def test_performance(self):
        cfg = default_cfg(algo='APPO', env='doom_battle')
        cfg.num_envs_per_worker = 20
        num_agents = 4

        tmp_env = make_env_func(cfg, env_config=None)
        obs_space = tmp_env.observation_space
        action_space = tmp_env.action_space
        tmp_env.close()

        traj_buffers = SharedBuffers(cfg, num_agents, obs_space, action_space)

        # t = traj_buffers.tensors['obs']['obs']
        t = traj_buffers.tensors_individual_transitions['obs']['obs']
        shape = t.shape

        total_elems = 0

        timing = Timing()

        with timing.add_time('loop'):
            for _ in range(10):
                for n_wor in range(shape[0]):
                    for n_spl in range(shape[1]):
                        for n_env in range(shape[2]):
                            for n_age in range(shape[3]):
                                for n_traj in range(shape[4]):
                                    for n_roll in range(shape[5]):
                                        tt = t[n_wor, n_spl, n_env, n_age, n_traj, n_roll]
                                        total_elems += tt.nelement()
        # loop1: 0.7s

        # with timing.add_time('loop2'):
        #     for _ in range(1):
        #         for n_wor in range(shape[0]):
        #             t_wor = t[n_wor]
        #             for n_spl in range(shape[1]):
        #                 t_spl = t_wor[n_spl]
        #                 for n_env in range(shape[2]):
        #                     t_env = t_spl[n_env]
        #                     for n_age in range(shape[3]):
        #                         t_age = t_env[n_age]
        #                         for n_traj in range(shape[4]):
        #                             t_traj = t_age[n_traj]
        #                             for n_roll in range(shape[5]):
        #                                 t_roll = t_traj[n_roll]
        #                                 total_elems += t_roll.nelement()
        # loop2: 0.15s

        # arr = np.ndarray(shape[:6], dtype=object)
        # for n_wor in range(shape[0]):
        #     t_wor = t[n_wor]
        #     for n_spl in range(shape[1]):
        #         t_spl = t_wor[n_spl]
        #         for n_env in range(shape[2]):
        #             t_env = t_spl[n_env]
        #             for n_age in range(shape[3]):
        #                 t_age = t_env[n_age]
        #                 for n_traj in range(shape[4]):
        #                     t_traj = t_age[n_traj]
        #                     for n_roll in range(shape[5]):
        #                         t_roll = t_traj[n_roll]
        #                         arr[n_wor, n_spl, n_env, n_age, n_traj, n_roll] = t_roll
        #
        # with timing.add_time('loop3'):
        #     for _ in range(10):
        #         for n_wor in range(shape[0]):
        #             for n_spl in range(shape[1]):
        #                 for n_env in range(shape[2]):
        #                     for n_age in range(shape[3]):
        #                         for n_traj in range(shape[4]):
        #                             for n_roll in range(shape[5]):
        #                                 total_elems += arr[n_wor, n_spl, n_env, n_age, n_traj, n_roll].nelement()
        # loop3: 0.036s (!!!)

        # d = dict()
        # for n_wor in range(shape[0]):
        #     t_wor = t[n_wor]
        #     for n_spl in range(shape[1]):
        #         t_spl = t_wor[n_spl]
        #         for n_env in range(shape[2]):
        #             t_env = t_spl[n_env]
        #             for n_age in range(shape[3]):
        #                 t_age = t_env[n_age]
        #                 for n_traj in range(shape[4]):
        #                     t_traj = t_age[n_traj]
        #                     for n_roll in range(shape[5]):
        #                         t_roll = t_traj[n_roll]
        #                         d[n_wor, n_spl, n_env, n_age, n_traj, n_roll] = t_roll
        # with timing.add_time('loop4'):
        #     for n_wor in range(shape[0]):
        #         for n_spl in range(shape[1]):
        #             for n_env in range(shape[2]):
        #                 for n_age in range(shape[3]):
        #                     for n_traj in range(shape[4]):
        #                         for n_roll in range(shape[5]):
        #                             total_elems += d[n_wor, n_spl, n_env, n_age, n_traj, n_roll].nelement()
        # loop4: 0.0390

        log.debug('Num elements %d', total_elems)
        log.debug('Timing: %s', timing)
