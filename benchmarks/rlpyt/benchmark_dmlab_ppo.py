
"""
Runs multiple instances of the

Doom environment and optimizes
using PPO algorithm

and a recurrent agent.

Uses GPU parallel sampler, with option for
whether to reset environments in middle of sampling batch.

Standard recurrent agents cannot train with a reset in the middle of a
sequence, so all data after the environment 'done' signal will be ignored (see
variable 'valid' in algo).  So it may be preferable to pause those environments
and wait to reset them for the beginning of the next iteration.

If the environment takes a long time to reset relative to step, this may also
give a slight speed boost, as resets will happen in the workers while the master
is optimizing.  Feedforward agents are compatible with this arrangement by same
use of 'valid' mask.

"""
import random
import time

import torch
import torch.nn.functional as F
from rlpyt.agents.pg.categorical import AlternatingRecurrentCategoricalPgAgent
from rlpyt.algos.pg.ppo import PPO
from rlpyt.envs.base import Env, EnvStep
from rlpyt.models.conv2d import Conv2dHeadModel
from rlpyt.models.pg.atari_lstm_model import RnnState
from rlpyt.runners.minibatch_rl import MinibatchRl
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler
from rlpyt.samplers.parallel.gpu.collectors import (GpuResetCollector, GpuWaitResetCollector)
from rlpyt.samplers.parallel.gpu.sampler import GpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.spaces.int_box import IntBox
from rlpyt.utils.logging.context import logger_context

from algorithms.utils.arguments import default_cfg
from benchmarks.rlpyt.benchmark_doom_ppo import DoomLstmAgent
from envs.create_env import create_env


env_idx = 0  # for debugging


class DmlabEnv(Env):

    def __init__(self,
                 game=None,
                 frame_skip=4,  # Frames per step (>=1).
                 num_img_obs=4,  # Number of (past) frames in observation (>=1).

                 clip_reward=True,
                 episodic_lives=True,
                 max_start_noops=30,
                 repeat_action_probability=0.,
                 horizon=27000,):

        cfg = default_cfg(env=game)
        cfg.res_w = 96
        cfg.res_h = 72
        cfg.dmlab_throughput_benchmark = True
        cfg.dmlab_renderer = 'software'

        self.env = create_env(game, cfg=cfg)
        self._observation_space = self.env.observation_space

        gym_action_space = self.env.action_space
        self._action_space = IntBox(low=0, high=gym_action_space.n)  # only for discrete space

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        res = self.env.reset()
        return res

    # Changed last line
    def step(self, actions):
        """
        Action is either a single value (discrete, one-hot), or a tuple with an action for each of the
        discrete action subspaces.
        """
        action = actions.item()
        obs, rew, done, info = self.env.step(action)
        # print('Obs shape as returned by the env:', obs.shape)
        fake_info = tuple()  # do not need for testing
        return EnvStep(obs, rew, done, fake_info)

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        self.env.close()


def build_and_train(game="doom_benchmark", run_ID=0, cuda_idx=None, n_parallel=-1,
                    n_env=-1, n_timestep=-1, sample_mode=None):
    affinity = dict(cuda_idx=cuda_idx, workers_cpus=list(range(n_parallel)))

    gpu_cpu = "CPU" if cuda_idx is None else f"GPU {cuda_idx}"
    if sample_mode == "serial":
        Sampler = SerialSampler  # (Ignores workers_cpus.)
        print(f"Using serial sampler, {gpu_cpu} for sampling and optimizing.")
    elif sample_mode == "cpu":
        Sampler = CpuSampler
        print(f"Using CPU parallel sampler (agent in workers), {gpu_cpu} for optimizing.")
    elif sample_mode == "gpu":
        Sampler = GpuSampler
        print(f"Using GPU parallel sampler (agent in master), {gpu_cpu} for sampling and optimizing.")
    elif sample_mode == "alternating":
        Sampler = AlternatingSampler
        affinity["workers_cpus"] += affinity["workers_cpus"]  # (Double list)
        affinity["alternating"] = True  # Sampler will check for this.
        print(f"Using Alternating GPU parallel sampler, {gpu_cpu} for sampling and optimizing.")

        # !!!
        # COMMENT: to use alternating sampler here we had to comment lines 126-127 in action_server.py
        # if "bootstrap_value" in self.samples_np.agent:
        #     self.bootstrap_value_pair[alt][:] = self.agent.value(*agent_inputs_pair[alt])
        # otherwise it crashes
        # !!!

    sampler = Sampler(
        EnvCls=DmlabEnv,
        env_kwargs=dict(game=game),
        batch_T=n_timestep,
        batch_B=n_env,
        max_decorrelation_steps=0,
    )
    # using decorrelation here completely destroys the performance, because episodes will reset at different times and the learner will wait for 1-2 workers to complete, wasting a lot of time
    # this should not be an issue with asynchronous implementation, but it is not supported at the moment

    algo = PPO(minibatches=1, epochs=1)

    agent = DoomLstmAgent()

    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e5,
        affinity=affinity,
    )
    config = dict(game=game)
    name = "ppo_" + game
    log_dir = "dmlab_ppo"

    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Dmlab game', default='dmlab_benchmark')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=None)
    parser.add_argument('--n_parallel', help='number of sampler workers', type=int, default=12)
    parser.add_argument('--n_env', help='number of environments', type=int, default=16)
    parser.add_argument('--n_timestep', help='number of time steps', type=int, default=20)
    parser.add_argument('--sample_mode', help='serial or parallel sampling',
                        type=str, default='serial', choices=['serial', 'cpu', 'gpu', 'alternating'])

    args = parser.parse_args()
    build_and_train(
        game=args.game,
        run_ID=args.run_ID,
        cuda_idx=args.cuda_idx,
        n_parallel=args.n_parallel,
        n_env=args.n_env,
        n_timestep=args.n_timestep,
        sample_mode=args.sample_mode,
    )
