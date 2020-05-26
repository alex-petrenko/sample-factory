
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
from envs.create_env import create_env


env_idx = 0  # for debugging


class VizdoomEnv(Env):

    def __init__(self,
                 game=None,
                 frame_skip=4,  # Frames per step (>=1).
                 num_img_obs=4,  # Number of (past) frames in observation (>=1).

                 clip_reward=True,
                 episodic_lives=True,
                 max_start_noops=30,
                 repeat_action_probability=0.,
                 horizon=27000,):

        if not game:
            game = 'doom_battle'

        cfg=default_cfg(env=game)
        cfg.wide_aspect_ratio = False

        self.env = create_env(game, cfg=cfg)
        self._observation_space = self.env.observation_space

        gym_action_space = self.env.action_space
        self._action_space = IntBox(low=0, high=gym_action_space.n)  # only for discrete space

        self.first_reset = True

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        if self.first_reset:
            global env_idx
            print('Resetting doom env...', env_idx)
            env_idx += 1
            self.first_reset = False
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


class DoomLstmModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            output_size,
            fc_sizes=512,  # Between conv and lstm.
            lstm_size=512,
            use_maxpool=False,
            channels=None,  # None uses default.
            kernel_sizes=None,
            strides=None,
            paddings=None,
            ):
        super().__init__()

        # same model architecture as Sample-Factory
        self.conv = Conv2dHeadModel(
            image_shape=image_shape,
            channels=channels or [32, 64, 128],
            kernel_sizes=kernel_sizes or [8, 4, 3],
            strides=strides or [4, 2, 2],
            paddings=paddings or [0, 1, 1],
            use_maxpool=use_maxpool,
            hidden_sizes=fc_sizes,  # Applies nonlinearity at end.
        )
        self.lstm = torch.nn.LSTM(self.conv.output_size + output_size + 1, lstm_size)
        self.pi = torch.nn.Linear(lstm_size, output_size)
        self.value = torch.nn.Linear(lstm_size, 1)

    def forward(self, image, prev_action, prev_reward, init_rnn_state):
        """Feedforward layers process as [T*B,H], recurrent ones as [T,B,H].
        Return same leading dims as input, can be [T,B], [B], or [].
        (Same forward used for sampling and training.)"""

        img = image.type(torch.float)  # Expect doom_torch.uint8 inputs
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.

        # Infer (presence of) leading dimensions: [T,B], [B], or [].
        from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
        lead_dim, T, B, img_shape = infer_leading_dims(img, 3)

        fc_out = self.conv(img.view(T * B, *img_shape))
        lstm_input = torch.cat([
            fc_out.view(T, B, -1),
            prev_action.view(T, B, -1),  # Assumed onehot.
            prev_reward.view(T, B, 1),
            ], dim=2)
        init_rnn_state = None if init_rnn_state is None else tuple(init_rnn_state)
        lstm_out, (hn, cn) = self.lstm(lstm_input, init_rnn_state)
        pi = F.softmax(self.pi(lstm_out.view(T * B, -1)), dim=-1)
        v = self.value(lstm_out.view(T * B, -1)).squeeze(-1)

        # Restore leading dimensions: [T,B], [B], or [], as input.
        pi, v = restore_leading_dims((pi, v), lead_dim, T, B)
        # Model should always leave B-dimension in rnn state: [N,B,H].
        next_rnn_state = RnnState(h=hn, c=cn)

        return pi, v, next_rnn_state


class DoomMixin:
    def make_env_to_model_kwargs(self, env_spaces):
        return dict(image_shape=env_spaces.observation.shape, output_size=env_spaces.action.n)


class DoomLstmAgent(DoomMixin, AlternatingRecurrentCategoricalPgAgent):
    def __init__(self, ModelCls=DoomLstmModel, **kwargs):
        super().__init__(ModelCls=ModelCls, **kwargs)


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
        EnvCls=VizdoomEnv,
        env_kwargs=dict(game=game),
        batch_T=n_timestep,
        batch_B=n_env,
        max_decorrelation_steps=0,
    )
    algo = PPO(minibatches=1, epochs=1)

    agent = DoomLstmAgent()

    # Maybe AsyncRL could give better performance?
    # In the current version however PPO + AsyncRL does not seem to be working (not implemented)
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=50e6,
        log_interval_steps=1e5,
        affinity=affinity,
    )
    config = dict(game=game)
    name = "ppo_" + game + str(n_env)
    log_dir = "doom_ppo"

    with logger_context(log_dir, run_ID, name, config):
        runner.train()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Doom game', default='doom_benchmark')
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
