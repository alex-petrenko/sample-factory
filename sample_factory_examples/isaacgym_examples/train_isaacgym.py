import os
import sys
from os.path import join
from typing import List

# this is here just to guarantee that isaacgym is imported before PyTorch
# noinspection PyUnresolvedReferences
import isaacgym

import gym
import torch
from isaacgymenvs.tasks import isaacgym_task_map
from isaacgymenvs.utils.reformat import omegaconf_to_dict
from torch import nn, Tensor

from sample_factory.algo.utils.context import global_env_registry
from sample_factory.model.model_utils import register_custom_encoder, EncoderBase, get_obs_shape
from sample_factory.cfg.arguments import parse_args
from sample_factory.train import run_rl
from sample_factory.utils.utils import str2bool


class IsaacGymVecEnv(gym.Env):
    def __init__(self, isaacgym_env, obs_key):
        self.env = isaacgym_env
        self.num_agents = self.env.num_envs  # TODO: what about vectorized multi-agent envs? should we take num_agents into account also?

        self.action_space = self.env.action_space

        # isaacgym_examples environments actually return dicts
        if obs_key == 'obs':
            self.observation_space = gym.spaces.Dict(dict(obs=self.env.observation_space))
            self._proc_obs_func = lambda obs_dict: obs_dict
        elif obs_key == 'states':
            self.observation_space = gym.spaces.Dict(dict(obs=self.env.state_space))
            self._proc_obs_func = self._use_states_as_obs
        else:
            raise ValueError(f'Unknown observation key: {obs_key}')

    @staticmethod
    def _use_states_as_obs(obs_dict):
        obs_dict['obs'] = obs_dict['states']
        return obs_dict

    def reset(self, *args, **kwargs):
        obs_dict = self.env.reset()
        return self._proc_obs_func(obs_dict)

    def step(self, actions):
        obs, rew, dones, infos = self.env.step(actions)
        return self._proc_obs_func(obs), rew, dones, infos

    def render(self, mode='human'):
        pass


def make_isaacgym_env(full_env_name, cfg, env_config=None):
    task_name = '_'.join(full_env_name.split('_')[1:])
    overrides = [f'task={task_name}']
    if cfg.env_agents > 0:
        overrides.append(f'num_envs={cfg.env_agents}')

    from hydra import compose, initialize
    import isaacgymenvs
    # this will register resolvers for the hydra config
    # noinspection PyUnresolvedReferences
    from isaacgymenvs import train

    module_dir = isaacgymenvs.__path__[0]
    cfg_dir = join(module_dir, 'cfg')
    curr_file_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_dir = os.path.relpath(cfg_dir, curr_file_dir)
    initialize(config_path=cfg_dir, job_name='sf_isaacgym')
    ige_cfg = compose(config_name='config', overrides=overrides)

    rl_device = ige_cfg.rl_device
    sim_device = ige_cfg.sim_device
    graphics_device_id = ige_cfg.graphics_device_id

    ige_cfg_dict = omegaconf_to_dict(ige_cfg)
    task_cfg = ige_cfg_dict['task']

    env = isaacgym_task_map[task_cfg['name']](
        cfg=task_cfg,
        sim_device=sim_device,
        rl_device=rl_device,
        graphics_device_id=graphics_device_id,
        headless=cfg.env_headless,
        virtual_screen_capture=False,
        force_render=False,
    )

    env = IsaacGymVecEnv(env, cfg.obs_key)
    return env


class _IsaacGymMlpEncoderImlp(nn.Module):
    def __init__(self, obs_space, mlp_layers: List[int]):
        super().__init__()

        obs_shape = get_obs_shape(obs_space)
        assert len(obs_shape.obs) == 1

        layer_input_width = obs_shape.obs[0]
        encoder_layers = []
        for layer_width in mlp_layers:
            encoder_layers.append(nn.Linear(layer_input_width, layer_width))
            layer_input_width = layer_width
            encoder_layers.append(nn.ELU(inplace=True))

        self.mlp_head = nn.Sequential(*encoder_layers)

    def forward(self, obs: Tensor):
        x = self.mlp_head(obs)
        return x


class IsaacGymMlpEncoder(EncoderBase):
    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        self._impl = _IsaacGymMlpEncoderImlp(obs_space, cfg.mlp_layers)
        self._impl = torch.jit.script(self._impl)
        self.encoder_out_size = cfg.mlp_layers[-1]  # TODO: we should make this an abstract method

    def forward(self, obs_dict):
        x = self._impl(obs_dict['obs'])
        return x


def add_extra_params_func(env, parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument('--env_agents', default=-1, type=int, help='Num agents in each env (default: -1, means use default value from isaacgymenvs env yaml config file)')
    p.add_argument('--env_headless', default=True, type=str2bool, help='Headless == no rendering')
    p.add_argument('--mlp_layers', default=[256, 128, 64], type=int, nargs='*', help='MLP layers to use with isaacgym_examples envs')
    p.add_argument(
        '--obs_key', default='obs', type=str,
        help='IsaacGym envs return dicts, some envs return just "obs", and some return "obs" and "states".'
             'States key denotes the full state of the environment, and obs key corresponds to limited observations '
             'available in real world deployment. If we use "states" here we can train will full information '
             '(although the original idea was to use asymmetric training - critic sees full state and policy only sees obs).',
    )


def override_default_params_func(env, parser):
    """
    Override default argument values for this family of environments.
    All experiments for environments from my_custom_env_ family will have these parameters unless
    different values are passed from command line.

    """
    parser.set_defaults(
        # we're using a single very vectorized env, no need to parallelize it further
        batched_sampling=True,
        num_workers=1,
        num_envs_per_worker=1,
        worker_num_splits=1,
        actor_worker_gpus=[0],  # obviously need a GPU

        train_for_env_steps=10000000,

        encoder_custom='isaac_gym_mlp_encoder',
        use_rnn=False,
        adaptive_stddev=False,
        policy_initialization='torch_default',
        env_gpu_actions=True,
        reward_scale=0.01,
        rollout=16,
        max_grad_norm=0.0,
        batch_size=32768,
        num_batches_per_epoch=2,
        num_epochs=4,
        ppo_clip_ratio=0.2,
        value_loss_coeff=2.0,
        exploration_loss_coeff=0.0,
        nonlinearity='elu',
        learning_rate=3e-4,
        lr_schedule='kl_adaptive_epoch',
        lr_schedule_kl_threshold=0.008,
        shuffle_minibatches=False,
        gamma=0.99,
        gae_lambda=0.95,
        with_vtrace=False,
        recurrence=1,
        value_bootstrap=True,  # assuming reward from the last step in the episode can generally be ignored
        normalize_input=True,
        experiment_summaries_interval=3,  # experiments are short so we should save summaries often
        save_every_sec=15,

        serial_mode=True,  # it makes sense to run isaacgym envs in serial mode since most of the parallelism comes from the env itself (although async mode works!)
        async_rl=False,
    )

    # environment specific overrides
    env_name = '_'.join(env.split('_')[1:]).lower()
    if env_name == 'ant':
        parser.set_defaults(mlp_layers=[256, 128, 64])
    elif env_name == 'humanoid':
        parser.set_defaults(
            train_for_env_steps=1310000000,  # to match how much it is trained in rl-games
            mlp_layers=[400, 200, 100],
            rollout=32,
            num_epochs=5,
            value_loss_coeff=4.0,
            max_grad_norm=1.0,
            num_batches_per_epoch=4,
        )
    elif env_name == 'allegrohand':
        parser.set_defaults(
            train_for_env_steps=10_000_000_000,
            mlp_layers=[512, 256, 128],
            gamma=0.99,
            rollout=16,
            recurrence=16,
            use_rnn=False,
            learning_rate=1e-4,
            lr_schedule_kl_threshold=0.02,
            reward_scale=0.01,
            num_epochs=5,
            max_grad_norm=1.0,
            num_batches_per_epoch=8,
        )
    elif env_name == 'allegrohandlstm':
        parser.set_defaults(
            train_for_env_steps=10_000_000_000,
            mlp_layers=[512, 256, 128],
            gamma=0.99,
            rollout=16,
            recurrence=16,
            use_rnn=True,
            learning_rate=1e-4,
            lr_schedule_kl_threshold=0.016,
            reward_scale=0.01,
            num_epochs=4,
            max_grad_norm=1.0,
            num_batches_per_epoch=8,

            obs_key='states',
        )


def register_isaacgym_custom_components():
    global_env_registry().register_env(
        env_name_prefix='isaacgym_',
        make_env_func=make_isaacgym_env,
        add_extra_params_func=add_extra_params_func,
        override_default_params_func=override_default_params_func,
    )

    register_custom_encoder('isaac_gym_mlp_encoder', IsaacGymMlpEncoder)


def main():
    """Script entry point."""
    register_isaacgym_custom_components()
    cfg = parse_args()
    status = run_rl(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
