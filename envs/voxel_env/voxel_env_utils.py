from voxel_env.voxel_env_gym import VoxelEnv


def make_voxel_env(env_name, cfg=None, **kwargs):
    env = VoxelEnv()
    return env


def voxel_env_override_defaults(env, parser):
    """RL params specific to VoxelEnv envs."""
    parser.set_defaults(
        encoder_type='conv',
        encoder_subtype='convnet_simple',
        hidden_size=512,
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        actor_worker_gpus=[0],
    )
