from utils.utils import str2bool


def dmlab_override_defaults(env, parser):
    """Currently use same parameters as Doom."""
    from envs.doom.doom_params import doom_override_defaults
    doom_override_defaults(env, parser)


# noinspection PyUnusedLocal
def add_dmlab_env_args(env, parser):
    p = parser

    p.add_argument('--res_w', default=96, type=int, help='Game frame width after resize')
    p.add_argument('--res_h', default=72, type=int, help='Game frame height after resize')
    p.add_argument('--dmlab_throughput_benchmark', default=False, type=str2bool, help='Execute random policy for performance measurements')
    p.add_argument('--dmlab_renderer', default='hardware', type=str, help='Type of renderer (GPU vs CPU)')
    p.add_argument('--dmlab_gpus', default=[0], nargs='+', type=int, help='Indices of GPUs to use for rendering, only works in hardware mode')
