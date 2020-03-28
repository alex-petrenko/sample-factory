from utils.utils import str2bool


def dmlab_override_defaults(env, parser):
    parser.set_defaults(
        encoder_type='conv',
        encoder_subtype='convnet_simple',
        encoder_custom=None,
        hidden_size=512,
        obs_subtract_mean=128.0,
        obs_scale=128.0,
        env_frameskip=4,
    )


# noinspection PyUnusedLocal
def add_dmlab_env_args(env, parser):
    p = parser

    p.add_argument('--res_w', default=96, type=int, help='Game frame width after resize')
    p.add_argument('--res_h', default=72, type=int, help='Game frame height after resize')
    p.add_argument('--dmlab_throughput_benchmark', default=False, type=str2bool, help='Execute random policy for performance measurements')
    p.add_argument('--dmlab_renderer', default='hardware', type=str, help='Type of renderer (GPU vs CPU)')
    p.add_argument('--dmlab_gpus', default=[0], nargs='+', type=int, help='Indices of GPUs to use for rendering, only works in hardware mode')
    p.add_argument('--dmlab30_dataset', default='~/datasets/brady_konkle_oliva2008', type=str, help='Path to dataset needed for some of the environments in DMLab-30')
    p.add_argument('--dmlab_with_instructions', default=True, type=str2bool, help='Whether to use text instructions or not')
    p.add_argument('--dmlab_extended_action_set', default=False, type=str2bool, help='Use larger action set from newer papers')
