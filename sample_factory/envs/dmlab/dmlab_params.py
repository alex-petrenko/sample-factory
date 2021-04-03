import os
from os.path import join

from sample_factory.utils.utils import str2bool


def dmlab_override_defaults(env, parser):
    parser.set_defaults(
        encoder_type='conv',
        encoder_subtype='convnet_impala',
        encoder_custom=None,
        hidden_size=512,
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        env_frameskip=4,
    )


# noinspection PyUnusedLocal
def add_dmlab_env_args(env, parser):
    p = parser

    p.add_argument('--res_w', default=96, type=int, help='Game frame width after resize')
    p.add_argument('--res_h', default=72, type=int, help='Game frame height after resize')
    p.add_argument('--dmlab_throughput_benchmark', default=False, type=str2bool, help='Execute random policy for performance measurements')
    p.add_argument('--dmlab_renderer', default='software', type=str, choices=['software', 'hardware'], help='Type of renderer (GPU vs CPU)')
    p.add_argument('--dmlab_gpus', default=[0], nargs='+', type=int, help='Indices of GPUs to use for rendering, only works in hardware mode')
    p.add_argument('--dmlab30_dataset', default='~/datasets/brady_konkle_oliva2008', type=str, help='Path to dataset needed for some of the environments in DMLab-30')
    p.add_argument('--dmlab_with_instructions', default=True, type=str2bool, help='Whether to use text instructions or not')
    p.add_argument('--dmlab_extended_action_set', default=False, type=str2bool, help='Use larger action set from newer papers (e.g. PopART and R2D2)')
    p.add_argument('--dmlab_use_level_cache', default=True, type=str2bool, help='Whether to use the local level cache (highly recommended)')
    p.add_argument('--dmlab_level_cache_path', default=join(os.getcwd(), '.dmlab_cache'), type=str, help='Location to store the cached levels (or path to pre-generated cache)')
    p.add_argument(
        '--dmlab_one_task_per_worker', default=False, type=str2bool,
        help='By default SampleFactory will run several tasks per worker. E.g. if num_envs_per_worker=30 then each and every worker'
             'will run all 30 tasks of DMLab-30. In such regime an equal amount of samples will be collected for all tasks'
             'throughout training. This can potentially limit the throughput, because in this case the system is forced to'
             'collect the same number of samples from slow and from fast environments (and the simulation speeds vary greatly, especially on CPU)'
             'This flag enables a different regime, where each worker is focused on a single task. In this case the total number of workers should'
             'be a multiple of 30 (for DMLab-30), and e.g. 17th task will be executed on 17th, 47th, 77th... worker',
    )
