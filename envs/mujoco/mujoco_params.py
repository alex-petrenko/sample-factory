from utils.utils import str2bool


def mujoco_override_defaults(env, parser):
    parser.set_defaults(
        encoder_type='mujoco',
        encoder_subtype='mujoco_mlp',
        hidden_size=64,
        encoder_extra_fc_layers=0,
        env_frameskip=1,
        nonlinearity='tanh'
    )


# noinspection PyUnusedLocal
def add_mujoco_env_args(env, parser):
    p = parser

    p.add_argument('--mujoco_discretize_actions', default=-1, type=int, help='Discretize actions into N bins for each individual action. Default (-1) means no discretization')
    p.add_argument('--mujoco_clip_input', default=False, type=str2bool, help='Whether to clip input to ensure it stays relatively small')
    p.add_argument('--mujoco_effort_reward', default=None, type=float, help='Override default value for effort reward')
