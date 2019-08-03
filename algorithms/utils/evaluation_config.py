from os.path import join

from utils.utils import project_root


def add_eval_args(parser):
    parser.add_argument(
        '--fps',
        default=0,
        type=int,
        help='Enable sync mode with adjustable FPS.'
             'Default (0) means default Doom FPS (~35). Leave at 0 for multiplayer evaluation'
             'This is used only for evaluation, not for training',
    )
    parser.add_argument(
        '--evaluation-env-frameskip',
        default=1,
        type=int,
        help='Environment frameskip during evaluation',
    )
    parser.add_argument(
        '--render-action-repeat',
        default=4,
        type=int,
        help='Repeat an action that many frames during evaluation.'
             'Use in combination with env-frameskip for smooth rendering.',
    )
    parser.add_argument(
        '--record-to',
        default=join(project_root(), '..', 'doom_episodes'),
        type=str,
        help='Record episodes to this folder',
    )
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Do not render the environment during evaluation',
    )
