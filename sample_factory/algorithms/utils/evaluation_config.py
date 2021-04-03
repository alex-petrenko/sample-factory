import os
from os.path import join

from sample_factory.utils.utils import str2bool


def add_eval_args(parser):
    parser.add_argument('--fps', default=0, type=int, help='Enable sync mode with adjustable FPS. Default (0) means default, e.g. for Doom its FPS (~35), or unlimited if not specified by env. Leave at 0 for Doom multiplayer evaluation')
    parser.add_argument('--render_action_repeat', default=None, type=int, help='Repeat an action that many frames during evaluation. By default uses the value from env config (used during training).')
    parser.add_argument('--no_render', action='store_true', help='Do not render the environment during evaluation')
    parser.add_argument('--policy_index', default=0, type=int, help='Policy to evaluate in case of multi-policy training')
    parser.add_argument('--record_to', default=join(os.getcwd(), '..', 'recs'), type=str, help='Record episodes to this folder. Only used for VizDoom!')

    parser.add_argument('--continuous_actions_sample', default=True, type=str2bool, help='True to sample from a continuous action distribution at test time, False to just take the mean')

