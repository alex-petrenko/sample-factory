from abc import ABC

from utils.utils import str2bool


class AlgorithmBase:
    def __init__(self, cfg):
        self.cfg = cfg

    def initialize(self):
        raise NotImplementedError()

    def run(self):
        raise NotImplementedError()

    def finalize(self):
        raise NotImplementedError()


class ReinforcementLearningAlgorithm(AlgorithmBase, ABC):
    """Basic things that most RL algorithms share."""

    @classmethod
    def add_cli_args(cls, parser):
        p = parser

        p.add_argument('--seed', default=None, type=int, help='Set a fixed seed value')

        p.add_argument('--initial_save_rate', default=1000, type=int, help='Save model every N train steps in the beginning of training')
        p.add_argument('--keep_checkpoints', default=2, type=int, help='Number of model checkpoints to keep')
        p.add_argument('--save_milestones_sec', default=-1, type=int, help='Save intermediate checkpoints in a separate folder for later evaluation (default=never)')

        p.add_argument('--stats_avg', default=200, type=int, help='How many episodes to average to measure performance (avg. reward etc)')

        p.add_argument('--learning_rate', default=1e-4, type=float, help='LR')

        p.add_argument('--train_for_env_steps', default=int(1e10), type=int, help='Stop after all policies are trained for this many env steps')
        p.add_argument('--train_for_seconds', default=int(1e10), type=int, help='Stop training after this many seconds')

        # observation preprocessing
        p.add_argument('--obs_subtract_mean', default=0.0, type=float, help='Observation preprocessing, mean value to subtract from observation (e.g. 128.0 for 8-bit RGB)')
        p.add_argument('--obs_scale', default=1.0, type=float, help='Observation preprocessing, divide observation tensors by this scalar (e.g. 128.0 for 8-bit RGB)')

        # RL
        p.add_argument('--gamma', default=0.99, type=float, help='Discount factor')
        p.add_argument(
            '--reward_scale', default=1.0, type=float,
            help=('Multiply all rewards but this factor before feeding into RL algorithm.'
                  'Sometimes the overall scale of rewards is too high which makes value estimation a harder regression task.'
                  'Loss values become too high which requires a smaller learning rate, etc.'),
        )
        p.add_argument('--reward_clip', default=10.0, type=float, help='Clip rewards between [-c, c]. Default [-10, 10] virtually means no clipping for most envs')

        # policy size and configuration
        p.add_argument('--encoder_type', default='conv', type=str, help='Type of the encoder')
        p.add_argument('--encoder_subtype', default='convnet_simple', type=str, help='Specific encoder design (see model.py)')
        p.add_argument('--encoder_custom', default=None, type=str, help='Use custom encoder class from the registry (see model_utils.py)')
        p.add_argument('--encoder_extra_fc_layers', default=1, type=int, help='Number of fully-connected layers of size "hidden size" to add after the basic encoder (e.g. convolutional)')
        p.add_argument('--hidden_size', default=512, type=int, help='Size of hidden layer in the model, or the size of RNN hidden state in recurrent model (e.g. GRU)')
        p.add_argument('--nonlinearity', default='elu', type=str, help='Type of nonlinearity to use')

    def __init__(self, cfg):
        super().__init__(cfg)
