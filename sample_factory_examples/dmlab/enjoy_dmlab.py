import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.enjoy import enjoy
from sample_factory.envs.env_utils import register_env
from sample_factory.model.model_utils import register_custom_encoder
from sample_factory_examples.dmlab.dmlab_env import DMLAB_ENVS, make_dmlab_env
from sample_factory_examples.dmlab.dmlab_model import DmlabEncoder
from sample_factory_examples.dmlab.dmlab_params import add_dmlab_env_args, dmlab_override_defaults
from sample_factory_examples.dmlab.train_dmlab import register_dmlab_components


def main():
    """Script entry point."""
    register_dmlab_components()

    parser, cfg = parse_sf_args(evaluation=True)
    # parameters specific to DMlab envs
    add_dmlab_env_args(parser)
    # override DMlab default values for algo parameters
    dmlab_override_defaults(parser)
    # second parsing pass yields the final configuration

    cfg = parse_full_cfg(parser)

    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
