import functools
import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.model.model_utils import register_custom_encoder
from sample_factory.train import run_rl
from sf_examples.vizdoom_examples.doom.doom_model import VizdoomEncoder
from sf_examples.vizdoom_examples.doom.doom_params import add_doom_env_args, doom_override_defaults
from sf_examples.vizdoom_examples.doom.doom_utils import DOOM_ENVS, make_doom_env_from_spec


def register_vizdoom_envs():
    for env_spec in DOOM_ENVS:
        make_env_func = functools.partial(make_doom_env_from_spec, env_spec)
        register_env(env_spec.name, make_env_func)


def register_vizdoom_models():
    register_custom_encoder("vizdoom", VizdoomEncoder)


def register_vizdoom_components():
    register_vizdoom_envs()
    register_vizdoom_models()


def main():
    """Script entry point."""
    register_vizdoom_components()

    parser, cfg = parse_sf_args()
    # parameters specific to Doom envs
    add_doom_env_args(parser)
    # override Doom default values for algo parameters
    doom_override_defaults(parser)
    # second parsing pass yields the final configuration
    cfg = parse_full_cfg(parser)

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
