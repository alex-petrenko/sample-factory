import sys

from sample_factory.cfg.arguments import checkpoint_override_defaults, parse_full_cfg, parse_sf_args
from sample_factory.eval import do_eval
from sf_examples.nethack.nethack_params import (
    add_extra_params_general,
    add_extra_params_model,
    add_extra_params_nethack_env,
    nethack_override_defaults,
)
from sf_examples.nethack.train_nethack import register_nethack_components


def main():  # pragma: no cover
    """Script entry point."""
    register_nethack_components()

    parser, cfg = parse_sf_args(evaluation=True)
    add_extra_params_nethack_env(parser)
    add_extra_params_model(parser)
    add_extra_params_general(parser)
    nethack_override_defaults(cfg.env, parser)

    # important, instead of `load_from_checkpoint` as in enjoy we want
    # to override it here to be able to use argv arguments
    checkpoint_override_defaults(cfg, parser)

    cfg = parse_full_cfg(parser)

    status = do_eval(cfg)
    return status


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
