import sys

from sample_factory.cfg.arguments import checkpoint_override_defaults, parse_full_cfg, parse_sf_args
from sample_factory.eval import do_eval
from sf_examples.mujoco.mujoco_params import add_mujoco_env_args, mujoco_override_defaults
from sf_examples.mujoco.train_mujoco import register_mujoco_components


def main():
    """Script entry point."""
    register_mujoco_components()
    parser, cfg = parse_sf_args(evaluation=True)
    add_mujoco_env_args(cfg.env, parser)
    mujoco_override_defaults(cfg.env, parser)

    # important, instead of `load_from_checkpoint` as in enjoy we want
    # to override it here to be able to use argv arguments
    checkpoint_override_defaults(cfg, parser)

    cfg = parse_full_cfg(parser)

    status = do_eval(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
