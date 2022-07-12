import argparse
import copy
import json
import os
import sys
from typing import List, Optional, Tuple

from sample_factory.cfg.cfg import (
    add_basic_cli_args,
    add_default_env_args,
    add_eval_args,
    add_model_args,
    add_rl_args,
    add_wandb_args,
)
from sample_factory.utils.utils import AttrDict, cfg_file, get_git_commit_hash, log


def parse_sf_args(
    argv: Optional[List[str]] = None, evaluation: bool = False
) -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    """
    Create a parser and parse the known arguments (default SF configuration, see cfg.py).
    Returns a parser that can be further extended with additional arguments before a final pass is made.
    This allows custom scripts to add any additional arguments they need depending on partially known configuration,
    such as the environment name.

    argv: list of arguments to parse. If None, use sys.argv.
    evaluation: if True, also add evaluation-only arguments.
    returns: (parser, args)
    """
    if argv is None:
        argv = sys.argv[1:]

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
    add_basic_cli_args(p)
    add_rl_args(p)
    add_model_args(p)
    add_default_env_args(p)
    add_wandb_args(p)

    if evaluation:
        add_eval_args(p)

    args, _ = p.parse_known_args(argv)
    return p, args


def parse_full_cfg(parser: argparse.ArgumentParser, argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Given a parser, parse all arguments and return the final configuration."""
    if argv is None:
        argv = sys.argv[1:]

    args = parser.parse_args(argv)
    args = postprocess_args(args, argv, parser)
    return args


def postprocess_args(args, argv, parser) -> argparse.Namespace:
    """
    Postprocessing after parse_args is called.
    Makes it easy to use SF within another codebase which might have its own parse_args call.

    """

    if args.help:
        parser.print_help()
        sys.exit(0)

    args.command_line = " ".join(argv)

    # following is the trick to get only the args passed from the command line
    # We copy the parser and set the default value of None for every argument. Since one cannot pass None
    # from command line, we can differentiate between args passed from command line and args that got initialized
    # from their default values. This will allow us later to load missing values from the config file without
    # overriding anything passed from the command line
    no_defaults_parser = copy.deepcopy(parser)
    for arg_name in vars(args).keys():
        no_defaults_parser.set_defaults(**{arg_name: None})
    cli_args = no_defaults_parser.parse_args(argv)

    for arg_name in list(vars(cli_args).keys()):
        if cli_args.__dict__[arg_name] is None:
            del cli_args.__dict__[arg_name]

    args.cli_args = vars(cli_args)
    args.git_hash, args.git_repo_name = get_git_commit_hash()
    return args


def verify_cfg(cfg: argparse.Namespace) -> None:
    """
    Do some checks to make sure this is a viable configuration.
    The fact that configuration passes these checks does not guarantee that it is 100% valid,
    there are more checks sprinkled throughout the codebase.
    It is better to add new checks here if possible since we check only once instead of doing this over and
    over again in the training loop.
    """
    if cfg.normalize_returns and cfg.with_vtrace:
        # When we use vtrace the logic for calculating returns is different - we need to recalculate them
        # on every minibatch, because important sampling depends on the trained policy.
        # Current implementation of normalized returns assumed that we can calculate returns once per experience
        # batch.
        raise ValueError("Normalized returns are not supported with vtrace!")

    if cfg.async_rl and cfg.serial_mode:
        log.warning(
            "In serial mode all components run on the same process. Only use async_rl "
            "and serial mode together for debugging."
        )


def default_cfg(algo="APPO", env="env", experiment="test"):
    """Useful for tests."""
    argv = [f"--algo={algo}", f"--env={env}", f"--experiment={experiment}"]
    parser, args = parse_sf_args(argv)
    args = parse_full_cfg(parser, argv)
    return args


def load_from_checkpoint(cfg):
    filename = cfg_file(cfg)
    if not os.path.isfile(filename):
        raise Exception(f"Could not load saved parameters for experiment {cfg.experiment} (file {filename} not found)")

    with open(filename, "r") as json_file:
        json_params = json.load(json_file)
        log.warning("Loading existing experiment configuration from %s", filename)
        loaded_cfg = AttrDict(json_params)

    # override the parameters in config file with values passed from command line
    for key, value in cfg.cli_args.items():
        if key in loaded_cfg and loaded_cfg[key] != value:
            log.debug("Overriding arg %r with value %r passed from command line", key, value)
            loaded_cfg[key] = value

    # incorporate extra CLI parameters that were not present in JSON file
    for key, value in vars(cfg).items():
        if key not in loaded_cfg:
            log.debug("Adding new argument %r=%r that is not in the saved config file!", key, value)
            loaded_cfg[key] = value

    return loaded_cfg


def maybe_load_from_checkpoint(cfg):
    filename = cfg_file(cfg)
    if not os.path.isfile(filename):
        log.warning("Saved parameter configuration for experiment %s not found!", cfg.experiment)
        log.warning("Starting experiment from scratch!")
        return AttrDict(vars(cfg))

    return load_from_checkpoint(cfg)
