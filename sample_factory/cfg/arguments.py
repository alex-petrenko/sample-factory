import argparse
import copy
import json
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.rl_utils import total_num_agents
from sample_factory.cfg.cfg import (
    add_basic_cli_args,
    add_default_env_args,
    add_dqn_args,
    add_eval_args,
    add_model_args,
    add_pbt_args,
    add_rl_args,
    add_wandb_args,
)
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config
from sample_factory.utils.utils import cfg_file, cfg_file_old, get_git_commit_hash, log


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
    add_pbt_args(p)
    add_dqn_args(p)

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


def preprocess_cfg(cfg: Config, env_info: EnvInfo) -> bool:
    if cfg.recurrence == -1:
        cfg.recurrence = cfg.rollout if cfg.use_rnn else 1
        log.debug(f"Automatically setting recurrence to {cfg.recurrence}")

    if str(cfg.algo).upper() == "DQN":
        dqn_batch_size = getattr(cfg, "dqn_batch_size", 0)
        if dqn_batch_size <= 0:
            cfg.dqn_batch_size = min(cfg.batch_size, 256)
            log.info(f"DQN: Use batch_size {cfg.dqn_batch_size}")
        if cfg.dqn_batch_size < cfg.rollout:
            log.warning("DQN: Set dqn_batch_size to rollout")
            cfg.dqn_batch_size = cfg.rollout
        if cfg.dqn_batch_size % cfg.rollout != 0:
            adjusted = max(cfg.rollout, (cfg.dqn_batch_size // cfg.rollout) * cfg.rollout)
            log.warning(
                f"DQN: dqn_batch_size ({cfg.dqn_batch_size}) not divisible by rollout ({cfg.rollout}), dqn_batch_size={adjusted}"
            )
            cfg.dqn_batch_size = adjusted

        # Big replay buffer causes overhead in memory allocation
        # So cap RAM usage to 1GB (1Bil bytes) by how many bytes each observation takes
        # Only when replay_buffer_size is not passed in cli args
        cli_args = getattr(cfg, "cli_args", {})  # as in config.json
        if "replay_buffer_size" not in cli_args and cfg.replay_buffer_size >= 1000000:
            obs_space = env_info.obs_space
            bytes_per_obs = 0
            if hasattr(obs_space, "spaces"):
                for space in obs_space.spaces.values():
                    if hasattr(space, "shape"):
                        dtype = np.dtype(space.dtype) if space.dtype is not None else np.float32
                        bytes_per_obs += int(np.prod(space.shape)) * dtype.itemsize
            elif hasattr(obs_space, "shape"):
                dtype = np.dtype(obs_space.dtype) if obs_space.dtype is not None else np.float32
                bytes_per_obs = int(np.prod(obs_space.shape)) * dtype.itemsize

            bytes_per_transition = bytes_per_obs * 2
            if bytes_per_transition > 0:
                max_size = max(1000, int(1_000_000_000 // bytes_per_transition))  # Approx 1GB
                if max_size < cfg.replay_buffer_size:
                    log.warning(f"DQN: Cap replay_buffer_size to {max_size}")
                    cfg.replay_buffer_size = max_size

        # learning_starts cant exceed buffer size
        if cfg.learning_starts > cfg.replay_buffer_size:
            old_learning_starts = cfg.learning_starts
            cfg.learning_starts = max(1000, cfg.replay_buffer_size // 2)
            log.warning(
                f"DQN: learning_starts ({old_learning_starts}) > replay_buffer_size ({cfg.replay_buffer_size}), capped to {cfg.learning_starts}"
            )

    return verify_cfg(cfg, env_info)


def verify_cfg(cfg: Config, env_info: EnvInfo) -> bool:
    """
    Do some checks to make sure this is a viable configuration.
    The fact that configuration passes these checks does not guarantee that it is 100% valid,
    there are more checks sprinkled throughout the codebase.
    It is better to add new checks here if possible since we check only once instead of doing this over and
    over again in the training loop.

    cfg: the configuration to verify
    returns: True if the configuration is valid, False otherwise.
    """
    good_config: bool = True

    def cfg_error(msg: str) -> None:
        nonlocal good_config
        good_config = False
        log.error(msg)

    if cfg.num_envs_per_worker % cfg.worker_num_splits != 0:
        cfg_error(
            f"{cfg.num_envs_per_worker=} must be a multiple of {cfg.worker_num_splits=}"
            f" (for double-buffered sampling you need to use even number of envs per worker)"
        )

    if cfg.normalize_returns and cfg.with_vtrace:
        # When we use vtrace the logic for calculating returns is different - we need to recalculate them
        # on every minibatch, because important sampling depends on the trained policy.
        # Current implementation of normalized returns assumed that we can calculate returns once per experience
        # batch.
        cfg_error("Normalized returns are not supported with vtrace!")

    if cfg.async_rl and cfg.serial_mode:
        log.warning(
            "In serial mode all components run on the same process. Only use async_rl "
            "and serial mode together for debugging."
        )

    if cfg.num_policies > 1 and cfg.batched_sampling:
        log.warning(
            "In batched mode we're using a single policy per worker which does not allow us to use multiple different policies in the same env (see agent_policy_mapping.py)."
        )

    sync_rl = not cfg.async_rl
    batch_size = cfg.batch_size
    if str(cfg.algo).upper() == "DQN":
        dqn_batch_size = getattr(cfg, "dqn_batch_size", 0)
        if dqn_batch_size > 0:
            batch_size = dqn_batch_size
            if dqn_batch_size < cfg.rollout:
                cfg_error(f"{dqn_batch_size=} must >= {cfg.rollout=}")
            if dqn_batch_size % cfg.rollout != 0:
                cfg_error(f"{dqn_batch_size=} must be divisible by {cfg.rollout=}")

    samples_per_training_iteration = cfg.num_batches_per_epoch * batch_size
    samples_from_all_workers_per_rollout = total_num_agents(cfg, env_info) * cfg.rollout // cfg.num_policies

    if sync_rl:
        if (
            samples_per_training_iteration % samples_from_all_workers_per_rollout == 0
            and samples_per_training_iteration >= samples_from_all_workers_per_rollout
        ):
            # everything is fine
            pass
        else:
            cfg_error(
                "In sync mode the goal is to avoid policy lag. In order to achieve this we "
                "alternate between collecting experience and training on it.\nThus sync mode requires "
                "the sampler to collect the exact amount of experience required for training in one "
                "or more iterations.\nConfiguration needs to be changed.\n"
                "The easiest option is to enable async mode using --async_rl=True.\n"
                "Alternatively you can use information below to change number of workers, or batch size, etc.:\n"
            )
            cfg_error(
                f"Number of samples collected per rollout by all workers: "
                f"{cfg.num_workers=} * {cfg.num_envs_per_worker=} * {env_info.num_agents=} * {cfg.rollout=} // {cfg.num_policies=} = {samples_from_all_workers_per_rollout}"
            )
            cfg_error(
                f"Number of samples processed per training iteration on one learner: "
                f"{cfg.num_batches_per_epoch=} * {cfg.batch_size=} = {samples_per_training_iteration}"
            )
            cfg_error(
                f"Ratio is {samples_per_training_iteration / samples_from_all_workers_per_rollout} (should be a positive integer)"
            )
            good_config = False

    if sync_rl and cfg.num_policies > 1:
        log.warning(
            "Sync mode is not fully tested with multi-policy training. Use at your own risk. "
            "Probably requires a deterministic policy to agent mapping to guarantee that we always collect the "
            "same amount of experience per policy."
        )

    if cfg.use_rnn:
        if cfg.recurrence <= 1:
            cfg_error(
                f"{cfg.recurrence=} must be > 1 to train an RNN. Recommeded value is recurrence == {cfg.rollout=}."
            )

        if cfg.with_vtrace and cfg.recurrence != cfg.rollout:
            cfg_error(f"{cfg.recurrence=} must be equal to {cfg.rollout=} when using vtrace.")
    else:
        if cfg.recurrence > 1:
            log.warning(
                f"{cfg.recurrence=} is set but {cfg.use_rnn=} is False. Consider setting {cfg.recurrence=} to 1 for maximum performance."
            )

    return good_config


def cfg_dict(cfg: Config) -> AttrDict:
    if isinstance(cfg, dict):
        return AttrDict(cfg)
    else:
        return AttrDict(vars(cfg))


def cfg_str(cfg: Config) -> str:
    cfg = cfg_dict(cfg)
    cfg_lines = []
    for k, v in cfg.items():
        cfg_lines.append(f"{k}={v}")
    return "\n".join(cfg_lines)


def default_cfg(algo="APPO", env="env", experiment="test"):
    """Useful for tests."""
    argv = [f"--algo={algo}", f"--env={env}", f"--experiment={experiment}"]
    parser, args = parse_sf_args(argv)
    args = parse_full_cfg(parser, argv)
    return args


def load_from_checkpoint(cfg: Config) -> AttrDict:
    cfg_filename = cfg_file(cfg)
    cfg_filename_old = cfg_file_old(cfg)

    if not os.path.isfile(cfg_filename) and os.path.isfile(cfg_filename_old):
        # rename old config file
        log.warning(f"Loading legacy config file {cfg_filename_old} instead of {cfg_filename}")
        os.rename(cfg_filename_old, cfg_filename)

    if not os.path.isfile(cfg_filename):
        raise Exception(
            f"Could not load saved parameters for experiment {cfg.experiment} "
            f"(file {cfg_filename} not found). Check that you have the correct experiment name "
            f"and --train_dir is set correctly."
        )

    with open(cfg_filename, "r") as json_file:
        json_params = json.load(json_file)
        log.warning("Loading existing experiment configuration from %s", cfg_filename)
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


def maybe_load_from_checkpoint(cfg: Config) -> AttrDict:
    """
    Will attempt to load experiment configuration from the checkpoint while preserving any new overrides passed
    from command line.
    """

    filename = cfg_file(cfg)
    if not os.path.isfile(filename):
        log.warning("Saved parameter configuration for experiment %s not found!", cfg.experiment)
        log.warning("Starting experiment from scratch!")
        return AttrDict(vars(cfg))

    return load_from_checkpoint(cfg)


def checkpoint_override_defaults(cfg: Config, parser) -> AttrDict:
    cfg_filename = cfg_file(cfg)

    if not os.path.isfile(cfg_filename):
        raise Exception(
            f"Could not load saved parameters for experiment {cfg.experiment} "
            f"(file {cfg_filename} not found). Check that you have the correct experiment name "
            f"and --train_dir is set correctly."
        )

    with open(cfg_filename, "r") as json_file:
        json_params = json.load(json_file)
        log.warning("Loading existing experiment configuration from %s", cfg_filename)
        loaded_cfg = AttrDict(json_params)

    parser.set_defaults(**loaded_cfg)

    return loaded_cfg
