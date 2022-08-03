import sys
from multiprocessing.context import BaseContext
from typing import Optional

from sample_factory.algo.runners.runner import Runner
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.multiprocessing_utils import get_mp_ctx
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.model.model_utils import register_custom_encoder
from sample_factory.train import make_runner
from sample_factory.utils.typing import Config, Env
from sample_factory.utils.utils import experiment_dir
from sf_examples.dmlab_examples.dmlab_env import (
    DMLAB_ENVS,
    dmlab_extra_episodic_stats_processing,
    dmlab_extra_summaries,
    list_all_levels_for_experiment,
    make_dmlab_env,
)
from sf_examples.dmlab_examples.dmlab_level_cache import DmlabLevelCaches, make_dmlab_caches
from sf_examples.dmlab_examples.dmlab_model import DmlabEncoder
from sf_examples.dmlab_examples.dmlab_params import add_dmlab_env_args, dmlab_override_defaults


class DmlabEnvWithCache:
    def __init__(self, level_caches: Optional[DmlabLevelCaches] = None):
        self.caches = level_caches

    def make_env(self, env_name, cfg, env_config) -> Env:
        return make_dmlab_env(env_name, cfg, env_config, self.caches)


def register_dmlab_envs(level_caches: Optional[DmlabLevelCaches] = None):
    env_factory = DmlabEnvWithCache(level_caches)
    for env in DMLAB_ENVS:
        register_env(env.name, env_factory.make_env)


def dmlab_register_models():
    register_custom_encoder("dmlab_instructions", DmlabEncoder)


def register_dmlab_components(level_caches: Optional[DmlabLevelCaches] = None):
    register_dmlab_envs(level_caches)
    dmlab_register_models()


def register_msg_handlers(cfg: Config, runner: Runner):
    if cfg.env == "dmlab_30":
        # extra functions to calculate human-normalized score etc.
        runner.register_episodic_stats_handler(dmlab_extra_episodic_stats_processing)
        runner.register_summary_handler(dmlab_extra_summaries)


def initialize_level_cache(cfg: Config, mp_ctx: BaseContext) -> Optional[DmlabLevelCaches]:
    if not cfg.dmlab_use_level_cache:
        return None

    env_name = cfg.env
    num_policies = cfg.num_policies if hasattr(cfg, "num_policies") else 1
    all_levels = list_all_levels_for_experiment(env_name)
    level_cache_dir = cfg.dmlab_level_cache_path
    caches = make_dmlab_caches(experiment_dir(cfg), all_levels, num_policies, level_cache_dir, mp_ctx)
    return caches


def parse_dmlab_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv, evaluation=evaluation)
    add_dmlab_env_args(parser)
    dmlab_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)
    return cfg


def main():
    """Script entry point."""
    cfg = parse_dmlab_args()

    # explicitly create the runner instead of simply calling run_rl()
    # this allows us to register additional message handlers
    cfg, runner = make_runner(cfg)
    register_msg_handlers(cfg, runner)

    level_caches = initialize_level_cache(cfg, get_mp_ctx(cfg.serial_mode))
    register_dmlab_components(level_caches)

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    return status


if __name__ == "__main__":
    sys.exit(main())
