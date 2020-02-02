import argparse
import math
import time
from collections import deque

import numpy as np
import ray
import yaml
from ray.rllib.agents.impala import ImpalaTrainer
from ray.rllib.agents.ppo import PPOTrainer, APPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tests.cluster_utils import Cluster
from ray.tune import Experiment
from ray.tune.config_parser import make_parser
from ray.tune.registry import register_trainable
# noinspection PyProtectedMember
from ray.tune.tune import _make_scheduler, run
from ray.tune.util import merge_dicts

from benchmarks.rllib.vizdoom_model import VizdoomVisionNetwork
from benchmarks.rllib.custom_appo_policy import CustomAPPOTFPolicy
from benchmarks.rllib.custom_ppo_policy import CustomPPOTFPolicy
from envs.ray_envs import register_doom_envs_rllib, register_dmlab_envs_rllib
from utils.utils import log

EXAMPLE_USAGE = """
Training example via RLlib CLI:
    rllib train --run DQN --env CartPole-v0

Grid search example via RLlib CLI:
    rllib train -f tuned_examples/cartpole-grid-search-example.yaml

Grid search example via executable:
    ./train_rllib.py -f tuned_examples/cartpole-grid-search-example.yaml

Note that -f overrides all other trial-specific command-line options.
"""


def create_parser(parser_creator=None):
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Train a reinforcement learning agent.",
        epilog=EXAMPLE_USAGE)

    # See also the base parser definition in ray/tune/config_parser.py
    parser.add_argument(
        "--redis-address",
        default=None,
        type=str,
        help="Connect to an existing Ray cluster at this address instead "
        "of starting a new one.")
    parser.add_argument(
        "--ray-num-cpus",
        default=None,
        type=int,
        help="--num-cpus to use if starting a new cluster.")
    parser.add_argument(
        "--ray-num-gpus",
        default=None,
        type=int,
        help="--num-gpus to use if starting a new cluster.")
    parser.add_argument(
        "--ray-num-nodes",
        default=None,
        type=int,
        help="Emulate multiple cluster nodes for debugging.")
    parser.add_argument(
        "--ray-redis-max-memory",
        default=None,
        type=int,
        help="--redis-max-memory to use if starting a new cluster.")
    parser.add_argument(
        "--ray-object-store-memory",
        default=None,
        type=int,
        help="--object-store-memory to use if starting a new cluster.")
    parser.add_argument(
        "--experiment-name",
        default="default",
        type=str,
        help="Name of the subdirectory under `local_dir` to put results in.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume previous Tune experiments.")
    parser.add_argument(
        "--env", default=None, type=str, help="The gym environment to use.")
    parser.add_argument(
        "--queue-trials",
        action="store_true",
        help=(
            "Whether to queue trials when the cluster does not currently have "
            "enough resources to launch one. This should be set to True when "
            "running on an autoscaling cluster to enable automatic scale-up."))
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="If specified, use config options from this file. Note that this "
        "overrides any trial-specific options set via flags above.")
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Local mode for debug",
    )
    parser.add_argument(
        "--dbg",
        action="store_true",
        help="Full debug mode (also enables local-mode)",
    )
    parser.add_argument(
        "--stop-seconds",
        default=-1,
        type=int,
        help="Stop experiment after this many seconds",
    )
    parser.add_argument(
        "--pbt",
        action="store_true",
        help="Experimental population-based training",
    )

    parser.add_argument(
        "--cfg-mixins",
        nargs='+',
        help='List of config files to override the default configuration (config "mixins")',
    )

    return parser


def make_custom_scheduler(args):
    # if args.pbt:
    #     return get_pbt_scheduler()
    # else:
    return _make_scheduler(args)


class FpsHelper:
    def __init__(self):
        self.last_num_samples = -1
        self.last_result = time.time()

        history_len = 30
        self.num_samples = deque(maxlen=history_len)
        self.durations = deque(maxlen=history_len)

    def record(self, num_samples):
        now = time.time()

        if self.last_num_samples > 0:
            delta_samples = num_samples - self.last_num_samples
            self.num_samples.append(delta_samples)
            duration = now - self.last_result
            self.durations.append(duration)

        self.last_result = now
        self.last_num_samples = num_samples

    def get_fps(self):
        if len(self.num_samples) > 0:
            fps = np.sum(self.num_samples) / max(float(np.sum(self.durations)), 1e-9)
            return fps
        else:
            return math.nan


def run_experiment(args, parser):
    # args.ray_object_store_memory = int(1e10)
    args.ray_redis_max_memory = int(2e9)

    if args.config_file:
        with open(args.config_file) as f:
            exp = yaml.load(f)
    else:
        raise Exception('No config file!')

    exp = merge_dicts(exp, args.config)
    log.info('Num workers: %d, num_envs_per_worker: %d', exp['config']['num_workers'], exp['config']['num_envs_per_worker'])

    if args.cfg_mixins is not None:
        for cfg_mixin_file in args.cfg_mixins:
            with open(cfg_mixin_file, 'r') as f:
                override_cfg = yaml.load(f)
                log.info('Overriding parameters from %s: %r', cfg_mixin_file, override_cfg)
                exp = merge_dicts(exp, override_cfg)

    if not exp.get("run"):
        parser.error("the following arguments are required: --run")
    if not exp.get("env") and not exp.get("config", {}).get("env"):
        parser.error("the following arguments are required: --env")

    if args.ray_num_nodes:
        cluster = Cluster()
        for _ in range(args.ray_num_nodes):
            cluster.add_node(
                num_cpus=args.ray_num_cpus or 1,
                num_gpus=args.ray_num_gpus or 0,
                object_store_memory=args.ray_object_store_memory,
                redis_max_memory=args.ray_redis_max_memory,
            )
        ray.init(redis_address=cluster.redis_address, local_mode=args.local_mode)
    else:
        ray.init(
            redis_address=args.redis_address,
            object_store_memory=args.ray_object_store_memory,
            redis_max_memory=args.ray_redis_max_memory,
            num_cpus=args.ray_num_cpus,
            num_gpus=args.ray_num_gpus,
            local_mode=args.local_mode,
        )

    exp = Experiment.from_json(args.experiment_name, exp)
    exp.spec['checkpoint_freq'] = 20
    if args.pbt:
        exp.spec['checkpoint_freq'] = 3

    exp.spec['checkpoint_at_end'] = True
    # exp.spec['checkpoint_score_attr'] = 'episode_reward_mean'
    exp.spec['keep_checkpoints_num'] = 5

    if args.stop_seconds > 0:
        exp.spec['stop'] = {'time_total_s': args.stop_seconds}

    # if 'multiagent' in exp.spec['config']:
    #     # noinspection PyProtectedMember
    #     make_env = ray.tune.registry._global_registry.get(ENV_CREATOR, exp.spec['config']['env'])
    #     temp_env = make_env(None)
    #     obs_space, action_space = temp_env.observation_space, temp_env.action_space
    #     temp_env.close()
    #     del temp_env
    #
    #     policies = dict(
    #         main=(None, obs_space, action_space, {}),
    #         dummy=(None, obs_space, action_space, {}),
    #     )
    #
    #     exp.spec['config']['multiagent'] = {
    #         'policies': policies,
    #         'policy_mapping_fn': function(lambda agent_id: 'main'),
    #         'policies_to_train': ['main'],
    #     }
    #
    # if args.dbg:
    #     exp.spec['config']['num_workers'] = 1
    #     exp.spec['config']['num_gpus'] = 1
    #     exp.spec['config']['num_envs_per_worker'] = 1
    #
    # if 'callbacks' not in exp.spec['config']:
    #     exp.spec['config']['callbacks'] = {}
    #
    # fps_helper = FpsHelper()
    #
    # def on_train_result(info):
    #     if 'APPO' in exp.spec['run']:
    #         samples = info['result']['info']['num_steps_sampled']
    #     else:
    #         samples = info['trainer'].optimizer.num_steps_trained
    #
    #     fps_helper.record(samples)
    #     fps = fps_helper.get_fps()
    #     info['result']['custom_metrics']['fps'] = fps
    #
    #     # remove this as currently
    #     skip_frames = exp.spec['config']['env_config']['skip_frames']
    #     info['result']['custom_metrics']['fps_frameskip'] = fps * skip_frames
    #
    # exp.spec['config']['callbacks']['on_train_result'] = function(on_train_result)
    #
    # def on_episode_end(info):
    #     episode = info['episode']
    #     stats = {
    #         'DEATHCOUNT': 0,
    #         'FRAGCOUNT': 0,
    #         'HITCOUNT': 0,
    #         'DAMAGECOUNT': 0,
    #         'KDR': 0,
    #         'FINAL_PLACE': 0,
    #         'LEADER_GAP': 0,
    #         'PLAYER_COUNT': 0,
    #         'BOT_DIFFICULTY': 0,
    #     }
    #
    #     # noinspection PyProtectedMember
    #     agent_to_last_info = episode._agent_to_last_info
    #     for agent in agent_to_last_info.keys():
    #         agent_info = agent_to_last_info[agent]
    #         for stats_key in stats.keys():
    #             stats[stats_key] += agent_info.get(stats_key, 0.0)
    #
    #     for stats_key in stats.keys():
    #         stats[stats_key] /= len(agent_to_last_info.keys())
    #
    #     episode.custom_metrics.update(stats)
    #
    # exp.spec['config']['callbacks']['on_episode_end'] = function(on_episode_end)

    extra_kwargs = {}
    if args.pbt:
        extra_kwargs['reuse_actors'] = False

    run(
        exp,
        name=args.experiment_name,
        scheduler=make_custom_scheduler(args),
        resume=args.resume,
        queue_trials=args.queue_trials,
        **extra_kwargs
    )


def main():
    register_doom_envs_rllib()
    register_dmlab_envs_rllib()

    ModelCatalog.register_custom_model('vizdoom_vision_model', VizdoomVisionNetwork)

    def custom_ppo():
        return PPOTrainer.with_updates(default_policy=CustomPPOTFPolicy)

    def custom_appo():
        return APPOTrainer.with_updates(
            default_policy=CustomAPPOTFPolicy,
            get_policy_class=lambda _: CustomAPPOTFPolicy,
        )

    register_trainable('CUSTOM_PPO', custom_ppo())
    register_trainable('CUSTOM_APPO', custom_appo())

    parser = create_parser()
    args = parser.parse_args()
    run_experiment(args, parser)


if __name__ == '__main__':
    main()
