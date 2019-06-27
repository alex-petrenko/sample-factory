import argparse

import ray
import yaml
from ray.rllib.models import ModelCatalog
from ray.tests.cluster_utils import Cluster
from ray.tune import Experiment, function
from ray.tune.config_parser import make_parser
from ray.tune.registry import ENV_CREATOR
# noinspection PyProtectedMember
from ray.tune.tune import _make_scheduler, run

from algorithms.models.vizdoom_model import VizdoomVisionNetwork
from envs.doom.doom_utils import register_doom_envs_rllib

EXAMPLE_USAGE = """
Training example via RLlib CLI:
    rllib train --run DQN --env CartPole-v0

Grid search example via RLlib CLI:
    rllib train -f tuned_examples/cartpole-grid-search-example.yaml

Grid search example via executable:
    ./train.py -f tuned_examples/cartpole-grid-search-example.yaml

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
        default=int(1e9),
        type=int,
        help="Stop experiment after this many seconds",
    )

    return parser


def run_experiment(args, parser):
    # args.ray_object_store_memory = int(1e10)
    args.ray_redis_max_memory = int(5e9)

    if args.dbg:
        args.local_mode = True

    if args.config_file:
        with open(args.config_file) as f:
            exp = yaml.load(f)
    else:
        raise Exception('No config file!')

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
    exp.spec['checkpoint_at_end'] = True
    exp.spec['keep_checkpoints_num'] = 3

    exp.spec['stop'] = {'time_total_s': args.stop_seconds}

    if 'multiagent' in exp.spec['config']:
        # noinspection PyProtectedMember
        make_env = ray.tune.registry._global_registry.get(ENV_CREATOR, exp.spec['config']['env'])
        temp_env = make_env(None)
        obs_space, action_space = temp_env.observation_space, temp_env.action_space
        temp_env.close()
        del temp_env

        policies = dict(
            main=(None, obs_space, action_space, {}),
            dummy=(None, obs_space, action_space, {}),
        )

        exp.spec['config']['multiagent'] = {
            'policies': policies,
            'policy_mapping_fn': function(lambda agent_id: 'main'),
            'policies_to_train': ['main'],
        }

    if args.dbg:
        exp.spec['config']['num_workers'] = 1
        exp.spec['config']['num_gpus'] = 1
        exp.spec['config']['num_envs_per_worker'] = 1

    run(
        exp,
        name=args.experiment_name,
        scheduler=_make_scheduler(args),
        resume=args.resume,
        queue_trials=args.queue_trials,
    )


def main():
    register_doom_envs_rllib()

    ModelCatalog.register_custom_model('vizdoom_vision_model', VizdoomVisionNetwork)

    parser = create_parser()
    args = parser.parse_args()
    run_experiment(args, parser)


if __name__ == '__main__':
    main()
