import sys

from sample_factory.algo.utils.context import global_env_registry
from sample_factory.cfg.arguments import parse_args
from sample_factory.train import run_rl
from sample_factory_examples.dmlab.dmlab_model import dmlab_register_models


def register_dmlab_components():
    from sample_factory_examples.dmlab.dmlab_env import make_dmlab_env
    from sample_factory_examples.dmlab.dmlab_params import add_dmlab_env_args, dmlab_override_defaults

    global_env_registry().register_env(
        env_name_prefix="dmlab_",
        make_env_func=make_dmlab_env,
        add_extra_params_func=add_dmlab_env_args,
        override_default_params_func=dmlab_override_defaults,
    )

    dmlab_register_models()


def main():
    """Script entry point."""
    register_dmlab_components()
    cfg = parse_args(
        argv=[
            "--env=dmlab_30",
            "--dmlab30_dataset=~/datasets/dmlab/brady_konkle_oliva2008",
            "--dmlab_use_level_cache=True",
            "--dmlab_level_cache_path=~/datasets/.dmlab_cache",
            "--gamma=0.99",
            "--use_rnn=True",
            "--num_workers=4",
            "--num_envs_per_worker=4",
            "--num_epochs=1",
            "--rollout=32",
            "--recurrence=32",
            "--batch_size=2048",
            "--benchmark=False",
            "--max_grad_norm=0.0",
            "--dmlab_renderer=software",
            "--decorrelate_experience_max_seconds=120",
            "--encoder_custom=dmlab_instructions",
            "--encoder_type=resnet",
            "--encoder_subtype=resnet_impala",
            "--encoder_extra_fc_layers=1",
            "--hidden_size=256",
            "--nonlinearity=relu",
            "--rnn_type=lstm",
            "--dmlab_extended_action_set=True",
            "--num_policies=1",
            "--set_workers_cpu_affinity=True",
            "--max_policy_lag=35",
            "--experiment=dmlab123",
        ]
    )
    cfg.num_workers = 4
    cfg.num_envs_per_worker = 2

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
