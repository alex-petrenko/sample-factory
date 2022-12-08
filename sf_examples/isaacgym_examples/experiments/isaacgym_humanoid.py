from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from sf_examples.isaacgym_examples.experiments.isaacgym_runs import base_cli, vstr

_params = ParamGrid(
    [
        ("seed", [1111, 2222, 3333, 4444, 5555]),
        (("serial_mode", "async_rl"), ([True, False], [False, True])),
        # ("value_bootstrap", [True, False]),
        (("use_rnn", "recurrence"), ([False, 1], [True, 32])),
    ]
)

vstr = f"humanoid_{vstr}"

humanoid_cli = f" --env=Humanoid --train_for_env_steps=100000000 --with_wandb=True --wandb_tags humanoid {vstr}"
cli = base_cli + humanoid_cli

_experiments = [
    Experiment(vstr, cli + f" --wandb_group=isaacgym_humanoid_sf2_{vstr}", _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription(vstr, experiments=_experiments)


# Run locally: python -m sample_factory.launcher.run --run=sf_examples.isaacgym_examples.experiments.isaacgym_humanoid --backend=processes --max_parallel=2 --experiments_per_gpu=2 --num_gpus=1
# Run on Slurm: python -m sample_factory.launcher.run --run=sf_examples.isaacgym_examples.experiments.isaacgym_humanoid --backend=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/launcher/slurm/sbatch_timeout.sh --pause_between=1 --slurm_print_only=False
