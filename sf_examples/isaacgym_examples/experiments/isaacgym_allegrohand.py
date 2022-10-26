from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from sf_examples.isaacgym_examples.experiments.isaacgym_runs import base_cli, vstr

_params = ParamGrid(
    [
        ("seed", [1111, 2222, 3333, 4444]),
        ("serial_mode", [True]),
        ("async_rl", [False]),
        ("normalize_returns", [True, False]),
    ]
)

vstr = f"{vstr}_norm_returns"

ahand_cli = (
    f" --env=AllegroHand --train_for_env_steps=150000000 "
    f"--with_wandb=True --wandb_group=isaacgym_allegrohand_sf2_{vstr} --wandb_tags allegrohand {vstr}"
)
cli = base_cli + ahand_cli

_experiments = [
    Experiment(f"allegrohand_{vstr}", cli, _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription(f"allegrohand_{vstr}", experiments=_experiments)


# Run locally: python -m sample_factory.launcher.run --run=sf_examples.isaacgym_examples.experiments.isaacgym_allegrohand --backend=processes --max_parallel=1 --experiments_per_gpu=1 --num_gpus=1
# Run on Slurm: python -m sample_factory.launcher.run --run=sf_examples.isaacgym_examples.experiments.isaacgym_allegrohand --backend=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/launcher/slurm/sbatch_timeout.sh --pause_between=1 --slurm_print_only=False
