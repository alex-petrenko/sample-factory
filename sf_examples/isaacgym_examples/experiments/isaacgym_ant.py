from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from sf_examples.isaacgym_examples.experiments.isaacgym_runs import base_cli, vstr

_params = ParamGrid(
    [
        ("seed", [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]),
        (("serial_mode", "async_rl"), ([True, False], [False, True])),
        ("value_bootstrap", [True, False]),
        # (("use_rnn", "recurrence"), ([False, 1], [True, 16])),
    ]
)

vstr = f"{vstr}_vb"

ant_cli = f" --env=Ant --train_for_env_steps=100000000 --with_wandb=True --wandb_tags ant {vstr}"
cli = base_cli + ant_cli

_experiments = [
    Experiment(f"ant_{vstr}", cli + f" --wandb_group=isaacgym_ant_sf2_{vstr}", _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription(f"ant_{vstr}", experiments=_experiments)


# Run locally: python -m sample_factory.launcher.run --run=sf_examples.isaacgym_examples.experiments.isaacgym_ant --backend=processes --max_parallel=2 --experiments_per_gpu=2 --num_gpus=1
# Run on Slurm: python -m sample_factory.launcher.run --run=sf_examples.isaacgym_examples.experiments.isaacgym_ant --backend=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/launcher/slurm/sbatch_timeout.sh --pause_between=1 --slurm_print_only=False
