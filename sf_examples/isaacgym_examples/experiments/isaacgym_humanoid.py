from sample_factory.runner.run_description import Experiment, ParamGrid, RunDescription
from sf_examples.isaacgym_examples.experiments.isaacgym_runs import base_cli, vstr

_params = ParamGrid(
    [
        ("seed", [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]),
        ("serial_mode", [True]),
        ("async_rl", [False]),
        ("normalize_returns", [True, False]),
    ]
)

vstr = f"{vstr}_norm_returns_v2"

humanoid_cli = (
    f" --env=Humanoid --train_for_env_steps=131000000 "
    f"--mlp_layers 400 200 100 --max_grad_norm=1.0 "
    f"--rollout=32 --num_epochs=5 --value_loss_coeff=2.0 "
    f"--with_wandb=True --wandb_tags humanoid brain {vstr}"
)

cli = base_cli + humanoid_cli

_experiments = [
    Experiment(
        f"humanoid_{vstr}", cli + f" --wandb_group=isaacgym_humanoid_sf2_{vstr}", _params.generate_params(False)
    ),
    Experiment(
        f"humanoid_{vstr}_rnn",
        cli + f" --use_rnn=True --recurrence=32 --wandb_group=isaacgym_humanoid_sf2_rnn_{vstr}",
        _params.generate_params(False),
    ),
]

RUN_DESCRIPTION = RunDescription(f"humanoid_{vstr}", experiments=_experiments)


# Run locally: python -m sample_factory.runner.run --run=sample_factory.runner.runs.isaacgym_humanoid --runner=processes --max_parallel=2 --experiments_per_gpu=2 --num_gpus=1
# Run on Slurm: python -m sample_factory.runner.run --run=sample_factory.runner.runs.isaacgym_humanoid --runner=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/runner/slurm/sbatch_template.sh --pause_between=1 --slurm_print_only=False
