from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid
from sample_factory.runner.runs.isaacgym_runs import vstr, base_cli

_params = ParamGrid([
    ('seed', [1111, 2222, 3333, 4444]),
    ('serial_mode', [True]),
    ('async_rl', [False]),
])

ahand_cli = f' --env=isaacgym_AllegroHandLSTM --train_for_env_steps=10000000000 --with_wandb=True --wandb_group=isaacgym_allegrohand_sf2_{vstr} --wandb_project=rlgpu-2022 --wandb_tags allegrohand {vstr}'
cli = base_cli + ahand_cli

_experiments = [
    Experiment(f'allegrohand_{vstr}', cli, _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription(f'ant_{vstr}', experiments=_experiments)


# Run locally: python -m sample_factory.runner.run --run=sample_factory.runner.runs.isaacgym_allegrohand --runner=processes --max_parallel=1 --experiments_per_gpu=1 --num_gpus=1
# Run on Slurm: python -m sample_factory.runner.run --run=sample_factory.runner.runs.isaacgym_allegrohand --runner=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/runner/slurm/sbatch_template.sh --pause_between=1 --slurm_print_only=False
