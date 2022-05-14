from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid
from sample_factory.runner.runs.isaacgym_runs import vstr, base_cli

_params = ParamGrid([
    ('seed', [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]),
    ('serial_mode', [True]),
    ('async_rl', [False]),
])

ant_cli = f' --env=isaacgym_ant --train_for_env_steps=100000000 --wandb_group=isaacgym_ant_sf2 --wandb_tags ant brain {vstr}'
cli = base_cli + ant_cli

_experiments = [
    Experiment(f'ant_{vstr}_async', cli, _params.generate_params(False)),
    Experiment(f'ant_{vstr}_async_rnn', cli + ' --use_rnn=True --recurrence=16', _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription(f'ant_{vstr}', experiments=_experiments)


# Run locally: python -m sample_factory.runner.run --run=sample_factory.runner.runs.isaacgym_ant --runner=processes --max_parallel=2 --experiments_per_gpu=2 --num_gpus=1
# Run on Slurm: python -m sample_factory.runner.run --run=sample_factory.runner.runs.isaacgym_ant --runner=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/runner/slurm/sbatch_template.sh --pause_between=1 --slurm_print_only=False
