from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]),
    ('serial_mode', [True, False]),
    ('async_rl', [True, False]),
])

_version = 35
_vstr = f'v{_version:03d}'
_cli = f'python -m sample_factory_examples.train_isaacgym ' \
       f'--algo=APPO --env=isaacgym_ant --actor_worker_gpus 0 --train_for_env_steps=100000000 --env_agents=4096 ' \
       f'--batch_size=32768 --env_headless=True --use_rnn=False --with_vtrace=False --recurrence=1 --with_wandb=True ' \
       f'--wandb_group=isaacgym_ant_sf2 --wandb_tags {_vstr}'

_experiments = [
    Experiment(f'ant_{_vstr}_async', _cli, _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription(f'ant_{_vstr}', experiments=_experiments)


# Run locally: python -m sample_factory.runner.run --run=sample_factory.runner.runs.isaacgym_ant --runner=processes --max_parallel=2 --experiments_per_gpu=2 --num_gpus=1
# Run on Slurm: python -m sample_factory.runner.run --run=sample_factory.runner.runs.isaacgym_ant --runner=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/runner/slurm/sbatch_template.sh --pause_between=1 --slurm_print_only=False


