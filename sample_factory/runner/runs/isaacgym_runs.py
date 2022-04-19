from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

version = 37
vstr = f'v{version:03d}'

base_cli = f'python -m sample_factory_examples.train_isaacgym ' \
           f'--algo=APPO --actor_worker_gpus 0 --env_agents=4096 ' \
           f'--batch_size=32768 --env_headless=True --with_vtrace=False --use_rnn=False --recurrence=1 --with_wandb=True'

_experiments = [
    Experiment(f'ant_{_vstr}_async', _cli, _params.generate_params(False)),
    Experiment(f'ant_{_vstr}_async_rnn', _cli + ' --use_rnn=True --recurrence=16', _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription(f'ant_{_vstr}', experiments=_experiments)


# Run locally: python -m sample_factory.runner.run --run=sample_factory.runner.runs.isaacgym_ant --runner=processes --max_parallel=2 --experiments_per_gpu=2 --num_gpus=1
# Run on Slurm: python -m sample_factory.runner.run --run=sample_factory.runner.runs.isaacgym_ant --runner=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/runner/slurm/sbatch_template.sh --pause_between=1 --slurm_print_only=False


