from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid
from sample_factory.runner.runs.isaacgym_runs import vstr, base_cli

_params = ParamGrid([
    ('seed', [1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]),
    ('serial_mode', [True]),
    ('async_rl', [False]),
])

humanoid_cli = f' --env=isaacgym_humanoid --train_for_env_steps=131000000 ' \
               f'--mlp_layers 400 200 100 --max_grad_norm=1.0 ' \
               f'--rollout=32 --ppo_epochs=5 --value_loss_coeff=4.0 ' \
               f'--wandb_group=isaacgym_humanoid_sf2 --wandb_tags humanoid brain {vstr}'

cli = base_cli + humanoid_cli

_experiments = [
    Experiment(f'humanoid_{vstr}', cli, _params.generate_params(False)),
    Experiment(f'humanoid_{vstr}_rnn', cli + ' --use_rnn=True --recurrence=32', _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription(f'humanoid_{vstr}', experiments=_experiments)


# Run locally: python -m sample_factory.runner.run --run=sample_factory.runner.runs.isaacgym_humanoid --runner=processes --max_parallel=2 --experiments_per_gpu=2 --num_gpus=1
# Run on Slurm: python -m sample_factory.runner.run --run=sample_factory.runner.runs.isaacgym_humanoid --runner=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/runner/slurm/sbatch_template.sh --pause_between=1 --slurm_print_only=False


