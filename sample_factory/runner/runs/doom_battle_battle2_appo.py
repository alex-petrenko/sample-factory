"""
Run this on Slurm with the following command (assuming you are in sample-factory repo root):
python -m sample_factory.runner.run --run=sample_factory.runner.runs.doom_battle_battle2_appo --runner=slurm --slurm_workdir=./slurm_doom_battle_battle2 --experiment_suffix=slurm --pause_between=1 --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/runner/slurm/sbatch_template.sh --slurm_print_only=False
"""

from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid
from sample_factory.runner.runs.run_utils import seeds

_params = ParamGrid([
    ('env', ['doom_battle', 'doom_battle2']),
    ('seed', seeds(4)),
])

_experiments = [
    Experiment(
        'battle_fs4',
        'python -m sample_factory.algorithms.appo.train_appo --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=16 --num_envs_per_worker=20 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('doom_battle_battle2_appo_v1.121.2', experiments=_experiments)
