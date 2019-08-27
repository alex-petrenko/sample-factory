from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_many import run

_params = ParamGrid([
    ('new_clip', ['True', 'False']),
    ('leaky_ppo', [0.5, 0.1]),
    ('ppo_epochs', [4]),
])

_experiment = Experiment(
    'bots_v19_fs2',
    'python -m train_pytorch --env=doom_dwango5_bots_experimental --train_for_seconds=360000 --algo=PPO --seed=42 --gamma=0.995 --recurrence=1 --use_rnn=False --env_frameskip=2',
    _params.generate_params(randomize=False),
)

gridsearch = RunDescription('doom_battle_bots_v19_fs2', experiments=[_experiment], pause_between_experiments=10, use_gpus=4, experiments_per_gpu=2, max_parallel=4)

run(gridsearch)
