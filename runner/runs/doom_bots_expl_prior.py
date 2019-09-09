from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_processes import run

_params_no_prior = ParamGrid([
    ('env', ['doom_dwango5_bots_experimental', 'doom_dwango5_bots_expl_reward']),
    ('seed', [42, 43, 44]),
])

_experiment_no_prior = Experiment(
    'bots_fs2_no_prior',
    'python -m train_pytorch --train_for_seconds=360000 --algo=PPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --rollout=64 --num_envs=96 --reward_scale=0.5 --recurrence=32 --ppo_epochs=1',
    _params_no_prior.generate_params(randomize=False),
)

_params_prior = ParamGrid([
    ('env', ['doom_dwango5_bots_experimental']),
    ('seed', [42, 43, 44]),
])

_experiment_prior = Experiment(
    'bots_fs2_prior',
    'python -m train_pytorch --train_for_seconds=360000 --algo=PPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --rollout=64 --num_envs=96 --reward_scale=0.5 --recurrence=32 --ppo_epochs=1 --learned_prior=./train_dir/doom_bots_exploration_v2/checkpoint',
    _params_prior.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('doom_bots_v36_fs2_prior', experiments=[_experiment_no_prior, _experiment_prior], pause_between_experiments=10, use_gpus=6, experiments_per_gpu=2, max_parallel=12)
