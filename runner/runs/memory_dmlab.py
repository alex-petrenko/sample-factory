from runner.run_description import RunDescription, Experiment, ParamGrid
from runner.run_processes import run

_params = ParamGrid([
    ('env', ['dmlab_nonmatch', 'dmlab_watermaze']),
    ('recurrence', [64]),
    ('use_rnn', [True, False]),
    ('ppo_epochs', [4]),
    ('mem_size', [4, 0]),
])

# IMPORTANT: for DMLAB number of workers better be equal to the number of envs, otherwise spurious crashes may occur!
_experiment = Experiment(
    'mem_dmlab_v21',
    'python -m train_pytorch --algo=PPO --rollout=64 --num_envs=64 --num_workers=64 --train_for_env_steps=1000000000 --normalize_advantage=False --prior_loss_coeff=0.01',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('mem_dmlab_v21', experiments=[_experiment])
