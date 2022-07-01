# Running Experiments

Here we provide command lines that can be used to reproduce the experiments from the paper, which also serve as an example on how to configure large-scale RL experiments.

#### VizDoom

```
Train for 4B env steps (also can be stopped at any time with Ctrl+C and resumed by using the same cmd).
This is more or less optimal training setup for a 10-core machine.
python -m sample_factory.algorithms.appo.train_appo --env=doom_battle --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=20 --num_policies=1  --experiment=doom_battle_w20_v20

Run at any point to visualize the experiment:
python -m sample_factory.algorithms.appo.enjoy_appo --env=doom_battle --algo=APPO --experiment=doom_battle_w20_v20
```

```
Train on one of the 6 "basic" VizDoom environments:
python -m sample_factory.algorithms.appo.train_appo --train_for_env_steps=500000000 --algo=APPO --env=doom_my_way_home --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False --experiment=doom_basic_envs
```

```
Doom "battle" and "battle2" environments, 36-core server (72 logical cores) with 4 GPUs:
python -m sample_factory.algorithms.appo.train_appo --env=doom_battle --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=72 --num_envs_per_worker=8 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0 --experiment=doom_battle
python -m sample_factory.algorithms.appo.train_appo --env=doom_battle2 --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=72 --num_envs_per_worker=8 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0 --experiment=doom_battle_2
```

```
Duel and deathmatch versus bots, population-based training, 36-core server:
python -m sample_factory.algorithms.appo.train_appo --env=doom_duel_bots --train_for_seconds=360000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --reward_scale=0.5 --num_workers=72 --num_envs_per_worker=32 --num_policies=8 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --benchmark=False --res_w=128 --res_h=72 --wide_aspect_ratio=False --pbt_replace_reward_gap=0.2 --pbt_replace_reward_gap_absolute=3.0 --pbt_period_env_steps=5000000 --save_milestones_sec=1800 --with_pbt=True --experiment=doom_duel_bots
python -m sample_factory.algorithms.appo.train_appo --env=doom_deathmatch_bots --train_for_seconds=3600000 --algo=APPO --use_rnn=True --gamma=0.995 --env_frameskip=2 --rollout=32 --num_workers=80 --num_envs_per_worker=24 --num_policies=8 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=False --with_pbt=True --pbt_period_env_steps=5000000 --experiment=doom_deathmatch_bots
```

```
Duel and deathmatch self-play, PBT, 36-core server:
python -m sample_factory.algorithms.appo.train_appo --env=doom_duel --train_for_seconds=360000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --num_workers=72 --num_envs_per_worker=16 --num_policies=8 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=False --benchmark=False --pbt_replace_reward_gap=0.5 --pbt_replace_reward_gap_absolute=0.35 --pbt_period_env_steps=5000000 --with_pbt=True --pbt_start_mutation=100000000 --experiment=doom_duel_full
python -m sample_factory.algorithms.appo.train_appo --env=doom_deathmatch_full --train_for_seconds=360000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --num_workers=72 --num_envs_per_worker=16 --num_policies=8 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=False --benchmark=False --pbt_replace_reward_gap=0.1 --pbt_replace_reward_gap_absolute=0.1 --pbt_period_env_steps=5000000 --with_pbt=True --pbt_start_mutation=100000000 --experiment=doom_deathmatch_full
```

Reproducing benchmarking results:

```
This achieves 50K+ framerate on a 10-core machine (Intel Core i9-7900X):
python -m sample_factory.algorithms.appo.train_appo --env=doom_benchmark --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=20 --num_envs_per_worker=32 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=4096 --experiment=doom_battle_appo_fps_20_32 --res_w=128 --res_h=72 --wide_aspect_ratio=False --policy_workers_per_policy=2 --worker_num_splits=2
```

```
This achieves 100K+ framerate on a 36-core machine:
python -m sample_factory.algorithms.appo.train_appo --env=doom_benchmark --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=72 --num_envs_per_worker=24 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=8192 --wide_aspect_ratio=False --experiment=doom_battle_appo_w72_v24 --policy_workers_per_policy=2
```

#### DMLab

DMLab-30 run on a 36-core server with 4 GPUs:

```
python -m sample_factory.algorithms.appo.train_appo --env=dmlab_30 --train_for_seconds=3600000 --algo=APPO --gamma=0.99 --use_rnn=True --num_workers=90 --num_envs_per_worker=12 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --benchmark=False --ppo_epochs=1 --max_grad_norm=0.0 --dmlab_renderer=software --decorrelate_experience_max_seconds=120 --reset_timeout_seconds=300 --encoder_custom=dmlab_instructions --encoder_type=resnet --encoder_subtype=resnet_impala --encoder_extra_fc_layers=1 --hidden_size=256 --nonlinearity=relu --rnn_type=lstm --dmlab_extended_action_set=True --num_policies=4 --pbt_replace_reward_gap=0.05 --pbt_replace_reward_gap_absolute=5.0 --pbt_period_env_steps=10000000 --pbt_start_mutation=100000000 --with_pbt=True --experiment=dmlab_30_resnet_4pbt_w90_v12 --dmlab_one_task_per_worker=True --set_workers_cpu_affinity=True --max_policy_lag=35 --pbt_target_objective=dmlab_target_objective --dmlab30_dataset=~/datasets/brady_konkle_oliva2008 --dmlab_use_level_cache=True --dmlab_level_cache_path=/home/user/dmlab_cache
```

##### DMLab level cache

Note `--dmlab_level_cache_path` parameter. This location will be used for level layout cache.
Subsequent DMLab experiments on envs that require level generation will become faster since environment files from
previous runs can be reused.

Generating environment levels for the first time can be really slow, especially for the full multi-task
benchmark like DMLab-30. On 36-core server generating enough environments for a 10B training session can take up to
a week. We provide a dataset of pre-generated levels to make training on DMLab-30 easier.
[Download here](https://drive.google.com/file/d/17JCp3DbuiqcfO9I_yLjbBP4a7N7Q4c2v/view?usp=sharing).

### Monitoring training sessions

Sample Factory uses Tensorboard summaries. Run Tensorboard to monitor your experiment: `tensorboard --logdir=train_dir --port=6006`

Additionally, we provide a helper script that has nice command line interface to monitor the experiment folders 
using wildcard masks: `python -m sample_factory.tb '*custom_experiment*' '*another*custom*experiment_name'`

#### WandB support

Sample Factory also supports experiment monitoring with Weights and Biases.
In order to setup WandB locally run `wandb login` in the terminal (https://docs.wandb.ai/quickstart#1.-set-up-wandb)

Example command line to run an experiment with WandB monitoring:

```
python -m sample_factory.algorithms.appo.train_appo --env=doom_basic --algo=APPO --train_for_env_steps=30000000 --num_workers=20 --num_envs_per_worker=20 --experiment=doom_basic --with_wandb=True --wandb_user=<your_wandb_user> --wandb_tags test benchmark doom appo
```

A total list of WandB settings: 
```
--with_wandb: Enables Weights and Biases integration (default: False)
--wandb_user: WandB username (entity). Must be specified from command line! Also see https://docs.wandb.ai/quickstart#1.-set-up-wandb (default: None)
--wandb_project: WandB "Project" (default: sample_factory)
--wandb_group: WandB "Group" (to group your experiments). By default this is the name of the env. (default: None)
--wandb_job_type: WandB job type (default: SF)
--wandb_tags: [WANDB_TAGS [WANDB_TAGS ...]] Tags can help with finding experiments in WandB web console (default: [])
```

Once the experiment is started the link to the monitored session is going to be available in the logs (or by searching in Wandb Web console).


### Runner interface

Sample Factory provides a simple interface that allows users to run experiments with multiple seeds
(or hyperparameter searches) with optimal distribution of work across GPUs.
The configuration of such experiments is done through Python scripts.

Here's an example runner script that we used to train agents for 6 basic VizDoom environments with 10 seeds each:

```
from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [0, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]),
    ('env', ['doom_my_way_home', 'doom_deadly_corridor', 'doom_defend_the_center', 'doom_defend_the_line', 'doom_health_gathering', 'doom_health_gathering_supreme']),
])

_experiments = [
    Experiment(
        'basic_envs_fs4',
        'python -m sample_factory.algorithms.appo.train_appo --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False',
        _params.generate_params(randomize=False),
    ),
]


RUN_DESCRIPTION = RunDescription('doom_basic_envs_appo', experiments=_experiments)

```

Runner script should be importable (i.e. be in your project or in PYTHONPATH), and should define a single variable
`RUN_DESCRIPTION`, which contains a list of experiments (each experiment can be a hyperparameter search), as well as some auxiliary parameters.

When such a script is saved i.e. at `myproject/train_10_seeds.py` in your project using Sample Factory, you can use this command to
execute it:

```
python -m sample_factory.runner.run --run=myproject.train_10_seeds --runner=processes --max_parallel=12 --pause_between=10 --experiments_per_gpu=3 --num_gpus=4
``` 

This will cycle through the requested configurations, training 12 experiments at the same time, 3 per GPU on 4 GPUs using local OS-level parallelism.

Runner supports other backends for parallel execution: `--runner=slurm` and `--runner=ngc` for Slurm and NGC support respectively.

Individual experiments will be stored in `train_dir/run_name` so the whole experiment can be easily monitored
with a single Tensorboard command.

Find more information on runner API in [runner/README.md](https://github.com/alex-petrenko/sample-factory/blob/master/sample_factory/runner/README.md).

### Dummy sampler

This tool can be useful if you want to estimate the upper bound on performance of any reinforcement learning
algorithm, i.e. how fast the environment can be sampled by a dumb random policy.

```
This achieves 90000+ FPS on a 10-core workstation:
python -m sample_factory.run_algorithm --algo=DUMMY_SAMPLER --env=doom_benchmark --num_workers=20 --num_envs_per_worker=1 --experiment=dummy_sampler --sample_env_frames=5000000

```

### Tests

To run unit tests execute `./all_tests.sh` from the root of the repo.
Consider installing VizDoom for a more comprehensive set of tests.

### Trained policies

See a separate [trained_policies/README.md](https://github.com/alex-petrenko/sample-factory/blob/master/trained_policies/README.md).

### Caveats

- Multiplayer VizDoom environments can freeze your console sometimes, simple `reset` takes care of this
- Sometimes VizDoom instances don't clear their internal shared memory buffers used to communicate between Python and
a Doom executable. The file descriptors for these buffers tend to pile up. `rm /dev/shm/ViZDoom*` will take care of this issue.
- It's best to use the standard `--fps=35` to visualize VizDoom results. `--fps=0` enables
Async execution mode for the Doom environments, although the results are not always reproducible between sync and async modes.
- Multiplayer VizDoom environments are significantly slower than single-player envs because actual network
communication between the environment instances is required which results in a lot of syscalls.
For prototyping and testing consider single-player environments with bots instead.
- Vectors of environments on rollout (actor) workers are instantiated on the same CPU thread.
This can create problems for certain types of environment that require global per-thread or per-process context
(e.g. OpenGL context). The solution should be an environment wrapper that starts the environment in a 
separate thread (or process if that's required) and communicates. `doom_multiagent_wrapper.py` is an example,
although not optimal.
