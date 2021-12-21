[![Build Status](https://travis-ci.com/alex-petrenko/sample-factory.svg?branch=master)](https://travis-ci.com/github/alex-petrenko/sample-factory)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/alex-petrenko/sample-factory/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/sample-factory)](https://pepy.tech/project/sample-factory)
<!-- [![codecov](https://codecov.io/gh/alex-petrenko/sample-factory/branch/master/graph/badge.svg)](https://codecov.io/gh/alex-petrenko/sample-factory) -->

# Sample Factory

Codebase for high throughput asynchronous reinforcement learning.

**Paper:** https://arxiv.org/abs/2006.11751

**Cite:** [BibTeX](https://github.com/alex-petrenko/sample-factory#citation)

**Talk:** https://youtu.be/lLG17LKKSZc

**Videos:** https://sites.google.com/view/sample-factory

VizDoom agents trained with Sample Factory playing in real time:

<p align="middle">
<img src="https://github.com/alex-petrenko/sample-factory/blob/master/gifs/battle.gif?raw=true" width="400">
<img src="https://github.com/alex-petrenko/sample-factory/blob/master/gifs/duel.gif?raw=true" width="400">
</p> 

#### When should I use Sample Factory?

1. Sample Factory is the fastest open source single-machine RL implementations (see paper for details).
If you plan to train RL agents on large amounts of experience, consider using it.
Sample Factory can significantly speed up
the experimentation or allow you to collect more samples in the same amount of time and achieve better performance.

2. Consider using Sample Factory for your multi-agent and population-based training experiments. 
Multi-agent and PBT setups are really simple with Sample Factory.

3. A lot of work went into our VizDoom and DMLab wrappers. For example, we include full support for
configurable VizDoom multi-agent environments and their interop with RL algorithms, which can open new interesting research directions.
Consider using Sample Factory if you train agents in these environments.

4. Sample Factory can be a good choice as a prototype for a single node in a distributed RL system or as a reference
codebase for other types of async RL algorithms.

## Recent releases

##### v1.121.3
* Fixed a small bug related to population-based training (a reward shaping dictionary was assumed to be a flat dict,
while it could be a nested dict in some envs)

##### v1.121.2
* Fixed a bug that prevented Vizdoom *.cfg and *.wad files from being copied to site-packages during installation from PyPI
* Added example on how to use custom Vizdoom envs without modifying the source code (`sample_factory_examples/train_custom_vizdoom_env.py`)

##### v1.121.0
* Added fixed KL divergence penalty as in https://arxiv.org/pdf/1707.06347.pdf 
Its usage is highly encouraged in environments with continuous action spaces (i.e. set --kl_loss_coeff=1.0).
Otherwise numerical instabilities can occur in certain environments, especially when the policy lag is high

* More summaries related to the new loss

##### v1.120.2
* More improvements and fixes in runner interface, including support for NGC cluster

##### v1.120.1
* Runner interface improvements for Slurm

##### v1.120.0
* Support inactive agents. Do deactivate an agent for a portion of the episode the environment should return `info={'is_active': False}` for the inactive agent. Useful for environments such as hide-n-seek.
* Improved memory consumption and performance with better shared memory management.
* Experiment logs are now saved into the experiment folder as `sf_log.txt`
* DMLab-related bug fixes (courtesy of @donghoonlee04 and @sungwoong. Thank you!)

## Installation

Just install from PyPI:

```pip install sample-factory```

#### Advanced installation

PyPI dependency resolution may result in suboptimal performance, i.e. some versions of MKL and Numpy are known to be slower.
To guarantee the maximum throughput (~10% faster than pip version) consider using our Conda environment with exact package versions:

- Clone the repo: `git clone https://github.com/alex-petrenko/sample-factory.git`

- Create and activate conda env:

```
cd sample-factory
conda env create -f environment.yml
conda activate sample-factory
```

SF is known to also work on macOS. There is no Windows support at this time.

### Environment support

Sample Factory has a runtime environment registry for _families of environments_. A family of environments
is defined by a name prefix (i.e. `atari_` or `doom_`) and a function that creates an instance of the environment
given its full name, including the prefix (i.e. `atari_breakout`).

Registering families of environments allows the user to add
and override configuration parameters (such as resolution, frameskip, default model type, etc.) for the whole family
of environments, i.e. all VizDoom envs can share their basic configuration parameters that don't need to be specified for each experiment.

Custom user-defined environment families and models can be added to the registry, see this example:
`sample_factory_examples/train_custom_env_custom_model.py`

Script `sample_factory_examples/train_gym_env.py` demonstrates how Sample Factory can be used with an environment defined in OpenAI Gym.

Sample Factory comes with a particularly comprehensive support for VizDoom and DMLab, see below.

#### VizDoom

To install VizDoom just follow system setup instructions from the original repository ([VizDoom linux_deps](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#linux_deps)),
after which the latest VizDoom can be installed from PyPI: ```pip install vizdoom```.
Version 1.1.9 or above is recommended as it fixes bugs related to multi-agent training.

#### DMLab
 
- Follow installation instructions from [DMLab Github](https://github.com/deepmind/lab/blob/master/docs/users/build.md).
- `pip install dm_env`
- To train on DMLab-30 you will need `brady_konkle_oliva2008` [dataset](https://github.com/deepmind/lab/tree/master/data/brady_konkle_oliva2008).
- To significantly speed up training on DMLab-30 consider downloading our [dataset](https://drive.google.com/file/d/17JCp3DbuiqcfO9I_yLjbBP4a7N7Q4c2v/view?usp=sharing)
of pre-generated environment layouts (see paper for details).
Command lines for running experiments with these datasets are provided in the sections below.
 
#### Atari
 
ALE envs are supported out-of-the-box, although the existing wrappers and hyperparameters
aren't well optimized for sample efficiency in Atari. Tuned Atari training examples would be a welcome contribution.
 
#### Custom multi-agent environments

Multi-agent environments are expected to return lists of observations/dones/rewards (one item for every agent).

It is expected that a multi-agent env exposes a property or a member variable `num_agents` that the algorithm uses
to allocate the right amount of memory during startup.

_Multi-agent environments require auto-reset._ I.e. they reset a particular agent when the corresponding `done` flag is `True` and return
the first observation of the next episode (because we have no use for the last observation of the previous
episode, we do not act based on it). See `multi_agent_wrapper.py` for example. For simplicity Sample Factory actually treats all
environments as multi-agent, i.e. single-agent environments are automatically treated as multi-agent environments with one agent.

Sample Factory uses this function to check if the environment is multi-agent. Make sure your environment provides the `num_agents` member:

```python
def is_multiagent_env(env):
    is_multiagent = hasattr(env, 'num_agents') and env.num_agents > 1
    if hasattr(env, 'is_multiagent'):
        is_multiagent = env.is_multiagent
    return is_multiagent
```
 
## Using Sample Factory
 
Once Sample Factpry is installed, it defines two main entry points, one for training, and one for algorithm evaluation:
* `sample_factory.algorithms.appo.train_appo`
* `sample_factory.algorithms.appo.enjoy_appo`

Some environments, such as VizDoom, DMLab, and Atari, are added to the env registry in the default installation, so training on these environments is as
simple as providing basic configuration parameters. I.e. to train and evaluate on the most basic VizDoom environment:

```
python -m sample_factory.algorithms.appo.train_appo --env=doom_basic --algo=APPO --train_for_env_steps=3000000 --num_workers=20 --num_envs_per_worker=20 --experiment=doom_basic
python -m sample_factory.algorithms.appo.enjoy_appo --env=doom_basic --algo=APPO --experiment=doom_basic
```
 
### Configuration

Sample Factory experiments are configured via command line parameters. The following command will print the help message
for the algorithm-environment combination:

```
python -m sample_factory.algorithms.appo.train_appo --algo=APPO --env=doom_battle --experiment=your_experiment --help
```

This will print the full list of parameters, their descriptions, and their default values.
Replace `doom_battle` with a different environment name (i.e. `atari_breakout`) to get information about parameters
specific to this particular environment. 

Once the new experiment is started, a directory containing experiment-related files is created in `--train_dir`
location (or `./train_dir` in `cwd` if `--train_dir` is not passed from command line). This directory contains a file
`cfg.json` where all the experiment parameters are saved (including those instantiated from their default values).

Most default parameter values and their help strings are defined in `sample_factory/algorithms/algorithm.py` and
`sample_factory/algorithms/appo/appo.py`. Besides that, additional parameters can be defined for specific families of environments.

The key parameters are:

- `--algo` (required) algorithm to use, pass value `APPO` to train agents with fast Async PPO.

- `--env` (required) full name that uniquely identifies the environment, starting with the env family prefix
(e.g. `doom_`, `dmlab_` or `atari_` for built-in Sample Factory envs). E.g. `doom_battle` or `atari_breakout`.

- `--experiment` (required) a name that uniquely identifies the experiment. E.g. `--experiment=my_experiment`.
If the experiment folder with the name already exists the experiment will be _resumed_!
Resuming experiments after a stop is the default behavior in Sample Factory. 
The parameters passed
from command line are taken into account, unspecified parameters will be loaded from the existing experiment
`cfg.json` file. If you want to start a new experiment, delete the old experiment folder or change the experiment name.

- `--train_dir` location for all experiments folders, defaults to `./train_dir`.

- `--num_workers` defaults to number of logical cores in the system, which will give the best throughput in
most scenarios.

- `--num_envs_per_worker` will greatly affect the performance. Large values (20-30) improve hardware utilization but
increase memory usage and policy lag. See example command lines below to find a value that works for your system.
_Must be even_ for the double-buffered sampling to work. Disable double-buffered sampling by setting `--worker_num_splits=1`
to use odd number of envs per worker (e.g. 1 env per worker). (Default: 2)

#### Configuring actor & critic architectures

`sample_factory/algorithms/algorithm.py` contains parameters that allow users to customize the architectures of neural networks
involved in the training process. Sample Factory includes a few popular NN architectures for RL, such as shallow
convnets for Atari and VizDoom, deeper ResNet models for DMLab, MLPs for continuous control tasks.
CLI parameters allow users to choose between
these existing architectures, as well as specify the type of the policy core (LSTM/GRU/feed-forward), nonlinearities,
etc. Consult experiment-specific `cfg.json` and the source code for full list of parameters.

`sample_factory.envs.dmlab.dmlab_model` and `sample_factory.envs.doom.doom_model` demonstrate how to handle environment-specific
additional input spaces (e.g. natural language and/or numerical vector inputs).
Script `sample_factory_examples/train_custom_env_custom_model.py` demonstrates how users can define a fully custom
environment-specific encoder. Whenever a fully custom actor-critic architecture is required, users are welcome
to override `_ActorCriticBase` following examples in `sample_factory/algorithms/appo/model.py`.

## Running experiments

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

## Citation

If you use this repository in your work or otherwise wish to cite it, please make reference to our ICML2020 paper.

```
@inproceedings{petrenko2020sf,
  title={Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with Asynchronous Reinforcement Learning},
  author={Petrenko, Aleksei and Huang, Zhehui and Kumar, Tushar and Sukhatme, Gaurav and Koltun, Vladlen},
  booktitle={ICML},
  year={2020}
}
```

For questions, issues, inquiries please email apetrenko1991@gmail.com. 
Github issues and pull requests are welcome.
