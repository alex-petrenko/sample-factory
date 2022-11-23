# Basic Usage

Use command line to train an agent using one of the existing integrations, e.g. Mujoco (might need to run `pip install sample-factory[mujoco]`):

```bash
python -m sf_examples.mujoco.train_mujoco --env=mujoco_ant --experiment=Ant --train_dir=./train_dir
```

Stop the experiment when the desired performance is reached and then evaluate the agent:

```bash
python -m sf_examples.mujoco.enjoy_mujoco --env=mujoco_ant --experiment=Ant --train_dir=./train_dir
```

Do the same in a pixel-based VizDoom environment (might need to run `pip install sample-factory[vizdoom]`, please also see docs for VizDoom-specific instructions):

```bash
python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --experiment=DoomBasic --train_dir=./train_dir --num_workers=16 --num_envs_per_worker=10 --train_for_env_steps=1000000
python -m sf_examples.vizdoom.enjoy_vizdoom --env=doom_basic --experiment=DoomBasic --train_dir=./train_dir
```

Monitor any running or completed experiment with Tensorboard:

```bash
tensorboard --logdir=./train_dir
```
(or see the docs for WandB integration).


[//]: # (### Configuration)

[//]: # ()
[//]: # (Sample Factory experiments are configured via command line parameters. The following command will print the help message)

[//]: # (for the algorithm-environment combination:)

[//]: # ()
[//]: # (```)

[//]: # (python -m sample_factory.algorithms.appo.train_appo --algo=APPO --env=doom_battle --experiment=your_experiment --help)

[//]: # (```)

[//]: # ()
[//]: # (This will print the full list of parameters, their descriptions, and their default values.)

[//]: # (Replace `doom_battle` with a different environment name &#40;i.e. `atari_breakout`&#41; to get information about parameters)

[//]: # (specific to this particular environment. )

[//]: # ()
[//]: # (Once the new experiment is started, a directory containing experiment-related files is created in `--train_dir`)

[//]: # (location &#40;or `./train_dir` in `cwd` if `--train_dir` is not passed from command line&#41;. This directory contains a file)

[//]: # (`cfg.json` where all the experiment parameters are saved &#40;including those instantiated from their default values&#41;.)

[//]: # ()
[//]: # (Most default parameter values and their help strings are defined in `sample_factory/algorithms/algorithm.py` and)

[//]: # (`sample_factory/algorithms/appo/appo.py`. Besides that, additional parameters can be defined for specific families of environments.)

[//]: # ()
[//]: # (The key parameters are:)

[//]: # ()
[//]: # (- `--algo` &#40;required&#41; algorithm to use, pass value `APPO` to train agents with fast Async PPO.)

[//]: # ()
[//]: # (- `--env` &#40;required&#41; full name that uniquely identifies the environment, starting with the env family prefix)

[//]: # (&#40;e.g. `doom_`, `dmlab_` or `atari_` for built-in Sample Factory envs&#41;. E.g. `doom_battle` or `atari_breakout`.)

[//]: # ()
[//]: # (- `--experiment` &#40;required&#41; a name that uniquely identifies the experiment. E.g. `--experiment=my_experiment`.)

[//]: # (If the experiment folder with the name already exists the experiment will be _resumed_!)

[//]: # (Resuming experiments after a stop is the default behavior in Sample Factory. )

[//]: # (The parameters passed)

[//]: # (from command line are taken into account, unspecified parameters will be loaded from the existing experiment)

[//]: # (`cfg.json` file. If you want to start a new experiment, delete the old experiment folder or change the experiment name.)

[//]: # ()
[//]: # (- `--train_dir` location for all experiments folders, defaults to `./train_dir`.)

[//]: # ()
[//]: # (- `--num_workers` defaults to number of logical cores in the system, which will give the best throughput in)

[//]: # (most scenarios.)

[//]: # ()
[//]: # (- `--num_envs_per_worker` will greatly affect the performance. Large values &#40;20-30&#41; improve hardware utilization but)

[//]: # (increase memory usage and policy lag. See example command lines below to find a value that works for your system.)

[//]: # (_Must be even_ for the double-buffered sampling to work. Disable double-buffered sampling by setting `--worker_num_splits=1`)

[//]: # (to use odd number of envs per worker &#40;e.g. 1 env per worker&#41;. &#40;Default: 2&#41;)

[//]: # ()
[//]: # (#### Configuring actor & critic architectures)

[//]: # ()
[//]: # (`sample_factory/algorithms/algorithm.py` contains parameters that allow users to customize the architectures of neural networks)

[//]: # (involved in the training process. Sample Factory includes a few popular NN architectures for RL, such as shallow)

[//]: # (convnets for Atari and VizDoom, deeper ResNet models for DMLab, MLPs for continuous control tasks.)

[//]: # (CLI parameters allow users to choose between)

[//]: # (these existing architectures, as well as specify the type of the policy core &#40;LSTM/GRU/feed-forward&#41;, nonlinearities,)

[//]: # (etc. Consult experiment-specific `cfg.json` and the source code for full list of parameters.)

[//]: # ()
[//]: # (`sample_factory.envs.dmlab.dmlab_model` and `sample_factory.envs.doom.doom_model` demonstrate how to handle environment-specific)

[//]: # (additional input spaces &#40;e.g. natural language and/or numerical vector inputs&#41;.)

[//]: # (Script `sf_examples/train_custom_env_custom_model.py` demonstrates how users can define a fully custom)

[//]: # (environment-specific encoder. Whenever a fully custom actor-critic architecture is required, users are welcome)

[//]: # (to override `_ActorCriticBase` following examples in `sample_factory/algorithms/appo/model.py`.)
