## Installation

To install MuJoCo environments to work with Sample-Factory v2, run `pip install sample-factory[mujoco]`
or `pip install -e .[mujoco]` in the sample-factory directory.

## Running Experiments

To run a single experiment, use the `sf_examples.mujoco.train_mujoco` script. An example command is
`python -m sf_examples.mujoco.train_mujoco --algo=APPO --env=mujoco_ant --experiment=experiment_name`.

To run multiple experiments in parallel, use the launcher script at `sf_examples.mujoco.experiments.mujoco_all_envs`.
An example command is `python -m sample_factory.launcher.run --run=sf_examples.mujoco.experiments.mujoco_all_envs --backend=processes --max_parallel=4  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1 --experiment_suffix=test`

## Showing Experiment Results

To display videos of experiments, use the `sf_examples.mujoco.enjoy_mujoco` script. An example command is 
`python -m sf_examples.mujoco.enjoy_mujoco --env=mujoco_ant --algo=APPO --experiment=experiment_name`

## Supported MuJoCo environments

SF2 supports the following MuJoCo v4 environments:

- mujoco_hopper
- mujoco_halfcheetah
- mujoco_humanoid
- mujoco_ant
- mujoco_standup
- mujoco_doublependulum
- mujoco_pendulum
- mujoco_reacher
- mujoco_walker
- mujoco_pusher
- mujoco_swimmer

More environments can be added in `sf_examples.mujoco.mujoco_utils`

## Benchmark Results

We benchmarked SF2 against CleanRL on a few environments. We achieved similar sample efficiency with the same parameters.
Results can be found here: https://wandb.ai/andrewzhang505/sample_factory/reports/MuJoCo-Sample-Factory-vs-CleanRL-w-o-EnvPool--VmlldzoyMjMyMTQ0