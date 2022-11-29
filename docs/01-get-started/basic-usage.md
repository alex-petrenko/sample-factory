# Basic Usage

## Usage examples

Use command line to train an agent using one of the existing integrations, e.g. Mujoco (might need to run `pip install sample-factory[mujoco]`):

```bash
python -m sf_examples.mujoco.train_mujoco --env=mujoco_ant --experiment=Ant --train_dir=./train_dir
```

Stop the experiment when the desired performance is reached and then evaluate the agent:

```bash
python -m sf_examples.mujoco.enjoy_mujoco --env=mujoco_ant --experiment=Ant --train_dir=./train_dir
```

Do the same in a pixel-based environment such as VizDoom (might need to run `pip install sample-factory[vizdoom]`, please also see docs for VizDoom-specific instructions):

```bash
python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --experiment=DoomBasic --train_dir=./train_dir --num_workers=16 --num_envs_per_worker=10 --train_for_env_steps=1000000
python -m sf_examples.vizdoom.enjoy_vizdoom --env=doom_basic --experiment=DoomBasic --train_dir=./train_dir
```

## Monitoring experiments

Monitor any running or completed experiment with Tensorboard:

```bash
tensorboard --logdir=./train_dir
```
(or see the docs for WandB integration).

## Next steps

* Read more about configuring experiments in the [Configuration](../02-configuration/configuration.md) guide.
* Follow the instructions in the [Customizing](../03-customization/custom-environments.md) guide to train an agent in your own environment.
