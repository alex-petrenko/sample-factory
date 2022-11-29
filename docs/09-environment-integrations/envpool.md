# Envpool

### Installation

Install Sample-Factory with Envpool dependencies with PyPI:

```
pip install sample-factory[atari,envpool]
pip install sample-factory[mujoco,envpool]
```

### Running Experiments
[EnvPool](https://envpool.readthedocs.io/en/latest/) is a C++-based batched environment pool with pybind11 and thread pool. It has high performance (~1M raw FPS with Atari games, ~3M raw FPS with Mujoco simulator).

We provide examples for envpool for Atari and Mujoco environments. The default parameters provide reasonable training speed, but can be tuning based on your machine configuration to achieve higher throughput.

To train a model with envpool in the `BreakoutNoFrameskip-v4` environment:

```
python -m sf_examples.envpool.atari.train_atari --algo=APPO --env=atari_breakout --experiment="Experiment Name"
```

To visualize the training results, use the `enjoy_atari` script:

```
python -m sf_examples.atari.enjoy_atari --algo=APPO --env=atari_breakout --experiment="Experiment Name"
```

Multiple experiments can be run in parallel with the launcher module. `atari_envs` is an example launcher script that runs atari envs with 4 seeds. 

```
python -m sample_factory.launcher.run --run=sf_examples.envpool.atari.experiments.atari_envs --backend=processes --max_parallel=8  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1
```
