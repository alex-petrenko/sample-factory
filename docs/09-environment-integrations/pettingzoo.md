# PettingZoo

[PettingZoo](https://pettingzoo.farama.org/) is a Python library for conducting research in multi-agent reinforcement learning. This guide explains how to use PettingZoo environments with Sample Factory.

## Installation

Install Sample Factory with PettingZoo dependencies with PyPI:

```bash
pip install -e sample-factory[pettingzoo]
```

## Running Experiments

Run PettingZoo experiments with the scripts in `sf_examples`.
The default parameters are not tuned for throughput.

To train a model in the `tictactoe_v3` environment:

```
python -m sf_examples.train_pettingzoo_env --algo=APPO --env=tictactoe_v3 --experiment="Experiment Name"
```

To visualize the training results, use the `enjoy_pettingzoo_env` script:

```
python -m sf_examples.enjoy_pettingzoo_env --env=tictactoe_v3 --experiment="Experiment Name"
```

Currently, the scripts in `sf_examples` are set up for the `tictactoe_v3` environment. To use other PettingZoo environments, you'll need to modify the scripts or add your own as explained below.

### Adding a new PettingZoo environment

To add a new PettingZoo environment, follow the instructions from [Custom environments](../03-customization/custom-environments.md), with the additional step of wrapping your PettingZoo environment with `sample_factory.envs.pettingzoo_envs.PettingZooParallelEnv`.

Here's an example of how to create a factory function for a PettingZoo environment:

```python
from sample_factory.envs.pettingzoo_envs import PettingZooParallelEnv
import some_pettingzoo_env # Import your desired PettingZoo environment

def make_pettingzoo_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    return PettingZooParallelEnv(some_pettingzoo_env.parallel_env(render_mode=render_mode))
```

Note: Sample Factory supports only the [Parallel API](https://pettingzoo.farama.org/api/parallel/) of PettingZoo. If your environment uses the AEC API, you can convert it to Parallel API using `pettingzoo.utils.conversions.aec_to_parallel` or `pettingzoo.utils.conversions.turn_based_aec_to_parallel`. Be aware that these conversions have some limitations. For more details, refer to the [PettingZoo documentation](https://pettingzoo.farama.org/api/wrappers/pz_wrappers/).