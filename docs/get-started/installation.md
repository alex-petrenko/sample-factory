# Installation

## Using pip
Just install from PyPI:

```pip install sample-factory```

SF is known to work on Linux and macOS. There is no Windows support at this time.

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

Since ~2022 some extra steps might be required to install atari: `pip install "gym[atari,accept-rom-license]"`
 
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
 