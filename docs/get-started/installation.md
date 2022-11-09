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

Sample Factory comes with comprehensive support Mujoco, Atari, VizDoom, DMLab, Megaverse and Envpool:

- [Mujoco](../environment-integrations/mujoco.md)
- [Atari](../environment-integrations/mujoco.md)
- [ViZDoom](../environment-integrations/mujoco.md)
- [DeepMind Lab](../environment-integrations/mujoco.md)
- [Megaverse](../environment-integrations/mujoco.md)
- [Envpool](../environment-integrations/mujoco.md)

Sample Factory allows users to easily add custom environments and models, refer to [Customizing Sample Factory](customizing.md) for more information.