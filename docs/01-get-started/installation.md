# Installation

Just install from PyPI:

```pip install sample-factory```

SF is known to work on Linux and macOS. There is no Windows support at this time.

## Install from sources

```bash
git clone git@github.com:alex-petrenko/sample-factory.git
cd sample-factory
pip install -e .

# or install with optional dependencies
pip install -e .[dev,mujoco,atari,vizdoom]
```

## Environment support

To run Sample Factory with one of the available environment integrations, please refer to the corresponding documentation sections: 

- [Mujoco](../09-environment-integrations/mujoco.md)
- [Atari](../09-environment-integrations/atari.md)
- [ViZDoom](../09-environment-integrations/vizdoom.md)
- [DeepMind Lab](../09-environment-integrations/dmlab.md)
- [Megaverse](../09-environment-integrations/megaverse.md)
- [Envpool](../09-environment-integrations/envpool.md)
- [Isaac Gym](../09-environment-integrations/isaacgym.md)
- [Quad-Swarm-RL](../09-environment-integrations/swarm-rl.md)

Sample Factory allows users to easily add custom environments and models, refer to [Customizing Sample Factory](../03-customization/custom-environments.md) for more information.