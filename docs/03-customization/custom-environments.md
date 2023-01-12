# Custom environments

Training agents in your own environment with Sample Factory is straightforward,
but if you get stuck feel free to raise an issue on our [GitHub Page](https://github.com/alex-petrenko/sample-factory/issues).

We recommend looking at our example environment integrations such as [Atari](../09-environment-integrations/atari.md)
or [MuJoCo](../09-environment-integrations/mujoco.md) before using your own environment.

## Custom environment template

In order to integrate your own environment with Sample Factory, the following steps are required:

* Define entry points for training and evaluation scripts, such as `train_custom_env.py` and `enjoy_custom_env.py`.
* Define a method that creates an instance of your environment, such as `make_custom_env()`.
* Override any default parameters that are specific to your environment, this way you can avoid passing them from the command line (optional).
* Add any custom parameters that will be parsed by Sample Factory alongside the default parameters (optional).

We provide the following template, which you can modify to intergrate your environment.
We assume your environment conforms to the [gym](https://github.com/openai/gym) 0.26 API (5-tuple).

```python3
from typing import Optional
import argparse
import sys

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl


def make_custom_env(full_env_name: str, cfg=None, env_config=None, render_mode: Optional[str] = None):
    # see the section below explaining arguments
    return CustomEnv(full_env_name, cfg, env_config, render_mode=render_mode)
    
def register_custom_env_envs():
    # register the env in sample-factory's global env registry
    # after this, you can use the env in the command line using --env=custom_env_name
    register_env("custom_env_name", make_custom_env)

def add_custom_env_args(_env, p: argparse.ArgumentParser, evaluation=False):
    # You can extend the command line arguments here
    p.add_argument("--custom_argument", default="value", type=str, help="")

def custom_env_override_defaults(_env, parser):
    # Modify the default arguments when using this env.
    # These can still be changed from the command line. See configuration guide for more details.
    parser.set_defaults(
        encoder_conv_architecture="convnet_atari",
        obs_scale=255.0,
        gamma=0.99,
        learning_rate=0.00025,
        lr_schedule="linear_decay",
        adam_eps=1e-5,  
    )

def parse_args(argv=None, evaluation=False):
    # parse the command line arguments to build
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_custom_env_args(partial_cfg.env, parser, evaluation=evaluation)
    custom_env_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg

def main():
    """Script entry point."""
    register_custom_env_envs()
    cfg = parse_args()

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())

```

Training can now be started with `python train_custom_env.py --env=custom_env_name --experiment=CustomEnv`. Note that this train script
can be defined in your own codebase, or in the Sample Factory codebase (in case you forked the repo).

### Environment factory function parameters

`register_env("custom_env_name", make_custom_env)` expects `make_custom_env` to be a Callable with the following signature:

```python3
def make_custom_env_func(full_env_name: str, cfg: Optional[Config] = None, env_config: Optional[AttrDict] = None, render_mode: Optional[str] = None) -> Env
```

Arguments:
* `full_env_name`: complete name of the environment as passed in the command line with `--env`
* `cfg`: full system configuration, output of argparser. Normally this is an `AttrDict` (dictionary where keys can be accessed as attributes)
* `env_config`: AttrDict with additional system information, for example: `env_config = AttrDict(worker_index=worker_idx, vector_index=vector_idx, env_id=env_id)`
Some custom environments will require this information, i.e. `env_id` is a unique identifier for each environment instance in 0..num_envs-1 range. 
* `render_mode`: if not None, environment will be rendered in this mode (e.g. 'human', 'rgb_array'). New parameter required after Gym 0.26.

See `sample_factory/envs/create_env.py` for more details.

## Evaluation script template

The evaluation script template is even more straightforward. Note that we just reuse functions already defined in the training script.

```python3
import sys

from sample_factory.enjoy import enjoy
from train_custom_env import parse_args, register_custom_env_envs


def main():
    """Script entry point."""
    register_custom_env_envs()
    cfg = parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
```

You can now run evaluation with `python enjoy_custom_env.py --env=custom_env_name --experiment=CustomEnv` to
measure the performance of the trained model, visualize agent's performance, or record a video file.

## Examples

* `sf_examples/train_custom_env_custom_model.py` - integrates an entirely custom toy environment.
* `sf_examples/train_gym_env.py` - trains an agent in a Gym environment. Environments registered in `gym` do not
get any special treatment, as it is just another way to define an environment. In this case the environment creation
function reduces to `gym.make(env_name)`.
* See environment integrations in `sf_examples/<env_name>` for additional examples.
