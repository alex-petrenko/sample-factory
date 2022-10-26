# Custom environments

Training agents in your own environment with sample-factory is straightforward, but if you get stuck feel free to raise an issue on our [GitHub Page](https://github.com/alex-petrenko/sample-factory/issues).

We recommend look at our example environment integrations such as [Atari](../../environment-integrations/atari/) or [ViZDoom](../../environment-integrations/vizdoom/) before using your own environment.

### Custom Env template
We provide the following template, which you can modify to intergrate your environment. We assume your environment conforms to a [gym](https://github.com/openai/gym) 0.26 API (5-tuple).

First make a file called `train_custom_env.py` and copy the following template.

```python

from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import register_env
from sample_factory.train import run_rl


def make_custom_env(env_name, cfg, env_config):
    # function that build your custom env
    # cfg and env_config can be used to further customize the env
    # env_config.env_id can be used to seed each env, for example
    # env = create_your_custom_env()
    return env
    
def register_custom_env_envs():
    # register the env in sample-factory's global env registry
    register_env("custom_env_name", make_custom_env)

def add_custom_env_args(_env, p: argparse.ArgumentParser, evaluation=False):
    # You can extend the command line arguments here
    p.add_argument("--custom_argument", default="value", type=str, help="")

def custom_env_override_defaults(_env, parser):
    # Modify the default arguments when using this env
    # these can still be changed from the command line
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

Training can now be started with `python train_custom_env --env=custom_env_name`
