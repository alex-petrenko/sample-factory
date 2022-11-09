# Customizing Sample Factory 



## Custom environments

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

## Custom models

Adding custom models in sample factory is simple, but if you get stuck feel free to raise an issue on our [GitHub Page](https://github.com/alex-petrenko/sample-factory/issues).


### Actor Critic models in sample factory
Actor Critic models in Sample Factory are composed of three components:

- Encoders - Process input observations (images, vectors) and map them to a vector.
- Cores - Intergrate vectors from one or more encoders, can optionally include an LSTM in a memory-based agent.
- Decoders - Apply addition linear layers to the output of the model core.
 
You can register custom versions of each component, or you can register an entire Actor Critic model.


### Custom model template

```python


from sample_factory.model.encoder import Encoder
from sample_factory.model.decoder import Decoder
from sample_factory.model.core import ModelCore
from sample_factory.model.actor_critic import ActorCritic
from sample_factory.algo.utils.context import global_model_factory


class CustomEncoder(Encoder):
    def __init__(self, cfg: Config, obs_space: ObsSpace):
        super().__init__(cfg)
        # build custom encoder architecture
        ...
    
    def forward(self, obs_dict):
        # custom forward logic
        ...

class CustomCore(ModelCore):
    def __init__(self, cfg: Config, input_size: int):
        super().__init__(cfg)
        # build custom core architecture
        ...
    
    def forward(self, head_output, rnn_states):
        # custom forward logic
        ...


class CustomDecoder(Decoder):
    def __init__(self, cfg: Config, decoder_input_size: int):
        super().__init__(cfg)
        # build custom decoder architecture
        ...
    
    def forward(self, core_output):
        # custom forward logic
        ...

class CustomActorCritic(ActorCritic):
    def __init__(
        self,
        model_factory,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        cfg: Config,
    ):
    super().__init__(obs_space, action_space, cfg)

    self.encoder = CustomEncoder(cfg, obs_space)
    self.core = CustomCore(cfg, self.encoder.get_out_size())
    self.decoder = CustomDecoder(cfg, self.core.get_out_size())
    self.critic_linear = nn.Linear(self.decoder.get_out_size())
    self.action_parameterization = self.get_action_parameterization(
        self.decoder.get_out_size()
    ) 

    def forward(self, normalized_obs_dict, rnn_states, values_only=False):
        # forward logic
        ...


def register_model_components():
    # register custom components with the factory
    # you can register an entire Actor Critic model
    global_model_factory().register_actor_critic_factory(CustomActorCritic)

    # or individual components
    global_model_factory().register_encoder_factory(CustomEncoder)
    global_model_factory().register_core_factory(CustomCore)
    global_model_factory().register_decoder_factory(CustomDecoder)

def main():
    """Script entry point."""
    register_model_components()
    cfg = parse_args()

    status = run_rl(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())


```


## Custom multi-agent environments

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
 