# Custom models

Adding custom models in Sample Factory is simple, but if you get stuck feel free to raise an issue on our [GitHub Page](https://github.com/alex-petrenko/sample-factory/issues).

## Actor Critic models in Sample Factory
Actor Critic models in Sample Factory are composed of three components:

- Encoder - Process input observations (images, vectors) and map them to a vector. This is the part of the model you will most likely want to customize.
- Core - Intergrate vectors from one or more encoders, can optionally include a single- or multi-layer LSTM/GRU in a memory-based agent.
- Decoder - Apply additional layers to the output of the model core before computing the policy and value outputs.

Regardless of the component customization, you can use your resulting model in "shared weights" or "separate weights" regime
(either sharing or not sharing the weights between the policy and value networks). This is controlled by
the `--actor_critic_share_weights=[True|False]` command line argument.
 
On top of that, you can register an entire custom Actor Critic model. This can be useful for more complex models, 
for example centralized critic for multi-agent envs,
asymmetric actor-critic where critic observes more information, which can be useful in sim-to-real, and so on.

## Custom model template

The following template demonstrates how different components of the model can be customized.
Feel free to combine this with the custom environment template above to create a fully custom environment & model combination.

```python3
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

## Examples

Examples of model customizations can be found in:

* `sf_examples/train_custom_env_custom_model.py`
* `sf_examples/isaacgym_examples/train_isaacgym.py`
* `sf_examples/dmlab/dmlab_model.py`
* `sf_examples/vizdoom/doom/doom_model.py`