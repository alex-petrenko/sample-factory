# Custom multi-agent environments

Multi-agent environments are expected to return lists (or tuples, arrays, tensors) of observations/rewards/etc, one item for every agent.

It is expected that a multi-agent env exposes a property or a member variable `num_agents` that the algorithm uses
to allocate the right amount of memory during startup.

**Multi-agent environments require auto-reset!** I.e. they reset a particular agent when the corresponding `terminated` or `truncated`
flag is `True` and return 
the first observation of the next episode (because we have no use for the last observation of the previous
episode, we do not act based on it).

For simplicity Sample Factory actually treats all
environments as multi-agent,
i.e. single-agent environments are automatically treated as multi-agent environments with one agent with the use of a wrapper.

In rare cases we may deal with an environment that should not be additionally wrapped, i.e. a single-agent version
of a multi-agent env may already return lists of length 1. In this case, your environment should define a member variable
`is_multiagent=True`, and Sample Factory will not wrap it.

## Examples

* `sf_examples/enjoy_custom_multi_env.py` - integrates and entirely custom toy example multi-agent env. Use this as a template for your own multi-agent env.
* `sf_examples/isaacgym_examples/train_isaacgym.py` - technically IsaacGym is not a multi-agent environment because different agents don't interact. 
It is a _vectorized_ environment simulating many agents with a single env instance, but is treated as a multi-agent environment by Sample Factory.
* `sf_examples/vizdoom/doom/multiplayer` - this is a rather advanced example, here we connect
multiple VizDoom instances into a single multi-agent match and expose a multi-agent env interface to Sample Factory. 

## Further reading

* Multi-agent environments can be combined with [multi-policy training](../07-advanced-topics/multi-policy-training.md) and
[Population Based Training (PBT)](../07-advanced-topics/pbt.md).
* Sometimes it makes sense to disable some of the agents in a multi-agent environment.
For example, in a multi-player game some agents might die in the middle of the episode and should not contribute
any rollouts until the episode reset. This can be achieved using [inactive agents feature](../07-advanced-topics/inactive-agents.md).