# Multi-Policy Training

Sample Factory supports training multiple policies at the same time with `--num_policies=N`, where `N` is the number of policies to train.

## Single-agent environments

Multi-policy training with single-agent environments is equivalent to just running multiple experiments with
different seeds. We actually recommend running separate experiments in this case because experiment monitoring is easier this way.

## Multi-agent environments

In multi-agent and self-play environments it can be beneficial to train multiple policies at once to avoid
overfitting to a single opponent (i.e. self).

If this is the desired training mode, it is important that we control multiple agents in the same environment with
different policies. This is controlled by the argument `--pbt_mix_policies_in_one_env`, which is set to `True` by default.
Although this argument has `--pbt` prefix, it actually applies regardless of whether we're training with PBT or not.

If `--pbt_mix_policies_in_one_env=True`, then we will periodically randomly resample policies controlling each agent in the environment.
This is implemented in `sample_factory.algo.utils.agent_policy_mapping`. Feel free to fork the repo and modify
this class to create your own custom mapping.

Exposing `agent_policy_mapping` through API to allow custom mapping is an obvious improvement, and contributions here are welcome!

## Implementation details

### GPU mapping

On a multi-GPU machine we assign each policy to a separate GPU. Or, if we have fewer GPUs than policies, we will 
fill the GPUs with policies until we run out of GPUs, and then start reusing GPUs.

For example, on a 2-GPU machine 4-policy training will look like this:

```
GPU 0: policy 0, policy 2
GPU 1: policy 1, policy 3
```

### Multi-policy training in different modes

All features of multi-policy training (including mixing different policies in one env) are only supported with
asynchronous (`--async_rl=True`) non-batched (`--batched_sampling=False`) training.

In synchronous mode we can still use multi-policy training, but the mapping between agents and policies is fixed
and deterministic because we need to guarantee the same amount of experience for all policies.

In batched mode we can also use multi-policy training, but mixing policies in one environment is not supported.
This would defeat the purpose of batched mode where we want to directly transfer a large vector of observations on the GPU
and do inference. Arbitrary mapping between agents and policies would make this significantly slower and more complicated.

That said, it might still make a lot of sense to use multi-policy training in batched mode/sync mode in the context of
Population-Based Training, i.e. to optimize hyperparameters of agents in the population.

See [Population-Based Training](../07-advanced-topics/pbt.md) for more details.