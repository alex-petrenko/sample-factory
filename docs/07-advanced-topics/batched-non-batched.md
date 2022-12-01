# Batched/Non-Batched Sampling

Sample Factory has two different implementations of the RolloutWorker: batched and non-batched.
You can switch between them using `--batched_sampling=[True|False]` argument.

## Non-Batched Sampling

Non-batched sampling is the default mode. It makes very few assumptions about the environment and is the most flexible.
We call it non-batched because it treats each agent in each environment independently and processes trajectories
one by one.

One advantage of this approach is that we can control each agent in each environment with any policy which can
be very useful for multi-agent, self-play, and PBT setups.

A downside of this mode is that we have to batch individual observations from rollout workers before we can do inference on
the GPU (because GPUs are most efficient with big batches).
This makes non-batched mode very inefficient for vectorized environments like [IsaacGym](../09-environment-integrations/isaacgym.md).

## Batched Sampling

Batched mode is perfect for massively vectorized environments like [IsaacGym](../09-environment-integrations/isaacgym.md) or
[EnvPool](../09-environment-integrations/envpool.md).
It assumes that the observations are available in one large tensor that we can directly give to the inference worker
for processing.

For GPU-accelerated environments we can sample thousands of observations in a single tensor that is already on GPU
and thus achieve the maximum possible throughput. It is common that batched mode is used with a single
rollout worker `--num_workers=1` and a single inference worker `--num_inference_workers=1` and the
parallelization of environment simulation is handled by the environment itself.

Although PBT can be used with batched sampling, we do not support controlling individual agents with different policies
in this mode.

For regular CPU-based envs (Atari, VizDoom, Mujoco) the difference between batched and non-batched sampling is negligible,
either mode should work fine.

## Observations

In Sample Factory for simplicity all environments have dictionary observations (or converted to dictionary with
a single key `obs`). In non-batched mode even with multi-agent envs each agent thus provides a separate observation dictionary.

In batched mode we want to work with big tensors, so instead of a list of dictionaries we have a single dictionary
with a tensor of observations for each key.
