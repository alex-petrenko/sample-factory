# Synchronous/Asynchronous RL

Since version 2.0 Sample Factory supports two training modes: synchronous and asynchronous.
You can switch between them by setting `--async_rl=False` or `--async_rl=True` in the command line.

## Synchronous RL

In synchronous mode we collect trajectories from all environments until we have just enough data
to fill a **dataset** (or **training batch**) on which the learner will perform one or more epochs of SGD.

We operate synchronously: the system either collects the experience or trains the policy, not both at the same time.
Rollout and Inference workers wait for the learner to finish the training before they start collecting more data.

## Asynchronous RL

In asynchronous (default) mode we collect trajectories all the time and train the policy in the background.
Once we have enough data to fill a dataset, we immediately start collecting new trajectories and will keep
doing so until `--num_batches_to_accumulate` training batches are accumulated.

## Pros and Cons

There is no clear winner between the two modes. Try both regimes and see which one works better for you.

* Async mode is often faster because we allow more computation to happen in parallel.
As a tradeoff it introduces more **policy-lag** because some of the experience is collected by older versions of the policy
(for example when we collect experience during training). So async mode enables faster training but might cost sample efficiency in some setups, for example 
LSTM/GRU training is usually more susceptible to policy-lag than non-recurrent policies.

* Sync mode has more strict requirements for the system configuration because we're looking
to collect the exact amount of data to fill a training batch.
Example: we have `--num_workers=16`, `--num_envs_per_worker=8`, `--rollout=32`. This means in one iteration
we collect 16 * 8 * 32 = 4096 steps of experience. Sync mode requires that training batch size is a multiple of 4096.
This would work with `--batch_size=4096` and `--num_batches_per_epoch=1` or `--batch_size=2048` and `--num_batches_per_epoch=2`, but
not with `--batch_size=512` and `--num_batches_per_epoch=3`.
**TLDR**: sync mode provides less flexibility in the training configuration. In async mode we can do pretty much anything.

* For multi-policy and PBT setups we recommend using async mode. Async mode allows different policies to collect different
amounts of experience per iteration, which allows us to use arbitrary mapping between agents and policies.

## Visualization

The following animations may provide further insight into the difference between the two modes.

* Sync RL: https://www.youtube.com/watch?v=FHRG0lHVa54
* Async RL: https://www.youtube.com/watch?v=ML2WAQNpF90

Note that the "Sync RL" animation is not 100% accurate to how SF works, we actually still do collect the
experience asynchronously within the rollout, but then pause during training.
"Sync RL" animation is closer to how a traditional RL implementation operates (e.g. OpenAI Baselines) and the comparison
between the two shows why Sample Factory is often much faster.

## Vectorized environments

In GPU-accelerated environments like [IsaacGym](09-environment-integrations/isaacgym.md) async mode does not provide
a significant speedup because we do everything on the same device anyway. For these environments it is recommended to use
sync mode for maximum sample efficiency.

This animation demonstrates how synchronous learning works in a vectorized environment like IsaacGym: https://www.youtube.com/watch?v=EyUyDs4AA1Y
