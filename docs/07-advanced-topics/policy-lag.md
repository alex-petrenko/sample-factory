# Policy Lag

Policy lag is the discrepancy between the policy that is used to collect samples and the policy that we train on this data.
Policy gradient algorithms (like PPO) are considered on-policy methods and typically suffer sample efficiency losses
when the policy lag is large (experience is stale, "off-policy").

In practice PPO is pretty robust to slightly off-policy data, but it is important to keep the policy lag under control.

Although this is not rigorous, we can measure the policy lag in the number of SGD steps between the policy
that collected the data (_behavior policy_) and the trained policy (_target policy_).

## Sources of policy lag

There are following main sources of policy lag:

* Multiple updates on the same data. This is the most common source of policy lag inherent to almost all
policy gradient implementations.
If `--num_batches_per_epoch > 1` and/or `--num_epochs > 1` then we need to do multiple SGD steps before we finish
training on the sampled data, causing the lag in the later epochs.

* Collecting more experience per sampling iteration (one rollout for all agents) than we use for one iteration of training
  (`num_workers * num_envs_per_worker * rollout >> batch_size * num_batches_per_epoch`). In this case we will inevitably
have some trajectories (or parts of trajectories) collected by the policy that is already outdated by the time we use them for training.

* Async sampling. With asynchronous sampling we collect new data while we are training on the old data, which will
inevitably cause some amount of lag. This is the smallest source of lag since we update the policy on the inference worker
as soon as new weights are available.

## Estimating and measuring policy lag

Policy lag for a particular RL experiment configuration is roughly proportional to the following value:

```math
Lag ~~ (num_epochs * num_workers * num_envs_per_worker * agents_per_env * rollout) / batch_size
```

Sample Factory reports empirical policy lag in two different ways.

### Policy lag measurements printed to the console

```text
[2022-11-30 19:48:19,509][07580] Updated weights for policy 0, policy_version 926 (0.0015)                                                                                                                  
[2022-11-30 19:48:21,166][07494] Fps is (10 sec: 22528.2, 60 sec: 20377.6, 300 sec: 20377.6). Total num frames: 3829760. Throughput: 0: 5085.4. Samples: 203415. Policy #0 lag: (min: 0.0, avg: 1.9, max: 5.0)                                                                                                           
[2022-11-30 19:48:21,166][07494] Avg episode reward: [(0, '0.824')]
```

Here message `Policy #0 lag: (min: 0.0, avg: 1.9, max: 5.0)` contains the minimum, average and maximum policy lag
for transitions encountered in the last minibatch processed by the learner at the moment of printing.
This can correspond to a minibatch in earlier or later epochs, so these values might fluctuate, but looking at 5-10 consecutive
printouts should give a good idea of the policy lag.

### Policy lag measurements in Tensorboard or Weights & Biases

`train/version_diff_avg`, `train/version_diff_max`, `train/version_diff_min` metrics represent policy lag values
measured in policy versions (SGD steps). See [Metrics Reference](../05-monitoring/metrics-reference.md).

## Minimizing policy lag

Policy lag can usually be traded off for higher throughput or sample efficiency (i.e. by doing many epochs of SGD on the same data).
But large values of policy lag can cause instability in training.
Each task will have its own sweet spot when it comes to configuration and policy lag. Very roughly speaking,
policy lag < 20-30 SGD steps is usually fine, but significantly larger values might be a reason to reconsider the configuration.

Empirically, LSTM/GRU policies and environments with very complex action spaces tend to be more sensitive
to policy lag. For RNNs this is true because not only the action distributions, but also the hidden states
change between the behavior and target policies.
With complex action spaces (i.e. tuple, multi-discrete) small changes to the policy can cause large changes to probabilities
of individual actions.

Following configuration options can be used to minimize policy lag:

* Increase `batch_size`,
decrease `num_epochs`, `num_batches_per_epoch`, `num_workers`, `num_envs_per_worker`, `rollout`, `num_batches_per_epoch` (see the formula above).
* Switch to synchronous sampling (`--async_rl=False`). Note that this will likely increase the training time.

### Achieving zero policy lag (A2C)

It is possible to achieve zero policy lag by using `--async_rl=False` and `--num_batches_per_epoch=1` and `--num_epochs=1`.
This will turn PPO into the algorithm known as A2C (Advantage Actor-Critic) which always trains on the most recent data.
This should typically yield stable training, although might not be the best option in terms of throughput or sample efficiency.
