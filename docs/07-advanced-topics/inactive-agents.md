# Inactive Agents

Sometimes it makes sense to disable some of the agents in a multi-agent environment.
For example, in a multi-player game some agents might die in the middle of the episode and should not contribute
any rollouts until the episode reset.

In order to disable (deactivate) the agent, add `info["is_active"] = False` in the `env.step()` call, i.e.
the agent's info dict should contain `is_active` key with `False` value.
Absent `is_active` key or `is_active=True` is treated as active agent.

When the agent is deactivated in the middle of the rollout, the inactive part of the rollout is treated as `invalid`
data by the learner (similar to any other invalid data, i.e. experience that exceeds `--max_policy_lag`).

We carefully mask this invalid data on the learner for loss & advantages calculations.
Therefore any inactive data makes the effective batch size smaller, so we decrease the learning rate accordingly,
otherwise batches with >90% invalid data would produce very noisy parameter updates.

It is generally advised that the portion of inactive data (`train/valids_fraction` on Tensorboard/WandB) does
not exceed 50%, otherwise it may seriously affect training dynamics and requires careful tuning.

There are also alternative ways to treat inactive agents, for example just feeding them some special observation (e.g. all zeros)
and zero rewards until the episode reset.

Inactive agents are currently only supported in non-batched sampling mode (`--batched_sampling=False`).

## Examples

* `sf_examples/train_custom_multi_env.py` - shows how to use inactive agents in a custom multi-agent environment.

Inactive agents are a new feature, suggestions & contributions are welcome!