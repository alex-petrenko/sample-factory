# Custom Summaries

## Environment-specific info

It is often useful to monitor custom training metrics, i.e. certain environment-specific aspects of agent's performance.

You can add custom monitored metrics by adding `info["episode_extra_stats"] = { ... }` to the environment's `info` dictionary returned from the `step()` function on the last step of the episode.

See `sf_examples/dmlab/wrappers/reward_shaping.py` for example. Here we add information about 
agent's performance on individual levels in DMLab-30.

## Custom metrics

You can add completely custom metrics that are calculated based on other metrics or the RL algorithm state.
To do this, add a custom algo observer that overrides `extra_summaries()` function.

See `sf_examples/dmlab/train_dmlab.py` where we define `DmlabExtraSummariesObserver` that aggregates custom 
environment metrics to produce a single "Human-normalized score" summary.