# Passing Info from RL Algo to Env

[Custom summary metrics](../05-monitoring/custom-metrics.md) provide a way to pass information
from the RL environment to the training system (i.e. success rate, etc.)

In some RL workflows it might be desirable to also pass information in the opposide direction: from the RL algorithm to
the environment. This can enable, for example, curriculum learning based on the number of training steps
consumed by the agent (or any other metric of the training progress).

We provide a way to do this by passing a `training_info` dictionary to the environment.
In order to do this, your environment needs to implement `TrainingInfoInterface`.

```python
class TrainingInfoInterface:
    def __init__(self):
        self.training_info: Dict[str, Any] = dict()

    def set_training_info(self, training_info):
        """
        Send the training information to the environment, i.e. number of training steps so far.
        Some environments rely on that i.e. to implement curricula.
        :param training_info: dictionary containing information about the current training session. Guaranteed to
        contain 'approx_total_training_steps' (approx because it lags a bit behind due to multiprocess synchronization)
        """
        self.training_info = training_info
```

Currently we only pass `approx_total_training_steps` to the environment which should be enough for simple curricula.
Feel free to fork the repo and add more information to this dictionary by modifying `_propagate_training_info()`
in `runner.py`. This is a new feature and further suggestions/extensions are welcome.

Note that if your environment uses a chain of wrappers (e.g. `env = Wrapper3(Wrapper2(Wrapper1(env)))`), then
it is sufficient that any Wrapper in the chain implements `TrainingInfoInterface`. Sample Factory will unwrap the
outer wrappers until it finds the first one that implements `TrainingInfoInterface`.

## Additional notes on curriculum learning

Curriculum based on the training progress is not the only way to implement curriculum learning. In most cases,
you can actually do it without knowing anything about the outer training loop.

An alternative approach is to implement curriculum based on the agent's performance in the current environment instance,
i.e. by averaging episodic statistics over the last N episodes. This way the resulting curriculum is more smooth and stochastic,
which can actually create more robust policies, since different environment instances can be at different levels of difficulty and
thus produce more diverse data.
We used this approach to train our agents against bots in the [original Sample Factory paper](https://arxiv.org/abs/2006.11751).
