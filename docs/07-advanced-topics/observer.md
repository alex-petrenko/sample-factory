# Observer Interface

Sample Factory version 2 introduces a new feature: you can wrap the RL algorithm with a custom `Observer` object
which allows you to interact with the RL training process in an arbitrary way.

## `AlgoObserver`

The `AlgoObserver` interface is defined as follows:

```python
class AlgoObserver:
    def on_init(self, runner: Runner) -> None:
        """Called after ctor, but before signal-slots are connected or any processes are started."""
        pass

    def on_connect_components(self, runner: Runner) -> None:
        """Connect additional signal-slot pairs in the observers if needed."""
        pass

    def on_start(self, runner: Runner) -> None:
        """Called right after sampling/learning processes are started."""
        pass

    def on_training_step(self, runner: Runner, training_iteration_since_resume: int) -> None:
        """Called after each training step."""
        pass

    def extra_summaries(self, runner: Runner, policy_id: PolicyID, env_steps: int, writer: SummaryWriter) -> None:
        pass

    def on_stop(self, runner: Runner) -> None:
        pass
```

Define your own class derived from `AlgoObserver` (i.e. `MyObserver`) and register it before starting the training process:

```python
runner.register_observer(MyObserver())
``` 

Our DMLab integration provides an example of how to use `AlgoObserver` to implement custom summaries that 
aggregate information from multiple custom metrics (see `sf_examples/dmlab/train_dmlab.py`).

`AlgoObserver` is a new feature and further suggestions/extensions are welcome!