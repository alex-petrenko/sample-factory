# Tensorboard

Sample Factory uses Tensorboard summaries.
Run Tensorboard to monitor any running or finished experiment:

```bash
tensorboard --logdir=<your_experiment_dir> --port=6006`
```

## Monitoring multiple experiments

Additionally, we provide a helper script that has nice command line interface to monitor the experiment folders using wildcards:

```bash
python -m sample_factory.utils.tb --dir=./train_dir '*name_mask*' '*another*mask*'
```

Here `--dir` parameter is the root directory with experiments, and the script will recursively search for experiment folders that match the masks.

## Monitoring experiments started by the Launcher

[Launcher API](../04-experiments/experiment-launcher.md) is a convenient way to start multiple experiments in parallel.
Such groups of experiments can be monitored with a single Tensorboard command, just specify `--logdir` pointing to the root directory with experiments.
