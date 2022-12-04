# Configuration

Sample Factory experiments are configured via command line parameters. The following command will print the help message
for the algorithm-environment combination containing the list of all parameters, their descriptions, and their default values:

```bash
python -m sf_examples.train_gym_env --env=CartPole-v1 --help
```

(replace `train_gym_env` with your own training script name and `CartPole-v1` with a different environment name to
get information about parameters specific to this particular environment).

Default parameter values and their help strings are defined in `sample_factory/cfg/cfg.py`.
Besides that, additional parameters can be defined in specific environment integrations, for example in
`sf_examples/envpool/mujoco/envpool_mujoco_params.py`.

## config.json

Once the new experiment is started, a directory containing experiment-related files is created in `--train_dir`
location (or `./train_dir` in `cwd` if `--train_dir` is not passed from command line). This directory contains a file
`config.json` where all the experiment parameters are saved (including those instantiated from their default values).

In addition to that, selected parameter values are printed to the console and thus are saved to `sf_log.txt` file in the experiment directory.
Running an experiment and then stopping it to check the parameter values is a good practice to make sure
that the experiment is configured as expected.

## Key parameters

- `--env` (required) full name that uniquely identifies the environment as it is registered in the environment registry
(see `register_env()` function).

- `--experiment` a name that uniquely identifies the experiment and the experiment folder. E.g. `--experiment=my_experiment`.
If the experiment folder with the name already exists the experiment (by default) will be _resumed_!
Resuming experiments after a stop is the default behavior in Sample Factory. 
When the experiment is resumed from command line are taken into account, unspecified parameters will be loaded from the existing experiment
`config.json` file. If you want to start a new experiment, delete the old experiment folder or change the experiment name.
You can also use `--restart_behavior=[resume|restart|overwrite]` to control this behavior.
        
- `--train_dir` location for all experiments folders, defaults to `./train_dir`.

- `--num_workers` defaults to number of logical cores in the system, which will give the best throughput in
most scenarios.

- `--num_envs_per_worker` will greatly affect the performance. Large values (15-30) improve hardware utilization but
increase memory usage and policy lag. _Must be even_ for the double-buffered sampling to work. Disable double-buffered sampling by setting `--worker_num_splits=1`
to use odd number of envs per worker (e.g. 1 env per worker). (Default: 2)
A good rule of thumb is to set this to relatively low value (e.g. 4 or 8 for common envs) and then increase it until you see
no more performance improvements or you start losing sample efficiency due to the [policy lag](../07-advanced-topics/policy-lag.md).

- `--rollout` is the length of trajectory collected by each agent.

- `--batch_size` is the minibatch size for SGD.
- `--num_batches_per_epoch` is the number of minibatches the training batch (dataset) is split into.
- `--num_epochs` is the number of epochs on the learner over one training batch (dataset).

The above six parameters (`batch_size, num_batches_per_epoch, rollout, num_epochs, num_workers, num_envs_per_worker`) have the
biggest influence on the data regime of the RL algorithm and thus on the sample efficiency and the training speed.

`num_workers`, `num_envs_per_worker`, and `rollout` define how many samples are collected per iteration (one rollout for all envs), which is
`sampling_size = num_workers * num_envs_per_worker * rollout` (note that this is further multiplied by env's `num_agents` for multi-agent envs).

`batch_size` and `num_batches_per_epoch` define how many samples are used for training per iteration.

If `sampling_size >> batch_size` then we will need many iterations of training to go through the data, which
will make some experience stale by the time it is used for training (**policy lag**). See [Policy Lag](../07-advanced-topics/policy-lag.md)
for additional information.

## Evaluation script parameters

Evaluation scripts (i.e. `sf_examples/atari/enjoy_atari.py`) use the same configuration parameters as training scripts
for simplicity, although of course many of them are ignored as they don't affect evaluation.

In addition to that, evaluation scripts provide additional parameters, see `add_eval_args()` in `sample_factory/cfg/cfg.py`.
[HuggingFace Hub integration guide](../10-huggingface/huggingface.md) provides a good overview of the important parameters
such as `--save_video`, check it out!

## Full list of parameters

Please see the [Full Parameter Reference](cfg-params.md) auto-generated using the `--help`
flag for the full list of available command line arguments.