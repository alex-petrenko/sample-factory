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

## cfg.json

Once the new experiment is started, a directory containing experiment-related files is created in `--train_dir`
location (or `./train_dir` in `cwd` if `--train_dir` is not passed from command line). This directory contains a file
`cfg.json` where all the experiment parameters are saved (including those instantiated from their default values).

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
`cfg.json` file. If you want to start a new experiment, delete the old experiment folder or change the experiment name.
You can also use `--restart_behavior=[resume|restart|overwrite]` to control this behavior.
        
- `--train_dir` location for all experiments folders, defaults to `./train_dir`.

- `--num_workers` defaults to number of logical cores in the system, which will give the best throughput in
most scenarios.

- `--num_envs_per_worker` will greatly affect the performance. Large values (20-30) improve hardware utilization but
increase memory usage and policy lag. See example command lines below to find a value that works for your system.
_Must be even_ for the double-buffered sampling to work. Disable double-buffered sampling by setting `--worker_num_splits=1`
to use odd number of envs per worker (e.g. 1 env per worker). (Default: 2)

## Full list of parameters

Please see the [Full Parameter Reference](cfg-params.md) auto-generated using the `--help`
flag for the full list of available command line arguments.