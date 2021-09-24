### SampleFactory Runner API reference

Sample Factory provides a simple interface that allows users to run experiments with multiple seeds (or hyperparameter searches) with optimal distribution of work across GPUs.
The configuration of such experiments is done through Python scripts.

See [README](https://github.com/alex-petrenko/sample-factory#runner-interface) for more general information.

#### Command-line interface

##### CLI Examples:

```
Parallelize with local multiprocessing:
$ python -m sample_factory.runner.run --run=paper_doom_battle2_appo --runner=processes --max_parallel=4 --pause_between=10 --experiments_per_gpu=1 --num_gpus=4

Parallelize with Slurm:
$ python -m sample_factory.runner.run --run=megaverse_rl.runs.single_agent --runner=slurm --slurm_workdir=./megaverse_single_agent --experiment_suffix=slurm --pause_between=1 --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=12 --slurm_sbatch_template=./megaverse_rl/slurm/sbatch_template.sh --slurm_print_only=False

Parallelize with NGC (https://ngc.nvidia.com/):
$ python -m sample_factory.runner.run --run=rlgpu.run_scripts.dexterous_manipulation --runner=ngc --ngc_job_template=run_scripts/ngc_job_16g_1gpu.template --ngc_print_only=False --train_dir=/workspace/train_dir
```

##### All command-line options:
```
usage: runner.py [-h] [--train_dir TRAIN_DIR] [--run RUN]
                 [--runner {processes,slurm,ngc}]
                 [--pause_between PAUSE_BETWEEN] [--num_gpus NUM_GPUS]
                 [--experiments_per_gpu EXPERIMENTS_PER_GPU]
                 [--max_parallel MAX_PARALLEL]
                 [--experiment_suffix EXPERIMENT_SUFFIX]

# Slurm-related:
                 [--slurm_gpus_per_job SLURM_GPUS_PER_JOB]
                 [--slurm_cpus_per_gpu SLURM_CPUS_PER_GPU]
                 [--slurm_print_only SLURM_PRINT_ONLY]
                 [--slurm_workdir SLURM_WORKDIR]
                 [--slurm_partition SLURM_PARTITION]
                 [--slurm_sbatch_template SLURM_SBATCH_TEMPLATE]

# NGC-related
                 [--ngc_job_template NGC_JOB_TEMPLATE]
                 [--ngc_print_only NGC_PRINT_ONLY]
```

```
Arguments:
  -h, --help            show this help message and exit
  --train_dir TRAIN_DIR
                        Directory for sub-experiments
  --run RUN             Name of the python module that describes the run, e.g.
                        sample_factory.runner.runs.doom_battle_hybrid
  --runner {processes,slurm,ngc}
  --pause_between PAUSE_BETWEEN
                        Pause in seconds between processes
  --num_gpus NUM_GPUS   How many GPUs to use (only for local multiprocessing)
  --experiments_per_gpu EXPERIMENTS_PER_GPU
                        How many experiments can we squeeze on a single GPU
                        (-1 for not altering CUDA_VISIBLE_DEVICES at all)
  --max_parallel MAX_PARALLEL
                        Maximum simultaneous experiments (only for local multiprocessing)
  --experiment_suffix EXPERIMENT_SUFFIX
                        Append this to the name of the experiment dir

Slurm-related:
  --slurm_gpus_per_job SLURM_GPUS_PER_JOB
                        GPUs in a single SLURM process
  --slurm_cpus_per_gpu SLURM_CPUS_PER_GPU
                        Max allowed number of CPU cores per allocated GPU
  --slurm_print_only SLURM_PRINT_ONLY
                        Just print commands to the console without executing
  --slurm_workdir SLURM_WORKDIR
                        Optional workdir. Used by slurm runner to store
                        logfiles etc.
  --slurm_partition SLURM_PARTITION
                        Adds slurm partition, i.e. for "gpu" it will add "-p
                        gpu" to sbatch command line
  --slurm_sbatch_template SLURM_SBATCH_TEMPLATE
                        Commands to run before the actual experiment (i.e.
                        activate conda env, etc.) Example: https://github.com/alex-petrenko/megaverse/blob/master/megaverse_rl/slurm/sbatch_template.sh
                        (typically a shell script)

NGC-related:
  --ngc_job_template NGC_JOB_TEMPLATE
                        NGC command line template, specifying instance type, docker container, etc.
  --ngc_print_only NGC_PRINT_ONLY
                        Just print commands to the console without executing
```


#### Runner script API

A typical runner script:

```
from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [0, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888, 9999]),
    ('env', ['doom_my_way_home', 'doom_deadly_corridor', 'doom_defend_the_center', 'doom_defend_the_line', 'doom_health_gathering', 'doom_health_gathering_supreme']),
])

_experiments = [
    Experiment(
        'basic_envs_fs4',
        'python -m sample_factory.algorithms.appo.train_appo --train_for_env_steps=500000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --wide_aspect_ratio=False',
        _params.generate_params(randomize=False),
    ),
]
RUN_DESCRIPTION = RunDescription('doom_basic_envs_appo', experiments=_experiments)
```

Runner script should expose a RunDescription object named `RUN_DESCRIPTION` that contains a list of experiments to run and some auxiliary parameters.
`RunDescription` parameter reference:

```
    def __init__(
            self, run_name, experiments, experiment_dirs_sf_format=True,
            experiment_arg_name='--experiment', experiment_dir_arg_name='--train_dir',
            customize_experiment_name=True, param_prefix='--',
    ):
        """
        :param run_name: overall name of the experiment and the name of the root folder
        :param experiments: a list of Experiment objects to run
        :param experiment_dirs_sf_format: adds an additional --experiments_root parameter, used only by SampleFactory.
               set to False for other applications.
        :param experiment_arg_name: CLI argument of the underlying experiment that determines it's unique name
               to be generated by the runner. Default: --experiment
        :param experiment_dir_arg_name: CLI argument for the root train dir of your experiment. Default: --train_dir
        :param customize_experiment_name: whether to add a hyperparameter combination to the experiment name
        :param param_prefix: most experiments will use "--" prefix for each parameter, but some apps don't have this
               prefix, i.e. with Hydra you should set it to empty string.
        """
```
