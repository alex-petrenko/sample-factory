# Sample Factory on Slurm

This section contains instructions for running Sample Factory experiments using Slurm.

## Setting up

Login to your Slurm login node using ssh with your username and password. Start an interactive job with `srun` to install files to your NFS. 

```
srun -c40 --gres=gpu:1 --pty bash
```

Note that you may get a message `groups: cannot find name for group ID XXXX` which is not an error.

Install Miniconda:

- Download installer using `wget` from https://docs.conda.io/en/latest/miniconda.html#linux-installers
- Run the installer with `bash {Miniconda...sh}`

Make new conda environment `conda create --name sf2` then `conda activate sf2`

Download Sample Factory and install dependencies, for example:

```bash
git clone https://github.com/alex-petrenko/sample-factory.git
cd sample-factory
git checkout sf2
pip install -e .
# install additional env dependencies here if needed
```

## Necessary scripts in Sample Factory

To run a custom launcher script for Sample Factory on slurm, you may need to write your own slurm_sbatch_template and/or launcher script.

slurm_sbatch_template is a bash script that run by slurm before your python script. It includes commands to activate your conda environment etc. See an example at `./sample_factory/launcher/slurm/sbatch_timeout.sh`. Variables in the bash script can be added in `sample_factory.launcher.run_slurm`.

The launcher script controls the Python command slurm will run. Examples are located in `sf_examples`. You can run multiple experiments with different parameters using `ParamGrid`.

### Timeout script

If your slurm cluster has time limits for jobs, you can use the `sbatch_timeout.sh` bash script to launch jobs that timeout and requeue themselves before the time limit. 

The time limit can be set with the `--slurm_timeout` command line argument. It defaults to `0` which runs the job with no time limit.
It is recommended the timeout be set to slightly less than the time limit of your job. For example, if the time limit is 24 hours, you should set `--slurm_timeout=23h`

## Running launcher scripts on Slurm

Activate your conda environment `conda activate sf2` then `cd sample-factory`

Run your launcher script - an example mujoco launcher (replace run, slurm_sbatch_template, and slurm_workdir with appropriate values)
```
python -m sample_factory.launcher.run --run=sf_examples.mujoco.experiments.mujoco_all_envs --backend=slurm --slurm_workdir=./slurm_mujoco --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/launcher/slurm/sbatch_timeout.sh --pause_between=1 --slurm_print_only=False 
```

The `slurm_gpus_per_job` and `slurm_cpus_per_gpu` determine the resources allocated to each job. You can view the jobs without running them by setting `slurm_print_only=True`.

You can view the status of your jobs on nodes or the queue with `squeue` and view the outputs of your experiments with `tail -f {slurm_workdir}/*.out`. Cancel your jobs with `scancel {job_id}`
