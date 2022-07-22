# How to use SF2 on Slurm

This doc contains instructions for running Sample-Factory v2 using slurm

### Setting up

Login to your slurm login node using ssh with your username and password.

Start an interactive job with `srun -c40 --gres=gpu:1 --pty bash` to install files to your NFS. 
Note that you may get a message `groups: cannot find name for group ID XXXX`

Git clone Sample-Factory

Install Miniconda
- Download installer using `wget` from https://docs.conda.io/en/latest/miniconda.html#linux-installers
- Run the installer with `bash {Miniconda...sh}`

Make new conda environment `conda create --name sf2` then `conda activate sf2`

Install dependencies for sf2
- `cd sample-factory`
- `git checkout sf2`
- `pip install -e .`

### Necessary scripts in SF2

To run a custom runner script for SF2 on slurm, you may need to write your own slurm_sbatch_template and/or runner script.

slurm_sbatch_template is a bash script that run by slurm before your python script. It includes commands to activate your conda environment etc. See and example at `./sample_factory/runner/slurm/sbatch_template.sh `

The runner script controls the python command slurm will run. Examples are located in `sf_examples`. You can run multiple experiments with different parameters using `ParamGrid`.   

### Running runner scripts

Return to the login node with `exit`

Setup slurm output folder `mkdir sf2` 

Activate your conda environment with `bash` and `conda activate sf2` then `cd sample-factory`

Run your runner script - an example mujuco runner (replace run, slurm_sbatch_template, and slurm_workdir with appropriate values)
```
python -m sample_factory.runner.run --runner=slurm --slurm_workdir=./slurm_mujoco --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/runner/slurm/sbatch_template.sh --pause_between=1 --slurm_print_only=False --run=sf_examples.mujoco_examples.experiments.mujoco_all_envs
```

The `slurm_gpus_per_job` and `slurm_cpus_per_gpu` determine the resources allocated to each job. You can view the jobs without running them by setting `slurm_print_only=True`.

You can view the status of your jobs on nodes or the queue with `squeue` and view the outputs of your experiments with `tail -f {slurm_workdir}/*.out`. Cancel your jobs with `scancel {job_id}`
