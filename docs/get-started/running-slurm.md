## How to use SF2 on Slurm

This doc contains instructions for running Sample-Factory v2 on USC's brain cluster

### Logging in

Login to `brain.usc.edu` using ssh with your username and password.

After logging in, change your password with the command `passwd`

### Installing files

Start an interactive job with `srun -c40 --gres=gpu:1 --pty bash`. 
Note that you will get a message `groups: cannot find name for group ID XXXX`

Git clone Sample-Factory

Install Miniconda
- Download installer using `wget` from https://docs.conda.io/en/latest/miniconda.html#linux-installers
- Run the installer with `bash {Miniconda...sh}`

Make new conda environment `conda create --name sf2` then `conda activate sf2`

Install dependencies for sf2
- `cd sample-factory`
- `git checkout sf2`
- `pip install -e .`

### Running runner scripts

Return to the login node with `exit`

Setup slurm output folder `mkdir sf2` 

Activate your conda environment with `bash` and `conda activate sf2` then `cd sample-factory`

Run your runner script - an example mujuco runner (replace run and slurm_workdir with appropriate values)
```
python -m sample_factory.runner.run --runner=slurm --slurm_workdir=./slurm_mujoco --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/runner/slurm/sbatch_template.sh --pause_between=1 --slurm_print_only=False --run=sf_examples.mujoco_examples.experiments.mujoco_all_envs
```

### Other Helpful Commands

`squeue` - List all jobs running or queued in the cluster
