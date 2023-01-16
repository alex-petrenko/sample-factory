# Brax

## Installation

Installing Brax with CUDA acceleration can be a bit tricky. There are some notes here: https://github.com/google/jax#pip-installation-gpu-cuda.

I had the best luck with the following steps:

```shell
# Create a Conda environment or use your existing one
conda create --name sf_brax python=3.9

# Activate the environment
conda activate sf_brax

# cuda-nvcc seems to be necessary, and the order of conda repos matters
conda install cudatoolkit cuda-nvcc -c conda-forge -c nvidia

# Install Jax/Jaxlib from a custom repo
pip install  --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install Brax
pip install brax
```

Then follow general instructions to install Sample Factory if you need to.

## Running Experiments

```shell
# to avoid OOM issues it is advised to disable vram preallocation (might not be necessary)
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# train for 100M steps with default hyperparameters
python -m sf_examples.brax.train_brax --env=ant --experiment=ant_brax

# evaluate the agent
python -m sf_examples.brax.enjoy_brax --env=ant --experiment=ant_brax

# Brax software renderer is quite slow, so you can render a video offscreen instead of visualizing it in a window
# Video will be saved to the experiment directory
python -m sf_examples.brax.enjoy_brax --env=ant --experiment=ant_brax --save_video --video_name=ant
```

## Results

### Reports

The following reports were created after running a [launcher script](https://github.com/alex-petrenko/sample-factory/blob/6aa87f2d416b9fad874b299d864a522c887c238a/sf_examples/brax/experiments/brax_basic_envs.py)
on a Slurm cluster with the following command:

```shell
python -m sample_factory.launcher.run --run=sf_examples.brax.experiments.brax_basic_envs --backend=slurm --slurm_workdir=./slurm_brax --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=8 --slurm_sbatch_template=./sf_examples/brax/experiments/sbatch_timeout_brax.sh --pause_between=0 --slurm_print_only=False
```

1. ant: https://api.wandb.ai/report/apetrenko/ji9jygss
2. humanoid: https://api.wandb.ai/report/apetrenko/m520i16m
3. halfcheetah: https://api.wandb.ai/report/apetrenko/7xlp3hh8
4. walker2d: https://api.wandb.ai/report/apetrenko/pvb9d11c

### Models

| Environment | HuggingFace Hub Models                                           | Evaluation Metrics   |
|-------------|------------------------------------------------------------------|----------------------|
| ant         | https://huggingface.co/apetrenko/sample_factory_brax_ant         | 12565.17 +/- 3350.51 |
| humanoid    | https://huggingface.co/apetrenko/sample_factory_brax_humanoid    | 33847.53 +/- 6327.36 |
| halfcheetah | https://huggingface.co/apetrenko/sample_factory_brax_halfcheetah | 22298.35 +/- 1882.48 |
| walker2d    | https://huggingface.co/apetrenko/sample_factory_brax_walker2d    | 5459.17 +/- 2198.74  |

Example command line used to generate a HuggingFace Hub model:

```shell
python -m sf_examples.brax.enjoy_brax \
  --env=humanoid --experiment=02_v083_brax_basic_benchmark_see_2322090_env_humanoid_u.rnn_False_n.epo_5 \
  --train_dir=/home/alex/all/projects/sf2/train_dir/v083_brax_basic_benchmark/v083_brax_basic_benchmark_slurm \
  --save_video --video_frames=500 --max_num_episodes=500 \
  --enjoy_script=sf_examples.brax.enjoy_brax --train_script=sf_examples.brax.train_brax \
  --push_to_hub --hf_repository=apetrenko/sample_factory_brax_humanoid --brax_render_res=320 --load_checkpoint_kind=best
```

### Videos

##### Ant Environment

<p align="center">
<video class="w-full" src="https://huggingface.co/apetrenko/sample_factory_brax_ant/resolve/main/replay.mp4" controls="" autoplay="" loop=""></video></p>

##### Humanoid Environment

<p align="center">
<video class="w-full" src="https://huggingface.co/apetrenko/sample_factory_brax_humanoid/resolve/main/replay.mp4" controls="" autoplay="" loop=""></video></p>
