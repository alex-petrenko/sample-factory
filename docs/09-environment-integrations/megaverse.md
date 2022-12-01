# Megaverse

<video width="800" controls autoplay><source src="https://huggingface.co/datasets/edbeeching/sample_factory_videos/resolve/main/megaverse_grid_4.mp4" type="video/mp4"></video>

Megaverse is a dedicated high-throughput RL environment with batched GPU rendering.

This document demonstrates an example of _external_ integration, i.e. another project _using_ Sample Factory as a library.
Very likely this is going to be the most common integration scenario.

### Installation

Install Megaverse according to the readme of the repo [Megaverse](https://github.com/alex-petrenko/megaverse).
Further instructions assume that you are in a Python (or Conda) environment with a working Megaverse installation.

### Running Experiments

Run Megaverse experiments with the scripts in `megaverse_rl`.

To train a model in the `TowerBuilding` environment:

```
python -m megaverse_rl.train_megaverse --train_for_seconds=360000000 --train_for_env_steps=2000000000 --algo=APPO --gamma=0.997 --use_rnn=True --rnn_num_layers=2 --num_workers=12 --num_envs_per_worker=2 --num_epochs=1 --rollout=32 --recurrence=32 --batch_size=4096 --actor_worker_gpus 0 --env_gpu_observations=False --num_policies=1 --with_pbt=False --max_grad_norm=0.0 --exploration_loss=symmetric_kl --exploration_loss_coeff=0.001 --megaverse_num_simulation_threads=1 --megaverse_num_envs_per_instance=32 --megaverse_num_agents_per_env=1 --megaverse_use_vulkan=True --policy_workers_per_policy=2 --reward_clip=30 --env=TowerBuilding --experiment=TowerBuilding
```

To visualize the training results, use the `enjoy_megaverse` script:

```
python -m megaverse_rl.enjoy_megaverse --algo=APPO --env=TowerBuilding --experiment=TowerBuilding --megaverse_num_envs_per_instance=1 --fps=20 --megaverse_use_vulkan=True
```

Multiple experiments can be run in parallel with the launcher module. `megaverse_envs` is an example launcher script that runs megaverse envs with 5 seeds. 

```
python -m sample_factory.launcher.run --run=megaverse_rl.runs.single_agent --backend=processes --max_parallel=2  --pause_between=1 --experiments_per_gpu=2 --num_gpus=1
```

Or you could run experiments on slurm:

```
python -m sample_factory.launcher.run --run=megaverse_rl.runs.single_agent --backend=slurm --slurm_workdir=./slurm_megaverse --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/launcher/slurm/sbatch_timeout.sh --pause_between=1 --slurm_print_only=False
```


### Results

#### Reports
- We trained models in the `TowerBuilding` environment in SF2 with single agent per env.
    - https://wandb.ai/wmfrank/megaverse-benchmark/reports/Megaverse-trained-Sample-Factory--VmlldzoyNTAxMDUz


#### Models

An example APPO model trained on Megaverse environments is uploaded to the HuggingFace Hub. The models have all been trained for 2G steps.

| Environment   | HuggingFace Hub Models                                    |
| ------------- | --------------------------------------------------------- |
| TowerBuilding | https://huggingface.co/wmFrank/sample-factory-2-megaverse |


#### Tower Building with single agent

<video width="500" controls="" autoplay="" loop=""><source src="https://huggingface.co/wmFrank/sample-factory-2-megaverse/resolve/main/replay.mp4" type="video/mp4"></video>
