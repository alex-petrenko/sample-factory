# Megaverse Integrations

### Installation

Install Megaverse according to the readme of the repo [Megaverse](https://github.com/alex-petrenko/megaverse).

### Running Experiments

Run Megaverse experiments with the scripts in `sf_examples.megaverse_examples`.

To train a model in the `TowerBuilding` environment:

```
python -m sf_examples.megaverse_examples.train_megaverse --train_for_seconds=360000000 --train_for_env_steps=2000000000 --algo=APPO --gamma=0.997 --use_rnn=True --rnn_num_layers=2 --num_workers=10 --num_envs_per_worker=2 --num_epochs=1 --rollout=32 --recurrence=32 --batch_size=4096 --actor_worker_gpus 0 --env_gpu_observations=False --num_policies=1 --with_pbt=False --max_grad_norm=0.0 --exploration_loss=symmetric_kl --exploration_loss_coeff=0.001 --megaverse_num_simulation_threads=1 --megaverse_num_envs_per_instance=30 --megaverse_num_agents_per_env=4 --megaverse_use_vulkan=True --policy_workers_per_policy=2 --reward_clip=30 --env=megaverse_TowerBuilding --experiment=TowerBuilding
```

To visualize the training results, use the `enjoy_megaverse` script:

```
python -m sf_examples.megaverse_examples.enjoy_megaverse --algo=APPO --env=megaverse_TowerBuilding --experiment=TowerBuilding --megaverse_num_envs_per_instance=1 --fps=20 --megaverse_use_vulkan=True
```

Multiple experiments can be run in parallel with the runner module. `megaverse_envs` is an example runner script that runs atari envs with 4 seeds. 

```
python -m sample_factory.runner.run --run=sf_examples.megaverse_examples.experiments.megaverse_envs --runner=processes --max_parallel=8  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1
```

Or you could run experiments on slurm:

```
python -m sample_factory.runner.run --runner=slurm --slurm_workdir=./slurm_megaverse --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/runner/slurm/sbatch_template.sh --pause_between=1 --slurm_print_only=False --run=sf_examples.megaverse_examples.experiments.megaverse_envs
```


### Results

#### Reports
- We trained models in the `TowerBuilding` environment in SF1 and SF2 with 4 agents per env.
    - https://wandb.ai/wmfrank/megaverse-benchmark/reports/Megaverse-trained-Sample-Factory--VmlldzoyNTAxMDUz


#### Models

An example APPO model trained on Megaverse environments is uploaded to the HuggingFace Hub. The models have all been trained for 2G steps.

| Environment | HuggingFace Hub Models                                    |
| -------- |-----------------------------------------------------------|
| TowerBuilding    | https://huggingface.co/wmFrank/sample-factory-2-megaverse |


https://user-images.githubusercontent.com/30235642/195955230-6fd36729-7356-41ca-87ce-bd231b01e8d4.mp4


https://user-images.githubusercontent.com/30235642/195955237-062e7c1c-1d0b-4ec7-8a0c-904f98f29c7b.mp4




#### Videos

##### Tower Building with single agent

<video width="500" controls><source src="https://user-images.githubusercontent.com/30235642/195955230-6fd36729-7356-41ca-87ce-bd231b01e8d4.mp4" type="video/mp4"></video>

##### Tower Building wiith four agents

<video width="1000" controls><source src="https://user-images.githubusercontent.com/30235642/195955237-062e7c1c-1d0b-4ec7-8a0c-904f98f29c7b.mp4" type="video/mp4"></video>
