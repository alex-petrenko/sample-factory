# IsaacGym

### Installation

Install IsaacGym from NVIDIA at https://developer.nvidia.com/isaac-gym. Installation instructions can be found in the package's docs folder. Python 3.8 is compatable with both IsaacGym and Sample-Factory

Install IsaacGymEnvs from https://github.com/NVIDIA-Omniverse/IsaacGymEnvs. 


### Running Experiments

Run IsaacGym experiments using scripts from the `sf_examples.isaacgym_examples` folder. Currently, we support the AllegroHand, Ant, Anymal, AnymalTerrain, BallBalance, Cartpole , Humanoid, and ShadowHand  environments out of the box, and more environments can be added in `train_isaacgym.py`.

To run an experiment in the Ant environment:
```
python -m sf_examples.isaacgym_examples.train_isaacgym --actor_worker_gpus 0 --env=Ant --train_for_env_steps=100000000  --experiment=isaacgym_ant
```

Multiple experiments can be run in parallel using the experiment launcher. See the `experiments` folder in `sf_examples.isaacgym_examples` for examples. To run multiple Ant and Humanoid experiments, run:
```
python -m sample_factory.launcher.run --run=sf_examples.isaacgym_examples.experiments.isaacgym_basic_envs --backend=processes --max_parallel=2 --experiments_per_gpu=2 --num_gpus=1
```

### Results

#### Reports

1. We tested the IsaacGym Ant and Humanoid environments with and without recurrence. When using an RNN and recurrence, the Ant and Humanoid environments see an improvement in sample efficiency. However, there is a decrease in wall time efficiency.
    - https://wandb.ai/andrewzhang505/sample_factory/reports/IsaacGym-Ant-and-Humanoid--VmlldzozMDUxNTky

2. The AllegroHand environment was tested with and without return normalization. Return normalization is essential to this environment as it improved the performance by around 200%
    - https://wandb.ai/andrewzhang505/sample_factory/reports/IsaacGym-AllegroHand--VmlldzozMDUxNjA2

#### Models

| Environment | HuggingFace Hub Models                                     | Evaluation Metrics  |
| ----------- | ---------------------------------------------------------- | ------------------- |
| Ant         | https://huggingface.co/andrewzhang505/isaacgym_ant         | 11830.10 +/- 875.26 |
| Humanoid    | https://huggingface.co/andrewzhang505/isaacgym_humanoid    | 8839.07 +/- 407.26  |
| AllegroHand | https://huggingface.co/andrewzhang505/isaacgym_allegrohand | 3608.18 +/- 1062.94 |

#### Videos

##### Ant Environment

<p align="center">
<video class="w-full" src="https://huggingface.co/andrewzhang505/isaacgym_ant/resolve/main/replay.mp4" controls="" autoplay="" loop=""></video></p>

##### Humanoid Environment

<p align="center">
<video class="w-full" src="https://huggingface.co/andrewzhang505/isaacgym_humanoid/resolve/main/replay.mp4" controls="" autoplay="" loop=""></video></p>

##### AllegroHand Environment

<p align="center">
<video class="w-full" src="https://huggingface.co/andrewzhang505/isaacgym_allegrohand/resolve/main/replay.mp4" controls="" autoplay="" loop=""></video></p>