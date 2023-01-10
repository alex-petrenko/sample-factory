# Brax

### Installation

```
pip install sample-factory[brax]
```

or if you're installing from sources:

```
pip install -e .[brax]
```

### Running Experiments



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