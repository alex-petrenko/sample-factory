# Quad-Swarm-RL Integrations

### Installation

Clone https://github.com/Zhehui-Huang/quad-swarm-rl into your home directory

Install dependencies in your conda environment
```
cd ~/quad-swarm-rl
pip install -e .
```

Note: if you have any error with bezier, run:
```
BEZIER_NO_EXTENSION=true pip install bezier==2020.5.19
pip install -e .
```

### Running Experiments

The environments can be run from the `quad_swarm_rl` folder in the downloaded `quad-swarm-rl` directory instead of from `sample-factory` directly. 

Experiments can be run with the `train` script and viewed with the `enjoy` script. If you are running custom experiments, it is recommended to use the `quad_multi_mix_baseline` runner script and make any modifications as needed. See `sf2_single_drone` and `sf2_multi_drone` runner scripts for an examples.

The quadrotor environments have many unique parameters that can be found in `quadrotor_params.py`. Some relevant params for rendering results include `--quads_view_mode` which can be set to local or global for viewing multi-drone experiments, and `--quads_mode` which determines which scenario(s) to train on, with `mix` using all scenarios.

### Results

#### Reports

1. Comparison using a single drone between normalized (input and return normalization) and un-normalized experiments. Normalization helped the drones learn in around half the number of steps.
    - https://wandb.ai/andrewzhang505/sample_factory/reports/Quad-Swarm-RL--VmlldzoyMzU1ODQ1
2. Experiments with 8 drones in scenarios with and without obstacles. All experiments used input and return normalization. Research and development are still being done on multi-drone scenarios to reduce the number of collisions.
    - https://wandb.ai/andrewzhang505/sample_factory/reports/Quad-Swarm-RL-Multi-Drone--VmlldzoyNDkwNDQ0

#### Models

| Description | HuggingFace Hub Models | Evaluation Metrics |
| ----------- | ---------------------- | ------------------ |
| Single drone with normalization | https://huggingface.co/andrewzhang505/quad-swarm-single-drone-sf2 | 0.03 +/- 1.86 |
| Multi drone without obstacles | https://huggingface.co/andrewzhang505/quad-swarm-rl-multi-drone-no-obstacles | -0.40 +/- 4.47 |
| Multi drone with obstacles | https://huggingface.co/andrewzhang505/quad-swarm-rl-multi-drone-obstacles | -2.84 +/- 3.71 |

#### Videos

Single drone with normalization flying between dynamic goals.

<p align="center">
<video class="w-full" src="https://huggingface.co/andrewzhang505/quad-swarm-single-drone-sf2/resolve/main/replay.mp4" controls="" autoplay="" loop=""></video></p>
