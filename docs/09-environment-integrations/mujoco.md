# MuJoCo
<video width="800" controls autoplay><source src="https://huggingface.co/datasets/edbeeching/sample_factory_videos/resolve/main/mujoco_grid_9.mp4" type="video/mp4"></video>
### Installation

Install Sample Factory with MuJoCo dependencies with PyPI:

```
pip install sample-factory[mujoco]
```

### Running Experiments

Run MuJoCo experiments with the scripts in `sf_examples.mujoco`.
The default parameters have been chosen to match CleanRL's results in the report below (please note
that we can achieve even faster training on a multi-core machine with more optimal parameters).

To train a model in the `Ant-v4` environment:

```
python -m sf_examples.mujoco.train_mujoco --algo=APPO --env=mujoco_ant --experiment=<experiment_name>
```

To visualize the training results, use the `enjoy_mujoco` script:

```
python -m sf_examples.mujoco.enjoy_mujoco --algo=APPO --env=mujoco_ant --experiment=<experiment_name>
```

Multiple experiments can be run in parallel with the launcher module. `mujoco_all_envs` is an example launcher script that runs all mujoco envs with 10 seeds. 

```
python -m sample_factory.launcher.run --run=sf_examples.mujoco.experiments.mujoco_all_envs --backend=processes --max_parallel=4  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1 --experiment_suffix=0
```

#### List of Supported Environments

Specify the environment to run with the `--env` command line parameter. The following MuJoCo v4 environments are supported out of the box, and more environments can be added as needed in `sf_examples.mujoco.mujoco.mujoco_utils`

| MuJoCo Environment Name   | Sample Factory Command Line Parameter |
| ------------------------- |---------------------------------------|
| Ant-v4                    | mujoco_ant                            |
| HalfCheetah-v4            | mujoco_halfcheetah                    |
| Hopper-v4                 | mujoco_hopper                         |
| Humanoid-v4               | mujoco_humanoid                       |
| Walker2d-v4               | mujoco_walker                         |
| InvertedDoublePendulum-v4 | mujoco_doublependulum                 |
| InvertedPendulum-v4       | mujoco_pendulum                       |
| Reacher-v4                | mujoco_reacher                        |
| Swimmer-v4                | mujoco_swimmer                        |


### Results

#### Reports

1. Sample Factory was benchmarked on MuJoCo against CleanRL. Sample-Factory was able to achieve similar sample efficiency as CleanRL using the same parameters.
    - https://wandb.ai/andrewzhang505/sample_factory/reports/MuJoCo-Sample-Factory-vs-CleanRL-w-o-EnvPool--VmlldzoyMjMyMTQ0

2. Sample Factory can run experiments synchronously or asynchronously, with asynchronous execution usually having worse sample efficiency but runs faster. MuJoCo's environments were compared using the two modes in Sample-Factory
    - https://wandb.ai/andrewzhang505/sample_factory/reports/MuJoCo-Synchronous-vs-Asynchronous--VmlldzoyMzEzNDUz

3. Sample Factory comparison with CleanRL in terms of wall time. Both experiments are run on a 16 core machine with 1 GPU. Sample-Factory was able to complete 10M samples 5 times as fast as CleanRL
    - https://wandb.ai/andrewzhang505/sample_factory/reports/MuJoCo-Sample-Factory-vs-CleanRL-Wall-Time--VmlldzoyMzg2MDA3

#### Models

Various APPO models trained on MuJoCo environments are uploaded to the HuggingFace Hub. The models have all been trained for 10M steps. Videos of the agents after training can be found on the HuggingFace Hub.

The models below are the best models from the experiment against CleanRL above. The evaluation metrics here are obtained by running the model 10 times.

| Environment               | HuggingFace Hub Models                                                       | Evaluation Metrics  |
| ------------------------- | ---------------------------------------------------------------------------- | ------------------- |
| Ant-v4                    | https://huggingface.co/andrewzhang505/sample-factory-2-mujoco-ant            | 5876.09 +/- 166.99  |
| HalfCheetah-v4            | https://huggingface.co/andrewzhang505/sample-factory-2-mujoco-halfcheetah    | 6262.56 +/- 67.29   |
| Humanoid-v4               | https://huggingface.co/andrewzhang505/sample-factory-2-mujoco-humanoid       | 5439.48 +/- 1314.24 |
| Walker2d-v4               | https://huggingface.co/andrewzhang505/sample-factory-2-mujoco-walker         | 5487.74 +/- 48.96   |
| Hopper-v4                 | https://huggingface.co/andrewzhang505/sample-factory-2-mujoco-hopper         | 2793.44 +/- 642.58  |
| InvertedDoublePendulum-v4 | https://huggingface.co/andrewzhang505/sample-factory-2-mujoco-doublependulum | 9350.13 +/- 1.31    |
| InvertedPendulum-v4       | https://huggingface.co/andrewzhang505/sample-factory-2-mujoco-pendulum       | 1000.00 +/- 0.00    |
| Reacher-v4                | https://huggingface.co/andrewzhang505/sample-factory-2-mujoco-reacher        | -4.53 +/- 1.79      |
| Swimmer-v4                | https://huggingface.co/andrewzhang505/sample-factory-2-mujoco-swimmer        | 117.28 +/- 2.91     |

#### Videos

Below are some video examples of agents in various MuJoCo envioronments. Videos for all environments can be found in the HuggingFace Hub pages linked above.

##### HalfCheetah-v4
<p align="center">
<video class="w-full" src="https://huggingface.co/andrewzhang505/sample-factory-2-mujoco-halfcheetah/resolve/main/replay.mp4" controls="" autoplay="" loop=""></video></p>

##### Ant-v4
<p align="center">
<video class="w-full" src="https://huggingface.co/andrewzhang505/sample-factory-2-mujoco-ant/resolve/main/replay.mp4" controls="" autoplay="" loop=""></video></p>

##### InvertedDoublePendulum-v4
<p align="center">
<video class="w-full" src="https://huggingface.co/andrewzhang505/sample-factory-2-mujoco-doublependulum/resolve/main/replay.mp4" controls="" autoplay="" loop=""></video></p>