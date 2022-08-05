# Atari Integrations

### Installation

Install Sample-Factory with Atari dependencies with PyPI:

```
pip install sample-factory[atari]
```

### Running Experiments

Run Atari experiments with the scripts in `sf_examples.atari_examples`.

The default parameters have been chosen to match CleanRL's configuration (see reports below) and are not tuned for throughput.
TODO: provide parameters that result in faster training.
 

To train a model in the `BreakoutNoFrameskip-v4` enviornment:

```
python -m sf_examples.atari_examples.train_atari --algo=APPO --env=atari_breakout --experiment="Experiment Name"
```

To visualize the training results, use the `enjoy_atari` script:

```
python -m sf_examples.atari_examples.enjoy_atari --algo=APPO --env=atari_breakout --experiment="Experiment Name"
```

Multiple experiments can be run in parallel with the runner module. `atari_envs` is an example runner script that runs atari envs with 4 seeds. 

```
python -m sample_factory.runner.run --run=sf_examples.atari_examples.experiments.atari_envs --runner=processes --max_parallel=8  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1
```

#### List of Supported Environments

Specify the environment to run with the `--env` command line parameter. The following Atari v4 environments are supported out of the box, and more enviornments can be added as needed in `sf_examples.atari_examples.atari.atari_utils`

| Atari Environment Name   | Atari Command Line Parameter |
| -----------------------   | ------------------------------------- |
| BreakoutNoFrameskip-v4    | atari_breakout                        |
| PongNoFrameskip-v4        | atari_pong                            |
| BeamRiderNoFrameskip-v4   | atari_beamrider                       |


### Results

#### Reports

- Sample-Factory was benchmarked on Atari against CleanRL and Baselines. Sample-Factory was able to achieve similar sample efficiency as CleanRL and Baselines using the same parameters.
    - https://wandb.ai/wmfrank/atari-benchmark/reports/Atari-Sample-Factory2-Baselines-CleanRL--VmlldzoyMzEyNjIw


#### Models

Various APPO models trained on Atari environments are uploaded to the HuggingFace Hub. The models have all been trained for 10M steps. Videos of the agents after training can be found on the HuggingFace Hub.

| Environment | HuggingFace Hub Models | Evaluation Metrics |
| ----------- | ---------------------- | ------------------ |
| BreakoutNoFrameskip-v4    | https://huggingface.co/wmFrank/sample-factory-2-atari-breakout | 30.20 ± 23.45 |
| PongNoFrameskip-v4        | https://huggingface.co/wmFrank/sample-factory-2-atari-pong | 13.50 ± 7.43 |
| BeamRiderNoFrameskip-v4   | https://huggingface.co/wmFrank/sample-factory-2-atari-beamrider | 3848.00 ± 308.00 |

#### Videos

Below are some video examples of agents in various Atari envioronments. Videos for all enviornments can be found in the HuggingFace Hub pages linked above.

##### BreakoutNoFrameskip-v4

<video width="500" controls><source src="https://huggingface.co/wmFrank/sample-factory-2-atari-breakout/resolve/main/replay.mp4" type="video/mp4"></video>

##### PongNoFrameskip-v4

<video width="500" controls><source src="https://huggingface.co/wmFrank/sample-factory-2-atari-pong/resolve/main/replay.mp4" type="video/mp4"></video>

##### BeamRiderNoFrameskip-v4

<video width="500" controls><source src="https://huggingface.co/wmFrank/sample-factory-2-atari-beamrider/resolve/main/replay.mp4" type="video/mp4"></video>
