# MuJoCo Integrations

### Installation

Install Sample-Factory with MuJoCo dependencies with PyPI:

```pip install sample-factory[mujoco]```

### Running Experiments

Run MuJoCo experiments with the scripts in `sf_examples.mujoco_examples`. The default parameters have been chosen to match CleanRL's results in the report below.

### Results

#### Video Examples

#### Reports

1. Sample-Factory was benchmarked on MuJoCo against CleanRL. Sample-Factory was able to achieve similar sample efficiency as CleanRL using the same parameters.
- https://wandb.ai/andrewzhang505/sample_factory/reports/MuJoCo-Sample-Factory-vs-CleanRL-w-o-EnvPool--VmlldzoyMjMyMTQ0

2. Sample-Factory can run experiments synchronously or asynchronously, with asynchronous execution usually having worse sample efficiency but runs faster. MuJoCo's environments were compared using the two modes in Sample-Factory
- https://wandb.ai/andrewzhang505/sample_factory/reports/MuJoCo-Synchronous-vs-Asynchronous--VmlldzoyMzEzNDUz

#### Models

Various APPO models trained on MuJoCo environments are uploaded to the HuggingFace Hub. 

- Ant: https://huggingface.co/andrewzhang505/sample-factory-2-mujoco-ant
- Half Cheetah: https://huggingface.co/andrewzhang505/sample-factory-2-mujoco-halfcheetah

To download the models, use Git LFS to clone the model repo. The TenserFlow training metrics can be viewed in the "Training Metrics" tab.