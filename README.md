[![tests](https://github.com/alex-petrenko/sample-factory/actions/workflows/test-ci.yml/badge.svg?branch=sf2)](https://github.com/alex-petrenko/sample-factory/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/alex-petrenko/sample-factory/branch/sf2/graph/badge.svg?token=9EHMIU5WYV)](https://codecov.io/gh/alex-petrenko/sample-factory)
[![pre-commit](https://github.com/alex-petrenko/sample-factory/actions/workflows/pre-commit.yml/badge.svg?branch=sf2)](https://github.com/alex-petrenko/sample-factory/actions/workflows/pre-commit.yml)
[![docs](https://github.com/alex-petrenko/sample-factory/actions/workflows/docs.yml/badge.svg)](https://samplefactory.dev)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/alex-petrenko/sample-factory/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/sample-factory)](https://pepy.tech/project/sample-factory)
[<img src="https://img.shields.io/discord/987232982798598164?label=discord">](https://discord.gg/BCfHWaSMkr)
<!-- [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/wmFrank/sample-factory/master.svg)](https://results.pre-commit.ci/latest/github/wmFrank/sample-factory/master)-->
<!-- [![wakatime](https://wakatime.com/badge/github/alex-petrenko/sample-factory.svg)](https://wakatime.com/badge/github/alex-petrenko/sample-factory)-->


# Sample Factory

High-throughput synchronous and asynchronous reinforcement learning.

This Readme provides only a brief overview of the library.
For documentation and other resources please refer to the links below:

* **Documentation:** [https://samplefactory.dev](https://samplefactory.dev) 

* **Paper:** https://arxiv.org/abs/2006.11751

* **Citation:** [BibTeX](https://github.com/alex-petrenko/sample-factory#citation)

* **Discord:** [https://discord.gg/BCfHWaSMkr](https://discord.gg/BCfHWaSMkr)

* **Talk (circa 2021):** https://youtu.be/lLG17LKKSZc

### What is Sample Factory?

Sample Factory is one of the fastest RL libraries.
Instead of implementing multiple algorithm families, we focus on one very efficient
implementation of policy gradients (PPO).

**Key features:**

* Highly optimized algorithm architecture for maximum learning throughput
* Synchronous and asynchronous training regimes
* Serial (single-process) mode for easy debugging
* Optimal performance in both CPU-based and GPU-accelerated environments
* Single- & multi-agent training, self-play, supports training multiple policies at once on one or many GPUs
* Population-Based Training (PBT)
* Discrete, continuous, hybrid action spaces
* Vector-based, image-based, dictionary observation spaces
* Automatically creates a model architecture by parsing action/observation space specification. Supports custom model architectures
* Library is designed to be imported into other projects, custom environments are first-class citizens
* Elaborate WandB and Tensorboard summaries, custom metrics
* HuggingFace integration (upload trained models and metrics to the Hub)
* Automatic testing with GitHub Actions
* Multiple example environment integrations with tuned parameters, i.e. below are ViZDoom, IsaacGym, DMLab-30, Megaverse, Mujoco, and Atari agents trained with Sample Factory:

<p align="middle">
<img src="https://github.com/alex-petrenko/sf_assets/blob/main/gifs/vizdoom.gif?raw=true" width="360" alt="VizDoom agents traned using Sample Factory 2.0">
<img src="https://github.com/alex-petrenko/sf_assets/blob/main/gifs/isaac.gif?raw=true" width="360" alt="IsaacGym agents traned using Sample Factory 2.0">
<br/>
<img src="https://github.com/alex-petrenko/sf_assets/blob/main/gifs/dmlab.gif?raw=true" width="380" alt="DMLab-30 agents traned using Sample Factory 2.0">
<img src="https://github.com/alex-petrenko/sf_assets/blob/main/gifs/megaverse.gif?raw=true" width="340" alt="Megaverse agents traned using Sample Factory 2.0">
<br/>
<img src="https://github.com/alex-petrenko/sf_assets/blob/main/gifs/mujoco.gif?raw=true" width="390" alt="Mujoco agents traned using Sample Factory 2.0">
<img src="https://github.com/alex-petrenko/sf_assets/blob/main/gifs/atari.gif?raw=true" width="330" alt="Atari agents traned using Sample Factory 2.0">
</p>


## Installation

Just install from PyPI:

```pip install sample-factory```

SF is known to work on Linux and macOS. There is no Windows support at this time.
Please refer to the [documentation](https://samplefactory.dev) for additional environment-specific installation notes.

## Quickstart

Use command line to train an agent using one of the existing integrations, e.g. Mujoco (might need to run `pip install sample-factory[mujoco]`):

```bash
python -m sf_examples.mujoco.train_mujoco --env=mujoco_ant --experiment=Ant --train_dir=./train_dir
```

Stop the experiment when the desired performance is reached and then evaluate the agent:

```bash
python -m sf_examples.mujoco.enjoy_mujoco --env=mujoco_ant --experiment=Ant --train_dir=./train_dir
```

Do the same in a pixel-based VizDoom environment (might need to run `pip install sample-factory[vizdoom]`, please also see docs for VizDoom-specific instructions):

```bash
python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --experiment=DoomBasic --train_dir=./train_dir --num_workers=16 --num_envs_per_worker=10 --train_for_env_steps=1000000
python -m sf_examples.vizdoom.enjoy_vizdoom --env=doom_basic --experiment=DoomBasic --train_dir=./train_dir
```

Monitor any running or completed experiment with Tensorboard:

```bash
tensorboard --logdir=./train_dir
```
(or see the docs for WandB integration).

To continue from here, copy and modify one of the existing env integrations to train agents in your own custom environment. We provide
examples for all kinds of supported environments, please refer to the [documentation](https://samplefactory.dev) for more details.

## Citation

If you use this repository in your work or otherwise wish to cite it, please make reference to our ICML2020 paper.

```
@inproceedings{petrenko2020sf,
  title={Sample Factory: Egocentric 3D Control from Pixels at 100000 FPS with Asynchronous Reinforcement Learning},
  author={Petrenko, Aleksei and Huang, Zhehui and Kumar, Tushar and Sukhatme, Gaurav and Koltun, Vladlen},
  booktitle={ICML},
  year={2020}
}
```

For questions, issues, inquiries please email apetrenko1991@gmail.com. 
Github issues and pull requests are welcome! Check out the [contribution guidelines](https://www.samplefactory.dev/community/contribution/).
