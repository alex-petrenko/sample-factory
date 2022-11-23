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

High-throughput reinforcement learning codebase.

**Resources:**

* **Paper:** https://arxiv.org/abs/2006.11751

* **Discord:** [https://discord.gg/BCfHWaSMkr](https://discord.gg/BCfHWaSMkr)

* **Talk (circa 2021):** https://youtu.be/lLG17LKKSZc

### What is Sample Factory?

Sample Factory is one of the fastest RL libraries. Instead of implementing multiple algorithm families,
we focused on very efficient synchronous and asynchronous implementations of policy gradients (PPO). 

Below are ViZDoom, IsaacGym, DMLab-30, Megaverse, Mujoco, and Atari agents trained with Sample Factory:

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
* Detailed WandB and Tensorboard summaries, custom metrics
* HuggingFace integration (upload trained models and metrics to the Hub)
* Multiple example environment integrations with tuned parameters
