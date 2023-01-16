[![tests](https://github.com/alex-petrenko/sample-factory/actions/workflows/test-ci.yml/badge.svg?branch=master)](https://github.com/alex-petrenko/sample-factory/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/alex-petrenko/sample-factory/branch/master/graph/badge.svg?token=9EHMIU5WYV)](https://codecov.io/gh/alex-petrenko/sample-factory)
[![pre-commit](https://github.com/alex-petrenko/sample-factory/actions/workflows/pre-commit.yml/badge.svg?branch=master)](https://github.com/alex-petrenko/sample-factory/actions/workflows/pre-commit.yml)
[![docs](https://github.com/alex-petrenko/sample-factory/actions/workflows/docs.yml/badge.svg)](https://samplefactory.dev)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/alex-petrenko/sample-factory/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/sample-factory)](https://pepy.tech/project/sample-factory)
[<img src="https://img.shields.io/discord/987232982798598164?label=discord">](https://discord.gg/BCfHWaSMkr)
<!-- [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/wmFrank/sample-factory/master.svg)](https://results.pre-commit.ci/latest/github/wmFrank/sample-factory/master)-->
<!-- [![wakatime](https://wakatime.com/badge/github/alex-petrenko/sample-factory.svg)](https://wakatime.com/badge/github/alex-petrenko/sample-factory)-->


# Sample Factory

High-throughput reinforcement learning codebase. Resources:

* **Paper:** https://arxiv.org/abs/2006.11751

* **Discord:** [https://discord.gg/BCfHWaSMkr](https://discord.gg/BCfHWaSMkr)

* **Twitter (for updates):** [@petrenko_ai](https://twitter.com/petrenko_ai)

[//]: # (* **Talk &#40;circa 2021&#41;:** https://youtu.be/lLG17LKKSZc)

## What is Sample Factory?

Sample Factory is one of the fastest RL libraries 
focused on very efficient synchronous and asynchronous implementations of policy gradients (PPO). 

Sample Factory is thoroughly tested, used by many researchers and practitioners, and is actively maintained.
Our implementation is known to reach SOTA performance in a variety of domains while minimizing RL experiment training time and hardware requirements.
Clips below demonstrate ViZDoom, IsaacGym, DMLab-30, Megaverse, Mujoco, and Atari agents trained with Sample Factory:

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

## Key features

* Highly optimized algorithm [architecture](06-architecture/overview.md) for maximum learning throughput
* [Synchronous and asynchronous](07-advanced-topics/sync-async.md) training regimes
* [Serial (single-process) mode](07-advanced-topics/serial-mode.md) for easy debugging
* Optimal performance in both CPU-based and [GPU-accelerated environments](09-environment-integrations/isaacgym.md)
* Single- & multi-agent training, self-play, supports [training multiple policies](07-advanced-topics/multi-policy-training.md) at once on one or many GPUs
* Population-Based Training ([PBT](07-advanced-topics/pbt.md))
* Discrete, continuous, hybrid action spaces
* Vector-based, image-based, dictionary observation spaces
* Automatically creates a model architecture by parsing action/observation space specification. Supports [custom model architectures](03-customization/custom-models.md)
* Library is designed to be imported into other projects, [custom environments](03-customization/custom-environments.md) are first-class citizens
* Detailed [WandB and Tensorboard summaries](05-monitoring/metrics-reference.md), [custom metrics](05-monitoring/custom-metrics.md)
* [HuggingFace 🤗 integration](10-huggingface/huggingface.md) (upload trained models and metrics to the Hub)
* [Multiple](09-environment-integrations/mujoco.md) [example](09-environment-integrations/atari.md) [environment](09-environment-integrations/vizdoom.md) [integrations](09-environment-integrations/dmlab.md) with tuned parameters and trained models

## Next steps

Check out the following guides to get started:

* [Installation](01-get-started/installation.md)
* [Basic Usage](01-get-started/basic-usage.md)