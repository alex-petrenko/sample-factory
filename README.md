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

High-throughput reinforcement learning codebase. Version 2.0.0 is out! 🤗

**Resources:**

* **Documentation:** [https://samplefactory.dev](https://samplefactory.dev) 

* **Paper:** https://arxiv.org/abs/2006.11751

* **Citation:** [BibTeX](https://github.com/alex-petrenko/sample-factory#citation)

* **Discord:** [https://discord.gg/BCfHWaSMkr](https://discord.gg/BCfHWaSMkr)

* **Twitter (for updates):** [@petrenko_ai](https://twitter.com/petrenko_ai)

* **Talk (circa 2021):** https://youtu.be/lLG17LKKSZc

### What is Sample Factory?

Sample Factory is one of the fastest RL libraries.
We focused on very efficient synchronous and asynchronous implementations of policy gradients (PPO). 

Sample Factory is thoroughly tested, used by many researchers and practitioners, and is actively maintained.
Our implementation is known to reach SOTA performance in a variety of domains in a short amount of time.
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

**Key features:**

* Highly optimized algorithm [architecture](https://www.samplefactory.dev/06-architecture/overview/) for maximum learning throughput
* [Synchronous and asynchronous](https://www.samplefactory.dev/07-advanced-topics/sync-async/) training regimes
* [Serial (single-process) mode](https://www.samplefactory.dev/07-advanced-topics/serial-mode/) for easy debugging
* Optimal performance in both CPU-based and [GPU-accelerated environments](https://www.samplefactory.dev/09-environment-integrations/isaacgym/)
* Single- & multi-agent training, self-play, supports [training multiple policies](https://www.samplefactory.dev/07-advanced-topics/multi-policy-training/) at once on one or many GPUs
* Population-Based Training ([PBT](https://www.samplefactory.dev/07-advanced-topics/multi-policy-training/))
* Discrete, continuous, hybrid action spaces
* Vector-based, image-based, dictionary observation spaces
* Automatically creates a model architecture by parsing action/observation space specification. Supports [custom model architectures](https://www.samplefactory.dev/03-customization/custom-models/)
* Library is designed to be imported into other projects, [custom environments](https://www.samplefactory.dev/03-customization/custom-environments/) are first-class citizens
* Detailed [WandB and Tensorboard summaries](https://www.samplefactory.dev/05-monitoring/metrics-reference/), [custom metrics](https://www.samplefactory.dev/05-monitoring/custom-metrics/)
* [HuggingFace 🤗 integration](https://www.samplefactory.dev/10-huggingface/huggingface/) (upload trained models and metrics to the Hub)
* [Multiple](https://www.samplefactory.dev/09-environment-integrations/mujoco/) [example](https://www.samplefactory.dev/09-environment-integrations/atari/) [environment](https://www.samplefactory.dev/09-environment-integrations/vizdoom/) [integrations](https://www.samplefactory.dev/09-environment-integrations/dmlab/) with tuned parameters and trained models

This Readme provides only a brief overview of the library.
Visit full documentation at [https://samplefactory.dev](https://samplefactory.dev) for more details.

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

Stop the experiment (Ctrl+C) when the desired performance is reached and then evaluate the agent:

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

## Acknowledgements

This project would not be possible without amazing contributions from many people. I would like to thank:

* [Vladlen Koltun](https://vladlen.info) for amazing guidance and support, especially in the early stages of the project, for
helping me solidify the ideas that eventually became this library.
* My academic advisor [Gaurav Sukhatme](https://viterbi.usc.edu/directory/faculty/Sukhatme/Gaurav) for supporting this project
over the years of my PhD and for being overall an awesome mentor.
* [Zhehui Huang](https://zhehui-huang.github.io/) for his contributions to the original ICML submission, his diligent work on
testing and evaluating the library and for adopting it in his own research.
* [Edward Beeching](https://edbeeching.github.io/) for his numerous awesome contributions to the codebase, including
hybrid action distributions, new version of the custom model builder, multiple environment integrations, and also
for promoting the library through the HuggingFace integration!
* [Andrew Zhang](https://andrewzhang505.github.io/) and [Ming Wang](https://www.mingwang.me/) for numerous contributions to the codebase and documentation during their HuggingFace internships!
* [Thomas Wolf](https://thomwolf.io/) and others at HuggingFace for the incredible (and unexpected) support and for the amazing
work they are doing for the open-source community.
* [Erik Wijmans](https://wijmans.xyz/) for feedback and insights and for his awesome implementation of RNN backprop using PyTorch's `PackedSequence`, multi-layer RNNs, and other features!
* [Tushar Kumar](https://www.linkedin.com/in/tushartk/) for contributing to the original paper and for his help
with the [fast queue implementation](https://github.com/alex-petrenko/faster-fifo).
* [Costa Huang](https://costa.sh/) for developing CleanRL, for his work on benchmarking RL algorithms, and for awesome feedback
and insights!
* [Denys Makoviichuk](https://github.com/Denys88/rl_games) for developing rl_games, a very fast RL library, for inspiration and 
feedback on numerous features of this library (such as return normalizations, adaptive learning rate, and others).
* [Eugene Vinitsky](https://eugenevinitsky.github.io/) for adopting this library in his own research and for his valuable feedback.
* All my labmates at RESL who used Sample Factory in their projects and provided feedback and insights!

Huge thanks to all the people who are not mentioned here for your code contributions, PRs, issues, and questions!
This project would not be possible without a community!

## Citation

If you use this repository in your work or otherwise wish to cite it, please make reference to our ICML2020 paper.

```
@inproceedings{petrenko2020sf,
  author    = {Aleksei Petrenko and
               Zhehui Huang and
               Tushar Kumar and
               Gaurav S. Sukhatme and
               Vladlen Koltun},
  title     = {Sample Factory: Egocentric 3D Control from Pixels at 100000 {FPS}
               with Asynchronous Reinforcement Learning},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning,
               {ICML} 2020, 13-18 July 2020, Virtual Event},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {7652--7662},
  publisher = {{PMLR}},
  year      = {2020},
  url       = {http://proceedings.mlr.press/v119/petrenko20a.html},
  biburl    = {https://dblp.org/rec/conf/icml/PetrenkoHKSK20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

For questions, issues, inquiries please join Discord. 
Github issues and pull requests are welcome! Check out the [contribution guidelines](https://www.samplefactory.dev/community/contribution/).
