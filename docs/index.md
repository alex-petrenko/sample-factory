[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/alex-petrenko/sample-factory/blob/master/LICENSE)
[![tests](https://github.com/alex-petrenko/sample-factory/actions/workflows/test-ci.yml/badge.svg?branch=sf2)](https://github.com/alex-petrenko/sample-factory/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/alex-petrenko/sample-factory/branch/sf2/graph/badge.svg?token=9EHMIU5WYV)](https://codecov.io/gh/alex-petrenko/sample-factory)
[![Downloads](https://pepy.tech/badge/sample-factory)](https://pepy.tech/project/sample-factory)
[<img src="https://img.shields.io/discord/987232982798598164?label=discord">](https://discord.gg/BCfHWaSMkr)

# Sample Factory

Codebase for high throughput asynchronous reinforcement learning.

**Paper:** https://arxiv.org/abs/2006.11751

**Cite:** [BibTeX](https://github.com/alex-petrenko/sample-factory#citation)

**Talk:** https://youtu.be/lLG17LKKSZc

**Videos:** https://sites.google.com/view/sample-factory

VizDoom agents trained with Sample Factory playing in real time:

<p align="middle">
<img src="https://github.com/alex-petrenko/sample-factory/blob/master/gifs/battle.gif?raw=true" width="400">
<img src="https://github.com/alex-petrenko/sample-factory/blob/master/gifs/duel.gif?raw=true" width="400">
</p> 

#### When should I use Sample Factory?

1. Sample Factory is the fastest open source single-machine RL implementations (see paper for details).
If you plan to train RL agents on large amounts of experience, consider using it.
Sample Factory can significantly speed up
the experimentation or allow you to collect more samples in the same amount of time and achieve better performance.

2. Consider using Sample Factory for your multi-agent and population-based training experiments. 
Multi-agent and PBT setups are really simple with Sample Factory.

3. A lot of work went into our VizDoom and DMLab wrappers. For example, we include full support for
configurable VizDoom multi-agent environments and their interop with RL algorithms, which can open new interesting research directions.
Consider using Sample Factory if you train agents in these environments.

4. Sample Factory can be a good choice as a prototype for a single node in a distributed RL system or as a reference
codebase for other types of async RL algorithms.
