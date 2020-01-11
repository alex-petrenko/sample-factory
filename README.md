# doom-neurobot
RL for classic Doom deathmatch

## Setup instructions

1) Install miniconda for python 3.7 on Ubuntu 18.04: https://docs.conda.io/en/latest/miniconda.html

2) Create conda env: `conda create -n doom-rl python=3.7`

3) Activate conda env: `conda activate doom-rl`

3.5) Install OpenCV and PyTorch from Conda:

```
conda install -c conda-forge opencv=4.1.0
conda install pytorch=1.3.1 -c pytorch
```

4) Install VizDoom system dependencies (from here https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#linux_deps):

```
# ZDoom dependencies
sudo apt install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip

# Boost libraries
sudo apt install libboost-all-dev

# Python 3 dependencies
sudo apt install python3-dev python3-pip
```

This list might not be entirely comprehensive, maybe on a clean system you need to install 1-2 more packages if you have compilation errors when installing VizDoom.

5) Clone the repo: `git clone https://github.com/alex-petrenko/doom-neurobot.git`

6) Install python dependencies: `pip install -r requirements.txt`

7) If you want to work with RLLIB scripts, install Ray with `pip install ray`

## Testing the RLLIB training scripts

From the doom-neurobot root, run the train script:

```
python -m train_rllib -f=experiments/doom-appo.yaml --experiment-name=test-appo
```

The model is checkpointed every 20 iterations, so wait at least a few minutes before stopping. Then you can see the agent's performance with:

```
python -m enjoy_rllib ~/ray_results/test-appo/CUSTOM_APPO_doom_battle_hybrid_0_2019-07-25_17-33-223ci_8euu/checkpoint_20/checkpoint-20 --run APPO
```

## Working with PyTorch implementation

```
python -m train_pytorch --algo=PPO --env=doom_basic --experiment=doom-test-ppo --recurrence=16
python -m enjoy_pytorch --algo=PPO --env=doom_basic --experiment=doom-test-ppo --fps=100
```
