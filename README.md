# Sample Factory

High throughput asynchronous reinforcement learning

## Setup instructions

- Install miniconda for python 3.7 on Ubuntu 18.04: https://docs.conda.io/en/latest/miniconda.html

- Clone the repo: `git clone https://github.com/alex-petrenko/sample-factory.git`

- Create and activate conda env:

```
cd sample-factory
conda env create -f environment.yml
conda activate sample-factory
```

- Install VizDoom system dependencies (from here https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#linux_deps):

```
# ZDoom dependencies
sudo apt install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
libopenal-dev timidity libwildmidi-dev unzip cmake

# Boost libraries
sudo apt install libboost-all-dev

# Python 3 dependencies
sudo apt install python3-dev python3-pip
```

This list might not be entirely comprehensive, on a clean system you might need to install 1-2 more packages if you have compilation errors when installing VizDoom.

- Install additional python dependencies: `pip install -r requirements.txt`

## Running experiments

```
The most basic experiment (preferrable number of workers is equal to number of logical cores):)
python -m algorithms.appo.train_appo --env=doom_basic --train_for_env_steps=3000000 --algo=APPO --env_frameskip=4 --use_rnn=True --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=20 --experiment=doom_basic

Wait until the end, or stop at any point to visualize the policy:
python -m algorithms.appo.enjoy_appo --env=doom_basic --algo=APPO --experiment=doom_basic

```



```
Train for 3B env steps (also can be stopped at any time with Ctrl+C and resumed by using the same cmd):
python -m algorithms.appo.train_appo --env=doom_battle --train_for_env_steps=3000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=20 --num_policies=1  --experiment=doom_battle_w20_v20

Run at any point to visualize the experiment:
python -m algorithms.appo.enjoy_appo --env=doom_battle --algo=APPO --experiment=doom_battle_w20_v20

```
