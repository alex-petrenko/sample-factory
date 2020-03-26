# Sample Factory

High throughput asynchronous reinforcement learning

## Setup instructions

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

- Install miniconda for python 3.7 on Ubuntu 18.04: https://docs.conda.io/en/latest/miniconda.html

- Clone the repo: `git clone https://github.com/alex-petrenko/sample-factory.git`

- Create and activate conda env:

```
cd sample-factory
conda env create -f environment.yml
conda activate sample-factory
```

- Build a fast C++ IPC queue extension

```
python setup.py build_ext --inplace --force
```

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

```
This achieves 50K+ framerate on a 10-core machine (Intel Core i9-7900X):
python -m algorithms.appo.train_appo --env=doom_benchmark --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=20 --num_envs_per_worker=32 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=4096 --batch_size=4096 --experiment=doom_battle_appo_fps_20_32 --res_w=128 --res_h=72 --wide_aspect_ratio=False --policy_workers_per_policy=2 --worker_num_splits=2
```

```
This achieves 100K+ framerate on a 36-core machine:
python -m algorithms.appo.train_appo --env=doom_benchmark --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=72 --num_envs_per_worker=24 --num_policies=1 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=8192 --batch_size=8192 --wide_aspect_ratio=False --experiment=doom_battle_appo_w72_v24 --policy_workers_per_policy=2
```

```
Doom Duel with bots, PBT with 8 policies, frameskip=2:
python -m algorithms.appo.train_appo --env=doom_duel_bots --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --num_workers=72 --num_envs_per_worker=32 --num_policies=8 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=False --pbt_replace_reward_gap=0.3 --pbt_replace_reward_gap_absolute=3.0 --pbt_period_env_steps=5000000 --experiment=doom_duel_bots_pbt
```

```
Doom Duel multi-agent, PBT with 8 policies, frameskip=2:
python -m algorithms.appo.train_appo --env=doom_duel --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --num_workers=72 --num_envs_per_worker=16 --num_policies=8 --ppo_epochs=1 --rollout=32 --recurrence=32 --macro_batch=2048 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=False --pbt_replace_reward_gap=0.5 --pbt_replace_reward_gap_absolute=0.35 --pbt_period_env_steps=5000000 --experiment=doom_duel_multiagent_pbt

```
