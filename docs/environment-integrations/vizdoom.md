# VizDoom Integrations

### Installation

To install VizDoom just follow system setup instructions from the original repository ([VizDoom linux_deps](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md#linux_deps)),
after which the latest VizDoom can be installed from PyPI: 
```pip install vizdoom```

### Running Experiments

Run MuJoCo experiments with the scripts in `sf_examples.vizdoom_examples`. 

Train for 4B env steps (also can be stopped at any time with Ctrl+C and resumed by using the same cmd).
This is more or less optimal training setup for a 10-core machine.

```
python -m sf_examples.vizdoom_examples.train_vizdoom --env=doom_battle --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --batch_size=2048 --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=20 --num_policies=1  --experiment=doom_battle_w20_v20
```

Run at any point to visualize the experiment:

```
python -m sf_examples.vizdoom_examples.enjoy_vizdoom --env=doom_battle --algo=APPO --experiment=doom_battle_w20_v20
```

Runner scripts are also provided in `sf_examples.vizdoom_examples.experiments` to run experiments in parallel or on slurm.

#### Reproducing Paper Results

Train on one of the 6 "basic" VizDoom environments:

```
python -m sf_examples.vizdoom_examples.train_vizdoom --train_for_env_steps=500000000 --algo=APPO --env=doom_my_way_home --env_frameskip=4 --use_rnn=True --num_workers=36 --num_envs_per_worker=8 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False --experiment=doom_basic_envs
```

Doom "battle" and "battle2" environments, 36-core server (72 logical cores) with 4 GPUs:
```
python -m sf_examples.vizdoom_examples.train_vizdoom --env=doom_battle --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=72 --num_envs_per_worker=8 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0 --experiment=doom_battle
python -m sf_examples.vizdoom_examples.train_vizdoom --env=doom_battle2 --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=72 --num_envs_per_worker=8 --num_policies=1 --batch_size=2048 --wide_aspect_ratio=False --max_grad_norm=0.0 --experiment=doom_battle_2
```

Duel and deathmatch versus bots, population-based training, 36-core server:

```
python -m sf_examples.vizdoom_examples.train_vizdoom --env=doom_duel_bots --train_for_seconds=360000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --reward_scale=0.5 --num_workers=72 --num_envs_per_worker=32 --num_policies=8 --batch_size=2048 --benchmark=False --res_w=128 --res_h=72 --wide_aspect_ratio=False --pbt_replace_reward_gap=0.2 --pbt_replace_reward_gap_absolute=3.0 --pbt_period_env_steps=5000000 --save_milestones_sec=1800 --with_pbt=True --experiment=doom_duel_bots
python -m sf_examples.vizdoom_examples.train_vizdoom --env=doom_deathmatch_bots --train_for_seconds=3600000 --algo=APPO --use_rnn=True --gamma=0.995 --env_frameskip=2 --num_workers=80 --num_envs_per_worker=24 --num_policies=8 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=False --with_pbt=True --pbt_period_env_steps=5000000 --experiment=doom_deathmatch_bots
```

Duel and deathmatch self-play, PBT, 36-core server:

```
python -m sf_examples.vizdoom_examples.train_vizdoom --env=doom_duel --train_for_seconds=360000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --num_workers=72 --num_envs_per_worker=16 --num_policies=8 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=False --benchmark=False --pbt_replace_reward_gap=0.5 --pbt_replace_reward_gap_absolute=0.35 --pbt_period_env_steps=5000000 --with_pbt=True --pbt_start_mutation=100000000 --experiment=doom_duel_full
python -m sf_examples.vizdoom_examples.train_vizdoom --env=doom_deathmatch_full --train_for_seconds=360000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --num_workers=72 --num_envs_per_worker=16 --num_policies=8 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=False --benchmark=False --pbt_replace_reward_gap=0.1 --pbt_replace_reward_gap_absolute=0.1 --pbt_period_env_steps=5000000 --with_pbt=True --pbt_start_mutation=100000000 --experiment=doom_deathmatch_full
```

Reproducing benchmarking results:

This achieves 50K+ framerate on a 10-core machine (Intel Core i9-7900X):

```
python -m sf_examples.vizdoom_examples.train_vizdoom --env=doom_benchmark --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=20 --num_envs_per_worker=32 --num_policies=1 --batch_size=4096 --experiment=doom_battle_appo_fps_20_32 --res_w=128 --res_h=72 --wide_aspect_ratio=False --policy_workers_per_policy=2 --worker_num_splits=2
```

This achieves 100K+ framerate on a 36-core machine:

```
python -m sf_examples.vizdoom_examples.train_vizdoom --env=doom_benchmark --algo=APPO --env_frameskip=4 --use_rnn=True --num_workers=72 --num_envs_per_worker=24 --num_policies=1 --batch_size=8192 --wide_aspect_ratio=False --experiment=doom_battle_appo_w72_v24 --policy_workers_per_policy=2
```

### Results

#### Reports

1. We reproduced the paper results in SF2 in the Battle and Battle2 and compared the results using input normalization. Input normalization has improved results in the Battle environment. This experiment with input normalization was run with `sf_examples.vizdoom_examples.experiments.sf2_doom_battle_envs`
    - https://wandb.ai/andrewzhang505/sample_factory/reports/VizDoom-Battle-Environments--VmlldzoyMzcyODQx

#### Models

| Environment | HuggingFace Hub Models                                              | Evaluation Metrics |
| ----------- | ------------------------------------------------------------------- | ------------------ |
| Battle      | https://huggingface.co/andrewzhang505/sample-factory-2-doom-battle  | 59.37 +/- 3.93     |
| Battle2     | https://huggingface.co/andrewzhang505/sample-factory-2-doom-battle2 | 25.72 +/- 5.51     |

#### Videos

##### Doom Battle

<p align="center">
<video class="w-full" src="https://huggingface.co/andrewzhang505/sample-factory-2-doom-battle/resolve/main/replay.mp4" controls="" autoplay="" loop=""></video></p>

##### Doom Battle2

<p align="center">
<video class="w-full" src="https://huggingface.co/andrewzhang505/sample-factory-2-doom-battle2/resolve/main/replay.mp4" controls="" autoplay="" loop=""></video></p>