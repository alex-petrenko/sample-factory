## Trained policies

### VizDoom APPO Battle and Battle2

<p align="left">
<img src="https://github.com/alex-petrenko/sample-factory/blob/master/gifs/battle.gif?raw=true" width="500">
</p>

[Download pretrained agents HERE](https://drive.google.com/drive/folders/1uRmivzqvXXQ5YSj18JAB3fh6IWsj3cZI?usp=sharing).
The link also includes learning traces (Tensorboard summaries).
Trained with Sample Factory 1.121.2 in VizDoom 1.1.11.

##### How to use:

* download and unzip into a local directory `./train_dir` (resulting tree should look like `train_dir/doom_battle_battle2_appo_v1.121.2`)
* make sure VizDoom and Sample Factory are installed (see main README.md)
* full configuration for the above experiments can be found in `cfg.json` in the experiment folder

##### `Battle` agent

Expected score: 30-70 points per episode, ~49 on average.

```
python -m sample_factory.algorithms.appo.enjoy_appo --algo=APPO --env_frameskip=4 --env=doom_battle --experiment=00_battle_fs4_env_doom_battle_see_6723621 --train_dir=./train_dir --experiments_root=doom_battle_battle2_appo_v1.121.2/battle_fs4_slurm
```

##### `Battle2` agent

Expected score: 10-30 points per episode, ~24 on average.

```
python -m sample_factory.algorithms.appo.enjoy_appo --algo=APPO --env_frameskip=4 --env=doom_battle2 --experiment=05_battle_fs4_env_doom_battle2_see_2530673 --train_dir=./train_dir --experiments_root=doom_battle_battle2_appo_v1.121.2/battle_fs4_slurm
```

##### View learning traces in Tensorboard

```
python -m sample_factory.tb '*doom_battle_battle2*' --port=6006 --reload_interval=1 --dir=./train_dir
```
