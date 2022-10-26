# DeepMind Lab
<video width="800" controls autoplay><source src="https://huggingface.co/datasets/edbeeching/sample_factory_videos/resolve/main/dmlab30_grid_30_30s.mp4" type="video/mp4"></video>
### Installation

Installation DeepMind Lab can be time consuming. If you are on a Linux system, we provide a prebuild [wheel](https://drive.google.com/file/d/1hAKAkl85HE8JsHXfXbdkF0CrLdiGyuoL/view?usp=sharing).

- Either `pip install deepmind_lab-1.0-py3-none-any.whl`
- Or alternatively, DMLab can be compiled from source by following the instructions on the [DMLab Github](https://github.com/deepmind/lab/blob/master/docs/users/build.md).
- `pip install dm_env`
- To train on DMLab-30 you will need `brady_konkle_oliva2008` [dataset](https://github.com/deepmind/lab/tree/master/data/brady_konkle_oliva2008).
- To significantly speed up training on DMLab-30 consider downloading our [dataset](https://drive.google.com/file/d/17JCp3DbuiqcfO9I_yLjbBP4a7N7Q4c2v/view?usp=sharing)
of pre-generated environment layouts (see paper for details).
Command lines for running experiments with these datasets are provided in the sections below.

### Running Experiments

Run DMLab experiments with the scripts in `sf_examples.dmlab_examples`. 

Example of training in the DMLab watermaze environment for 1B environment steps

```
python -m sf_examples.dmlab.train_dmlab --env=doom_battle --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --batch_size=2048 --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=20 --num_policies=1  --experiment=doom_battle_w20_v20
```


DMLab-30 run on a 36-core server with 4 GPUs:

```
python -m sf_examples.dmlab.train_dmlab --env=dmlab_30 --train_for_env_steps=10000000000 --algo=APPO --gamma=0.99 --use_rnn=True --num_workers=90 --num_envs_per_worker=12 --num_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --benchmark=False --max_grad_norm=0.0 --dmlab_renderer=software --decorrelate_experience_max_seconds=120 --reset_timeout_seconds=300 --encoder_conv_architecture=resnet_impala --encoder_conv_mlp_layers=512 --nonlinearity=relu --rnn_type=lstm --dmlab_extended_action_set=True --num_policies=4 --pbt_replace_reward_gap=0.05 --pbt_replace_reward_gap_absolute=5.0 --pbt_period_env_steps=10000000 --pbt_start_mutation=100000000 --with_pbt=True --experiment=dmlab_30_resnet_4pbt_w90_v12 --dmlab_one_task_per_worker=True --set_workers_cpu_affinity=True --max_policy_lag=35 --pbt_target_objective=dmlab_target_objective --dmlab30_dataset=~/datasets/brady_konkle_oliva2008 --dmlab_use_level_cache=True --dmlab_level_cache_path=/home/user/dmlab_cache
```

### Results

#### Reports

<!-- 1. We reproduced the paper results in SF2 in the Battle and Battle2 and compared the results using input normalization. Input normalization has improved results in the Battle environment. This experiment with input normalization was run with `sf_examples.dmlab_examples.experiments.sf2_doom_battle_envs`. Note that `normalize_input=True` is set compared to the results from the paper
    - https://wandb.ai/andrewzhang505/sample_factory/reports/VizDoom-Battle-Environments--VmlldzoyMzcyODQx

2. In SF2's bot environments (deathmatch_bots and duel_bots), we trained the agents against randomly generated bots as opposed to a curriculum of increasing bot difficulty. This is because the ViZDoom environment no longer provides the bots used in the curriculum, and SF2 no longer requires the curriculum to train properly. However, due to the differences in bot difficulty, the current training results are no longer comparable to the paper. An example training curve on deathmatch_bots with the same parameters as in the paper is shown below:
    - https://wandb.ai/andrewzhang505/sample_factory/reports/ViZDoom-Deathmatch-Bots--VmlldzoyNzY2NDI1 -->

#### Models
# TODO add models to hub
The models below are the best models from the input normalization experiment above. The evaluation metrics here are obtained by running the experiment 10 times with different seeds.  

| DMLab Command Line Parameter | DMLab Environment name           | Model Checkpooints                                                                             |
| ---------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------------- |
| dmlab_30                     | DMLab-30 Benchmark               | [🤗 Hub DMLab30 checkpoints TODO](https://huggingface.co/edbeeching/atari_2B_atari_alien_1111)  |
| dmlab_benchmark_slow_reset   | rooms_keys_doors_puzzle          | [🤗 Hub DMLab30 checkpoints TODO](https://huggingface.co/edbeeching/atari_2B_atari_amidar_1111) |
| dmlab_sparse                 | explore_goal_locations_large     | [🤗 Hub DMLab30 checkpoints TODO](https://huggingface.co/edbeeching/atari_2B_atari_alien_1111)  |
| dmlab_very_sparse            | explore_goal_locations_large     | [🤗 Hub DMLab30 checkpoints TODO](https://huggingface.co/edbeeching/atari_2B_atari_amidar_1111) |
| dmlab_sparse_doors           | explore_obstructed_goals_large   | [🤗 Hub DMLab30 checkpoints TODO](https://huggingface.co/edbeeching/atari_2B_atari_alien_1111)  |
| dmlab_nonmatch               | rooms_select_nonmatching_object  | [🤗 Hub DMLab30 checkpoints TODO](https://huggingface.co/edbeeching/atari_2B_atari_amidar_1111) |
| dmlab_watermaze              | rooms_watermaze                  | [🤗 Hub DMLab30 checkpoints TODO](https://huggingface.co/edbeeching/atari_2B_atari_alien_1111)  |
| dmlab_collect_good_objects   | rooms_collect_good_objects_train | [🤗 Hub DMLab30 checkpoints TODO](https://huggingface.co/edbeeching/atari_2B_atari_amidar_1111) |

