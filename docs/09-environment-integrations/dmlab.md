# DeepMind Lab
<img src="https://huggingface.co/datasets/edbeeching/sample_factory_videos/resolve/main/dmlab.gif" width="800" alt="DMLab-30 agents traned using Sample Factory 2.0">

## Installation

Installation DeepMind Lab can be time consuming. If you are on a Linux system, we provide a prebuilt [wheel](https://drive.google.com/file/d/1hAKAkl85HE8JsHXfXbdkF0CrLdiGyuoL/view?usp=sharing).

- Either `pip install deepmind_lab-1.0-py3-none-any.whl`
- Or alternatively, DMLab can be compiled from source by following the instructions on the [DMLab Github](https://github.com/deepmind/lab/blob/master/docs/users/build.md).
- `pip install dm_env`
- To train on DMLab-30 you will need `brady_konkle_oliva2008` [dataset](https://github.com/deepmind/lab/tree/master/data/brady_konkle_oliva2008).
- To significantly speed up training on DMLab-30 consider downloading our [dataset](https://drive.google.com/file/d/17JCp3DbuiqcfO9I_yLjbBP4a7N7Q4c2v/view?usp=sharing)
of pre-generated environment layouts (see [paper](https://proceedings.mlr.press/v119/petrenko20a.html) for details).
Command lines for running experiments with these datasets are provided in the sections below.

## Running Experiments

Run DMLab experiments with the scripts in `sf_examples.dmlab`. 

Example of training in the DMLab watermaze environment for 1B environment steps

```
python -m sf_examples.dmlab.train_dmlab --env=dmlab_watermaze --train_for_env_steps=1000000000 --gamma=0.99 --use_rnn=True --num_workers=90 --num_envs_per_worker=12 --num_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --benchmark=False --max_grad_norm=0.0 --dmlab_renderer=software --decorrelate_experience_max_seconds=120 --reset_timeout_seconds=300 --encoder_conv_architecture=resnet_impala --encoder_conv_mlp_layers=512 --nonlinearity=relu --rnn_type=lstm --dmlab_extended_action_set=True --num_policies=1 --experiment=dmlab_watermaze_resnet_w90_v12 --set_workers_cpu_affinity=True --max_policy_lag=35  --dmlab30_dataset=~/datasets/brady_konkle_oliva2008 --dmlab_use_level_cache=True --dmlab_level_cache_path=/home/user/dmlab_cache
```


DMLab-30 run on a 36-core server with 4 GPUs using Population-Based Training with 4 agents:

```
python -m sf_examples.dmlab.train_dmlab --env=dmlab_30 --train_for_env_steps=10000000000 --gamma=0.99 --use_rnn=True --num_workers=90 --num_envs_per_worker=12 --num_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --benchmark=False --max_grad_norm=0.0 --dmlab_renderer=software --decorrelate_experience_max_seconds=120 --reset_timeout_seconds=300 --encoder_conv_architecture=resnet_impala --encoder_conv_mlp_layers=512 --nonlinearity=relu --rnn_type=lstm --dmlab_extended_action_set=True --num_policies=4 --pbt_replace_reward_gap=0.05 --pbt_replace_reward_gap_absolute=5.0 --pbt_period_env_steps=10000000 --pbt_start_mutation=100000000 --with_pbt=True --experiment=dmlab_30_resnet_4pbt_w90_v12 --dmlab_one_task_per_worker=True --set_workers_cpu_affinity=True --max_policy_lag=35 --pbt_target_objective=dmlab_target_objective --dmlab30_dataset=~/datasets/brady_konkle_oliva2008 --dmlab_use_level_cache=True --dmlab_level_cache_path=/home/user/dmlab_cache
```


## Models
| DMLab Command Line Parameter | DMLab Environment name | Model Checkpooints                                                           |
| ---------------------------- | ---------------------- | ---------------------------------------------------------------------------- |
| dmlab_30                     | DMLab-30               | [🤗 Hub DMLab30 checkpoints](https://huggingface.co/edbeeching/dmlab_30_1111) |

