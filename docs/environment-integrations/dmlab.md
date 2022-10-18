# DeepMind Lab Integration

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
python -m sf_examples.dmlab_examples.train_dmlab --env=doom_battle --train_for_env_steps=4000000000 --algo=APPO --env_frameskip=4 --use_rnn=True --batch_size=2048 --wide_aspect_ratio=False --num_workers=20 --num_envs_per_worker=20 --num_policies=1  --experiment=doom_battle_w20_v20
```


DMLab-30 run on a 36-core server with 4 GPUs:
TODO CHECK COMMAND

```
python -m sample_factory.algorithms.appo.train_appo --env=dmlab_30 --train_for_env_steps=10000000000 --algo=APPO --gamma=0.99 --use_rnn=True --num_workers=90 --num_envs_per_worker=12 --ppo_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --benchmark=False --ppo_epochs=1 --max_grad_norm=0.0 --dmlab_renderer=software --decorrelate_experience_max_seconds=120 --reset_timeout_seconds=300 --encoder_custom=dmlab_instructions --encoder_type=resnet --encoder_subtype=resnet_impala --encoder_extra_fc_layers=1 --hidden_size=256 --nonlinearity=relu --rnn_type=lstm --dmlab_extended_action_set=True --num_policies=4 --pbt_replace_reward_gap=0.05 --pbt_replace_reward_gap_absolute=5.0 --pbt_period_env_steps=10000000 --pbt_start_mutation=100000000 --with_pbt=True --experiment=dmlab_30_resnet_4pbt_w90_v12 --dmlab_one_task_per_worker=True --set_workers_cpu_affinity=True --max_policy_lag=35 --pbt_target_objective=dmlab_target_objective --dmlab30_dataset=~/datasets/brady_konkle_oliva2008 --dmlab_use_level_cache=True --dmlab_level_cache_path=/home/user/dmlab_cache
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

<!-- | Environment | HuggingFace Hub Models                                              | Evaluation Metrics |
| ----------- | ------------------------------------------------------------------- | ------------------ |
| dmlab30      | https://huggingface.co/andrewzhang505/sample-factory-2-doom-battle  | 59.37 +/- 3.93     |
| Battle2     | https://huggingface.co/andrewzhang505/sample-factory-2-doom-battle2 | 36.40 +/- 4.20     | -->

#### Videos

TODO ADD VIDEOS
##### DMLab 30

<!-- <p align="center">
<video class="w-full" src="https://huggingface.co/andrewzhang505/sample-factory-2-doom-battle/resolve/main/replay.mp4" controls="" autoplay="" loop=""></video></p> -->

##### DMLab Watermaze ...

<!-- <p align="center">
<video class="w-full" src="https://huggingface.co/andrewzhang505/sample-factory-2-doom-battle2/resolve/main/replay.mp4" controls="" autoplay="" loop=""></video></p> -->