# NetHack
<video width="800" controls autoplay><source src="https://huggingface.co/LLParallax/sample_factory_human_monk/resolve/main/replay.mp4" type="video/mp4"></video>
## Installation
Works in `Python 3.10`. Higher versions have problems with building NLE.

To install NetHack, you need nle and its dependencies.

```bash
# nle dependencies
apt-get install build-essential python3-dev python3-pip python3-numpy autoconf libtool pkg-config libbz2-dev
conda install cmake flex bison lit

# install nle locally and modify it to enable seeding and handle rendering with gymnasium
git clone https://github.com/facebookresearch/nle.git nle && cd nle \
&& git checkout v0.9.0 && git submodule init && git submodule update --recursive \
&& sed '/#define NLE_ALLOW_SEEDING 1/i#define NLE_ALLOW_SEEDING 1' include/nleobs.h -i \
&& sed '/self\.nethack\.set_initial_seeds = f/d' nle/env/tasks.py -i \
&& sed '/self\.nethack\.set_current_seeds = f/d' nle/env/tasks.py -i \
&& sed '/self\.nethack\.get_current_seeds = f/d' nle/env/tasks.py -i \
&& sed '/def seed(self, core=None, disp=None, reseed=True):/d' nle/env/tasks.py -i \
&& sed '/raise RuntimeError("NetHackChallenge doesn.t allow seed changes")/d' nle/env/tasks.py -i \
&& python setup.py install && cd .. 

# install sample factory with nethack extras
pip install -e .[nethack]
conda install -c conda-forge pybind11
pip install -e sf_examples/nethack/nethack_render_utils
```

## Running Experiments

Run NetHack experiments with the scripts in `sf_examples.nethack`.
The default parameters have been chosen to match [dungeons & data](https://github.com/dungeonsdatasubmission/dungeonsdata-neurips2022) which is based on [nle sample factory baseline](https://github.com/Miffyli/nle-sample-factory-baseline). By moving from D&D to sample factory we've managed to increase the APPO score from 2k to 2.8k.

To train a model in the `nethack_challenge` environment:

```
python -m sf_examples.nethack.train_nethack \
    --env=nethack_challenge \
    --batch_size=4096 \
    --num_workers=16 \
    --num_envs_per_worker=32 \
    --worker_num_splits=2 \
    --rollout=32 \
    --character=mon-hum-neu-mal \
    --model=ChaoticDwarvenGPT5 \
    --rnn_size=512 \
    --experiment=nethack_monk
```

To visualize the training results, use the `enjoy_nethack` script:

```
python -m sf_examples.nethack.enjoy_nethack --env=nethack_challenge --character=mon-hum-neu-mal --experiment=nethack_monk
```

Additionally it's possible to use an alternative `fast_eval_nethack` script which is much faster

```
python -m sf_examples.nethack.fast_eval_nethack --env=nethack_challenge --sample_env_episodes=128 --num_workers=16 --num_envs_per_worker=2 --character=mon-hum-neu-mal --experiment=nethack_monk 
```

### List of Supported Environments

- nethack_staircase
- nethack_score
- nethack_pet
- nethack_oracle
- nethack_gold
- nethack_eat
- nethack_scout
- nethack_challenge

## Results

### Reports
1. Sample Factory was benchmarked on `nethack_challenge` against Dungeons and Data. Sample-Factory was able to achieve similar sample efficiency as D&D using the same parameters and get better running returns (2.8k vs 2k). Training was done on `nethack_challenge` with human-monk character for 2B env steps.
    - https://api.wandb.ai/links/bartekcupial/w69fid1w

### Models
Sample Factory APPO model trained on `nethack_challenge` environment is uploaded to the HuggingFace Hub. The model have been trained for 2B steps.

The model below is the best model from the experiment against Dungeons and Data above. The evaluation metrics here are obtained by running the model 1024 times. 

Model card: https://huggingface.co/LLParallax/sample_factory_human_monk
Evaluation results:
```
{
    "reward/reward": 3245.3828125,
    "reward/reward_min": 20.0,
    "reward/reward_max": 18384.0,
    "len/len": 2370.4560546875,
    "len/len_min": 27.0,
    "len/len_max": 21374.0,
    "policy_stats/avg_score": 3245.4716796875,
    "policy_stats/avg_turns": 14693.970703125,
    "policy_stats/avg_dlvl": 1.13671875,
    "policy_stats/avg_max_hitpoints": 46.42578125,
    "policy_stats/avg_max_energy": 34.00390625,
    "policy_stats/avg_armor_class": 4.68359375,
    "policy_stats/avg_experience_level": 6.13671875,
    "policy_stats/avg_experience_points": 663.375,
    "policy_stats/avg_eating_score": 14063.2587890625,
    "policy_stats/avg_gold_score": 76.033203125,
    "policy_stats/avg_scout_score": 499.0478515625,
    "policy_stats/avg_sokobanfillpit_score": 0.0,
    "policy_stats/avg_staircase_pet_score": 0.005859375,
    "policy_stats/avg_staircase_score": 4.9970703125,
    "policy_stats/avg_episode_number": 1.5,
    "policy_stats/avg_true_objective": 3245.3828125,
    "policy_stats/avg_true_objective_min": 20.0,
    "policy_stats/avg_true_objective_max": 18384.0
}
```