# NetHack
<video width="800" controls autoplay><source src="https://github.com/BartekCupial/sample-factory/assets/92169405/47884b73-beeb-4303-a72f-75d202aa87a8" type="video/mp4"></video>
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
&& sed -i '/def render(self, mode="human"):/a\        if not self.last_observation:\n            return' nle/env/base.py \
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
python -m sf_examples.nethack.train_nethack --env=nethack_challenge --batch_size=4096 --num_workers=16 --num_envs_per_worker=30 --character=mon-hum-neu-mal --experiment=nethack_monk
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
Example training on human-monk for 2B env steps.
- https://api.wandb.ai/links/bartekcupial/w69fid1w

### Models
TODO:
