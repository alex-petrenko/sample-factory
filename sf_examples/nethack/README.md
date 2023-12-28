## Installation

To install NetHack, you need nle and its dependencies.

```bash
# nle dependencies
apt-get install build-essential python3-dev python3-pip python3-numpy autoconf libtool pkg-config libbz2-dev
conda install cmake flex bison lit

# install nle locally and modify it to enable seeding
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
pip install -e sf_examples/nethack/render_utils
```

## Running Experiments

To run a single experiment, use the `sf_examples.nethack.train_nethack` script. An example command is
`python -m sf_examples.nethack.train_nethack --env=nethack_challenge --experiment=experiment_name`.


## Showing Experiment Results

To display videos of experiments, use the `sf_examples.nethack.enjoy_nethack` script. An example command is 
`python -m sf_examples.nethack.enjoy_nethack --env=nethack_challenge --experiment=experiment_name`.


## Supported NetHack environments

SF2 supports the NetHack environments:

- nethack_staircase
- nethack_score
- nethack_pet
- nethack_oracle
- nethack_gold
- nethack_eat
- nethack_scout
- nethack_challenge


More environments can be added through the creation of new tasks see examples in `sf_examples.nethack.utils.tasks`
