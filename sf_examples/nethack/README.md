## Installation

To install NetHack, you need nle and its dependencies.

```bash
# nle dependencies
apt-get install build-essential python3-dev python3-pip python3-numpy autoconf libtool pkg-config libbz2-dev
conda install cmake flex bison lit
# install sample factory with nethack extras
pip install -e .[nethack]
```

## Running Experiments

To run a single experiment, use the `sf_examples.nethack.train_nethack` script. An example command is
`python -m sf_examples.nethack.train_nethack --env=nethack_challenge --experiment=experiment_name`.


## Showing Experiment Results

To display videos of experiments, use the `sf_examples.nethack.enjoy_nethack` script. An example command is 
`python -m sf_examples.nethack.enjoy_nethack --env=nethack_challenge --experiment=experiment_name`


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
