## Installing swarm-rl
1. Clone https://github.com/Zhehui-Huang/quad-swarm-rl into your home directory
2. Follow the instructions in the repo to create the conda environment
3. Install the gym_art folder by adding    `export PATH=$PATH:~/quad_swarm_rl` to your .bashrc

## Running an example
```
python -m sample_factory_examples.swarm_rl_examples.train_swarm_rl --env=quadrotor_multi --algo=APPO --experiment=swarm_rl 
```

Enjoy 
```
python -m sample_factory_examples.swarm_rl_examples.enjoy_swarm_rl --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --continuous_actions_sample=False --quads_use_numba=False --experiment=swarm_rl
```