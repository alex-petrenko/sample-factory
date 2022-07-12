## Installing swarm-rl
1. Clone https://github.com/Zhehui-Huang/quad-swarm-rl into your home directory
2. Follow the instructions in the repo to create the conda environment
3. Install the gym_art folder by adding  `export PATH=$PATH:~/quad-swarm-rl` to your .bashrc

## Running an example
```
python -m sample_factory_examples.swarm_rl_examples.train_swarm_rl --env=quadrotor_multi --algo=APPO --experiment=swarm_rl 
```

Enjoy 
```
python -m sample_factory_examples.swarm_rl_examples.enjoy_swarm_rl --algo=APPO --env=quadrotor_multi --replay_buffer_sample_prob=0 --continuous_actions_sample=False --quads_use_numba=False --experiment=swarm_rl
```

Using the runner with a single drone
```
python -m sample_factory.runner.run --run=swarm_rl.runs.single_drone --runner=processes --max_parallel=1 --pause_between=1 --experiments_per_gpu=1 --num_gpus=1
```