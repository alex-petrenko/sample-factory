# Quad-Swarm-RL Integrations

### Installation

Clone https://github.com/Zhehui-Huang/quad-swarm-rl into your home directory

Install dependencies in your conda environment
```
cd ~/quad-swarm-rl
pip install -e .
```

Note: if you have any error with bezier, run:
```
BEZIER_NO_EXTENSION=true pip install bezier==2020.5.19
pip install -e .
```

Install the gym_art folder by adding it to your PATH
```
echo "export PATH=$PATH:~/quad-swarm-rl" >> ~/.bashrc
```

### Running Experiments

The environments can be run from the `sf_examples.swarm_rl_examples` folder. 

Experiments can be run with the `train_swarm_rl` script and viewed with the `enjoy_swarm_rl` script. If you are running custom experiments, it is recommended to use the `quad_multi_mix_baseline` runner script and make any modifications as needed. See `single_drone` and `multi_drone` runner scripts for an examples.

### Results

#### Reports

