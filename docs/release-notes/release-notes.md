# Recent releases

##### v1.121.4
* Support Weights and Biases (see section "WandB support")
* More configurable population-based training: 
can set from command line whether or not to mutate gamma, plus the perturbation magnitude for all float hyperparams can also be set from command line:
```
--pbt_optimize_gamma: Whether to optimize gamma, discount factor, or not (experimental) (default: False)
--pbt_perturb_min: When PBT mutates a float hyperparam, it samples the change magnitude randomly from the uniform distribution [pbt_perturb_min, pbt_perturb_max] (default: 1.05)
--pbt_perturb_max: When PBT mutates a float hyperparam, it samples the change magnitude randomly from the uniform distribution [pbt_perturb_min, pbt_perturb_max] (default: 1.5)
```

##### v1.121.3
* Fixed a small bug related to population-based training (a reward shaping dictionary was assumed to be a flat dict,
while it could be a nested dict in some envs)

##### v1.121.2
* Fixed a bug that prevented Vizdoom *.cfg and *.wad files from being copied to site-packages during installation from PyPI
* Added example on how to use custom Vizdoom envs without modifying the source code (`sample_factory_examples/train_custom_vizdoom_env.py`)

##### v1.121.0
* Added fixed KL divergence penalty as in https://arxiv.org/pdf/1707.06347.pdf 
Its usage is highly encouraged in environments with continuous action spaces (i.e. set --kl_loss_coeff=1.0).
Otherwise numerical instabilities can occur in certain environments, especially when the policy lag is high

* More summaries related to the new loss

##### v1.120.2
* More improvements and fixes in runner interface, including support for NGC cluster

##### v1.120.1
* Runner interface improvements for Slurm

##### v1.120.0
* Support inactive agents. To deactivate an agent for a portion of the episode the environment should return `info={'is_active': False}` for the inactive agent. Useful for environments such as hide-n-seek.
* Improved memory consumption and performance with better shared memory management.
* Experiment logs are now saved into the experiment folder as `sf_log.txt`
* DMLab-related bug fixes (courtesy of @donghoonlee04 and @sungwoong. Thank you!)
