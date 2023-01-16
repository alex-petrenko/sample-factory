# Recent releases

##### v2.0.3

* Added cfg parameters `--lr_adaptive_min` and `--lr_adaptive_max` to control the minimum and maximum adaptive learning rate
* Added Brax environment support + custom brax renderer for enjoy scripts
* Automatically set `--recurrence` based on feed-forward vs RNN training
* Added `--enjoy_script` and `--train_script` for generating the model card when uploading to the Hugging Face Hub (thank you Andrew!)
* Fixed video name when generating Hugging Face model card
* Fixed small DMLab-related bug (thank you Lucy!)

##### v2.0.2

* `cfg.json` renamed to `config.json` for consistency with other HuggingFace integrations
* We can still load from legacy checkpoints (`cfg.json` will be renamed to `config.json`)
* Fixed a bug in enjoy.py with multi-agent envs

##### v2.0.1

* Added MuJoCo & IsaacGym examples to the PyPI package
* Added missing `__init__.py` files

##### v2.0.0

**Major update, adds new functionality, changes API and configuration parameters**

* Major API update, codebase rewritten from scratch for better maintainability and clarity
* Synchronous and asynchronous training modes
* Serial and parallel execution modes
* Support for vectorized and GPU-accelerated environments in batched sampling mode
* Integration with Hugging Face Hub
* New environment integrations, CI, and 40+ documentation pages

See [v1 to v2](../08-miscellaneous/v1-to-v2.md) transition guide for details.

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
