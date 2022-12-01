# v1 to v2 Migration

The repository has changed very significantly from the original Sample Factory v1,
which makes it pretty much impossible to track all the changes.

Perhaps the most obvious change that will affect everyone
is that we removed generic entry points such as `train_appo.py` and `enjoy_appo.py`.
As a consequence, now there's no difference between how "custom" and "built-in" environments are handled
(custom envs are first-class citizens now).
See train/enjoy scripts in `sf_examples` to see how to use the new API.

If you have any custom code that registers custom environments of model architectures,
please check [Custom Environments](../03-customization/custom-environments.md) and 
[Custom Models](../03-customization/custom-models.md) for the new API.

Some configuration parameters were renamed:

* `--ppo_epochs` -> `--num_epochs`
* `--num_batches_per_iteration` -> `--num_batches_per_epoch`
* `--num_minibatches_to_accumulate` -> `--num_batches_to_accumulate` (also changed semantically, check the cfg reference)

`Runner` class we used to launch groups of experiments such as hyperparameter searches got renamed to `Launcher`.
The name `Runner` now refers to an entirely different concept, a class that handles the main loop of the algorithm.

Entities `ActorWorker` and `PolicyWorker` were renamed to `RolloutWorker` and `InferenceWorker` respectively.

Due to the gravity of the changes it is difficult to provide a comprehensive migration guide. If you recently
updated your codebase to use Sample Factory v2.0+, please consider sharing your experience and [contribute](../12-community/contribution.md)
to this guide! :)