import random

from ray.tune.schedulers import PopulationBasedTraining

from utils.utils import log


def explore(config):
    """We can modify the config further in this function."""
    log.info('New config: %r', config)
    return config


def get_pbt_scheduler():
    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=600,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.8, 1.0),
            "clip_param": lambda: random.uniform(0.05, 0.2),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 10),
            "train_batch_size": lambda: random.randint(256, 4096),
        },
        custom_explore_fn=explore,
    )

    return pbt
