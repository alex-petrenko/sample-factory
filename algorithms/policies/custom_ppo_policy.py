import tensorflow as tf

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy


# noinspection PyUnusedLocal
def grad_stats(policy, grads):
    return dict(
        grad_gnorm=tf.global_norm(grads),
    )


CustomPPOTFPolicy = PPOTFPolicy.with_updates(grad_stats_fn=grad_stats)
