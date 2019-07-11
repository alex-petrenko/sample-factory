import tensorflow as tf
from ray.rllib.agents.ppo.appo_policy import AsyncPPOTFPolicy

from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy, kl_and_loss_stats

from algorithms.policies.custom_ppo_policy import stats, grad_stats

CustomAPPOTFPolicy = AsyncPPOTFPolicy.with_updates(
    stats_fn=stats,
    grad_stats_fn=grad_stats,
)
