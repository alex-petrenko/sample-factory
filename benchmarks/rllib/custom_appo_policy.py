import tensorflow as tf

from ray.rllib.agents.ppo.appo_policy import AsyncPPOTFPolicy

from benchmarks.rllib.custom_ppo_policy import grad_stats, _mean_min_max


def stats(policy, batch_tensors):
    # stats_dict = dict(
    #     cur_lr=tf.cast(policy.cur_lr, tf.float64),
    #     policy_loss=policy.loss.pi_loss,
    #     entropy=policy.loss.entropy,
    #     var_gnorm=tf.global_norm(policy.var_list),
    #     vf_loss=policy.loss.vf_loss,
    # )
    #
    # if policy.config['vtrace']:
    #     is_stat_mean, is_stat_var = tf.nn.moments(policy.loss.is_ratio, [0, 1])
    #     stats_dict.update(dict(mean_IS=is_stat_mean))
    #     stats_dict.update(dict(var_IS=is_stat_var))
    #
    # if 'use_kl_loss' in policy.config and policy.config['use_kl_loss']:
    #     stats_dict.update(dict(KL=policy.loss.mean_kl))
    #     stats_dict.update(dict(KL_Coeff=policy.kl_coeff))
    #
    # stats_dict.update(_mean_min_max(policy.value_function, '_summ_value_estimate'))
    # stats_dict.update(_mean_min_max(batch_tensors['rewards'], '_summ_rewards'))

    stats_dict = dict()
    return stats_dict


CustomAPPOTFPolicy = AsyncPPOTFPolicy.with_updates(
    stats_fn=stats,
    grad_stats_fn=grad_stats,
)
