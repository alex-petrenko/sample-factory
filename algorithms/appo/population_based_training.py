import copy
import json
import math
import numbers
import os
import random
import time
from enum import Enum
from os.path import join

import numpy as np

from algorithms.appo.appo_utils import TaskType, iterate_recursively
from algorithms.utils.algo_utils import EPS
from utils.utils import log, experiment_dir


def perturb_float(x):
    # mutation amount
    amount = 1.2

    # mutation direction
    new_value = x / amount if random.random() < 0.5 else x * amount
    return new_value


def perturb_exponential_decay(x):
    perturbed = perturb_float(1.0 - x)
    new_value = 1.0 - perturbed
    new_value = max(EPS, new_value)
    return new_value


class PbtTask(Enum):
    SAVE_MODEL, LOAD_MODEL, UPDATE_CFG, UPDATE_REWARD_SCHEME = range(4)


HYPERPARAMS_TO_TUNE = {'learning_rate', 'prior_loss_coeff', 'adam_beta1'}
SPECIAL_PERTURBATION = dict(gamma=perturb_exponential_decay, adam_beta1=perturb_exponential_decay)
REWARD_CATEGORIES_TO_TUNE = {'delta', 'selected_weapon'}


def policy_cfg_file(cfg, policy_id):
    return join(experiment_dir(cfg=cfg), f'policy_{policy_id:02d}_cfg.json')


def policy_reward_shaping_file(cfg, policy_id):
    return join(experiment_dir(cfg=cfg), f'policy_{policy_id:02d}_reward_shaping.json')


class PopulationBasedTraining:
    def __init__(self, cfg, default_reward_shaping, summary_writers):
        self.cfg = cfg
        self.last_update = [0] * self.cfg.num_policies

        self.policy_cfg = [dict() for _ in range(self.cfg.num_policies)]
        self.policy_reward_shaping = [dict() for _ in range(self.cfg.num_policies)]

        self.default_reward_shaping = default_reward_shaping

        self.summary_writers = summary_writers
        self.last_pbt_summaries = 0

        self.learner_workers = self.actor_workers = None

    def init(self, learner_workers, actor_workers):
        self.learner_workers = learner_workers
        self.actor_workers = actor_workers

        for policy_id in range(self.cfg.num_policies):
            # save the policy-specific configs if they don't exist, or else load them from files
            policy_cfg_filename = policy_cfg_file(self.cfg, policy_id)
            if os.path.exists(policy_cfg_filename):
                with open(policy_cfg_filename, 'r') as json_file:
                    log.debug('Loading initial policy %d configuration from file %s', policy_id, policy_cfg_filename)
                    json_params = json.load(json_file)
                    self.policy_cfg[policy_id] = json_params
            else:
                self.policy_cfg[policy_id] = dict()
                for param_name in HYPERPARAMS_TO_TUNE:
                    self.policy_cfg[policy_id][param_name] = self.cfg[param_name]

                if policy_id > 0:  # keep one policy with default settings in the beginning
                    log.debug('Initial cfg mutation for policy %d', policy_id)
                    self.policy_cfg[policy_id] = self._perturb_cfg(self.policy_cfg[policy_id])

        for policy_id in range(self.cfg.num_policies):
            # save the policy-specific reward shaping if it doesn't exist, or else load from file
            policy_reward_shaping_filename = policy_reward_shaping_file(self.cfg, policy_id)

            if os.path.exists(policy_reward_shaping_filename):
                with open(policy_reward_shaping_filename, 'r') as json_file:
                    log.debug(
                        'Loading policy %d reward shaping from file %s', policy_id, policy_reward_shaping_filename,
                    )
                    json_params = json.load(json_file)
                    self.policy_reward_shaping[policy_id] = json_params
            else:
                self.policy_reward_shaping[policy_id] = copy.deepcopy(self.default_reward_shaping)
                if policy_id > 0:  # keep one policy with default settings in the beginning
                    log.debug('Initial rewards mutation for policy %d', policy_id)
                    self.policy_reward_shaping[policy_id] = self._perturb_reward(self.policy_reward_shaping[policy_id])

        # send initial configuration to the system components
        for policy_id in range(self.cfg.num_policies):
            self._save_cfg(policy_id)
            self._save_reward_shaping(policy_id)
            self._learner_update_cfg(policy_id)
            self._actors_update_shaping_scheme(policy_id)

    def _save_cfg(self, policy_id):
        policy_cfg_filename = policy_cfg_file(self.cfg, policy_id)
        with open(policy_cfg_filename, 'w') as json_file:
            log.debug('Saving policy-specific configuration %d to file %s', policy_id, policy_cfg_filename)
            json.dump(self.policy_cfg[policy_id], json_file)

    def _save_reward_shaping(self, policy_id):
        policy_reward_shaping_filename = policy_reward_shaping_file(self.cfg, policy_id)
        with open(policy_reward_shaping_filename, 'w') as json_file:
            log.debug('Saving policy-specific reward shaping %d to file %s', policy_id, policy_reward_shaping_filename)
            json.dump(self.policy_reward_shaping[policy_id], json_file)

    def _perturb_param(self, param, param_name, default_param):
        # toss a coin whether we perturb the parameter at all
        if random.random() > self.cfg.pbt_mutation_rate:
            return param

        if param != default_param and random.random() < 0.1:
            # chance to replace parameter with a default value
            log.debug('%s changed to default value %r', param_name, default_param)
            return default_param

        if param_name in SPECIAL_PERTURBATION:
            new_value = SPECIAL_PERTURBATION[param_name](param)
        elif type(param) is bool:
            new_value = not param
        elif isinstance(param, numbers.Number):
            new_value = perturb_float(float(param))
        else:
            raise RuntimeError('Unsupported parameter type')

        log.debug('Param %s changed from %.6f to %.6f', param_name, param, new_value)
        return new_value

    def _perturb(self, old_params, default_params):
        """Params assumed to be a flat dict."""
        params = copy.deepcopy(old_params)

        for key, value in params.items():
            if isinstance(value, (tuple, list)):
                # this is the case for reward shaping delta params
                params[key] = tuple(
                    self._perturb_param(p, f'{key}_{i}', default_params[key][i])
                    for i, p in enumerate(value)
                )
            else:
                params[key] = self._perturb_param(value, key, default_params[key])

        return params

    def _perturb_cfg(self, original_cfg):
        replacement_cfg = copy.deepcopy(original_cfg)
        return self._perturb(replacement_cfg, default_params=self.cfg)

    def _perturb_reward(self, original_reward_shaping):
        replacement_shaping = copy.deepcopy(original_reward_shaping)
        for category in REWARD_CATEGORIES_TO_TUNE:
            if category in replacement_shaping:
                replacement_shaping[category] = self._perturb(
                    replacement_shaping[category], default_params=self.default_reward_shaping[category],
                )
        return replacement_shaping

    def _force_learner_to_save_model(self, policy_id):
        learner_worker = self.learner_workers[policy_id]
        learner_worker.model_saved_event.clear()
        save_task = (PbtTask.SAVE_MODEL, policy_id)
        learner_worker.task_queue.put((TaskType.PBT, save_task))
        log.debug('Wait while learner %d saves the model...', policy_id)
        learner_worker.model_saved_event.wait()
        log.debug('Learner %d saved the model!', policy_id)
        learner_worker.model_saved_event.clear()

    def _learner_load_model(self, policy_id, replacement_policy):
        log.debug('Asking learner %d to load model from %d', policy_id, replacement_policy)

        load_task = (PbtTask.LOAD_MODEL, (policy_id, replacement_policy))
        learner_worker = self.learner_workers[policy_id]
        learner_worker.task_queue.put((TaskType.PBT, load_task))

    def _learner_update_cfg(self, policy_id):
        learner_worker = self.learner_workers[policy_id]

        log.debug('Sending learning configuration to learner %d...', policy_id)
        cfg_task = (PbtTask.UPDATE_CFG, (policy_id, self.policy_cfg[policy_id]))
        learner_worker.task_queue.put((TaskType.PBT, cfg_task))

    def _actors_update_shaping_scheme(self, policy_id):
        log.debug('Sending latest reward scheme to actors for policy %d...', policy_id)
        for actor_worker in self.actor_workers:
            reward_scheme_task = (PbtTask.UPDATE_REWARD_SCHEME, (policy_id, self.policy_reward_shaping[policy_id]))
            actor_worker.task_queue.put((TaskType.PBT, reward_scheme_task))

    @staticmethod
    def _write_dict_summaries(dictionary, writer, name, env_steps):
        for d, key, value in iterate_recursively(dictionary):
            if isinstance(value, bool):
                value = int(value)

            if isinstance(value, (int, float)):
                writer.add_scalar(f'zz_pbt/{name}_{key}', value, env_steps)
            elif isinstance(value, (tuple, list)):
                for i, tuple_value in enumerate(value):
                    writer.add_scalar(f'zz_pbt/{name}_{key}_{i}', tuple_value, env_steps)
            else:
                log.error('Unsupported type in pbt summaries %r', type(value))

    def _write_pbt_summaries(self, policy_id, env_steps):
        writer = self.summary_writers[policy_id]
        self._write_dict_summaries(self.policy_cfg[policy_id], writer, 'cfg', env_steps)
        self._write_dict_summaries(self.policy_reward_shaping[policy_id], writer, 'rew', env_steps)

    def _update_policy(self, policy_id, policy_stats):
        if 'true_reward' not in policy_stats:
            return

        true_rewards = policy_stats['true_reward']

        # not enough data to perform PBT yet
        for rewards in true_rewards:
            if len(rewards) <= 0:
                return

        true_rewards = [np.mean(r) for r in true_rewards]

        policies = list(range(self.cfg.num_policies))
        policies_sorted = sorted(zip(true_rewards, policies), reverse=True)
        policies_sorted = [p for rew, p in policies_sorted]

        replace_fraction = self.cfg.pbt_replace_fraction
        replace_number = math.ceil(replace_fraction * self.cfg.num_policies)

        best_policies = policies_sorted[:replace_number]
        worst_policies = policies_sorted[-replace_number:]

        if policy_id in best_policies:
            # don't touch the policies that are doing very well
            return

        log.debug('PBT best policies: %r, worst policies %r', best_policies, worst_policies)

        # to make the code below uniform, this means keep our own parameters and cfg
        # we only take parameters and cfg from another policy if certain conditions are met (see below)
        replacement_policy = policy_id

        if policy_id in worst_policies:
            log.debug('Current policy %d is among the worst policies %r', policy_id, worst_policies)

            replacement_policy_candidate = random.choice(best_policies)
            reward_delta = true_rewards[replacement_policy_candidate] - true_rewards[policy_id]
            reward_delta_relative = abs(reward_delta / (true_rewards[replacement_policy_candidate] + EPS))

            if abs(reward_delta) > self.cfg.pbt_replace_reward_gap_absolute and reward_delta_relative > self.cfg.pbt_replace_reward_gap:
                replacement_policy = replacement_policy_candidate
                log.debug(
                    'Difference in reward is %.4f (%.4f), policy %d weights to be replaced by %d',
                    reward_delta, reward_delta_relative, policy_id, replacement_policy,
                )

        if policy_id == 0:
            # Do not ever mutate the 1st policy, leave it for the reference
            # Still we allow replacements in case it's really bad
            self.policy_cfg[policy_id] = self.policy_cfg[replacement_policy]
            self.policy_reward_shaping[policy_id] = self.policy_reward_shaping[replacement_policy]
        else:
            self.policy_cfg[policy_id] = self._perturb_cfg(self.policy_cfg[replacement_policy])
            self.policy_reward_shaping[policy_id] = self._perturb_reward(self.policy_reward_shaping[replacement_policy])

        if replacement_policy != policy_id:
            # force replacement policy learner to save the model and wait until it's done
            self._force_learner_to_save_model(replacement_policy)

            # now that the latest "replacement" model is saved to disk, we ask the learner to load the replacement policy
            self._learner_load_model(policy_id, replacement_policy)

        self._save_cfg(policy_id)
        self._save_reward_shaping(policy_id)
        self._learner_update_cfg(policy_id)
        self._actors_update_shaping_scheme(policy_id)

    def update(self, env_steps, policy_stats):
        if not self.cfg.with_pbt or self.cfg.num_policies <= 1:
            return

        for policy_id in range(self.cfg.num_policies):
            if policy_id not in env_steps:
                continue

            steps_since_last_update = env_steps[policy_id] - self.last_update[policy_id]
            if steps_since_last_update > self.cfg.pbt_period_env_steps:
                self._update_policy(policy_id, policy_stats)
                self._write_pbt_summaries(policy_id, env_steps[policy_id])
                self.last_update[policy_id] = env_steps[policy_id]

        now = time.time()
        if now - self.last_pbt_summaries > 60 * 60:
            self.last_pbt_summaries = now

            for policy_id in range(self.cfg.num_policies):
                if policy_id in env_steps:
                    self._write_pbt_summaries(policy_id, env_steps[policy_id])
