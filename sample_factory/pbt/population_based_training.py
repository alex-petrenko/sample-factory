"""Population-Based Training implementation, inspired by https://arxiv.org/abs/1807.01281."""


import copy
import json
import math
import os
import random
import time
from os.path import join
from typing import Dict, Optional, SupportsFloat

import numpy as np
from signal_slot.signal_slot import EventLoopObject
from tensorboardX import SummaryWriter

from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.misc import EPS
from sample_factory.utils.dicts import iter_dicts_recursively, iterate_recursively
from sample_factory.utils.typing import Config, PolicyID
from sample_factory.utils.utils import experiment_dir, log


def perturb_float(x, perturb_amount=1.2):
    # mutation direction
    new_value = x / perturb_amount if random.random() < 0.5 else x * perturb_amount
    return new_value


def perturb_vtrace(x, _cfg):
    return perturb_float(x, perturb_amount=1.005)


def perturb_exponential_decay(x, _cfg, perturb_amount_min=1.01, perturb_amount_max=1.2):
    # very conservative values, things like gamma should not change quickly
    perturb_amount = random.uniform(perturb_amount_min, perturb_amount_max)
    perturbed = perturb_float(1.0 - x, perturb_amount=perturb_amount)
    new_value = 1.0 - perturbed
    new_value = max(EPS, new_value)
    return new_value


def perturb_batch_size(x, cfg):
    new_value = perturb_float(x, perturb_amount=1.2)
    initial_batch_size = cfg.batch_size
    max_batch_size = initial_batch_size * 1.5
    min_batch_size = cfg.rollout

    new_value = min(new_value, max_batch_size)

    # round down to whole number of rollouts
    new_value = (int(new_value) // cfg.rollout) * cfg.rollout

    new_value = max(new_value, min_batch_size)
    return new_value


HYPERPARAMS_TO_TUNE = {
    "learning_rate",
    "exploration_loss_coeff",
    "value_loss_coeff",
    "max_grad_norm",
    "ppo_clip_ratio",
    "ppo_clip_value",
    # gamma can be added with a CLI parameter (--pbt_optimize_gamma=True)
}

# if not specified then tune all rewards
REWARD_CATEGORIES_TO_TUNE = {
    "doom_": ["delta", "selected_weapon"],
}

# HYPERPARAMS_TO_TUNE_EXTENDED = {
#     'learning_rate', 'exploration_loss_coeff', 'value_loss_coeff', 'adam_beta1', 'max_grad_norm',
#     'ppo_clip_ratio', 'ppo_clip_value', 'vtrace_rho', 'vtrace_c',
# }

SPECIAL_PERTURBATION = dict(
    gamma=perturb_exponential_decay,
    adam_beta1=perturb_exponential_decay,
    vtrace_rho=perturb_vtrace,
    vtrace_c=perturb_vtrace,
    batch_size=perturb_batch_size,
)


def policy_cfg_file(cfg, policy_id):
    return join(experiment_dir(cfg=cfg), f"policy_{policy_id:02d}_cfg.json")


def policy_reward_shaping_file(cfg, policy_id):
    return join(experiment_dir(cfg=cfg), f"policy_{policy_id:02d}_reward_shaping.json")


def update_cfg_signal(policy_id: PolicyID) -> str:
    return f"update_cfg{policy_id}"


def save_model_signal(policy_id: PolicyID) -> str:
    return f"save_model{policy_id}"


def load_model_signal(policy_id: PolicyID) -> str:
    return f"load_model{policy_id}"


class PopulationBasedTraining(AlgoObserver, EventLoopObject):
    def __init__(self, cfg: Config, runner: Runner):
        EventLoopObject.__init__(self, runner.event_loop, "PBT")

        self.cfg: Config = cfg
        self.env_info: Optional[EnvInfo] = None

        self.runner: Runner = runner

        # currently not supported, would require changes on the batcher
        # if cfg.pbt_optimize_batch_size:
        #     HYPERPARAMS_TO_TUNE.add("batch_size")

        if cfg.pbt_optimize_gamma:
            HYPERPARAMS_TO_TUNE.add("gamma")

        self.last_update = [0] * self.cfg.num_policies

        self.policy_cfg = [dict() for _ in range(self.cfg.num_policies)]
        self.policy_reward_shaping = [dict() for _ in range(self.cfg.num_policies)]

        self.default_reward_shaping: Optional[Dict] = None

        self.last_pbt_summaries = 0

        self.reward_categories_to_tune = []
        for env_prefix, categories in REWARD_CATEGORIES_TO_TUNE.items():
            if cfg.env.startswith(env_prefix):
                self.reward_categories_to_tune = categories

        # Set to non-None when policy x has to be replaced by replacement_policy[x]
        self.replacement_policy: Dict[PolicyID, Optional[PolicyID]] = {p: None for p in range(self.cfg.num_policies)}

    def on_init(self, runner: Runner) -> None:
        self.env_info = runner.env_info
        self.default_reward_shaping = self.env_info.reward_shaping_scheme

        for policy_id in range(self.cfg.num_policies):
            # save the policy-specific configs if they don't exist, or else load them from files
            policy_cfg_filename = policy_cfg_file(self.cfg, policy_id)
            if os.path.exists(policy_cfg_filename):
                with open(policy_cfg_filename, "r") as json_file:
                    log.debug("Loading initial policy %d configuration from file %s", policy_id, policy_cfg_filename)
                    json_params = json.load(json_file)
                    self.policy_cfg[policy_id] = json_params
            else:
                self.policy_cfg[policy_id] = dict()
                for param_name in HYPERPARAMS_TO_TUNE:
                    self.policy_cfg[policy_id][param_name] = self.cfg[param_name]

                if policy_id > 0:  # keep one policy with default settings in the beginning
                    log.debug("Initial cfg mutation for policy %d", policy_id)
                    self.policy_cfg[policy_id] = self._perturb_cfg(self.policy_cfg[policy_id])

        for policy_id in range(self.cfg.num_policies):
            # save the policy-specific reward shaping if it doesn't exist, or else load from file
            policy_reward_shaping_filename = policy_reward_shaping_file(self.cfg, policy_id)

            if os.path.exists(policy_reward_shaping_filename):
                with open(policy_reward_shaping_filename, "r") as json_file:
                    log.debug(
                        "Loading policy %d reward shaping from file %s",
                        policy_id,
                        policy_reward_shaping_filename,
                    )
                    json_params = json.load(json_file)
                    self.policy_reward_shaping[policy_id] = json_params
            else:
                self.policy_reward_shaping[policy_id] = copy.deepcopy(self.default_reward_shaping)
                if policy_id > 0:  # keep one policy with default settings in the beginning
                    log.debug("Initial rewards mutation for policy %d", policy_id)
                    self.policy_reward_shaping[policy_id] = self._perturb_reward(self.policy_reward_shaping[policy_id])

        for policy_id in range(self.cfg.num_policies):
            self._save_cfg(policy_id)
            self._save_reward_shaping(policy_id)

    def on_connect_components(self, runner: Runner) -> None:
        for policy_id, learner_worker in runner.learners.items():
            self.connect(update_cfg_signal(policy_id), learner_worker.on_update_cfg)
            self.connect(save_model_signal(policy_id), learner_worker.save)
            self.connect(load_model_signal(policy_id), learner_worker.load)
            learner_worker.saved_model.connect(self.on_saved_model)

    def on_start(self, runner: Runner) -> None:
        # send initial configuration to the system components
        for policy_id in range(self.cfg.num_policies):
            self._learner_update_cfg(policy_id)
            runner.update_reward_shaping(policy_id, self.policy_reward_shaping[policy_id])

    def _save_cfg(self, policy_id):
        policy_cfg_filename = policy_cfg_file(self.cfg, policy_id)
        with open(policy_cfg_filename, "w") as json_file:
            log.debug("Saving policy-specific configuration %d to file %s", policy_id, policy_cfg_filename)
            json.dump(self.policy_cfg[policy_id], json_file)

    def _save_reward_shaping(self, policy_id):
        policy_reward_shaping_filename = policy_reward_shaping_file(self.cfg, policy_id)
        with open(policy_reward_shaping_filename, "w") as json_file:
            log.debug("Saving policy-specific reward shaping %d to file %s", policy_id, policy_reward_shaping_filename)
            json.dump(self.policy_reward_shaping[policy_id], json_file)

    def _perturb_param(self, param, param_name, default_param):
        # toss a coin whether we perturb the parameter at all
        if random.random() > self.cfg.pbt_mutation_rate:
            return param

        if param != default_param and random.random() < 0.01:
            # small chance to replace parameter with a default value
            log.debug("%s changed to default value %r", param_name, default_param)
            return default_param

        if param_name in SPECIAL_PERTURBATION:
            new_value = SPECIAL_PERTURBATION[param_name](param, self.cfg)
        elif type(param) is bool:
            new_value = not param
        elif isinstance(param, SupportsFloat):
            perturb_amount = random.uniform(self.cfg.pbt_perturb_min, self.cfg.pbt_perturb_max)
            new_value = perturb_float(float(param), perturb_amount=perturb_amount)
        else:
            raise RuntimeError("Unsupported parameter type")

        log.debug("Param %s changed from %.6f to %.6f", param_name, param, new_value)
        return new_value

    def _perturb(self, old_params, default_params):
        """Params assumed to be a flat dict or a dict of dicts, etc."""
        params = copy.deepcopy(old_params)

        # this will iterate over all leaf nodes in two identical (potentially nested) dicts
        for d_params, d_default, key, value, value_default in iter_dicts_recursively(params, default_params):
            if isinstance(value, (tuple, list)):
                # this is the case for reward shaping delta params
                # i.e. where reward is characterized by two values (one corresponding to a negative or a positive change
                # of something in the environment, like health).
                # See envs/doom/wrappers/reward_shaping.py:39 for example
                d_params[key] = tuple(
                    self._perturb_param(p, f"{key}_{i}", value_default[i]) for i, p in enumerate(value)
                )
            else:
                d_params[key] = self._perturb_param(value, key, value_default)

        return params

    def _perturb_cfg(self, original_cfg):
        replacement_cfg = copy.deepcopy(original_cfg)
        return self._perturb(replacement_cfg, default_params=self.cfg)

    def _perturb_reward(self, original_reward_shaping):
        if original_reward_shaping is None:
            return None

        replacement_shaping = copy.deepcopy(original_reward_shaping)

        if len(self.reward_categories_to_tune) > 0:
            for category in self.reward_categories_to_tune:
                if category in replacement_shaping:
                    replacement_shaping[category] = self._perturb(
                        replacement_shaping[category],
                        default_params=self.default_reward_shaping[category],
                    )
        else:
            replacement_shaping = self._perturb(replacement_shaping, default_params=self.default_reward_shaping)

        return replacement_shaping

    def _learner_update_cfg(self, policy_id: PolicyID) -> None:
        log.debug(f"Sending learning configuration to learner {policy_id}...")
        self.emit(update_cfg_signal(policy_id), self.policy_cfg[policy_id])

    @staticmethod
    def _write_dict_summaries(dictionary, writer, name, env_steps):
        for d, key, value in iterate_recursively(dictionary):
            if isinstance(value, bool):
                value = int(value)

            if isinstance(value, (int, float)):
                writer.add_scalar(f"zz_pbt/{name}_{key}", value, env_steps)
            elif isinstance(value, (tuple, list)):
                for i, tuple_value in enumerate(value):
                    writer.add_scalar(f"zz_pbt/{name}_{key}_{i}", tuple_value, env_steps)
            else:
                log.error("Unsupported type in pbt summaries %r", type(value))

    def _write_pbt_summaries(self, policy_id, env_steps, writer: SummaryWriter):
        self._write_dict_summaries(self.policy_cfg[policy_id], writer, "cfg", env_steps)
        if self.policy_reward_shaping[policy_id] is not None:
            self._write_dict_summaries(self.policy_reward_shaping[policy_id], writer, "rew", env_steps)

    def _update_policy(self, policy_id, policy_stats):
        if self.cfg.pbt_target_objective not in policy_stats:
            return

        target_objectives = policy_stats[self.cfg.pbt_target_objective]

        # not enough data to perform PBT yet
        for objectives in target_objectives:
            if len(objectives) <= 0:
                return

        target_objectives = [np.mean(o) for o in target_objectives]

        policies = list(range(self.cfg.num_policies))
        policies_sorted = sorted(zip(target_objectives, policies), reverse=True)
        policies_sorted = [p for objective, p in policies_sorted]

        replace_fraction = self.cfg.pbt_replace_fraction
        replace_number = math.ceil(replace_fraction * self.cfg.num_policies)

        best_policies = policies_sorted[:replace_number]
        worst_policies = policies_sorted[-replace_number:]

        if policy_id in best_policies:
            # don't touch the policies that are doing well
            return

        log.debug("PBT best policies: %r, worst policies %r", best_policies, worst_policies)

        # to make the code below uniform, this means keep our own parameters and cfg
        # we only take parameters and cfg from another policy if certain conditions are met (see below)
        replacement_policy = policy_id

        if policy_id in worst_policies:
            log.debug("Current policy %d is among the worst policies %r", policy_id, worst_policies)

            replacement_policy_candidate = random.choice(best_policies)
            reward_delta = target_objectives[replacement_policy_candidate] - target_objectives[policy_id]
            reward_delta_relative = abs(
                reward_delta / (target_objectives[replacement_policy_candidate] + EPS)
            )  # TODO: this might not work correctly with negative rewards

            if (
                abs(reward_delta) > self.cfg.pbt_replace_reward_gap_absolute
                and reward_delta_relative > self.cfg.pbt_replace_reward_gap
            ):
                replacement_policy = replacement_policy_candidate
                log.debug(
                    "Difference in reward is %.4f (%.4f), policy %d weights to be replaced by %d",
                    reward_delta,
                    reward_delta_relative,
                    policy_id,
                    replacement_policy,
                )
            else:
                log.debug("Difference in reward is not enough %.3f %.3f", abs(reward_delta), reward_delta_relative)

        if policy_id == 0:
            # Do not ever mutate the 1st policy, leave it for the reference
            # Still we allow replacements in case it's really bad
            self.policy_cfg[policy_id] = self.policy_cfg[replacement_policy]
            self.policy_reward_shaping[policy_id] = self.policy_reward_shaping[replacement_policy]
        else:
            self.policy_cfg[policy_id] = self._perturb_cfg(self.policy_cfg[replacement_policy])
            self.policy_reward_shaping[policy_id] = self._perturb_reward(self.policy_reward_shaping[replacement_policy])

        # force replacement policy learner to save its model so we get the latest version
        # for simplicity we do this even if the policy is replaced by itself (so no replacement happens)
        self.replacement_policy[policy_id] = replacement_policy
        self.emit(save_model_signal(replacement_policy))

    def on_saved_model(self, replacement_policy: PolicyID) -> None:
        """
        Called when learner saves its model. At this point we're free to use this model to
        replace other policies.
        """
        for policy_id in range(self.cfg.num_policies):
            if self.replacement_policy[policy_id] != replacement_policy:
                continue

            if replacement_policy != policy_id:
                # only load the model if it's not already loaded
                log.debug(f"Asking learner {policy_id} to load model from {replacement_policy}")
                self.emit(load_model_signal(policy_id), replacement_policy)

            self.replacement_policy[policy_id] = None

            self._save_cfg(policy_id)
            self._save_reward_shaping(policy_id)
            self._learner_update_cfg(policy_id)

            self.runner.update_reward_shaping(policy_id, self.policy_reward_shaping[policy_id])

    def on_training_step(self, runner: Runner, training_iteration_since_resume: int) -> None:
        if not self.cfg.with_pbt or self.cfg.num_policies <= 1:
            return

        env_steps = runner.env_steps
        policy_avg_stats = runner.policy_avg_stats

        for policy_id in range(self.cfg.num_policies):
            if policy_id not in env_steps:
                continue

            if env_steps[policy_id] < self.cfg.pbt_start_mutation:
                continue

            steps_since_last_update = env_steps[policy_id] - self.last_update[policy_id]
            if steps_since_last_update > self.cfg.pbt_period_env_steps:
                self._update_policy(policy_id, policy_avg_stats)
                self._write_pbt_summaries(policy_id, env_steps[policy_id], runner.writers[policy_id])
                self.last_update[policy_id] = env_steps[policy_id]

        # also periodically dump a pbt summary even if we didn't change anything
        now = time.time()
        if now - self.last_pbt_summaries > 5 * 60:
            for policy_id in range(self.cfg.num_policies):
                if policy_id in env_steps:
                    self._write_pbt_summaries(policy_id, env_steps[policy_id], runner.writers[policy_id])
                    self.last_pbt_summaries = now
