import copy
import operator
from collections import deque

import gym

from sample_factory.algorithms.utils.algo_utils import EPS
from sample_factory.envs.env_utils import RewardShapingInterface
from sample_factory.utils.utils import log

NUM_WEAPONS = 8

# these are somewhat arbitrary, but can be optimized via PBT
WEAPON_PREFERENCE = {
    2: 1,  # pistol
    3: 5,  # shotguns
    4: 5,  # machinegun
    5: 5,  # rocket launcher
    6: 10,  # plasmagun
    7: 10,  # bfg
}

WEAPON_DELTA_REWARDS = dict()
SELECTED_WEAPON_REWARDS = dict()
for weapon in range(NUM_WEAPONS):
    pref = WEAPON_PREFERENCE.get(weapon, 1)
    # reward/penalty for finding/losing a weapon
    WEAPON_DELTA_REWARDS[f'WEAPON{weapon}'] = (+0.02 * pref, -0.01 * pref)
    # reward/penalty for picking up/spending weapon ammo
    WEAPON_DELTA_REWARDS[f'AMMO{weapon}'] = (+0.0002 * pref, -0.0001 * pref)

    # reward for choosing a weapon and sticking to it; really helps learning in the beginning, otherwise the agent
    # just keeps changing weapons all the time, unable to shoot. Towards the later stages of the training agents
    # tend to ignore this, and change weapons at will
    SELECTED_WEAPON_REWARDS[f'SELECTED{weapon}'] = 0.0002 * pref


# reward shaping scheme to convert env info into scalar reward
REWARD_SHAPING_DEATHMATCH_V0 = dict(
    delta=dict(
        FRAGCOUNT=(+1, -1.5),  # reward per unit of positive or negative change
        DEATHCOUNT=(-0.75, +0.75),
        HITCOUNT=(+0.01, -0.01),
        DAMAGECOUNT=(+0.003, -0.003),
        HEALTH=(+0.005, -0.003),
        ARMOR=(+0.005, -0.001),
        **WEAPON_DELTA_REWARDS,
    ),
    selected_weapon=SELECTED_WEAPON_REWARDS,
)

# "zero-sum" scheme for self-play scenarios
REWARD_SHAPING_DEATHMATCH_V1 = copy.deepcopy(REWARD_SHAPING_DEATHMATCH_V0)
REWARD_SHAPING_DEATHMATCH_V1['delta'].update(dict(
    FRAGCOUNT=(+1, -0.001),
    DEATHCOUNT=(-1, +1),
    HITCOUNT=(0, 0),
    DAMAGECOUNT=(+0.01, -0.01),
    HEALTH=(+0.01, -0.01),
))


# just the same reward scheme for consistency, only battle does not have most game variables,
# so only a very small reward shaping for collecting Health and Ammo will be applied.
# It works pretty much the same without this.
REWARD_SHAPING_BATTLE = copy.deepcopy(REWARD_SHAPING_DEATHMATCH_V0)


def true_reward_final_position(info):
    if info['LEADER_GAP'] == 0:
        # tied with the leader for the win, we don't reward for ties, only for the win
        return 0.0
    elif info['FINAL_PLACE'] > 1:
        # lost the match (don't care about the place, losing is losing)
        return 0.0
    else:
        # won the match!
        assert info['FINAL_PLACE'] == 1
        return 1.0


def true_reward_frags(info):
    return info['FRAGCOUNT']


class DoomRewardShapingWrapper(gym.Wrapper, RewardShapingInterface):
    """Convert game info variables into scalar reward using a reward shaping scheme."""

    def __init__(self, env, reward_shaping_scheme=None, true_reward_func=None):
        gym.Wrapper.__init__(self, env)
        RewardShapingInterface.__init__(self)

        self.reward_shaping_scheme = reward_shaping_scheme
        self.true_reward_func = true_reward_func

        # without this we reward using BFG and shotguns too much
        self.reward_delta_limits = dict(DAMAGECOUNT=200, HITCOUNT=5)

        self.prev_vars = dict()
        self.prev_dead = True

        self.orig_env_reward = self.total_shaping_reward = 0.0

        self.selected_weapon = deque([], maxlen=5)

        self.reward_structure = {}

        self.verbose = False
        self.print_once = False

        # save a reference to this wrapper in the actual env class, for other wrappers
        self.env.unwrapped.reward_shaping_interface = self

    def get_default_reward_shaping(self):
        return self.reward_shaping_scheme

    def get_current_reward_shaping(self, agent_idx: int):
        return self.reward_shaping_scheme

    def set_reward_shaping(self, reward_shaping: dict, agent_idx: int):
        self.reward_shaping_scheme = reward_shaping

    def _delta_rewards(self, info):
        reward = 0.0
        deltas = []

        for var_name, rewards in self.reward_shaping_scheme['delta'].items():
            if var_name not in self.prev_vars:
                continue

            # generate reward based on how the env variable values changed
            new_value = info.get(var_name, 0.0)
            prev_value = self.prev_vars[var_name]
            delta = new_value - prev_value

            if var_name in self.reward_delta_limits:
                delta = min(delta, self.reward_delta_limits[var_name])

            if abs(delta) > EPS:
                if delta > EPS:
                    reward_delta = delta * rewards[0]
                else:
                    reward_delta = -delta * rewards[1]

                reward += reward_delta
                deltas.append((var_name, reward_delta, delta))
                self.reward_structure[var_name] = self.reward_structure.get(var_name, 0.0) + reward_delta

        return reward, deltas

    def _selected_weapon_rewards(self, selected_weapon, selected_weapon_ammo, deltas):
        # we must keep the weapon ready for a certain number of frames to get rewards
        unholstered = len(self.selected_weapon) > 4 and all(sw == selected_weapon for sw in self.selected_weapon)
        reward = 0.0

        if selected_weapon_ammo > 0 and unholstered:
            try:
                reward = self.reward_shaping_scheme['selected_weapon'][f'SELECTED{weapon}']
            except KeyError:
                log.error('%r', self.reward_shaping_scheme)
                log.error('%r', selected_weapon)
            weapon_key = f'weapon{selected_weapon}'
            deltas.append((weapon_key, reward))
            self.reward_structure[weapon_key] = self.reward_structure.get(weapon_key, 0.0) + reward

        return reward

    def _parse_info(self, info, done):
        if self.reward_shaping_scheme is None:
            # skip reward calculation
            return 0.0

        # by default these are negative values if no weapon is selected
        selected_weapon = info.get('SELECTED_WEAPON', 0.0)
        selected_weapon = int(max(0, selected_weapon))
        selected_weapon_ammo = float(max(0.0, info.get('SELECTED_WEAPON_AMMO', 0.0)))
        self.selected_weapon.append(selected_weapon)

        was_dead = self.prev_dead
        is_alive = not info.get('DEAD', 0.0)
        just_respawned = was_dead and is_alive

        shaping_reward = 0.0
        if not done and not just_respawned:
            shaping_reward, deltas = self._delta_rewards(info)

            shaping_reward += self._selected_weapon_rewards(
                selected_weapon, selected_weapon_ammo, deltas,
            )

            if abs(shaping_reward) > 2.5 and not self.print_once:
                log.info('Large shaping reward %.3f for %r', shaping_reward, deltas)
                self.print_once = True

        if done and 'FRAGCOUNT' in self.reward_structure:
            sorted_rew = sorted(self.reward_structure.items(), key=operator.itemgetter(1))
            sum_rew = sum(r for key, r in sorted_rew)
            sorted_rew = {key: f'{r:.3f}' for key, r in sorted_rew}
            log.info('Sum rewards: %.3f, reward structure: %r', sum_rew, sorted_rew)

        return shaping_reward

    def reset(self):
        obs = self.env.reset()

        self.prev_vars = dict()
        self.prev_dead = True
        self.reward_structure = dict()
        self.selected_weapon.clear()

        self.orig_env_reward = self.total_shaping_reward = 0.0

        self.print_once = False
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if obs is None:
            return obs, rew, done, info

        self.orig_env_reward += rew

        shaping_rew = self._parse_info(info, done)
        rew += shaping_rew
        self.total_shaping_reward += shaping_rew

        if self.verbose:
            log.info('Original env reward before shaping: %.3f', self.orig_env_reward)
            player_id = 1
            if hasattr(self.env.unwrapped, 'player_id'):
                player_id = self.env.unwrapped.player_id

            log.info(
                'Total shaping reward is %.3f for %d (done %d)',
                self.total_shaping_reward, player_id, done,
            )

        # remember new variable values
        for var_name in self.reward_shaping_scheme['delta'].keys():
            self.prev_vars[var_name] = info.get(var_name, 0.0)

        self.prev_dead = not not info.get('DEAD', 0.0)  # float -> bool

        if done:
            if self.true_reward_func is None:
                true_reward = self.orig_env_reward
            else:
                true_reward = self.true_reward_func(info)

            info['true_reward'] = true_reward

        return obs, rew, done, info

    def close(self):
        self.env.unwrapped.reward_shaping_interface = None
        return self.env.close()
