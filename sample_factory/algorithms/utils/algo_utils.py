import numpy as np


EPS = 1e-8


class ExperimentStatus:
    SUCCESS, FAILURE, INTERRUPTED = range(3)


# custom experiments can define functions to this list to do something extra with the raw episode summaries
# coming from the environments
EXTRA_EPISODIC_STATS_PROCESSING = []

# custom experiments or environments can append functions to this list to postprocess some summaries, or aggregate
# summaries, or do whatever else the user wants
EXTRA_PER_POLICY_SUMMARIES = []


class RunningMeanStd(object):
    """
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    Courtesy of OpenAI Baselines.

    """

    def __init__(self, max_past_samples=None, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon
        self.max_past_samples = max_past_samples

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count, self.max_past_samples,
        )


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count, max_past_samples):
    """Courtesy of OpenAI Baselines."""
    if max_past_samples is not None:
        # pretend we never have more than n past samples, this will guarantee a constant convergence rate
        count = min(count, max_past_samples)

    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    m_2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = m_2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def main_observation(data):
    obs = maybe_extract_key(data, 'obs')
    if obs is None:
        return data
    else:
        return obs


def goal_observation(data):
    return maybe_extract_key(data, 'goal')


def maybe_extract_key(data, key):
    if isinstance(data, (list, tuple)) and isinstance(data[0], dict):
        if key in data[0]:
            return extract_key(data, key)
        else:
            return None
    elif isinstance(data, dict):
        return data.get(key, None)
    else:
        return None


def extract_keys(list_of_dicts, *keys):
    """Turn a lists of dicts into a tuple of lists, with one entry for every given key."""
    res = []
    for k in keys:
        res.append([d[k] for d in list_of_dicts])
    return tuple(res)


def extract_key(list_of_dicts, key):
    return extract_keys(list_of_dicts, key)[0]


def calculate_discounted_sum(x, dones, discount, x_last=None):
    """
    Computing cumulative sum (of something) for the trajectory, taking episode termination into consideration.
    :param x: ndarray of shape [num_steps, num_envs]
    :param dones: ndarray of shape [num_steps, num_envs]
    :param discount: float in range [0,1]
    :param x_last: iterable of shape [num_envs], value at the end of trajectory. None interpreted as zero(s).
    """
    x_last = np.zeros_like(x[0]) if x_last is None else np.array(x_last, dtype=np.float32)
    cumulative = x_last

    discounted_sum = np.zeros_like(x)
    for i in reversed(range(len(x))):
        cumulative = x[i] + discount * cumulative * (1 - dones[i])
        discounted_sum[i] = cumulative
    return discounted_sum


def calculate_gae(rewards, dones, values, gamma, gae_lambda):
    """
    Computing discounted cumulative sum, taking episode terminations into consideration. Follows the
    Generalized Advantage Estimation algorithm.
    See unit tests for details.

    :param rewards: actual environment rewards
    :param dones: True if absorbing state is reached
    :param values: estimated values
    :param gamma: discount factor [0,1]
    :param gae_lambda: lambda-factor for GAE (discounting for longer-horizon advantage estimations), [0,1]
    :return: advantages and discounted returns
    """
    assert len(rewards) == len(dones)
    assert len(rewards) + 1 == len(values)

    # section 3 in GAE paper: calculating advantages
    deltas = rewards + (1 - dones) * (gamma * values[1:]) - values[:-1]
    advantages = calculate_discounted_sum(deltas, dones, gamma * gae_lambda)

    # targets for value function - this is just a simple discounted sum of rewards
    discounted_returns = calculate_discounted_sum(rewards, dones, gamma, values[-1])

    return advantages.astype(np.float32), discounted_returns.astype(np.float32)


def num_env_steps(infos):
    """Calculate number of environment frames in a batch of experience."""

    total_num_frames = 0
    for info in infos:
        total_num_frames += info.get('num_frames', 1)
    return total_num_frames


def list_to_string(x, limit=6):
    if len(x) <= limit:
        return str(x)
    else:
        res = str(x[:3]).replace(']', ',')
        res += ' ... ,'
        res += str(x[-2:]).replace('[', ' ')
        return res


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


def choice_weighted(arr, logits):
    assert len(arr) == len(logits)

    probs = softmax(logits)
    return np.random.choice(arr, p=probs)
