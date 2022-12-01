# Population-Based Training

Sample Factory contains an implementation of the Population-Based Training algorithm.
See [PBT paper](https://arxiv.org/abs/1711.09846) and original [Sample Factory paper](https://arxiv.org/abs/2006.11751) for more details.

PBT is a hyperparameter optimization algorithm that can be used to train RL agents.
Instead of manually tuning all hyperparameters, you can let an optimization method do it for you. This can include 
not only learning parameters (e.g. learning rate, entropy coefficient), but also environment parameters (e.g. reward function coefficients).

It is common in RL to have a sophisticated (shaped) reward function which guides the exploration process.
As a result such reward function can distract the agent from the actual final goal.

PBT allows you to optimize with respect to some sparse final objective (which we call "true_objective") while still using a shaped reward function.
Theoretically the algorithm should find hyperparameters (including shaping coefficients) that lead to the best final objective.
This can be, for example, directly optimizing for just winning a match in a multiplayer game, which would be very difficult to do with just regular RL because of
the sparsity of such objective.
This type of PBT algorithm is implemented in the [FTW agent](https://www.deepmind.com/blog/capture-the-flag-the-emergence-of-complex-cooperative-agents) by DeepMind.

## Algorithm

PBT works similar to a genetic algorithm. A population of agents is trained simultaneously with roughly the following approach:

* Each agent is assigned a set of hyperparameters (e.g. learning rate, entropy coefficient, reward function coefficients, etc.)
* Each agent is trained for a fixed number of steps (e.g. 5M steps)
* At the end of this meta-training epoch, the performance of all agents is ranked:
    * Agents with top K % of performance are unchanged, we just keep training them
    * Agents with bottom K % of performance are replaced by a copy of a random top-K % agent with mutated hyperparameters.
    * Agents in the middle keep their weights but also get mutated hyperparameters.
* Proceed to the next meta-training epoch.

Current version of PBT is implemented for a single machine. The perfect setup is a multi-GPU server that can train multiple agents at the same time.
For example, we can train a population of 8 agents on a 4-GPU machine, training 2 agents on each GPU.

PBT is perfect for multiplayer game scenarios where training a population of agents against one another
yields much more robust results compared to self-play with a single policy.

## Providing "True Objective" to PBT

In order to optimize for a true objective, you need to return it from the environment.
Just add it to the `info` dictionary returned by the environment at the last step of the episode, e.g.:

```python
def step(self, action):
    info = {}
    ...
    info['true_objective'] = self.compute_true_objective()
    return obs, reward, terminated, truncated, info
```

In the absence of `true_objective` in the `info` dictionary, PBT will use the regular reward as the objective.

## Learning parameters optimized by PBT

See `population_based_training.py`:

```python
HYPERPARAMS_TO_TUNE = {
    "learning_rate",
    "exploration_loss_coeff",
    "value_loss_coeff",
    "max_grad_norm",
    "ppo_clip_ratio",
    "ppo_clip_value",
    # gamma can be added with a CLI parameter (--pbt_optimize_gamma=True)
}
```

During training current learning parameters are saved in `f"policy_{policy_id:02d}_cfg.json"` files in the experiment directory.

## Optimizing environment parameters

Besides learning parameters we can also optimize parameters of the environment with respect to some "true objective".

In order to do that, your environment should implement `RewardShapingInterface` interface in addition to `gym.Env` interface.

```python
class RewardShapingInterface:
    def get_default_reward_shaping(self) -> Optional[Dict[str, Any]]:
        """Should return a dictionary of string:float key-value pairs defining the current reward shaping scheme."""
        raise NotImplementedError

    def set_reward_shaping(self, reward_shaping: Dict[str, Any], agent_idx: int | slice) -> None:
        """
        Sets the new reward shaping scheme.
        :param reward_shaping dictionary of string-float key-value pairs
        :param agent_idx: integer agent index (for multi-agent envs). Can be a slice if we're training in batched mode
        (set a single reward shaping scheme for a range of agents)
        """
        raise NotImplementedError
```

Any parameters in the dictionary returned by `get_default_reward_shaping` will be optimized by PBT.
Note that although the dictionary is called "reward shaping", it can be used to optimize any environment parameters.

It is important that none of these parameters should directly affect the objective calculation, otherwise
all PBT will do is increase the coefficients all the way to infinity.

An example of how this can be used. Suppose your shaped reward function contains a term for picking up a weapon in a game like Quake or VizDoom.
If the true objective is `1.0` for winning the game and `0.0` otherwise then PBT can optimize these weapon preference coefficients to maximize success.
But if true objective is not specified (so just the env reward itself is used as objective), then you can just increase
the coefficients to increase the reward unboundedly.

## Configuring PBT

Please see [Configuration parameter reference](../02-configuration/cfg-params.md). Parameters with `--pbt_` prefix are related to PBT.
Use `--with_pbt=True` to enable PBT. It is important also to set `--num_policies` to the number of agents in the population.

### Command-line examples

Training on DMLab-30 with a 4-agent population on a 4-GPU machine:

```bash
python -m sf_examples.dmlab.train_dmlab --env=dmlab_30 --train_for_env_steps=10000000000 --algo=APPO --gamma=0.99 --use_rnn=True --num_workers=90 --num_envs_per_worker=12 --num_epochs=1 --rollout=32 --recurrence=32 --batch_size=2048 --benchmark=False --max_grad_norm=0.0 --dmlab_renderer=software --decorrelate_experience_max_seconds=120 --reset_timeout_seconds=300 --encoder_conv_architecture=resnet_impala --encoder_conv_mlp_layers=512 --nonlinearity=relu --rnn_type=lstm --dmlab_extended_action_set=True --num_policies=4 --pbt_replace_reward_gap=0.05 --pbt_replace_reward_gap_absolute=5.0 --pbt_period_env_steps=10000000 --pbt_start_mutation=100000000 --with_pbt=True --experiment=dmlab_30_resnet_4pbt_w90_v12 --dmlab_one_task_per_worker=True --set_workers_cpu_affinity=True --max_policy_lag=35 --pbt_target_objective=dmlab_target_objective --dmlab30_dataset=~/datasets/brady_konkle_oliva2008 --dmlab_use_level_cache=True --dmlab_level_cache_path=/home/user/dmlab_cache
```

PBT for VizDoom (8 agents, 4-GPU machine):

```bash
python -m sf_examples.vizdoom.train_vizdoom --env=doom_deathmatch_bots --train_for_seconds=3600000 --algo=APPO --use_rnn=True --gamma=0.995 --env_frameskip=2 --num_workers=80 --num_envs_per_worker=24 --num_policies=8 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=False --with_pbt=True --pbt_period_env_steps=5000000 --experiment=doom_deathmatch_bots
```
