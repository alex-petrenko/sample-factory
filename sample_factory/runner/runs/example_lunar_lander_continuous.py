from sample_factory.runner.run_description import RunDescription, Experiment, ParamGrid

_params_earlystop = ParamGrid([
    ('seed', [0000, 1111, 2222, 3333, 4444]),
])

_experiment_earlystop = Experiment(
    'lunar_lander_cont',
    'python -m sample_factory_examples.train_gym_env --train_for_env_steps=500000000 --algo=APPO --num_workers=20 --num_envs_per_worker=6 --seed 0 --gae_lambda 0.99 --experiment=lunar_lander_2 --env=gym_LunarLanderContinuous-v2 --exploration_loss_coeff=0.0 --max_grad_norm=0.0 --encoder_type=mlp --encoder_subtype=mlp_mujoco --encoder_extra_fc_layers=0 --hidden_size=128 --policy_initialization=xavier_uniform --actor_critic_share_weights=False --adaptive_stddev=False --recurrence=1 --use_rnn=False --batch_size=256 --ppo_epochs=4 --with_vtrace=False --reward_scale=0.05 --max_policy_lag=100000 --save_every_sec=15 --experiment_summaries_interval=10',
    _params_earlystop.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('lunar_lander_cont_v100', experiments=[_experiment_earlystop])
