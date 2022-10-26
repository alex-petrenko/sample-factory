from sample_factory.launcher.run_description import Experiment, RunDescription

_experiment = Experiment(
    "bots_ssl2_fs2",
    "python -m sf_examples.vizdoom.train_vizdoom --env=doom_duel --train_for_seconds=360000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --reward_scale=0.5 --num_workers=72 --num_envs_per_worker=16 --num_policies=8 --batch_size=2048 --res_w=128 --res_h=72 --wide_aspect_ratio=False --benchmark=False --pbt_replace_reward_gap=0.5 --pbt_replace_reward_gap_absolute=0.35 --pbt_period_env_steps=5000000 --with_pbt=True --pbt_start_mutation=100000000",
)

RUN_DESCRIPTION = RunDescription("paper_doom_duel_fs2", experiments=[_experiment])
