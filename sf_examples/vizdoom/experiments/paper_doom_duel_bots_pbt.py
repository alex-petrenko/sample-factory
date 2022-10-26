from sample_factory.launcher.run_description import Experiment, RunDescription

_experiment = Experiment(
    "bots_ssl2_fs2",
    "python -m sf_examples.vizdoom.train_vizdoom --env=doom_duel_bots --train_for_seconds=360000 --algo=APPO --gamma=0.995 --env_frameskip=2 --use_rnn=True --reward_scale=0.5 --num_workers=72 --num_envs_per_worker=32 --num_policies=8 --batch_size=2048 --benchmark=False --res_w=128 --res_h=72 --wide_aspect_ratio=False --pbt_replace_reward_gap=0.2 --pbt_replace_reward_gap_absolute=3.0 --pbt_period_env_steps=5000000 --save_milestones_sec=1800 --with_pbt=True",
)

RUN_DESCRIPTION = RunDescription("paper_doom_duel_bots_fs2", experiments=[_experiment])
