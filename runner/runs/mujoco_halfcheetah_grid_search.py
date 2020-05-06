from runner.run_description import RunDescription, Experiment, ParamGrid

_params = ParamGrid([
    ('seed', [0, 1, 2, 3]),
    ('with_vtrace', ['False']),
    ('learning_rate', [0.0003]),
    ('max_grad_norm', [0.5]),
    ('use_rnn', ['False']),
    ('recurrence', [1]),
    ('num_minibatches_to_accumulate', [0]),
    ('device', ['cpu']),
    ('actor_critic_share_weights', ['False']),
    ('max_policy_lag', [1000000]),
    ('adaptive_stddev', ['False']),

    ('ppo_epochs', [10]),
    ('ppo_clip_ratio', [0.2, 4]),
    ('batch_size', [4096]),
    ('num_batches_per_iteration', [16]),
    ('rollout', [64]),
])

_experiment = Experiment(
    'mujoco_halfcheetah_gridsearch',
    'run_algorithm --env=mujoco_halfcheetah --train_for_env_steps=4000000 --algo=APPO --num_workers=10 --num_envs_per_worker=4 --benchmark=False --with_pbt=False',
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription('mujoco_halfcheetah_gridsearch_v89_seeds_v1', experiments=[_experiment])

# python -m runner.run --run=mujoco_halfcheetah_grid_search --runner=processes --max_parallel=8  --pause_between=1 --experiments_per_gpu=10000 --num_gpus=1

# Avg episode reward: [(0, '-298.590')
# lse_lea_0.0003_bat_4096_num_16_max_0.5_use_False_rec_1_num__0_dev_cpu_act_False_max__1000000_ppo_8_rol_128_ada_False/.summary/0

# [36m[2020-04-29 22:12:31,035][16355] Fps is (10 sec: 0.0, 60 sec: 2166.8, 300 sec: 1322.3). Total num frames: 983040. Throughput: 0: 2348.7. Samples: 1046959. Policy #0 lag: (min: 156.0, avg: 156.6, max: 412.0)[0m
# [36m[2020-04-29 22:12:31,036][16355] Avg episode reward: [(0, '124.200')]
# mujoco_halfcheetah_gridsearch_v89_seeds_v2/mujoco_halfcheetah_gridsearch/
# 08_mujoco_halfcheetah_gridsearch_wit_False_lea_0.0003_bat_4096_num_16_max_0.5_use_False_rec_1_num__0_dev_cpu_act_False_max__1000000_ada_False_
# ppo_16_rol_64_ppo__0.4/

# [36m[2020-04-29 22:27:53,926][18362] Fps is (10 sec: 0.0, 60 sec: 2165.8, 300 sec: 2423.0). Total num frames: 983040. Throughput: 0: 2768.9. Samples: 1039030. Policy #0 lag: (min: 240.0, avg: 240.9, max: 496.0)[0m
# [36m[2020-04-29 22:27:53,927][18362] Avg episode reward: [(0, '450.674')][0m
#mujoco_halfcheetah_gridsearch_v89_seeds_v3_ppo_clip_ratio/mujoco_halfcheetah_gridsearch/02_mujoco_halfcheetah_gridsearch_
# wit_False_lea_0.0003_bat_4096_num_16_max_0.5_use_False_rec_1_num__0_dev_cpu_act_False_max__1000000_ada_False_
# ppo_16_rol_64_ppo__3.2/.summary/0

# [36m[2020-04-29 22:42:36,124][19233] Fps is (10 sec: 3254.8, 60 sec: 2714.5, 300 sec: 2706.0). Total num frames: 999424. Throughput: 0: 2444.4. Samples: 999711. Policy #0 lag: (min: 240.0, avg: 243.5, max: 496.0)[0m
# [36m[2020-04-29 22:42:36,124][19233] Avg episode reward: [(0, '822.773')][0m
# mujoco_halfcheetah_gridsearch_v89_seeds_v4-batch_size/mujoco_halfcheetah_gridsearch/01_mujoco_halfcheetah_gridsearch_wit_False_lea_0.0003_num_16_max_0.5_use_False_rec_1_num__0_dev_cpu_act_False_max__1000000_ada_False_
# ppo_16_rol_64_ppo__3.2_bat_1024/.summary/0

# [36m[2020-04-29 23:09:11,506][20023] Fps is (10 sec: 1626.7, 60 sec: 1629.8, 300 sec: 935.4). Total num frames: 999424. Throughput: 0: 2424.5. Samples: 1015792. Policy #0 lag: (min: 480.0, avg: 487.0, max: 992.0)[0m
# [36m[2020-04-29 23:09:11,506][20023] Avg episode reward: [(0, '936.148')][0m
# mujoco_halfcheetah_gridsearch_v89_seeds_v5-total_size/mujoco_halfcheetah_gridsearch/03_mujoco_halfcheetah_gridsearch_wit_False_lea_0.0003_max_0.5_use_False_rec_1_num_0_dev_cpu_act_False_max__1000000_ada_False_
# ppo_16_rol_64_ppo__3.2_bat_512_num__32/.summary/0

# [36m[2020-04-30 01:02:46,152][28036] Fps is (10 sec: 2858.7, 60 sec: 2941.3, 300 sec: 2951.0). Total num frames: 989184. Throughput: 0: 3161.7. Samples: 987516. Policy #0 lag: (min: 36.0, avg: 42.6, max: 76.0)[0m
# [36m[2020-04-30 01:02:46,153][28036] Avg episode reward: [(0, '1302.775')][0m
# mujoco_halfcheetah_gridsearch_v89_seeds_v8-total_size/mujoco_halfcheetah_gridsearch/00_mujoco_halfcheetah_gridsearch_wit_False_lea_0.0003_max_0.5_use_False_rec_1_num_0_dev_cpu_act_False_max__1000000_ada_False_
# ppo_10_rol_64_ppo__4_bat_512_num__4/.summary/0
