# Full Parameter Reference

The command line arguments / config parameters for training using Sample Factory can be found by running your training script with the `--help` flag.
The list of config parameters below was obtained from running `python -m sf_examples.train_gym_env --env=CartPole-v1 --help`. These params can be used in any environment.
Other environments may have other custom params than can also be viewed with `--help` flag when running the environment-specific training script.

```
usage: train_gym_env.py [-h] [--algo ALGO] --env ENV [--experiment EXPERIMENT]
                        [--train_dir TRAIN_DIR]
                        [--restart_behavior {resume,restart,overwrite}]
                        [--device {gpu,cpu}] [--seed SEED]
                        [--num_policies NUM_POLICIES] [--async_rl ASYNC_RL]
                        [--serial_mode SERIAL_MODE]
                        [--batched_sampling BATCHED_SAMPLING]
                        [--num_batches_to_accumulate NUM_BATCHES_TO_ACCUMULATE]
                        [--worker_num_splits WORKER_NUM_SPLITS]
                        [--policy_workers_per_policy POLICY_WORKERS_PER_POLICY]
                        [--max_policy_lag MAX_POLICY_LAG]
                        [--num_workers NUM_WORKERS]
                        [--num_envs_per_worker NUM_ENVS_PER_WORKER]
                        [--batch_size BATCH_SIZE]
                        [--num_batches_per_epoch NUM_BATCHES_PER_EPOCH]
                        [--num_epochs NUM_EPOCHS] [--rollout ROLLOUT]
                        [--recurrence RECURRENCE]
                        [--shuffle_minibatches SHUFFLE_MINIBATCHES]
                        [--gamma GAMMA] [--reward_scale REWARD_SCALE]
                        [--reward_clip REWARD_CLIP]
                        [--value_bootstrap VALUE_BOOTSTRAP]
                        [--normalize_returns NORMALIZE_RETURNS]
                        [--exploration_loss_coeff EXPLORATION_LOSS_COEFF]
                        [--value_loss_coeff VALUE_LOSS_COEFF]
                        [--kl_loss_coeff KL_LOSS_COEFF]
                        [--exploration_loss {entropy,symmetric_kl}]
                        [--gae_lambda GAE_LAMBDA]
                        [--ppo_clip_ratio PPO_CLIP_RATIO]
                        [--ppo_clip_value PPO_CLIP_VALUE]
                        [--with_vtrace WITH_VTRACE] [--vtrace_rho VTRACE_RHO]
                        [--vtrace_c VTRACE_C] [--optimizer {adam,lamb}]
                        [--adam_eps ADAM_EPS] [--adam_beta1 ADAM_BETA1]
                        [--adam_beta2 ADAM_BETA2]
                        [--max_grad_norm MAX_GRAD_NORM]
                        [--learning_rate LEARNING_RATE]
                        [--lr_schedule {constant,kl_adaptive_minibatch,kl_adaptive_epoch}]
                        [--lr_schedule_kl_threshold LR_SCHEDULE_KL_THRESHOLD]
                        [--lr_adaptive_min LR_ADAPTIVE_MIN]
                        [--lr_adaptive_max LR_ADAPTIVE_MAX]
                        [--obs_subtract_mean OBS_SUBTRACT_MEAN]
                        [--obs_scale OBS_SCALE]
                        [--normalize_input NORMALIZE_INPUT]
                        [--normalize_input_keys [NORMALIZE_INPUT_KEYS [NORMALIZE_INPUT_KEYS ...]]]
                        [--decorrelate_experience_max_seconds DECORRELATE_EXPERIENCE_MAX_SECONDS]
                        [--decorrelate_envs_on_one_worker DECORRELATE_ENVS_ON_ONE_WORKER]
                        [--actor_worker_gpus [ACTOR_WORKER_GPUS [ACTOR_WORKER_GPUS ...]]]
                        [--set_workers_cpu_affinity SET_WORKERS_CPU_AFFINITY]
                        [--force_envs_single_thread FORCE_ENVS_SINGLE_THREAD]
                        [--default_niceness DEFAULT_NICENESS]
                        [--log_to_file LOG_TO_FILE]
                        [--experiment_summaries_interval EXPERIMENT_SUMMARIES_INTERVAL]
                        [--flush_summaries_interval FLUSH_SUMMARIES_INTERVAL]
                        [--stats_avg STATS_AVG]
                        [--summaries_use_frameskip SUMMARIES_USE_FRAMESKIP]
                        [--heartbeat_interval HEARTBEAT_INTERVAL]
                        [--heartbeat_reporting_interval HEARTBEAT_REPORTING_INTERVAL]
                        [--train_for_env_steps TRAIN_FOR_ENV_STEPS]
                        [--train_for_seconds TRAIN_FOR_SECONDS]
                        [--save_every_sec SAVE_EVERY_SEC]
                        [--keep_checkpoints KEEP_CHECKPOINTS]
                        [--load_checkpoint_kind {latest,best}]
                        [--save_milestones_sec SAVE_MILESTONES_SEC]
                        [--save_best_every_sec SAVE_BEST_EVERY_SEC]
                        [--save_best_metric SAVE_BEST_METRIC]
                        [--save_best_after SAVE_BEST_AFTER]
                        [--benchmark BENCHMARK]
                        [--encoder_mlp_layers [ENCODER_MLP_LAYERS [ENCODER_MLP_LAYERS ...]]]
                        [--encoder_conv_architecture {convnet_simple,convnet_impala,convnet_atari,resnet_impala}]
                        [--encoder_conv_mlp_layers [ENCODER_CONV_MLP_LAYERS [ENCODER_CONV_MLP_LAYERS ...]]]
                        [--use_rnn USE_RNN] [--rnn_size RNN_SIZE]
                        [--rnn_type {gru,lstm}]
                        [--rnn_num_layers RNN_NUM_LAYERS]
                        [--decoder_mlp_layers [DECODER_MLP_LAYERS [DECODER_MLP_LAYERS ...]]]
                        [--nonlinearity {elu,relu,tanh}]
                        [--policy_initialization {orthogonal,xavier_uniform,torch_default}]
                        [--policy_init_gain POLICY_INIT_GAIN]
                        [--actor_critic_share_weights ACTOR_CRITIC_SHARE_WEIGHTS]
                        [--adaptive_stddev ADAPTIVE_STDDEV]
                        [--continuous_tanh_scale CONTINUOUS_TANH_SCALE]
                        [--initial_stddev INITIAL_STDDEV]
                        [--use_env_info_cache USE_ENV_INFO_CACHE]
                        [--env_gpu_actions ENV_GPU_ACTIONS]
                        [--env_gpu_observations ENV_GPU_OBSERVATIONS]
                        [--env_frameskip ENV_FRAMESKIP]
                        [--env_framestack ENV_FRAMESTACK]
                        [--pixel_format PIXEL_FORMAT]
                        [--use_record_episode_statistics USE_RECORD_EPISODE_STATISTICS]
                        [--with_wandb WITH_WANDB] [--wandb_user WANDB_USER]
                        [--wandb_project WANDB_PROJECT]
                        [--wandb_group WANDB_GROUP]
                        [--wandb_job_type WANDB_JOB_TYPE]
                        [--wandb_tags [WANDB_TAGS [WANDB_TAGS ...]]]
                        [--with_pbt WITH_PBT]
                        [--pbt_mix_policies_in_one_env PBT_MIX_POLICIES_IN_ONE_ENV]
                        [--pbt_period_env_steps PBT_PERIOD_ENV_STEPS]
                        [--pbt_start_mutation PBT_START_MUTATION]
                        [--pbt_replace_fraction PBT_REPLACE_FRACTION]
                        [--pbt_mutation_rate PBT_MUTATION_RATE]
                        [--pbt_replace_reward_gap PBT_REPLACE_REWARD_GAP]
                        [--pbt_replace_reward_gap_absolute PBT_REPLACE_REWARD_GAP_ABSOLUTE]
                        [--pbt_optimize_gamma PBT_OPTIMIZE_GAMMA]
                        [--pbt_target_objective PBT_TARGET_OBJECTIVE]
                        [--pbt_perturb_min PBT_PERTURB_MIN]
                        [--pbt_perturb_max PBT_PERTURB_MAX]

optional arguments:
  -h, --help            Print the help message (default: False)
  --algo ALGO           Algorithm to use (default: APPO)
  --env ENV             Name of the environment to use (default: None)
  --experiment EXPERIMENT
                        Unique experiment name. This will also be the name for
                        the experiment folder in the train dir.If the
                        experiment folder with this name aleady exists the
                        experiment will be RESUMED!Any parameters passed from
                        command line that do not match the parameters stored
                        in the experiment config.json file will be overridden.
                        (default: default_experiment)
  --train_dir TRAIN_DIR
                        Root for all experiments (default:
                        /home/alex/all/projects/sf2/train_dir)
  --restart_behavior {resume,restart,overwrite}
                        How to handle the experiment if the directory with the
                        same name already exists. "resume" (default) will
                        resume the experiment, "restart" will preserve the
                        existing experiment folder under a different name
                        (with "old" suffix) and will start training from
                        scratch, "overwrite" will delete the existing
                        experiment folder and start from scratch. This
                        parameter does not have any effect if the experiment
                        directory does not exist. (default: resume)
  --device {gpu,cpu}    CPU training is only recommended for smaller e.g. MLP
                        policies (default: gpu)
  --seed SEED           Set a fixed seed value (default: None)
  --num_policies NUM_POLICIES
                        Number of policies to train jointly, i.e. for multi-
                        agent environments (default: 1)
  --async_rl ASYNC_RL   Collect experience asynchronously while learning on
                        the previous batch. This is significantly different
                        from standard synchronous actor-critic (or PPO)
                        because not all of the experience will be collected by
                        the latest policy thus increasing policy lag. Negative
                        effects of using async_rl can range from negligible
                        (just grants you throughput boost) to quite serious
                        where you can consider switching it off. It all
                        depends how sensitive your experiment is to policy
                        lag. Envs with complex action spaces and RNN policies
                        tend to be particularly sensitive. (default: True)
  --serial_mode SERIAL_MODE
                        Enable serial mode: run everything completely
                        synchronously in the same process (default: False)
  --batched_sampling BATCHED_SAMPLING
                        Batched sampling allows the data to be processed in
                        big batches on the rollout worker.This is especially
                        important for GPU-accelerated vectorized environments
                        such as Megaverse or IsaacGym. As a downside, in
                        batched mode we do not support (for now) some of the
                        features, such as population-based self-play or
                        inactive agents, plus each batched sampler (rollout
                        worker) process only collects data for a single
                        policy. Another issue between batched/non-batched
                        sampling is handling of infos. In batched mode we
                        assume that infos is a single dictionary of
                        lists/tensors containing info for each environment in
                        a vector. If you need some complex info dictionary
                        handling and your environment might return dicts with
                        different keys, on different rollout steps, you
                        probably need non-batched mode. (default: False)
  --num_batches_to_accumulate NUM_BATCHES_TO_ACCUMULATE
                        This parameter governs the maximum number of training
                        batches the learner can accumulate before further
                        experience collection is stopped. The default value
                        will set this to 2, so if the experience collection is
                        faster than the training, the learner will accumulate
                        enough minibatches for 2 iterations of training but no
                        more. This is a good balance between policy-lag and
                        throughput. When the limit is reached, the learner
                        will notify the actor workers that they ought to stop
                        the experience collection until accumulated
                        minibatches are processed. Set this parameter to 1 to
                        further reduce policy-lag. If the experience
                        collection is very non-uniform, increasing this
                        parameter can increase overall throughput, at the cost
                        of increased policy-lag. (default: 2)
  --worker_num_splits WORKER_NUM_SPLITS
                        Typically we split a vector of envs into two parts for
                        "double buffered" experience collection Set this to 1
                        to disable double buffering. Set this to 3 for triple
                        buffering! (default: 2)
  --policy_workers_per_policy POLICY_WORKERS_PER_POLICY
                        Number of policy workers that compute forward pass
                        (per policy) (default: 1)
  --max_policy_lag MAX_POLICY_LAG
                        Max policy lag in policy versions. Discard all
                        experience that is older than this. (default: 1000)
  --num_workers NUM_WORKERS
                        Number of parallel environment workers. Should be less
                        than num_envs and should divide num_envs.Use this in
                        async mode. (default: 12)
  --num_envs_per_worker NUM_ENVS_PER_WORKER
                        Number of envs on a single CPU actor, in high-
                        throughput configurations this should be in 10-30
                        range for Atari/VizDoomMust be even for double-
                        buffered sampling! (default: 2)
  --batch_size BATCH_SIZE
                        Minibatch size for SGD (default: 1024)
  --num_batches_per_epoch NUM_BATCHES_PER_EPOCH
                        This determines the training dataset size for each
                        iteration of training. We collect this many
                        minibatches before performing any SGD. Example: if
                        batch_size=128 and num_batches_per_epoch=2, then
                        learner will process 2*128=256 environment transitions
                        in one training iteration. (default: 1)
  --num_epochs NUM_EPOCHS
                        Number of training epochs on a dataset of collected
                        experiences of size batch_size x num_batches_per_epoch
                        (default: 1)
  --rollout ROLLOUT     Length of the rollout from each environment in
                        timesteps.Once we collect this many timesteps on actor
                        worker, we send this trajectory to the learner.The
                        length of the rollout will determine how many
                        timesteps are used to calculate bootstrappedMonte-
                        Carlo estimates of discounted rewards, advantages,
                        GAE, or V-trace targets. Shorter rolloutsreduce
                        variance, but the estimates are less precise (bias vs
                        variance tradeoff).For RNN policies, this should be a
                        multiple of --recurrence, so every rollout will be
                        splitinto (n = rollout / recurrence) segments for
                        backpropagation. V-trace algorithm currently requires
                        thatrollout == recurrence, which what you want most of
                        the time anyway.Rollout length is independent from the
                        episode length. Episode length can be both shorter or
                        longer thanrollout, although for PBT training it is
                        currently recommended that rollout << episode_len(see
                        function finalize_trajectory in actor_worker.py)
                        (default: 32)
  --recurrence RECURRENCE
                        Trajectory length for backpropagation through time.
                        Default value (-1) sets recurrence to rollout length
                        for RNNs and to 1 (no recurrence) for feed-forward
                        nets. If you train with V-trace recurrence should be
                        equal to rollout length. (default: -1)
  --shuffle_minibatches SHUFFLE_MINIBATCHES
                        Whether to randomize and shuffle minibatches between
                        iterations (this is a slow operation when batches are
                        large, disabling this increases learner throughput
                        when training with multiple epochs/minibatches per
                        epoch) (default: False)
  --gamma GAMMA         Discount factor (default: 0.99)
  --reward_scale REWARD_SCALE
                        Multiply all rewards by this factor before feeding
                        into RL algorithm.Sometimes the overall scale of
                        rewards is too high which makes value estimation a
                        harder regression task.Loss values become too high
                        which requires a smaller learning rate, etc. (default:
                        1.0)
  --reward_clip REWARD_CLIP
                        Clip rewards between [-c, c]. Default [-1000, 1000]
                        should mean no clipping for most envs (unless rewards
                        are very large/small) (default: 1000.0)
  --value_bootstrap VALUE_BOOTSTRAP
                        Bootstrap returns from value estimates if episode is
                        terminated by timeout. More info here:
                        https://github.com/Denys88/rl_games/issues/128
                        (default: False)
  --normalize_returns NORMALIZE_RETURNS
                        Whether to use running mean and standard deviation to
                        normalize discounted returns (default: True)
  --exploration_loss_coeff EXPLORATION_LOSS_COEFF
                        Coefficient for the exploration component of the loss
                        function. (default: 0.003)
  --value_loss_coeff VALUE_LOSS_COEFF
                        Coefficient for the critic loss (default: 0.5)
  --kl_loss_coeff KL_LOSS_COEFF
                        Coefficient for fixed KL loss (as used by Schulman et
                        al. in https://arxiv.org/pdf/1707.06347.pdf). Highly
                        recommended for environments with continuous action
                        spaces. (default: 0.0)
  --exploration_loss {entropy,symmetric_kl}
                        Usually the exploration loss is based on maximizing
                        the entropy of the probability distribution. Note that
                        mathematically maximizing entropy of the categorical
                        probability distribution is exactly the same as
                        minimizing the (regular) KL-divergence between this
                        distribution and a uniform prior. The downside of
                        using the entropy term (or regular asymmetric KL-
                        divergence) is the fact that penalty does not increase
                        as probabilities of some actions approach zero. I.e.
                        numerically, there is almost no difference between an
                        action distribution with a probability epsilon > 0 for
                        some action and an action distribution with a
                        probability = zero for this action. For many tasks the
                        first (epsilon) distribution is preferrable because we
                        keep some (albeit small) amount of exploration, while
                        the second distribution will never explore this action
                        ever again.Unlike the entropy term, symmetric KL
                        divergence between the action distribution and a
                        uniform prior approaches infinity when entropy of the
                        distribution approaches zero, so it can prevent the
                        pathological situations where the agent stops
                        exploring. Empirically, symmetric KL-divergence
                        yielded slightly better results on some problems.
                        (default: entropy)
  --gae_lambda GAE_LAMBDA
                        Generalized Advantage Estimation discounting (only
                        used when V-trace is False) (default: 0.95)
  --ppo_clip_ratio PPO_CLIP_RATIO
                        We use unbiased clip(x, 1+e, 1/(1+e)) instead of
                        clip(x, 1+e, 1-e) in the paper (default: 0.1)
  --ppo_clip_value PPO_CLIP_VALUE
                        Maximum absolute change in value estimate until it is
                        clipped. Sensitive to value magnitude (default: 1.0)
  --with_vtrace WITH_VTRACE
                        Enables V-trace off-policy correction. If this is
                        True, then GAE is not used (default: False)
  --vtrace_rho VTRACE_RHO
                        rho_hat clipping parameter of the V-trace algorithm
                        (importance sampling truncation) (default: 1.0)
  --vtrace_c VTRACE_C   c_hat clipping parameter of the V-trace algorithm. Low
                        values for c_hat can reduce variance of the advantage
                        estimates (similar to GAE lambda < 1) (default: 1.0)
  --optimizer {adam,lamb}
                        Type of optimizer to use (default: adam)
  --adam_eps ADAM_EPS   Adam epsilon parameter (1e-8 to 1e-5 seem to reliably
                        work okay, 1e-3 and up does not work) (default: 1e-06)
  --adam_beta1 ADAM_BETA1
                        Adam momentum decay coefficient (default: 0.9)
  --adam_beta2 ADAM_BETA2
                        Adam second momentum decay coefficient (default:
                        0.999)
  --max_grad_norm MAX_GRAD_NORM
                        Max L2 norm of the gradient vector, set to 0 to
                        disable gradient clipping (default: 4.0)
  --learning_rate LEARNING_RATE
                        LR (default: 0.0001)
  --lr_schedule {constant,kl_adaptive_minibatch,kl_adaptive_epoch}
                        Learning rate schedule to use. Constant keeps constant
                        learning rate throughout training.kl_adaptive*
                        schedulers look at --lr_schedule_kl_threshold and if
                        KL-divergence with behavior policyafter the last
                        minibatch/epoch significantly deviates from this
                        threshold, lr is apropriatelyincreased or decreased
                        (default: constant)
  --lr_schedule_kl_threshold LR_SCHEDULE_KL_THRESHOLD
                        Used with kl_adaptive_* schedulers (default: 0.008)
  --lr_adaptive_min LR_ADAPTIVE_MIN
                        Minimum learning rate (default: 1e-06)
  --lr_adaptive_max LR_ADAPTIVE_MAX
                        Maximum learning rate. This is the best value tuned
                        for IsaacGymEnvs environments such as Ant/Humanoid,
                        but it can be too high for some other envs. Set this
                        to 1e-3 if you see instabilities with adaptive LR,
                        especially if reported LR on Tensorboard reaches this
                        max value before the instability happens. (default:
                        0.01)
  --obs_subtract_mean OBS_SUBTRACT_MEAN
                        Observation preprocessing, mean value to subtract from
                        observation (e.g. 128.0 for 8-bit RGB) (default: 0.0)
  --obs_scale OBS_SCALE
                        Observation preprocessing, divide observation tensors
                        by this scalar (e.g. 128.0 for 8-bit RGB) (default:
                        1.0)
  --normalize_input NORMALIZE_INPUT
                        Whether to use running mean and standard deviation to
                        normalize observations (default: True)
  --normalize_input_keys [NORMALIZE_INPUT_KEYS [NORMALIZE_INPUT_KEYS ...]]
                        Which observation keys to use for normalization. If
                        None, all observation keys are used (be careful with
                        this!) (default: None)
  --decorrelate_experience_max_seconds DECORRELATE_EXPERIENCE_MAX_SECONDS
                        Decorrelating experience serves two benefits. First:
                        this is better for learning because samples from
                        workers come from random moments in the episode,
                        becoming more "i.i.d".Second, and more important one:
                        this is good for environments with highly non-uniform
                        one-step times, including long and expensive episode
                        resets. If experience is not decorrelatedthen training
                        batches will come in bursts e.g. after a bunch of
                        environments finished resets and many iterations on
                        the learner might be required,which will increase the
                        policy-lag of the new experience collected. The
                        performance of the Sample Factory is best when
                        experience is generated as more-or-lessuniform stream.
                        Try increasing this to 100-200 seconds to smoothen the
                        experience distribution in time right from the
                        beginning (it will eventually spread out and settle
                        anyways) (default: 0)
  --decorrelate_envs_on_one_worker DECORRELATE_ENVS_ON_ONE_WORKER
                        In addition to temporal decorrelation of worker
                        processes, also decorrelate envs within one worker
                        process. For environments with a fixed episode length
                        it can prevent the reset from happening in the same
                        rollout for all envs simultaneously, which makes
                        experience collection more uniform. (default: True)
  --actor_worker_gpus [ACTOR_WORKER_GPUS [ACTOR_WORKER_GPUS ...]]
                        By default, actor workers only use CPUs. Changes this
                        if e.g. you need GPU-based rendering on the actors
                        (default: [])
  --set_workers_cpu_affinity SET_WORKERS_CPU_AFFINITY
                        Whether to assign workers to specific CPU cores or
                        not. The logic is beneficial for most workloads
                        because prevents a lot of context switching.However
                        for some environments it can be better to disable it,
                        to allow one worker to use all cores some of the time.
                        This can be the case for some DMLab environments with
                        very expensive episode resetthat can use parallel CPU
                        cores for level generation. (default: True)
  --force_envs_single_thread FORCE_ENVS_SINGLE_THREAD
                        Some environments may themselves use parallel
                        libraries such as OpenMP or MKL. Since we parallelize
                        environments on the level of workers, there is no need
                        to keep this parallel semantic.This flag uses
                        threadpoolctl to force libraries such as OpenMP and
                        MKL to use only a single thread within the
                        environment.Enabling this is recommended unless you
                        are running fewer workers than CPU cores.
                        threadpoolctl has caused a bunch of crashes in the
                        past, so this feature is disabled by default at this
                        moment. (default: False)
  --default_niceness DEFAULT_NICENESS
                        Niceness of the highest priority process (the
                        learner). Values below zero require elevated
                        privileges. (default: 0)
  --log_to_file LOG_TO_FILE
                        Whether to log to a file (sf_log.txt in the experiment
                        folder) or not. If False, logs to stdout only. It can
                        make sense to disable this in a slow server filesystem
                        environment like NFS. (default: True)
  --experiment_summaries_interval EXPERIMENT_SUMMARIES_INTERVAL
                        How often in seconds we write avg. statistics about
                        the experiment (reward, episode length, extra
                        stats...) (default: 10)
  --flush_summaries_interval FLUSH_SUMMARIES_INTERVAL
                        How often do we flush tensorboard summaries (set to
                        higher value for slow NFS-based server filesystems)
                        (default: 30)
  --stats_avg STATS_AVG
                        How many episodes to average to measure performance
                        (avg. reward etc) (default: 100)
  --summaries_use_frameskip SUMMARIES_USE_FRAMESKIP
                        Whether to multiply training steps by frameskip when
                        recording summaries, FPS, etc. When this flag is set
                        to True, x-axis for all summaries corresponds to the
                        total number of simulated steps, i.e. with frameskip=4
                        the x-axis value of 4 million will correspond to 1
                        million frames observed by the policy. (default: True)
  --heartbeat_interval HEARTBEAT_INTERVAL
                        How often in seconds components send a heartbeat
                        signal to the runner to verify they are not stuck
                        (default: 20)
  --heartbeat_reporting_interval HEARTBEAT_REPORTING_INTERVAL
                        How often in seconds the runner checks for heartbeats
                        (default: 180)
  --train_for_env_steps TRAIN_FOR_ENV_STEPS
                        Stop after all policies are trained for this many env
                        steps (default: 10000000000)
  --train_for_seconds TRAIN_FOR_SECONDS
                        Stop training after this many seconds (default:
                        10000000000)
  --save_every_sec SAVE_EVERY_SEC
                        Checkpointing rate (default: 120)
  --keep_checkpoints KEEP_CHECKPOINTS
                        Number of model checkpoints to keep (default: 2)
  --load_checkpoint_kind {latest,best}
                        Whether to load from latest or best checkpoint
                        (default: latest)
  --save_milestones_sec SAVE_MILESTONES_SEC
                        Save intermediate checkpoints in a separate folder for
                        later evaluation (default=never) (default: -1)
  --save_best_every_sec SAVE_BEST_EVERY_SEC
                        How often we check if we should save the policy with
                        the best score ever (default: 5)
  --save_best_metric SAVE_BEST_METRIC
                        Save "best" policies based on this metric (just env
                        reward by default) (default: reward)
  --save_best_after SAVE_BEST_AFTER
                        Start saving "best" policies after this many env steps
                        to filter lucky episodes that succeed and dominate the
                        statistics early on (default: 100000)
  --benchmark BENCHMARK
                        Benchmark mode (default: False)
  --encoder_mlp_layers [ENCODER_MLP_LAYERS [ENCODER_MLP_LAYERS ...]]
                        In case of MLP encoder, sizes of layers to use. This
                        is ignored if observations are images. (default: [512,
                        512])
  --encoder_conv_architecture {convnet_simple,convnet_impala,convnet_atari,resnet_impala}
                        Architecture of the convolutional encoder. See
                        models.py for details. VizDoom and DMLab examples
                        demonstrate how to define custom architectures.
                        (default: convnet_simple)
  --encoder_conv_mlp_layers [ENCODER_CONV_MLP_LAYERS [ENCODER_CONV_MLP_LAYERS ...]]
                        Optional fully connected layers after the
                        convolutional encoder head. (default: [512])
  --use_rnn USE_RNN     Whether to use RNN core in a policy or not (default:
                        True)
  --rnn_size RNN_SIZE   Size of the RNN hidden state in recurrent model (e.g.
                        GRU or LSTM) (default: 512)
  --rnn_type {gru,lstm}
                        Type of RNN cell to use if use_rnn is True (default:
                        gru)
  --rnn_num_layers RNN_NUM_LAYERS
                        Number of RNN layers to use if use_rnn is True
                        (default: 1)
  --decoder_mlp_layers [DECODER_MLP_LAYERS [DECODER_MLP_LAYERS ...]]
                        Optional decoder MLP layers after the policy core. If
                        empty (default) decoder is identity function.
                        (default: [])
  --nonlinearity {elu,relu,tanh}
                        Type of nonlinearity to use. (default: elu)
  --policy_initialization {orthogonal,xavier_uniform,torch_default}
                        NN weight initialization (default: orthogonal)
  --policy_init_gain POLICY_INIT_GAIN
                        Gain parameter of PyTorch initialization schemas (i.e.
                        Xavier) (default: 1.0)
  --actor_critic_share_weights ACTOR_CRITIC_SHARE_WEIGHTS
                        Whether to share the weights between policy and value
                        function (default: True)
  --adaptive_stddev ADAPTIVE_STDDEV
                        Only for continuous action distributions, whether
                        stddev is state-dependent or just a single learned
                        parameter (default: True)
  --continuous_tanh_scale CONTINUOUS_TANH_SCALE
                        Only for continuous action distributions, whether to
                        use tanh squashing and what scale to use. Applies
                        tanh(mu / scale) * scale to distribution means.
                        Experimental. Currently only works with
                        adaptive_stddev=False (TODO). (default: 0.0)
  --initial_stddev INITIAL_STDDEV
                        Initial value for non-adaptive stddev. Only makes
                        sense for continuous action spaces (default: 1.0)
  --use_env_info_cache USE_ENV_INFO_CACHE
                        Whether to use cached env info (default: False)
  --env_gpu_actions ENV_GPU_ACTIONS
                        Set to true if environment expects actions on GPU
                        (i.e. as a GPU-side PyTorch tensor) (default: False)
  --env_gpu_observations ENV_GPU_OBSERVATIONS
                        Setting this to True together with non-empty
                        --actor_worker_gpus will make observations GPU-side
                        PyTorch tensors. Otherwise data will be on CPU. For
                        CPU-based envs just set --actor_worker_gpus to empty
                        list then this parameter does not matter. (default:
                        True)
  --env_frameskip ENV_FRAMESKIP
                        Number of frames for action repeat (frame skipping).
                        Setting this to >1 will not add any wrappers that will
                        do frame-skipping, although this can be used in the
                        environment factory function to add these wrappers or
                        to tell the environment itself to skip a desired
                        number of frames i.e. as it is done in VizDoom. FPS
                        metrics will be multiplied by the frameskip value,
                        i.e. 100000FPS with frameskip=4 actually corresponds
                        to 100000/4=25000 samples per second observed by the
                        policy. Frameskip=1 (default) means no frameskip, we
                        process every frame. (default: 1)
  --env_framestack ENV_FRAMESTACK
                        Frame stacking (only used in Atari, and it is usually
                        set to 4) (default: 1)
  --pixel_format PIXEL_FORMAT
                        PyTorch expects CHW by default, Ray & TensorFlow
                        expect HWC (default: CHW)
  --use_record_episode_statistics USE_RECORD_EPISODE_STATISTICS
                        Whether to use gym RecordEpisodeStatistics wrapper to
                        keep track of reward (default: False)
  --with_wandb WITH_WANDB
                        Enables Weights and Biases integration (default:
                        False)
  --wandb_user WANDB_USER
                        WandB username (entity). Must be specified from
                        command line! Also see
                        https://docs.wandb.ai/quickstart#1.-set-up-wandb
                        (default: None)
  --wandb_project WANDB_PROJECT
                        WandB "Project" (default: sample_factory)
  --wandb_group WANDB_GROUP
                        WandB "Group" (to group your experiments). By default
                        this is the name of the env. (default: None)
  --wandb_job_type WANDB_JOB_TYPE
                        WandB job type (default: SF)
  --wandb_tags [WANDB_TAGS [WANDB_TAGS ...]]
                        Tags can help with finding experiments in WandB web
                        console (default: [])
  --with_pbt WITH_PBT   Enables population-based training (PBT) (default:
                        False)
  --pbt_mix_policies_in_one_env PBT_MIX_POLICIES_IN_ONE_ENV
                        For multi-agent envs, whether we mix different
                        policies in one env. (default: True)
  --pbt_period_env_steps PBT_PERIOD_ENV_STEPS
                        Periodically replace the worst policies with the best
                        ones and perturb the hyperparameters (default:
                        5000000)
  --pbt_start_mutation PBT_START_MUTATION
                        Allow initial diversification, start PBT after this
                        many env steps (default: 20000000)
  --pbt_replace_fraction PBT_REPLACE_FRACTION
                        A portion of policies performing worst to be replace
                        by better policies (rounded up) (default: 0.3)
  --pbt_mutation_rate PBT_MUTATION_RATE
                        Probability that a parameter mutates (default: 0.15)
  --pbt_replace_reward_gap PBT_REPLACE_REWARD_GAP
                        Relative gap in true reward when replacing weights of
                        the policy with a better performing one (default: 0.1)
  --pbt_replace_reward_gap_absolute PBT_REPLACE_REWARD_GAP_ABSOLUTE
                        Absolute gap in true reward when replacing weights of
                        the policy with a better performing one (default:
                        1e-06)
  --pbt_optimize_gamma PBT_OPTIMIZE_GAMMA
                        Whether to optimize gamma, discount factor, or not
                        (experimental) (default: False)
  --pbt_target_objective PBT_TARGET_OBJECTIVE
                        Policy stat to optimize with PBT. true_objective
                        (default) is equal to raw env reward if not specified,
                        but can also be any other per-policy stat.For DMlab-30
                        use value "dmlab_target_objective" (which is capped
                        human normalized score) (default: true_objective)
  --pbt_perturb_min PBT_PERTURB_MIN
                        When PBT mutates a float hyperparam, it samples the
                        change magnitude randomly from the uniform
                        distribution [pbt_perturb_min, pbt_perturb_max]
                        (default: 1.1)
  --pbt_perturb_max PBT_PERTURB_MAX
                        When PBT mutates a float hyperparam, it samples the
                        change magnitude randomly from the uniform
                        distribution [pbt_perturb_min, pbt_perturb_max]
                        (default: 1.5)
```
