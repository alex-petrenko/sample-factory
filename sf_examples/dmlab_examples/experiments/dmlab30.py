from sample_factory.runner.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid([("seed", [1111])])

vstr = "dmlab30"

cli = (
    "python -m sf_examples.dmlab_examples.train_dmlab "
    "--env=dmlab_30 "
    "--train_for_seconds=3600000 "
    "--algo=APPO "
    "--gamma=0.99 "
    "--use_rnn=True "
    "--num_workers=60 "
    "--num_envs_per_worker=12 "
    "--num_epochs=1 "
    "--rollout=32 "
    "--recurrence=32 "
    "--batch_size=2048 "
    "--benchmark=False "
    "--max_grad_norm=0.0 "
    "--dmlab_renderer=software "
    "--decorrelate_experience_max_seconds=1 "
    "--encoder_custom=dmlab_instructions "
    "--encoder_type=resnet "
    "--encoder_subtype=resnet_impala "
    "--encoder_extra_fc_layers=1 "
    "--hidden_size=256 "
    "--nonlinearity=relu "
    "--rnn_type=lstm "
    "--dmlab_extended_action_set=True "
    "--num_policies=1"
)
_experiments = [
    Experiment("dm30", cli, _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription(f"{vstr}", experiments=_experiments)


# Run locally: python -m sample_factory.runner.run --runner=processes --max_parallel=1 --experiments_per_gpu=1 --num_gpus=1 --run=sf_examples.dmlab_examples.experiments.dmlab30
# Run on Slurm: python -m sample_factory.runner.run --runner=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/runner/slurm/sbatch_template.sh --pause_between=1 --slurm_print_only=False --run=sf_examples.dmlab_examples.experiments.dmlab30
