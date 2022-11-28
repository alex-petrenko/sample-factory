from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription

_params = ParamGrid([("seed", [1111])])

vstr = "dmlab30"

cli = (
    "python -m sf_examples.dmlab.train_dmlab "
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
    "--nonlinearity=relu "
    "--rnn_type=lstm "
    "--dmlab_extended_action_set=True "
    "--num_policies=1"
)
_experiments = [
    Experiment("dm30", cli, _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription(f"{vstr}", experiments=_experiments)


# Run locally: python -m sample_factory.launcher.run --backend=processes --max_parallel=1 --experiments_per_gpu=1 --num_gpus=1 --run=sf_examples.dmlab.experiments.dmlab30
# Run on Slurm: python -m sample_factory.launcher.run --backend=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/launcher/slurm/sbatch_timeout.sh --pause_between=1 --slurm_print_only=False --run=sf_examples.dmlab.experiments.dmlab30
