from sample_factory.launcher.launcher_utils import seeds
from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from sf_examples.isaacgym_examples.experiments.isaacgym_runs import base_cli, vstr

_params = ParamGrid(
    [
        ("seed", seeds(3)),
        ("env", ["Ant", "Humanoid"]),
        # should be 16 recurrence for Ant and 32 for Humanoid
        # (("use_rnn", "recurrence"), ([False, 1], [True, 16])),  # train recurrent and non-recurrent models
    ]
)

vstr = f"{vstr}_basic_ige"
cli = base_cli + f" --train_for_env_steps=100000000 --with_wandb=True --wandb_tags {vstr} --wandb_group=sf2_{vstr}"
_experiments = [Experiment(f"{vstr}", cli, _params.generate_params(False))]
RUN_DESCRIPTION = RunDescription(f"{vstr}", experiments=_experiments)


# Run locally: python -m sample_factory.launcher.run --run=sf_examples.isaacgym_examples.experiments.isaacgym_basic_envs --backend=processes --max_parallel=2 --experiments_per_gpu=2 --num_gpus=1
# Run on Slurm: python -m sample_factory.launcher.run --run=sf_examples.isaacgym_examples.experiments.isaacgym_basic_envs --backend=slurm --slurm_workdir=./slurm_isaacgym --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/launcher/slurm/sbatch_timeout.sh --pause_between=1 --slurm_print_only=False
