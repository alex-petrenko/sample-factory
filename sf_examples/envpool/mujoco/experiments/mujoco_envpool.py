from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from sample_factory.utils.algo_version import ALGO_VERSION

_params = ParamGrid(
    [
        ("seed", [00, 11, 22]),
        (
            "env",
            [
                "mujoco_ant",
                "mujoco_halfcheetah",
                "mujoco_hopper",
                "mujoco_humanoid",
                "mujoco_doublependulum",
                "mujoco_pendulum",
                "mujoco_reacher",
                "mujoco_swimmer",
                "mujoco_walker",
            ],
        ),
    ]
)

vstr = f"mujoco_envpool_v{ALGO_VERSION}"
cli = (
    f"python -m sf_examples.envpool.mujoco.train_envpool_mujoco "
    f"--train_for_env_steps=100000000 --with_wandb=True --wandb_tags {vstr} --wandb_group=sf2_{vstr}"
)

_experiments = [Experiment(f"{vstr}", cli, _params.generate_params(False))]
RUN_DESCRIPTION = RunDescription(f"{vstr}", experiments=_experiments)

# Run locally: python -m sample_factory.launcher.run --run=sf_examples.envpool.mujoco.experiments.mujoco_envpool --backend=processes --max_parallel=2 --experiments_per_gpu=2 --num_gpus=1
# Run on Slurm: python -m sample_factory.launcher.run --run=sf_examples.envpool.mujoco.experiments.mujoco_envpool --backend=slurm --slurm_workdir=./slurm_envpool --experiment_suffix=slurm --slurm_gpus_per_job=1 --slurm_cpus_per_gpu=16 --slurm_sbatch_template=./sample_factory/launcher/slurm/sbatch_timeout.sh --pause_between=1 --slurm_print_only=False
