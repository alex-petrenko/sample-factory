from sample_factory.utils.algo_version import ALGO_VERSION

version = ALGO_VERSION
vstr = f"v{version:03d}"

# wandb_project = 'rlgpu-2022'
wandb_project = "sample_factory"

base_cli = (
    f"python -m sf_examples.isaacgym_examples.train_isaacgym "
    f"--actor_worker_gpus 0 "
    f"--wandb_project={wandb_project}"
)
