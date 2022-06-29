version = 60
vstr = f'v{version:03d}'

# wandb_project = 'rlgpu-2022'
wandb_project = 'sample_factory'

base_cli = f'python -m sample_factory_examples.isaacgym_examples.train_isaacgym ' \
           f'--actor_worker_gpus 0 ' \
           f'--env_headless=True ' \
           f'--wandb_project={wandb_project}'
