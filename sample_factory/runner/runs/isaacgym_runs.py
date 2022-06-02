version = 48
vstr = f'v{version:03d}'

base_cli = f'python -m sample_factory_examples.isaacgym_examples.train_isaacgym ' \
           f'--algo=APPO --actor_worker_gpus 0 ' \
           f'--env_headless=True'
