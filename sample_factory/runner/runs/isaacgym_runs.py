version = 37
vstr = f'v{version:03d}'

base_cli = f'python -m sample_factory_examples.train_isaacgym ' \
           f'--algo=APPO --actor_worker_gpus 0 --env_agents=4096 ' \
           f'--batch_size=32768 --env_headless=True --with_vtrace=False --use_rnn=False --recurrence=1 --with_wandb=True'
