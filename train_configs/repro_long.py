import os

from mrunner.helpers.specification_helper import create_experiments_helper

name = globals()["script"][:-3]

num_minibatches = 1
num_epochs = 1
num_envs = 256
batch_size = 4096
num_steps = 32
num_workers = 16

# params for all exps
config = {
    "exp_tag": name,
    "env": "nethack_score",
    "run_script": "sf_examples.nethack.train_nethack",
    "train_for_env_steps": 1_000_000_000,
    "num_workers": num_workers,
    "num_envs_per_worker": num_envs // num_workers,
    "worker_num_splits": 2,
    "rollout": num_steps,
    "batch_size": batch_size,
    "num_batches_per_epoch": num_minibatches,
    "num_epochs": num_epochs,
    "penalty_step": 0.0,
    "penalty_time": 0.0,
    "async_rl": True,
    "serial_mode": False,
    "wandb_user": "bartekcupial",
    "wandb_project": "nle_simba",
    "wandb_group": "ideas-ncbr",
    "with_wandb": True,
    "decorrelate_envs_on_one_worker": True,
    "character": "mon-hum-neu-mal",
    "max_grad_norm": 40.0,
    "learning_rate": 1e-4,
    "exploration_loss_coeff": 0.001,
    "gamma": 0.999,
    "gae_lambda": 1.0,
    "value_loss_coeff": 1.0,
    "model": "cnn_pixels",
    "add_image_observation": True,
}

# params different between exps
params_grid = [
    {
        "seed": list(range(3)),
    }
]

experiments_list = create_experiments_helper(
    experiment_name=name,
    project_name="nle_simba",
    with_neptune=False,
    script="python3 mrunner_run.py",
    python_path=".",
    tags=[name],
    env={
        "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
    },
    base_config=config,
    params_grid=params_grid,
    mrunner_ignore=".mrunnerignore",
)
