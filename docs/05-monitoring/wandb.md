# Weights and Biases

Sample Factory also supports experiment monitoring with Weights and Biases.
In order to setup WandB locally run `wandb login` in the terminal (https://docs.wandb.ai/quickstart#1.-set-up-wandb)

Example command line to run an experiment with WandB monitoring:

```
python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --experiment=DoomBasic --train_dir=./train_dir --num_workers=20 --num_envs_per_worker=16 --train_for_env_steps=1000000 \\
 --with_wandb=True --wandb_user=<your_wandb_user> --wandb_tags test doom appo
```

A total list of WandB settings: 
```
--with_wandb: Enables Weights and Biases integration (default: False)
--wandb_user: WandB username (entity). Must be specified from command line! Also see https://docs.wandb.ai/quickstart#1.-set-up-wandb (default: None)
--wandb_project: WandB "Project" (default: sample_factory)
--wandb_group: WandB "Group" (to group your experiments). By default this is the name of the env. (default: None)
--wandb_job_type: WandB job type (default: SF)
--wandb_tags: [WANDB_TAGS [WANDB_TAGS ...]] Tags can help with finding experiments in WandB web console (default: [])
```

Once the experiment is started the link to the monitored session is going to be available in the logs (or you can find it by searching in Wandb Web console).