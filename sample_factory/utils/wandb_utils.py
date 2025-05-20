from datetime import datetime

from sample_factory.utils.utils import log, retry


def init_wandb(cfg):
    """
    Must call initialization of Wandb before summary writer is initialized, otherwise
    sync_tensorboard does not work.
    """

    if not cfg.with_wandb:
        log.debug("Weights and Biases integration disabled")
        return

    if "wandb_unique_id" not in cfg:
        # if we're going to restart the experiment, this will be saved to a json file
        cfg.wandb_unique_id = f'{cfg.experiment}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'

    wandb_unique_id = cfg.wandb_unique_id
    wandb_group = cfg.env if cfg.wandb_group is None else cfg.wandb_group

    log.debug(
        "Weights and Biases integration enabled. Project: %s, user: %s, group: %s, unique_id: %s",
        cfg.wandb_project,
        cfg.wandb_user,
        cfg.wandb_group,
        wandb_unique_id,
    )

    import wandb

    # this can fail occasionally, so we try a couple more times
    @retry(3, exceptions=(Exception,))
    def init_wandb_func():
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_user,
            sync_tensorboard=True,
            id=wandb_unique_id,
            name=wandb_unique_id,
            group=wandb_group,
            job_type=cfg.wandb_job_type,
            tags=cfg.wandb_tags,
            resume="allow",
            settings=wandb.Settings(start_method="fork"),
        )

    log.debug("Initializing WandB...")
    try:
        init_wandb_func()
    except Exception as exc:
        log.error(f"Could not initialize WandB! {exc}")
        raise

    wandb.config.update(cfg, allow_val_change=True)


def finish_wandb(cfg):
    if cfg.with_wandb:
        import wandb

        wandb.run.finish()
