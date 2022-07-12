from sample_factory.algo.runners.runner_parallel import ParallelRunner
from sample_factory.algo.runners.runner_serial import SerialRunner
from sample_factory.cfg.arguments import maybe_load_from_checkpoint, verify_cfg


def run_rl(cfg):
    cfg = maybe_load_from_checkpoint(cfg)

    # check for any incompatible arguments
    verify_cfg(cfg)

    if cfg.serial_mode:
        runner_cls = SerialRunner
    else:
        runner_cls = ParallelRunner

    runner = runner_cls(cfg)
    runner.init()
    status = runner.run()
    return status
