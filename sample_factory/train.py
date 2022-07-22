from sample_factory.algo.runners.runner_parallel import ParallelRunner
from sample_factory.algo.runners.runner_serial import SerialRunner
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.cfg.arguments import maybe_load_from_checkpoint, verify_cfg


def run_rl(cfg):
    if cfg.restart_behavior == "resume":
        cfg = maybe_load_from_checkpoint(cfg)

    # check for any incompatible arguments
    if not verify_cfg(cfg):
        return ExperimentStatus.FAILURE

    if cfg.serial_mode:
        runner_cls = SerialRunner
    else:
        runner_cls = ParallelRunner

    runner = runner_cls(cfg)
    runner.init()
    status = runner.run()
    return status
