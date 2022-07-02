from sample_factory.algo.runners.runner_parallel import ParallelRunner
from sample_factory.algo.runners.runner_serial import SerialRunner


def run_rl(cfg):
    if cfg.serial_mode:
        runner_cls = SerialRunner
    else:
        runner_cls = ParallelRunner

    runner = runner_cls(cfg)
    runner.init()
    status = runner.run()
    return status
