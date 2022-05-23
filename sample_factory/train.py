import sys

from sample_factory.algo.runners.runner_parallel import ParallelRunner
from sample_factory.algo.runners.runner_serial import SerialRunner
from sample_factory.algorithms.utils.arguments import parse_args


def run_rl(cfg):
    if cfg.serial_mode:
        runner_cls = SerialRunner
    else:
        runner_cls = ParallelRunner

    runner = runner_cls(cfg)
    runner.init()
    status = runner.run()
    return status


# TODO: remove duplicate code
# def run_rl_async(cfg):
#     policy_id = 0  # TODO: multiple policies
#     ctx = multiprocessing.get_context('spawn')
#     learner_proc = EventLoopProcess('learner_proc', ctx, init_func=init_learner_process, args=(sf_global_context(), cfg, policy_id))
#     batcher = SequentialBatcher(learner_proc.event_loop, buffer_mgr.trajectories_per_batch, buffer_mgr.total_num_trajectories, env_info)
#     learner = Learner(learner_proc.event_loop, cfg, env_info, buffer_mgr, policy_id=0, mp_ctx=ctx)  # currently support only single-policy learning
#
#     sampler_proc = EventLoopProcess('sampler_proc', ctx, init_func=init_sampler_process, args=(sf_global_context(), cfg, policy_id))
#     sampler = SyncSampler(sampler_proc.event_loop, cfg, env_info, learner.param_server, buffer_mgr, batcher.sampling_batches_queue)
#
#     runner.init(sampler, batcher, learner)
#
#     sampler_proc.start()
#     learner_proc.start()
#     status = runner.run()
#     learner_proc.join()
#     sampler_proc.join()
#
#     return status


def main():
    """RL training entry point."""
    cfg = parse_args()
    status = run_rl(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
