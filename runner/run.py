import importlib
import sys
import argparse

from runner.run_slurm import add_slurm_args


def runner_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=None, type=str, help='Name of the python script that describes the run, e.g. doom_battle_hybrid')
    parser.add_argument('--runner', default='processes', choices=['processes', 'slurm'])
    parser.add_argument('--pause_between', default=10, type=int, help='Pause in seconds between processes')
    parser.add_argument('--num_gpus', default=1, type=int, help='How many GPUs to use')
    parser.add_argument('--experiments_per_gpu', default=-1, type=int, help='How many experiments can we squeeze on a single GPU (-1 for not altering CUDA_VISIBLE_DEVICES at all)')
    parser.add_argument('--max_parallel', default=4, type=int, help='Maximum simultaneous experiments')
    parser.add_argument('--experiment_suffix', default='', type=str, help='Append this to the name of the experiment dir')

    parser = add_slurm_args(parser)

    return parser


def parse_args():
    args = runner_argparser().parse_args(sys.argv[1:])
    return args


def main():
    args = parse_args()

    run_module = importlib.import_module(f'runner.runs.{args.run}')
    run_description = run_module.RUN_DESCRIPTION
    run_description.experiment_suffix = args.experiment_suffix

    if args.runner == 'processes':
        from runner.run_processes import run
        run(run_description, args)
    elif args.runner == 'slurm':
        from runner.run_slurm import run_slurm
        run_slurm(run_description, args)


if __name__ == '__main__':
    sys.exit(main())
