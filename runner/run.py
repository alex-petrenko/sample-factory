import importlib
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=None, type=str, help='Name of the python script that describes the run, e.g. doom_battle_hybrid')
    parser.add_argument('--runner', default='processes', choices=['processes', 'slurm'])
    parser.add_argument('--workdir', default=None, type=str, help='Optional workdir. Used by slurm runner to store logfiles etc.')
    parser.add_argument('--pause_between', default=10, type=int, help='Pause in seconds between processes')
    parser.add_argument('--num_gpus', default=1, type=int, help='How many GPUs to use')
    parser.add_argument('--experiments_per_gpu', default=-1, type=int, help='How many experiments can we squeeze on a single GPU (-1 for not altering CUDA_VISIBLE_DEVICES at all)')
    parser.add_argument('--max_parallel', default=2, type=int, help='Maximum simultaneous experiments')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    run_module = importlib.import_module(f'runner.runs.{args.run}')
    run_description = run_module.RUN_DESCRIPTION

    if args.runner == 'processes':
        from runner.run_processes import run
        run(run_description, args)
    elif args.runner == 'slurm':
        from runner.run_slurm import run_slurm
        run_slurm(run_description, args.workdir, args.pause_between)


if __name__ == '__main__':
    sys.exit(main())
