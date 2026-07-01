"""Run groups of experiments, hyperparameter sweeps, etc."""

import argparse
import os
import subprocess
import sys
import time
from os.path import join

from sample_factory.utils.utils import ensure_dir_exists, log


def add_os_parallelism_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--num_gpus", default=1, type=int, help="How many local GPUs to use")
    parser.add_argument("--max_parallel", default=4, type=int, help="Maximum simultaneous experiments")
    parser.add_argument(
        "--experiments_per_gpu",
        default=-1,
        type=int,
        help="How many experiments can we squeeze on a single GPU. "
        "Specify this option if and only if you are using launcher to run several experiments using OS-level"
        "parallelism (--backend=processes)."
        "In any other case use default value (-1) for not altering CUDA_VISIBLE_DEVICES at all."
        "This will allow your experiments to use all GPUs available (as many as --num_gpu allows)"
        "Helpful when e.g. you are running a single big PBT experiment.",
    )
    return parser


def find_least_busy_gpu(num_gpus, experiments_per_gpu, processes_per_gpu):
    """Return GPU with the largest remaining process capacity."""
    least_busy_gpu = None
    gpu_available_processes = 0

    for gpu_id in range(num_gpus):
        available_processes = experiments_per_gpu - len(processes_per_gpu[gpu_id])
        if available_processes > gpu_available_processes:
            gpu_available_processes = available_processes
            least_busy_gpu = gpu_id

    return least_busy_gpu, gpu_available_processes


def can_squeeze_another_process(processes, processes_per_gpu, args):
    """Check whether another experiment process can start now."""
    if len(processes) >= args.max_parallel:
        return False

    if args.experiments_per_gpu <= 0:
        return True

    _, gpu_available_processes = find_least_busy_gpu(
        args.num_gpus, args.experiments_per_gpu, processes_per_gpu
    )
    return gpu_available_processes > 0


def prepare_process_command(cmd):
    """Build command tokens and replace Python executable with current interpreter."""
    cmd_tokens = cmd.split(" ")

    # workaround to make sure we're running the correct python executable from our virtual env
    if cmd_tokens[0].startswith("python"):
        cmd_tokens[0] = sys.executable
        log.debug("Using Python executable %s", cmd_tokens[0])

    return cmd_tokens


def prepare_process_env(exp_env_vars, processes_per_gpu, args):
    """Prepare environment variables and optional GPU assignment."""
    envvars = os.environ.copy()
    best_gpu = None

    if args.experiments_per_gpu > 0:
        best_gpu, best_gpu_available_processes = find_least_busy_gpu(
            args.num_gpus, args.experiments_per_gpu, processes_per_gpu
        )
        log.info(
            "The least busy gpu is %d where we can run %d more processes",
            best_gpu,
            best_gpu_available_processes,
        )
        envvars["CUDA_VISIBLE_DEVICES"] = f"{best_gpu}"

    if exp_env_vars is not None:
        for key, value in exp_env_vars.items():
            log.info("Adding env variable %r %r", key, value)
            envvars[str(key)] = str(value)

    return envvars, best_gpu


def start_experiment_process(experiment, processes_per_gpu, args):
    """Start a single experiment process."""
    cmd, _, root_dir, exp_env_vars = experiment
    cmd_tokens = prepare_process_command(cmd)

    ensure_dir_exists(join(args.train_dir, root_dir))
    envvars, best_gpu = prepare_process_env(exp_env_vars, processes_per_gpu, args)

    log.info("Starting process %r", cmd_tokens)

    process = subprocess.Popen(cmd_tokens, stdout=None, stderr=None, env=envvars)
    process.gpu_id = best_gpu
    process.proc_cmd = cmd

    if process.gpu_id is not None:
        processes_per_gpu[process.gpu_id].append(process.proc_cmd)

    log.info("Started process %s on GPU %r", process.proc_cmd, process.gpu_id)
    return process


def collect_finished_processes(processes, processes_per_gpu, failed_processes):
    """Collect still-running processes and record failures."""
    remaining_processes = []
    for process in processes:
        if process.poll() is None:
            remaining_processes.append(process)
            continue

        if process.gpu_id is not None:
            processes_per_gpu[process.gpu_id].remove(process.proc_cmd)
        log.info("Process %r finished with code %r", process.proc_cmd, process.returncode)
        if process.returncode != 0:
            failed_processes.append((process.proc_cmd, process.pid, process.returncode))
            log.error("WARNING: RETURN CODE IS %r", process.returncode)

    return remaining_processes


def report_failed_processes(failed_processes, last_log_time, log_interval):
    """Log failed processes at a throttled interval."""
    if time.time() - last_log_time <= log_interval:
        return last_log_time

    if failed_processes:
        log.error(
            "Failed processes: %s",
            ", ".join([f"PID: {p[1]} code: {p[2]}" for p in failed_processes]),
        )

    return time.time()


def run(run_description, args):
    """Run generated experiments as OS-level processes."""
    experiments = run_description.experiments

    log.info("Starting processes with base cmds: %r", [e.cmd for e in experiments])
    log.info("Max parallel processes is %d", args.max_parallel)
    log.info(
        "Monitor log files using\n\n\ttail -f train_dir/%s/**/**/sf_log.txt\n\n",
        run_description.run_name,
    )

    processes = []
    processes_per_gpu = {g: [] for g in range(args.num_gpus)}

    experiments = run_description.generate_experiments(args.train_dir)
    next_experiment = next(experiments, None)

    failed_processes = []
    last_log_time = 0
    log_interval = 3  # seconds

    while len(processes) > 0 or next_experiment is not None:
        while (
            can_squeeze_another_process(processes, processes_per_gpu, args)
            and next_experiment is not None
        ):
            process = start_experiment_process(next_experiment, processes_per_gpu, args)

            processes.append(process)

            log.info("Waiting for %d seconds before starting next process", args.pause_between)
            time.sleep(args.pause_between)

            next_experiment = next(experiments, None)

        processes = collect_finished_processes(processes, processes_per_gpu, failed_processes)
        last_log_time = report_failed_processes(failed_processes, last_log_time, log_interval)

        time.sleep(0.1)

    log.info("Done!")

    return 0
