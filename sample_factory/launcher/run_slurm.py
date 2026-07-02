"""
Run many experiments with SLURM: hyperparameter sweeps, etc.
This isn't production code, but feel free to use as an example for your SLURM setup.

"""

import os
import time
from os.path import join
from string import Template
from subprocess import PIPE, Popen

from sample_factory.utils.utils import log, str2bool

SBATCH_TEMPLATE_DEFAULT = (
    "#!/bin/bash\n"
    # "source /homes/petrenko/miniconda3/etc/profile.d/conda.sh\n"
    "conda activate sf2\n"
    "cd ~/sample-factory\n"
)


def add_slurm_args(parser):
    parser.add_argument("--slurm_gpus_per_job", default=1, type=int, help="GPUs in a single SLURM process")
    parser.add_argument(
        "--slurm_cpus_per_gpu", default=14, type=int, help="Max allowed number of CPU cores per allocated GPU"
    )
    parser.add_argument(
        "--slurm_print_only", default=False, type=str2bool, help="Just print commands to the console without executing"
    )
    parser.add_argument(
        "--slurm_workdir",
        default=None,
        type=str,
        help="Optional workdir. Used by slurm launcher to store logfiles etc.",
    )
    parser.add_argument(
        "--slurm_partition",
        default=None,
        type=str,
        help='Adds slurm partition, i.e. for "gpu" it will add "-p gpu" to sbatch command line',
    )

    parser.add_argument(
        "--slurm_sbatch_template",
        default=None,
        type=str,
        help="Commands to run before the actual experiment (i.e. activate conda env, etc.)",
    )

    parser.add_argument(
        "--slurm_timeout",
        default="0",
        type=str,
        help="Time to run jobs before timing out job and requeuing the job. Defaults to 0, which does not time out the job",
    )

    return parser


def ensure_slurm_workdir(workdir):
    """Create SLURM working directory if it does not exist."""
    if not os.path.exists(workdir):
        log.info("Creating %s...", workdir)
        os.makedirs(workdir)


def load_sbatch_template(template_path):
    """Load a custom sbatch template or return the default one."""
    if template_path is None:
        return SBATCH_TEMPLATE_DEFAULT

    with open(template_path, "r", encoding="utf-8") as template_file:
        return template_file.read()


def slurm_partition_arg(partition):
    """Build optional SLURM partition argument."""
    if partition is None:
        return ""
    return f"-p {partition} "


def slurm_num_cpus(args):
    """Calculate CPU count requested for one SLURM job."""
    return args.slurm_cpus_per_gpu * args.slurm_gpus_per_job


def create_sbatch_files(experiments, workdir, sbatch_template, partition, args):
    """Create sbatch files for all generated experiments."""
    num_cpus = slurm_num_cpus(args)
    sbatch_files = []
    for experiment in experiments:
        cmd, name, *_ = experiment

        sbatch_fname = f"sbatch_{name}.sh"
        sbatch_fname = join(workdir, sbatch_fname)
        sbatch_fname = os.path.abspath(sbatch_fname)

        file_content = Template(sbatch_template).substitute(
            CMD=cmd,
            FILENAME=sbatch_fname,
            PARTITION=partition,
            GPU=args.slurm_gpus_per_job,
            CPU=num_cpus,
            TIMEOUT=args.slurm_timeout,
        )
        with open(sbatch_fname, "w", encoding="utf-8") as sbatch_f:
            sbatch_f.write(file_content)

        sbatch_files.append(sbatch_fname)

    return sbatch_files


def submit_sbatch_file(sbatch_file, idx, workdir, partition, args):
    """Submit one sbatch file and return the generated job id."""
    num_cpus = slurm_num_cpus(args)
    sbatch_fname = os.path.basename(sbatch_file)
    cmd = (
        f"sbatch {partition}--gres=gpu:{args.slurm_gpus_per_job} "
        f"-c {num_cpus} --parsable --output {workdir}/{sbatch_fname}-slurm-%j.out {sbatch_file}"
    )
    log.info("Executing %s...", cmd)

    if args.slurm_print_only:
        return idx

    cmd_tokens = cmd.split()
    process = Popen(cmd_tokens, stdout=PIPE)
    output, err = process.communicate()
    exit_code = process.wait()
    log.info("Output: %s, err: %s, exit code: %r", output, err, exit_code)

    if exit_code != 0:
        log.error("sbatch process failed!")
        time.sleep(5)

    return output


def submit_sbatch_files(sbatch_files, workdir, partition, args, pause_between):
    """Submit all sbatch files and collect job ids."""
    job_ids = []
    for idx, sbatch_file in enumerate(sbatch_files, start=1):
        output = submit_sbatch_file(sbatch_file, idx, workdir, partition, args)
        job_id = int(output)
        job_ids.append(str(job_id))

        time.sleep(pause_between)

    return job_ids


def write_scancel_script(workdir, job_ids):
    """Write a helper script that cancels all submitted jobs."""
    scancel_cmd = f'scancel {" ".join(job_ids)}'

    log.info("Jobs queued: %r", job_ids)
    log.info("Use this command to cancel your jobs: \n\t %s \n", scancel_cmd)

    with open(join(workdir, "scancel.sh"), "w", encoding="utf-8") as fobj:
        fobj.write(scancel_cmd)


def run_slurm(run_description, args):
    """Run a generated set of experiments through SLURM."""
    workdir = args.slurm_workdir
    pause_between = args.pause_between

    experiments = run_description.experiments

    log.info("Starting processes with base cmds: %r", [e.cmd for e in experiments])

    ensure_slurm_workdir(workdir)

    sbatch_template = load_sbatch_template(args.slurm_sbatch_template)
    log.info("Sbatch template: %s", sbatch_template)

    partition = slurm_partition_arg(args.slurm_partition)

    experiments = run_description.generate_experiments(args.train_dir)
    sbatch_files = create_sbatch_files(experiments, workdir, sbatch_template, partition, args)
    job_ids = submit_sbatch_files(sbatch_files, workdir, partition, args, pause_between)

    tail_cmd = f"tail -f {workdir}/*.out"
    log.info("Monitor log files using\n\n\t %s \n\n", tail_cmd)

    write_scancel_script(workdir, job_ids)

    log.info("Done!")
    return 0
