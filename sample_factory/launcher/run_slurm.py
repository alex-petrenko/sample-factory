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


def run_slurm(run_description, args):
    workdir = args.slurm_workdir
    pause_between = args.pause_between

    experiments = run_description.experiments

    log.info("Starting processes with base cmds: %r", [e.cmd for e in experiments])

    if not os.path.exists(workdir):
        log.info("Creating %s...", workdir)
        os.makedirs(workdir)

    if args.slurm_sbatch_template is not None:
        with open(args.slurm_sbatch_template, "r") as template_file:
            sbatch_template = template_file.read()
    else:
        sbatch_template = SBATCH_TEMPLATE_DEFAULT

    log.info("Sbatch template: %s", sbatch_template)

    partition = ""
    if args.slurm_partition is not None:
        partition = f"-p {args.slurm_partition} "

    num_cpus = args.slurm_cpus_per_gpu * args.slurm_gpus_per_job

    experiments = run_description.generate_experiments(args.train_dir)
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
        with open(sbatch_fname, "w") as sbatch_f:
            sbatch_f.write(file_content)

        sbatch_files.append(sbatch_fname)

    job_ids = []
    idx = 0
    for sbatch_file in sbatch_files:
        idx += 1
        sbatch_fname = os.path.basename(sbatch_file)
        cmd = f"sbatch {partition}--gres=gpu:{args.slurm_gpus_per_job} -c {num_cpus} --parsable --output {workdir}/{sbatch_fname}-slurm-%j.out {sbatch_file}"
        log.info("Executing %s...", cmd)

        if args.slurm_print_only:
            output = idx
        else:
            cmd_tokens = cmd.split()
            process = Popen(cmd_tokens, stdout=PIPE)
            output, err = process.communicate()
            exit_code = process.wait()
            log.info("Output: %s, err: %s, exit code: %r", output, err, exit_code)

            if exit_code != 0:
                log.error("sbatch process failed!")
                time.sleep(5)

        job_id = int(output)
        job_ids.append(str(job_id))

        time.sleep(pause_between)

    tail_cmd = f"tail -f {workdir}/*.out"
    log.info("Monitor log files using\n\n\t %s \n\n", tail_cmd)

    scancel_cmd = f'scancel {" ".join(job_ids)}'

    log.info("Jobs queued: %r", job_ids)

    log.info("Use this command to cancel your jobs: \n\t %s \n", scancel_cmd)

    with open(join(workdir, "scancel.sh"), "w") as fobj:
        fobj.write(scancel_cmd)

    log.info("Done!")
    return 0
