"""
Run many experiments with SLURM: hyperparameter sweeps, etc.
This isn't production code, but feel free to use as an example for your SLURM setup.

"""


import os
import time
from os.path import join
from subprocess import Popen, PIPE

from sample_factory.utils.utils import log, str2bool

# TODO: this is not portable, a hack
SBATCH_TEMPLATE = (
    '#!/bin/bash\n'
    'source /homes/petrenko/miniconda3/etc/profile.d/conda.sh\n'
    'conda activate sample-factory\n'
    'cd ~/sample-factory\n'
)


def add_slurm_args(parser):
    parser.add_argument('--slurm_gpus_per_job', default=1, type=int, help='GPUs in a single SLURM process')
    parser.add_argument('--slurm_cpus_per_gpu', default=14, type=int, help='Max allowed number of CPU cores per allocated GPU')
    parser.add_argument('--slurm_print_only', default=False, type=str2bool, help='Just print commands to the console without executing')
    parser.add_argument('--slurm_workdir', default=None, type=str, help='Optional workdir. Used by slurm runner to store logfiles etc.')
    return parser


def run_slurm(run_description, args):
    workdir = args.slurm_workdir
    pause_between = args.pause_between

    experiments = run_description.experiments

    log.info('Starting processes with base cmds: %r', [e.cmd for e in experiments])

    if not os.path.exists(workdir):
        log.info('Creating %s...', workdir)
        os.makedirs(workdir)

    experiments = run_description.generate_experiments()
    sbatch_files = []
    for experiment in experiments:
        cmd, name, *_ = experiment

        sbatch_fname = f'sbatch_{name}.sh'
        sbatch_fname = join(workdir, sbatch_fname)

        file_content = SBATCH_TEMPLATE + cmd + '\n\necho "Done!!!"'
        with open(sbatch_fname, 'w') as sbatch_f:
            sbatch_f.write(file_content)

        sbatch_files.append(sbatch_fname)

    job_ids = []
    idx = 0
    for sbatch_file in sbatch_files:
        idx += 1
        sbatch_fname = os.path.basename(sbatch_file)
        num_cpus = args.slurm_cpus_per_gpu * args.slurm_gpus_per_job
        cmd = f'sbatch -p gpu --gres=gpu:{args.slurm_gpus_per_job} -c {num_cpus} --parsable --output {workdir}/{sbatch_fname}-slurm-%j.out {sbatch_file}'
        log.info('Executing %s...', cmd)

        if args.slurm_print_only:
            output = idx
        else:
            cmd_tokens = cmd.split()
            process = Popen(cmd_tokens, stdout=PIPE)
            output, err = process.communicate()
            exit_code = process.wait()
            log.info('Output: %s, err: %s, exit code: %r', output, err, exit_code)
        job_id = int(output)
        job_ids.append(str(job_id))

        time.sleep(pause_between)

    tail_cmd = f'tail -f {workdir}/*.out'
    log.info('Monitor log files using\n\n\t %s \n\n', tail_cmd)

    scancel_cmd = f'scancel {" ".join(job_ids)}'

    log.info(f'Cancel with: \n\t %s \n', scancel_cmd)

    with open(join(workdir, 'scancel.sh'), 'w') as fobj:
        fobj.write(scancel_cmd)

    log.info('Done!')
    return 0
