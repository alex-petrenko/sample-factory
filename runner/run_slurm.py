"""Run many experiments with SLURM: hyperparameter sweeps, etc."""
import os
import time
from os.path import join
from subprocess import Popen, PIPE

from utils.utils import log

# TODO: this is not portable, a hack
SBATCH_TEMPLATE = (
    '#!/bin/bash\n'
    'source /home/apetrenk/anaconda3/etc/profile.d/conda.sh\n'
    'conda activate doom-rl\n'
    'cd ~/doom-neurobot\n'
)


def run_slurm(run_description, workdir):
    experiments = run_description.experiments

    log.info('Starting processes with base cmds: %r', [e.cmd for e in experiments])

    if not os.path.exists(workdir):
        log.info('Creating %s...', workdir)
        os.makedirs(workdir)

    experiments = run_description.generate_experiments()
    sbatch_files = []
    for experiment in experiments:
        cmd, name, _ = experiment

        sbatch_fname = f'sbatch_{name}.sh'
        sbatch_fname = join(workdir, sbatch_fname)

        file_content = SBATCH_TEMPLATE + cmd + '\n\necho "Done!!!"'
        with open(sbatch_fname, 'w') as sbatch_f:
            sbatch_f.write(file_content)

        sbatch_files.append(sbatch_fname)

    job_ids = []
    for sbatch_file in sbatch_files:
        cmd = f'sbatch -p gpu --gres=gpu:1 -c 14 --parsable --output {workdir}/slurm-%j.out {sbatch_file}'
        log.info('Executing %s...', cmd)
        cmd_tokens = cmd.split()
        process = Popen(cmd_tokens, stdout=PIPE)
        output, err = process.communicate()
        exit_code = process.wait()
        log.info('Output: %s, err: %s, exit code: %r', output, err, exit_code)

        job_id = int(output)
        job_ids.append(str(job_id))

        time.sleep(2)

    tail_cmd = f'tail -f {workdir}/*.out'
    log.info('Monitor log files using\n\n\t %s \n\n', tail_cmd)

    scancel_cmd = f'scancel {" ".join(job_ids)}'

    log.info(f'Cancel with: \n\t %s \n', scancel_cmd)

    with open(join(workdir, 'scancel.sh'), 'w') as fobj:
        fobj.write(scancel_cmd)

    log.info('Done!')
    return 0
