"""
Run many experiments with NGC: hyperparameter sweeps, etc.
This isn't production code, but feel free to use as an example for your NGC setup.

"""

import time
from subprocess import PIPE, Popen

from sample_factory.utils.utils import log, str2bool


def add_ngc_args(parser):
    parser.add_argument(
        "--ngc_job_template",
        default=None,
        type=str,
        help="NGC command line template, specifying instance type, docker container, etc.",
    )
    parser.add_argument(
        "--ngc_print_only", default=False, type=str2bool, help="Just print commands to the console without executing"
    )
    return parser


def run_ngc(run_description, args):
    pause_between = args.pause_between
    experiments = run_description.experiments

    log.info("Starting processes with base cmds: %r", [e.cmd for e in experiments])

    if args.ngc_job_template is not None:
        with open(args.ngc_job_template, "r") as template_file:
            ngc_template = template_file.read()

    ngc_template = ngc_template.replace("\\", " ")
    ngc_template = " ".join(ngc_template.split())

    log.info("NGC template: %s", ngc_template)
    experiments = run_description.generate_experiments(args.train_dir, makedirs=False)
    for experiment_idx, experiment in enumerate(experiments):
        cmd, name, *_ = experiment

        job_name = name
        log.info("Job name: %s", job_name)

        ngc_job_cmd = ngc_template.replace("{{ name }}", job_name).replace("{{ experiment_cmd }}", cmd)

        log.info("Executing %s...", ngc_job_cmd)

        if not args.ngc_print_only:
            process = Popen(ngc_job_cmd, stdout=PIPE, shell=True)
            output, err = process.communicate()
            exit_code = process.wait()
            log.info("Output: %s, err: %s, exit code: %r", output, err, exit_code)

        time.sleep(pause_between)

    log.info("Done!")
    return 0
