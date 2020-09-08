#!/usr/bin/env python

"""
Little handy script to launch tensorboard using a wildcard mask.
Inspired by code originally written by Shao-Hua Sun https://github.com/shaohua0116
"""

import sys
import glob
import time
import psutil
import os.path
import argparse
import subprocess

from os.path import join

from utils.utils import kill


def main():
    parser = argparse.ArgumentParser(description=r'Launch tensorboard on multiple directories in an easy way.')
    parser.add_argument('--dir', default='./train_dir', help='Base folder with summaries')
    parser.add_argument('--port', default=6006, type=int, help='The port to use for tensorboard')
    parser.add_argument('--quiet', '-q', action='store_true', help='Run in silent mode')
    parser.add_argument('--refresh_every', '-r', dest='refresh', type=int, default=36000,
                        help='Restart tensorboard process every x seconds to prevent mem leaks (default 36000 sec, which is 10 hours)')
    parser.add_argument('--reload_interval', type=int, default=60, help='How often to reload data')
    parser.add_argument('filters', nargs='+', type=str, help='directories in train_dir to monitor')
    args = parser.parse_args()

    train_dirs = []
    for f in args.filters:
        matches = glob.glob(join(os.path.expanduser(args.dir), f))
        for match in matches:
            if os.path.isdir(match):
                train_dirs.append(match)
                print('Monitoring', match, '...')

    train_dirs = ','.join([s for s in train_dirs])
    cmd = (
        f'tensorboard '
        f'--port={args.port} '
        f'--logdir={train_dirs} '
        f'--reload_interval={args.reload_interval} '
        f'--max_reload_threads=8 '
        f'--samples_per_plugin="scalars=200"'
    )

    if args.quiet:
        cmd += ' 2>/dev/null'

    num_restarts = 0
    while True:
        num_restarts += 1
        print(cmd)
        p = subprocess.Popen(cmd, shell=True)
        time.sleep(args.refresh)
        print('Kill current tensorboard process', p.pid, '...')
        kill(p.pid)
        print('Restarting tensorboard', num_restarts, '...')


if __name__ == '__main__':
    sys.exit(main())
