import shutil
from os.path import isdir

from sample_factory.utils.typing import Config
from sample_factory.utils.utils import experiment_dir


def clean_test_dir(cfg: Config) -> str:
    directory = experiment_dir(cfg=cfg, mkdir=False)
    if isdir(directory):
        # remove any previous unfinished test dirs so they don't interfere with this test
        shutil.rmtree(directory, ignore_errors=True)

    return directory
