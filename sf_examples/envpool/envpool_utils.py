import multiprocessing

from sample_factory.utils.utils import is_module_available


def envpool_available():
    return is_module_available("envpool")


def add_envpool_common_args(env, parser) -> None:
    parser.add_argument(
        "--envpool_num_threads",
        default=multiprocessing.cpu_count(),
        type=int,
        help="Num threads to use for envpool, defaults to the number of logical CPUs",
    )

    parser.add_argument(
        "--envpool_thread_affinity_offset",
        default=-1,
        type=int,
        help="The start id of binding thread. -1 means not to use thread affinity in thread pool. More information: https://envpool.readthedocs.io/en/latest/content/python_interface.html",
    )
