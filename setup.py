import platform

import setuptools
from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()
    descr_lines = long_description.split("\n")
    descr_no_gifs = []  # gifs are not supported on PyPI web page
    for dl in descr_lines:
        if not ("<img src=" in dl and "gif" in dl):
            descr_no_gifs.append(dl)

    long_description = "\n".join(descr_no_gifs)


_atari_deps = ["gymnasium[atari, accept-rom-license]"]
_mujoco_deps = ["gymnasium[mujoco]", "mujoco<2.5"]
_nethack_deps = [
    "numba ~= 0.58",
    "pandas ~= 2.1",
    "matplotlib ~= 3.8",
    "seaborn ~= 0.12",
    "scipy ~= 1.11",
    "shimmy",
    "tqdm ~= 4.66",
    "debugpy ~= 1.6",
]
_envpool_deps = ["envpool"]
_pettingzoo_deps = ["pettingzoo[classic]"]
_onnx_deps = ["onnx", "onnxruntime"]

_docs_deps = [
    "mkdocs-material",
    "mkdocs-minify-plugin",
    "mkdocs-redirects",
    "mkdocs-git-revision-date-localized-plugin",
    "mkdocs-git-committers-plugin-2",
    "mkdocs-git-authors-plugin",
]


def is_macos():
    return platform.system() == "Darwin"


setup(
    # Information
    name="sample-factory",
    description="High throughput asynchronous reinforcement learning framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version="2.1.3",
    url="https://github.com/alex-petrenko/sample-factory",
    author="Aleksei Petrenko",
    license="MIT",
    keywords="asynchronous reinforcement learning policy gradient ppo appo impala ai",
    project_urls={
        "Github": "https://github.com/alex-petrenko/sample-factory",
        "Videos": "https://sites.google.com/view/sample-factory",
    },
    install_requires=[
        "numpy>=1.18.1,<2.0",
        "torch>=1.9,<3.0,!=1.13.0",
        "gymnasium>=0.27,<1.0",
        "pyglet",  # gym dependency
        "tensorboard>=1.15.0",
        "tensorboardx>=2.0",
        "psutil>=5.7.0",
        "threadpoolctl>=2.0.0",
        "colorlog",
        # "faster-fifo>=1.4.2,<2.0",  <-- installed by signal-slot-mp
        "signal-slot-mp>=1.0.3,<2.0",
        "filelock",
        "wandb>=0.12.9",
        "huggingface-hub>=0.10.0,<1.0",
        "pandas",
        "opencv-python",
    ],
    extras_require={
        # some tests require Atari and Mujoco so let's make sure dev environment has that
        "dev": ["black", "isort>=5.12", "pytest<8.0", "flake8", "pre-commit", "twine"]
        + _docs_deps
        + _atari_deps
        + _mujoco_deps
        + _onnx_deps
        + _pettingzoo_deps,
        "atari": _atari_deps,
        "envpool": _envpool_deps,
        "mujoco": _mujoco_deps,
        "nethack": _nethack_deps,
        "onnx": _onnx_deps,
        "pettingzoo": _pettingzoo_deps,
        "vizdoom": ["vizdoom<2.0", "gymnasium[classic_control]"],
        # "dmlab": ["dm_env"],  <-- these are just auxiliary packages, the main package has to be built from sources
    },
    package_dir={"": "./"},
    packages=setuptools.find_packages(where="./", include=["sample_factory*", "sf_examples*"]),
    include_package_data=True,
    python_requires=">=3.8",
)
