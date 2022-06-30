## Installation

### Method 1 - Install from setup.py on sf2

1. Clone sample factory and checkout the sf2 branch and `cd sample-factory`
2. Create new conda environment with python=3.9 `conda create -n {myenv} python=3.9`
3. Install pytorch with `conda install pytorch==1.11.0 torchvision -c pytorch`
4. Install other dependencies with `pip install -e .`
5. Install gym environments `pip install "gym[mujoco]"`
6. `pip install mujoco`
7. `pip install mujoco-py`

### Method 2 - Install from environment.yaml from master (also steps for debugging on existing environment)

1. Clone sample factory and `cd sample-factory`
2. Install the environment `conda env create -f environment.yml` and `conda activate sample-factory`
3. `pip install mujoco`
4. `pip install mujoco-py`
5. Update gym by `pip uninstall gym[mujoco]` then `pip install gym[mujoco]`
6. Update pytorch to version 1.11.0 `conda install pytorch==1.11.0 -c pytorch`
7. Uninstall and reinstall cffi `pip uninstall cffi` then `pip install cffi`