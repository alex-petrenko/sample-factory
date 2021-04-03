## Setup instructions

Follow instructions in the repo to install Mujoco environments

- Download [MuJoCo200 linux](https://www.roboti.us/download/mujoco200_linux.zip)

- Unzip mujoco200_linux.zip file to $HOME/.mujoco/mjpro200/

- Export Environment Variable

```
vim ~/.bashrc

paste command below to above file
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro200/bin
```

- If you have MuJoCo Institutional License, just skip this step.
  
  Get 30-day Trial license or MuJoCo Personal License [link](https://www.roboti.us/license.html)

- Put the Mujoco key under $HOME/.mujoco/

- Install mujoco-py environments

```
pip install mujoco-py
```

- If you can not run above command successfully, then you can try commands below.

```
cd ~/Downloads
git clone https://github.com/openai/mujoco-py.git
bash -c "cd mujoco-py && conda activate sample-factory && python setup.py install"
```

- Install extra dependencies to support evaluation and rendering

```
sudo apt-get update -y
sudo apt-get install -y patchelf
sudo apt-get install -y libglew-dev
```
