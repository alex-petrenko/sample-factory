## Setup instructions

1) Complete all system-wide installation, including VizDoom dependencies

2) Create conda env: `conda env create -f rlpyt_linux_cuda10.yml`

3) Activate conda env: `conda activate rlpyt`

4)

```
cd ~/
git clone https://github.com/astooke/rlpyt.git

# last commit at the time
git checkout 75e96cda433626868fd2a30058be67b99bbad810

cd ~/rlpyt
pip install -e .
```