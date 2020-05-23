## Setup instructions

1) Complete all system-wide installation, including VizDoom dependencies

2) Create conda env: `conda env create -f rlpyt_linux_cuda10.yml`

3) Activate conda env: `conda activate rlpyt`

4) To install rlpyt:

```
cd ~/
git clone https://github.com/alex-petrenko/rlpyt.git
cd rlpyt
pip install -e .
```

Then run one of the scripts in this folder 