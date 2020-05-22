## Setup instructions

1) Complete all system-wide installation, including VizDoom dependencies

2) Create conda env: `conda env create -f rlpyt_linux_cuda10.yml`

3) Activate conda env: `conda activate rlpyt`

4)

```
cd ~/
git clone https://github.com/astooke/rlpyt.git

# last commit at the time
cd ~/rlpyt
git checkout 75e96cda433626868fd2a30058be67b99bbad810

pip install -e .
```

```
COMMENT: to use alternating sampler here we had to comment lines 126-127 in samplers/parallel/gpu/action_server.py
if "bootstrap_value" in self.samples_np.agent:
    self.bootstrap_value_pair[alt][:] = self.agent.value(*agent_inputs_pair[alt])
otherwise it crashes
```

5) Run one of the scripts in this folder 