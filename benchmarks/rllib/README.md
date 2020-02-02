## Setup instructions

1) conda create --name rllib python=3.7

2)
```
conda install -c anaconda cudatoolkit=10.0
conda install -c conda-forge opencv=4.1.0
conda install pytorch=1.3.1 -c pytorch
```

3) install main project dependencies:

```
pip install -r requirements.txt
```

4) From the root of the repo:

```
pip install setup/ray-0.8.0.dev2-cp37-cp37m-manylinux1_x86_64.whl

```

This is the version the scripts were written for.

5) Example command line for dmlab throughput test:

```
python -m benchmarks.rllib.train_rllib -f=benchmarks/rllib/rllib_fps_benchmark/dmlab-impala-fps.yaml --experiment-name=rllib-impala-fps --config='{"config": {"num_workers": 20, "num_envs_per_worker": 32, "train_batch_size": 2048}}'
```