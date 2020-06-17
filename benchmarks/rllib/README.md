## Setup instructions

1) Use standard sample-factory Conda environment setup (main README file for instructions)

2) Install Ray from the wheel you will find next to this readme:

```
pip install ray-0.8.0.dev2-cp37-cp37m-manylinux1_x86_64.whl

```

This is the version the scripts were written for.

3) Example command line for dmlab throughput test:

```
python -m benchmarks.rllib.train_rllib -f=benchmarks/rllib/rllib_fps_benchmark/doom-impala-fps.yaml --experiment-name=rllib-impala-fps --config='{"config": {"num_workers": 20, "num_envs_per_worker": 16}}'
```