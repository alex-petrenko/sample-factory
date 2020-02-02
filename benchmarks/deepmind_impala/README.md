## Setup instructions

Please use our forked "scalable_agent" repo to reproduce the measurements
https://github.com/alex-petrenko/scalable_agent


Environment setup:

```
Conda env with Python 3.6 (not 3.7)                                                                                                                                                                                                                                                
Install conda install -c anaconda cudatoolkit=9.0
Install pip install tensorflow-gpu==1.9.0
install lab                                                                                                                                                                                                                                          
pip install dm_env                                                                                                                                                                                                                                                                

```

Example command line for dmlab:

```
python -m experiment --level_name=rooms_collect_good_objects_train --batch_size=64 --unroll_length=32 --num_actors=20 --renderer=software --benchmark_mode=1 --logdir=/tmp/dmlab_agent
```

Upgrade bazel to 2.0.0 to build DMLAB