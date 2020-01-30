## Setup instructions

Please use our forked "scalable_agent" repo to reproduce the measurements
https://github.com/alex-petrenko/scalable_agent


Environment setup:

```
Conda env with Python 3.6 (not 3.7)                                                                                                                                                                                                                                                
Install conda install -c anaconda cudatoolkit=9.0
Install pip install tensorflow-gpu==1.9.0                                                                                                                                                                                                                                          
pip install dm_env                                                                                                                                                                                                                                                                 

```

Upgrade bazel to 2.0.0 to build DMLAB