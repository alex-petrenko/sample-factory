[//]: # (# Running Experiments)

[//]: # ()
[//]: # (Here we provide command lines that can be used to reproduce the experiments from the paper, which also serve as an example on how to configure large-scale RL experiments.)

[//]: # ()
[//]: # ()
[//]: # (#### DMLab)

[//]: # (##### DMLab level cache)

[//]: # ()
[//]: # (Note `--dmlab_level_cache_path` parameter. This location will be used for level layout cache.)

[//]: # (Subsequent DMLab experiments on envs that require level generation will become faster since environment files from)

[//]: # (previous runs can be reused.)

[//]: # ()
[//]: # (Generating environment levels for the first time can be really slow, especially for the full multi-task)

[//]: # (benchmark like DMLab-30. On 36-core server generating enough environments for a 10B training session can take up to)

[//]: # (a week. We provide a dataset of pre-generated levels to make training on DMLab-30 easier.)

[//]: # ([Download here]&#40;https://drive.google.com/file/d/17JCp3DbuiqcfO9I_yLjbBP4a7N7Q4c2v/view?usp=sharing&#41;.)

[//]: # ()


[//]: # (### Caveats)

[//]: # ()
[//]: # (- Multiplayer VizDoom environments can freeze your console sometimes, simple `reset` takes care of this)

[//]: # (- Sometimes VizDoom instances don't clear their internal shared memory buffers used to communicate between Python and)

[//]: # (a Doom executable. The file descriptors for these buffers tend to pile up. `rm /dev/shm/ViZDoom*` will take care of this issue.)

[//]: # (- It's best to use the standard `--fps=35` to visualize VizDoom results. `--fps=0` enables)

[//]: # (Async execution mode for the Doom environments, although the results are not always reproducible between sync and async modes.)

[//]: # (- Multiplayer VizDoom environments are significantly slower than single-player envs because actual network)

[//]: # (communication between the environment instances is required which results in a lot of syscalls.)

[//]: # (For prototyping and testing consider single-player environments with bots instead.)

