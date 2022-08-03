# HuggingFace Hub Integration

Sample Factory has integrations with HuggingFace Hub to push models with evaluation results and training metrics to the hub. 

### Setting Up

The HuggingFace Hub requires `git lfs` to download model files.

```
sudo apt install git-lfs
git lfs install
```

To upload files to the HuggingFace Hub, you need to log in to your HuggingFace account with:

```
huggingface-cli login
```

As part of the `huggingface-cli login`, you should generate a token with write access at https://huggingface.co/settings/tokens

### Downloading Models

#### Using the load_from_hub Scipt

To download a model from the HuggingFace Hub to use with Sample-Factory, use the `load_from_hub` script:

```
python -m sample_factory.huggingface.load_from_hub -r <HuggingFace_repo_id> -d <train_dir_path>
```

The command line arguments are:

- `-r`: The repo ID for the HF repository to download. The repo ID should be in the format `<username>/<repo_name>`

- `-d`: An optional argument to specify the directory to save the experiment to. Defaults to `./train_dir` which will save the repo to `./train_dir/<repo_name>`

#### Download Model Repository Directly

HuggingFace repositories can be downloaded directly using `git clone`:

```
git clone <URL of HuggingFace Repo>
```

#### Using Downloaded Models with Sample-Factory

After downloading the model, you can run the models in the repo with the enjoy script corresponding to your environment. For example, if you are downloading a `mujoco-ant` model, it can be run with:

```
python -m sf_examples.mujoco_examples.enjoy_mujoco --algo=APPO --env=mujoco_ant --experiment=<repo_name> --train_dir=./train_dir
```

Note, you may have to specify the `--train_dir` if your local train_dir has a different path than the one in the `cfg.json`

### Uploading Models

#### Using enjoy.py

You can upload your models to the Hub using your environment's `enjoy` script with the `--push_to_hub` flag. Uploading using `enjoy` can also generate evaluation metrics and a replay video.

Other relevant command line arguments are:

- `--hf_username`: Your HuggingFace username

- `--hf_repository`: The repository to push to. The model will be saved to `https://huggingface.co/hf_username/hf_repository`

- `--max_num_episodes`: Number of episodes to evaluate on before uploading. Used to generate evaluation metrics. It is recommended to use multiple episodes to generate an accurate mean and std.

- `--max_num_frames`: Number of episodes to evaluate on before uploading. An alternative to `max_num_episodes`

- `--no_render`: A flag that disables rendering and showing the environment steps. It is recommended to set this flag to speed up the evaluation process.

You can also save a video of the model during evaluation to upload to the hub with the `--save_video` flag

- `--video_frames`: The number of frames to be rendered in the video. Defaults to -1 which renders an entire episode

- `--video_name`: The name of the video to save as. If `None`, will save to `replay.mp4` in your experiment directory

For example:

```
python -m sf_examples.mujoco_examples.enjoy_mujoco --algo=APPO --env=mujoco_ant --experiment=<repo_name> --train_dir=./train_dir --max_num_episodes=10 --push_to_hub --hf_username=<username> --hf_repository=<hf_repo_name> --save_video --no_render
```

#### Using the push_to_hub Script

If you want to upload without generating evaluation metrics or a replay video, you can use the `push_to_hub` script:

```
python -m sample_factory.huggingface.push_to_hub -r <hf_repo_name> -u <hf_username> -d <experiment_dir_path>
```

The command line arguments are:

- `-r`: The name of the repo to save on HF Hub. This is the same as `hf_repository` in the enjoy script

- `-u`: Your HuggingFace username

- `-d`: The full path to your experiment directory to upload