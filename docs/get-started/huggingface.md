# HuggingFace Hub Integration

Sample Factory has integrations with HuggingFace Hub to push models with evaluation results and training metrics to the hub. 

### Setting Up

The HuggingFace Hub requires `git lfs` to download model files.

```
sudo apt install git-lfs
git lfs install
```

To upload files to the HuggingFace Hub, you need to log in to your HuggingFace account with:

```huggingface-cli login```

### Downloading Models

#### Using the load_from_hub Scipt

To download a model from the HuggingFace Hub to use with Sample-Factory, use the `load_from_hub` script:

```python -m sample_factory.huggingface.load_from_hub -r "HuggingFace Repo ID" -d "Path to train_dir"```

The command line arguments are:
- `-r`: The repo ID for the HF repository to download. The repo ID should be in the format `"username"/"repo_name"`
- `-d`: An optional argument to specify the directory to save the experiment to. Defaults to `./train_dir` which will save the repo to `./train_dir/<repo_name>`

#### Download Model Repository Directly

HuggingFace repositories can be downloaded directly using `git clone`:

```git clone "URL of HuggingFace Repo"```

#### Using Downloaded Models with Sample-Factory

After downloading the model, you can run the models in the repo with the enjoy script corresponding to your environment. For example, if you are downloading a `mujoco-ant` model, it can be run with:

```python -m sf_examples.mujoco_examples.enjoy_mujoco --algo=APPO --env=mujoco_ant --experiment="Repo Name" --train_dir=./train_dir```

Note, you may have to specify the `--train_dir` if your local train_dir has a different path than the one in the `cfg.json`

### Uploading Models

#### Using enjoy.py

You can upload your models to the Hub using your environment's `enjoy` script with the `--push_to_hub` flag. Uploading using `enjoy` can also generate evaluation metics and a replay video.

Other relevant command line arguments are:
- `--hf_username`: Your HuggingFace username
- `--hf_repository`: The repository to push to. The model will be saved to `https://huggingface.co/hf_username/hf_repository`
- `--max_num_frames`: Number of frames to evaluate on before uploading. Used to generate evaluation metrics

You can also save a video of the model during evaluation to upload to the hub with the `--save_video` flag
- `--video_frames`: The number of frames to be rendered in the video
- `--video_name`: The name of the video to save as. If `None`, will save to `replay.mp4` in your experiment directory

For example:

```python -m sf_examples.mujoco_examples.enjoy_mujoco --algo=APPO --env=mujoco_ant --experiment="Repo Name" --train_dir=./train_dir --max_num_frames=10000 --push_to_hub --hf_username="User Name" --hf_repository="Repo Name" --save_video --video_frames=500```

#### Using the push_to_hub Script

If you want to upload without generating evaluation metrics or a replay video, you can use the `push_to_hub` script:

```python -m sample_factory.huggingface.push_to_hub -r "Repo Name" -u "User Name" -d "Path to Experiment Folder"```

The command line arguments are:
- `-r`: The name of the repo to save on HF Hub. This is the same as `hf_repository` in the enjoy script
- `-u`: Your HuggingFace username
- `-d`: The full path to your experiment directory to upload