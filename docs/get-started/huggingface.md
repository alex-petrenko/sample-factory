# HuggingFace Hub Integration

Sample Factory has integrations with HuggingFace Hub to push models with evaluation results and training metrics to the hub. 

### Downloading Models

To download models from the HuggingFace Hub, use `git lfs` to clone the repo into your `train_dir`

1. Download `git lfs` with `sudo apt install git-lfs`
2. `git lfs install`
3. Download the repo with `git clone `

Afterwards, you can run the models in the repo with the enjoy script corresponding to your environment.

Note, you may have to specify the `--train_dir` if your local train_dir has a different path than the one in the `cfg.json`

### Uploading Models

Before uploading models to the hub, log in to huggingface with `huggingface-cli login`

Upload your model with your environment's enjoy script with the `--push_to_hub` flag. Other relevant command line arguments are:
- `--hf_username`: Your HuggingFace username
- `--hf_repository`: The repository to push to
- `--max_num_frames`: Number of frames to evaluate on before uploading. Used to generate evaluation metrics

You can also save a video of the model during evaluation to upload to the hub with the `--save_video` flag
- `--video_frames`: The number of frames to be rendered in the video
- `--video_name`: The name of the video to save as. If `None`, will save to `replay.mp4` in your experiment directory