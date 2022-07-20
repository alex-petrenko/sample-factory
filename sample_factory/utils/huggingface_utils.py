import os

import cv2
import numpy as np
from huggingface_hub import HfApi, repocard, upload_file, upload_folder

from sample_factory.utils.utils import experiment_dir


def generate_replay_video(dir_path: str, frames: list, fps: int):
    tmp_name = "/tmp/replay.mp4"
    video_name = os.path.join(dir_path, "replay.mp4")
    frame_size = (frames[0].shape[0], frames[0].shape[1])

    video = cv2.VideoWriter(tmp_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    for frame in frames:
        video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    video.release()
    os.system(f"ffmpeg -y -i {tmp_name} -vcodec libx264 {video_name}")


def generate_model_card(cfg, rewards: list):
    readme_path = os.path.join(experiment_dir(cfg=cfg), "README.md")

    readme = f"""
A(n) **{cfg.algo}** model trained on the **{cfg.env}** environment.
This model was trained using Sample Factory 2.0: https://github.com/alex-petrenko/sample-factory
    """

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    metadata = {}
    metadata["library_name"] = "sample-factory"
    metadata["tags"] = [
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "sample-factory",
    ]

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    eval = repocard.metadata_eval_result(
        model_pretty_name=cfg.algo,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=cfg.env,
        dataset_id=cfg.env,
    )

    metadata = {**metadata, **eval}

    repocard.metadata_save(readme_path, metadata)


def push_model_to_repo(dir_path: str, repo_name: str):
    HfApi().create_repo(
        repo_id=repo_name,
        private=False,
        exist_ok=True,
    )

    # Upload folders
    folders = ["checkpoint_p0", ".summary"]
    for f in folders:
        upload_folder(
            repo_id=repo_name,
            folder_path=os.path.join(dir_path, f),
            path_in_repo=f,
        )

    # Upload files
    files = ["cfg.json", "README.md", "replay.mp4"]
    for f in files:
        upload_file(
            repo_id=repo_name,
            path_or_fileobj=os.path.join(dir_path, f),
            path_in_repo=f,
        )
