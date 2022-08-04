import os

import cv2
import numpy as np
from huggingface_hub import HfApi, Repository, repocard, upload_file, upload_folder

from sample_factory.utils.utils import log, project_tmp_dir

MIN_FRAME_SIZE = 180


def generate_replay_video(dir_path: str, frames: list, fps: int):
    tmp_name = os.path.join(project_tmp_dir(), "replay.mp4")
    video_name = os.path.join(dir_path, "replay.mp4")
    frame_size = (frames[0].shape[1], frames[0].shape[0])
    resize = False

    if min(frame_size) < MIN_FRAME_SIZE:
        resize = True
        scaling_factor = MIN_FRAME_SIZE / min(frame_size)
        frame_size = (int(frame_size[0] * scaling_factor), int(frame_size[1] * scaling_factor))

    video = cv2.VideoWriter(tmp_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    for frame in frames:
        if resize:
            frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
        video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    video.release()
    os.system(f"ffmpeg -y -i {tmp_name} -vcodec libx264 {video_name}")


def generate_model_card(dir_path: str, algo: str, env: str, rewards: list = None):
    readme_path = os.path.join(dir_path, "README.md")

    readme = f"""
A(n) **{algo}** model trained on the **{env}** environment.
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

    if rewards is not None:
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        eval = repocard.metadata_eval_result(
            model_pretty_name=algo,
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=env,
            dataset_id=env,
        )

        metadata = {**metadata, **eval}

    repocard.metadata_save(readme_path, metadata)


def push_to_hf(dir_path: str, repo_name: str, num_policies: int = 1):
    repo_url = HfApi().create_repo(
        repo_id=repo_name,
        private=False,
        exist_ok=True,
    )

    # Upload folders
    folders = [".summary"]
    for policy_id in range(num_policies):
        folders.append(f"checkpoint_p{policy_id}")
    for f in folders:
        if os.path.exists(os.path.join(dir_path, f)):
            upload_folder(
                repo_id=repo_name,
                folder_path=os.path.join(dir_path, f),
                path_in_repo=f,
            )

    # Upload files
    files = ["cfg.json", "README.md", "replay.mp4"]
    for f in files:
        if os.path.exists(os.path.join(dir_path, f)):
            upload_file(
                repo_id=repo_name,
                path_or_fileobj=os.path.join(dir_path, f),
                path_in_repo=f,
            )

    log.info(f"The model has been pushed to {repo_url}")


def load_from_hf(dir_path: str, repo_id: str):
    temp = repo_id.split("/")
    repo_name = temp[1]

    local_dir = os.path.join(dir_path, repo_name)
    Repository(local_dir, repo_id)
    log.info(f"The repository {repo_id} has been cloned to {local_dir}")
