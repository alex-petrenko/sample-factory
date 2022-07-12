from huggingface_hub import HfApi, upload_folder, repocard, snapshot_download
import os
import json
from argparse import ArgumentParser


def generate_metadata():
    metadata = {}
    metadata["library_name"] = "sample-factory"
    metadata["tags"] = [
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "sample-factory",
    ]
    return metadata


def generate_model_card(dir_path : str):
    readme_path = dir_path + "/README.md"
    cfg_path = dir_path + "/cfg.json"
    if os.path.exists(readme_path):
        # Skip creating README if one already exists
        return
    
    with open(cfg_path, "r") as json_file:
        cfg = json.load(json_file)

    readme = f"""
A(n) **{cfg["algo"]}** model trained on the **{cfg["env"]}** environment.
This model was trained using Sample Factory 2.0: https://github.com/alex-petrenko/sample-factory
    """    

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)
    
    metadata = generate_metadata()
    repocard.metadata_save(readme_path, metadata)



def push_model_to_new_repo(dir_path : str, repo_name : str):
    HfApi().create_repo(
        repo_id=repo_name,
        private=False,
        exist_ok=True,
    )

    generate_model_card(dir_path)

    upload_folder(
        repo_id=repo_name,
        folder_path=dir_path,
        path_in_repo="",
    )


if __name__ == "__main__":
    # Login with `huggingface-cli login`

    p = ArgumentParser()
    p.add_argument("-d", "--directory", help="Directory of experiment")
    p.add_argument("-u", "--username", help="Huggingface username")
    p.add_argument("-r", "--repository", help="Name of Huggingface repository")
    args = p.parse_args()
    push_model_to_new_repo(args.directory, f"{args.username}/{args.repository}")
