import argparse

from sample_factory.huggingface.huggingface_utils import load_from_hf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--hf_repository",
        help="Repo id of the model repository from the Hugging Face Hub in the form user_name/repo_name",
        type=str,
    )
    parser.add_argument(
        "-d",
        "--train_dir",
        help="Local destination of the repository. Will save repo to train_dir/repo_name",
        type=str,
        default="./train_dir",
    )
    args = parser.parse_args()

    load_from_hf(args.train_dir, args.hf_repository)


if __name__ == "__main__":
    main()
