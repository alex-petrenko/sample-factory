import argparse
import json
import os
import sys

from sample_factory.huggingface.huggingface_utils import generate_model_card, push_to_hf
from sample_factory.utils.attr_dict import AttrDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--hf_repository",
        help="The full repo_id to push to on the HuggingFace Hub. Must be of the form <username>/<repo_name>",
        type=str,
    )
    parser.add_argument("-d", "--experiment_dir", help="Path to your experiment directory", type=str)
    parser.add_argument(
        "--train_script",
        default=None,
        type=str,
        help="Module name used to run training script. Used to generate HF model card",
    )
    parser.add_argument(
        "--enjoy_script",
        default=None,
        type=str,
        help="Module name used to run training script. Used to generate HF model card",
    )
    args = parser.parse_args()

    cfg_file = os.path.join(args.experiment_dir, "config.json")

    if not os.path.isfile(cfg_file):
        old_cfg_file = os.path.join(args.experiment_dir, "cfg.json")
        if os.path.isfile(old_cfg_file):
            os.rename(old_cfg_file, cfg_file)
        else:
            raise Exception(
                f"Could not load saved parameters for experiment {args.experiment_dir} "
                f"(file {cfg_file} not found). Check that you have the correct experiment directory."
            )

    with open(cfg_file, "r") as json_file:
        json_params = json.load(json_file)
        cfg = AttrDict(json_params)

    generate_model_card(
        args.experiment_dir,
        cfg.algo,
        cfg.env,
        args.hf_repository,
        enjoy_name=args.enjoy_script,
        train_name=args.train_script,
    )
    push_to_hf(args.experiment_dir, args.hf_repository)


if __name__ == "__main__":
    sys.exit(main())
