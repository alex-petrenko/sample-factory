import argparse
import json
import os

from sample_factory.huggingface.huggingface_utils import generate_model_card, push_to_hf
from sample_factory.utils.attr_dict import AttrDict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--hf_repository", help="Name of the model repository from the Hugging Face Hub", type=str
    )
    parser.add_argument("-u", "--hf_username", help="Username for the Hugging Face Hub", type=str)
    parser.add_argument("-d", "--experiment_dir", help="Path to your experiment directory", type=str)
    args = parser.parse_args()

    cfg_file = os.path.join(args.experiment_dir, "cfg.json")
    with open(cfg_file, "r") as json_file:
        json_params = json.load(json_file)
        cfg = AttrDict(json_params)

    generate_model_card(args.experiment_dir, cfg.algo, cfg.env)
    push_to_hf(args.experiment_dir, f"{args.hf_username}/{args.hf_repository}", cfg.num_policies)


if __name__ == "__main__":
    main()
