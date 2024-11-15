"""
An example that shows how to export a SampleFactory model to the ONNX format.

Example command line for CartPole-v1 that exports to "./example_gym_cartpole-v1.onnx"
python -m sf_examples.export_onnx_gym_env --experiment=example_gym_cartpole-v1 --env=CartPole-v1 --use_rnn=False

"""

import sys

from sample_factory.export_onnx import export_onnx
from sf_examples.train_gym_env import parse_custom_args, register_custom_components


def main():
    register_custom_components()
    cfg = parse_custom_args(evaluation=True)
    status = export_onnx(cfg, f"{cfg.experiment}.onnx")
    return status


if __name__ == "__main__":
    sys.exit(main())
