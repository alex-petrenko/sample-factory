import argparse
import os
from os.path import join

from sample_factory.utils.utils import str2bool


def hipposlam_override_defaults(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(
        encoder_conv_architecture="convnet_impala",
        obs_subtract_mean=0.0,
        obs_scale=255.0,
        env_frameskip=4,
        nonlinearity="relu",
        rollout=32,
        recurrence=32,
        rnn_type="lstm",
        rnn_size=256,
        num_epochs=1,
        # if observation normalization is used, it is important that we do not normalize INSTRUCTIONS observation
        normalize_input_keys=["obs"],
        decoder_mlp_layers=[128,128],
    )


def add_hipposlam_env_args(parser: argparse.ArgumentParser) -> None:
    p = parser
    p.add_argument("--Hippo_n_feature", default=64, type=int, help="number of sequences/features")
    p.add_argument("--Hippo_R", default=8, type=int, help="number of repeats in a sequence")
    p.add_argument("--Hippo_L", default=48, type=int, help="sequence length")

    p.add_argument("--simple_sequence", default=False, type=bool, help="simple sequence, simply shrinking feature dimensions and expanding features to include their history")
    p.add_argument("--core_name", default=None, type=str, help="simple sequence, simply shrinking feature dimensions and expanding features to include their history")
    p.add_argument("--encoder_name", default=None, type=str, help="actually using dmlab encoders")
    p.add_argument("--encoder_load_path", default=None, type=str, help="if loading encoder, the path")


    p.add_argument("--DG_lr", default=None, type=float, help="Dentate Gyrus Pattern separation learning rate")
    p.add_argument("--DG_temperature", default=None, type=float, help="Dentate Gyrus output temperature")
    p.add_argument("--DG_batch_q", default=None, type=bool, help="Dentate Gyrus batch quantile, momentum 0.2, quantile 0.98")
    p.add_argument("--DG_softmax", default=None, type=bool, help="Dentate Gyrus softmax")
    p.add_argument("--DG_name", default=None, type=str, help="model name for the last encoder layer, i.e. Dentate Gyrus")
    p.add_argument("--DG_detect", default=None, type=float, help="batch novelty detection threshold (to activate a sequence)")
    p.add_argument("--DG_novelty", default=None, type=float, help="batch novelty novelty threshold to store a new pattern")
    # p.add_argument("--dense", default=None, type=bool, help="whether encoder gives additional dense output")
    p.add_argument("--head_l1_coef", default=None, type=float, help="L1 penalty to encoder output")
    p.add_argument("--fix_encoder_when_load", default=True, type=bool, help="when loading an encoder, fix its weights at initialization")
    p.add_argument("--depth_sensor", default=False, type=bool, help="having extra depth sensor")
    p.add_argument("--dmlab_reduced_action_set", default=False, type=bool, help="reduced action set to facilitate learning")
    p.add_argument("--with_number_instruction", default=True, type=str2bool, help="instruction input is number, e.g. 1-3")
    p.add_argument("--number_instruction_coef", default=1, type=float, help="instruction strength")
    p.add_argument("--DG_BN_intercept", default=2, type=float, help="instruction strength")
    p.add_argument("--with_pos_obs", default=False, type=str2bool, help="get the true position of agent")
    p.add_argument("--use_jit", default=True, type=str2bool, help="use jit / pytorch script to accelerate decoder. disable it for hooking")

    p.add_argument("--refractory", default=0, type=int, help="when using bypassSS_binary, determine whether to block reentry and how much the refractory. 0: no refractory, -1: entire sequence")
    p.add_argument("--bypassLSTM_hidden", default=167, type=int, help="LSTM hidden size to match the number of learnable parameters in SS, default: matching L64, R8")    
    p.add_argument("--fix_DG", default=False, type=str2bool, help="fix DG weights after creating DG_projection obj")


    

