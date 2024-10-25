import types
from typing import List

import gymnasium as gym
import torch
import torch.nn as nn
import torch.onnx
from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import EnvInfo, extract_env_info
from sample_factory.algo.utils.make_env import BatchedVecEnv
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.enjoy import load_state_dict, make_env
from sample_factory.model.actor_critic import ActorCritic, create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config


class OnnxExporter(nn.Module):
    actor_critic: ActorCritic
    cfg: Config
    env_info: EnvInfo
    rnn_states: Tensor

    def __init__(self, cfg: Config, env_info: EnvInfo, actor_critic: ActorCritic):
        super(OnnxExporter, self).__init__()
        self.cfg = cfg
        self.env_info = env_info
        self.actor_critic = actor_critic

    def forward(self, **obs):
        if self.cfg.use_rnn:
            rnn_states = obs.pop("rnn_states")
        else:
            rnn_states = generate_rnn_states(self.cfg)

        action_mask = obs.pop("action_mask", None)
        normalized_obs = prepare_and_normalize_obs(self.actor_critic, obs)
        policy_outputs = self.actor_critic(normalized_obs, rnn_states, action_mask=action_mask)
        actions = policy_outputs["actions"]
        rnn_states = policy_outputs["new_rnn_states"]

        if self.cfg.eval_deterministic:
            action_distribution = self.actor_critic.action_distribution()
            actions = argmax_actions(action_distribution)

        if actions.ndim == 1:
            actions = unsqueeze_tensor(actions, dim=-1)

        actions = preprocess_actions(self.env_info, actions, to_numpy=False)

        if self.cfg.use_rnn:
            return actions, rnn_states
        else:
            return actions


def create_onnx_exporter(cfg: Config, env: BatchedVecEnv, enable_jit=False) -> OnnxExporter:
    env_info = extract_env_info(env, cfg)
    device = torch.device("cpu")

    if enable_jit:
        actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    else:
        try:
            # HACK: disable torch.jit to avoid the following problem:
            # https://github.com/pytorch/pytorch/issues/47887
            #
            # The other workaround is to use torch.jit.trace, but it requires
            # to change many things of models too
            torch.jit._state.disable()  # type: ignore[reportAttributeAccessIssue]
            actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
        finally:
            torch.jit._state.enable()  # type: ignore[reportAttributeAccessIssue]

    actor_critic.eval()
    actor_critic.model_to_device(device)
    load_state_dict(cfg, actor_critic, device)

    model = OnnxExporter(cfg, env_info, actor_critic)
    return model


def generate_args(space: gym.spaces.Space, batch_size: int = 1):
    args = [unsqueeze_args(sample_space(space)) for _ in range(batch_size)]
    args = [a for a in args if isinstance(a, dict)]
    args = {k: torch.cat(tuple(a[k] for a in args), dim=0) for k in args[0].keys()} if len(args) > 0 else {}
    return args


def generate_rnn_states(cfg):
    return torch.zeros([1, get_rnn_size(cfg)], dtype=torch.float32)


def sample_space(space: gym.spaces.Space):
    if isinstance(space, gym.spaces.Discrete):
        return int(space.sample())
    elif isinstance(space, gym.spaces.Box):
        return torch.from_numpy(space.sample())
    elif isinstance(space, gym.spaces.Dict):
        return {k: sample_space(v) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Tuple):
        return tuple(sample_space(s) for s in space.spaces)
    else:
        raise NotImplementedError(f"Unsupported space type: {type(space)}")


def unsqueeze_args(args):
    if isinstance(args, int):
        return torch.tensor(args).unsqueeze(0)
    if isinstance(args, torch.Tensor):
        return args.unsqueeze(0)
    if isinstance(args, dict):
        return {k: unsqueeze_args(v) for k, v in args.items()}
    elif isinstance(args, tuple):
        return (unsqueeze_args(v) for v in args)
    else:
        raise NotImplementedError(f"Unsupported args type: {type(args)}")


def create_forward(original_forward, arg_names: List[str]):
    args_str = ", ".join(arg_names)

    func_code = f"""
def forward(self, {args_str}):
    bound_args = locals()
    bound_args.pop('self')
    return original_forward(**bound_args)
    """

    globals_vars = {"original_forward": original_forward}
    local_vars = {}
    exec(func_code, globals_vars, local_vars)
    return local_vars["forward"]


def patch_forward(model: OnnxExporter, input_names: List[str]):
    """
    Patch the forward method of the model to dynamically define the input arguments
    since *args and **kwargs are not supported in `torch.onnx.export`

    see also: https://github.com/pytorch/pytorch/issues/96981 and https://github.com/pytorch/pytorch/issues/110439
    """
    forward = create_forward(model.forward, input_names)
    model.forward = types.MethodType(forward, model)


def export_onnx(cfg: Config, f: str) -> int:
    cfg = load_from_checkpoint(cfg)
    env = make_env(cfg)
    model = create_onnx_exporter(cfg, env)
    args = generate_args(env.observation_space)

    # The args dict is mapped to the inputs of the model
    # since usages of dictionaries is not recommended by pytorch
    # see also: https://github.com/pytorch/pytorch/blob/v2.4.1/torch/onnx/utils.py#L768-L772
    input_names = list(args.keys())

    # Append the "output_" prefix to avoid name confliction with the input names
    # that causes to add ".N" suffix to the input names.
    # see also: https://discuss.pytorch.org/t/onnx-export-same-input-and-output-names/93155
    output_names = ["output_actions"]

    if cfg.use_rnn:
        input_names.append("rnn_states")
        output_names.append("output_rnn_states")
        args["rnn_states"] = generate_rnn_states(cfg)

        # batch size must be 1 when rnn is used
        # See also https://github.com/onnx/onnx/issues/3182
        dynamic_axes = None
    else:
        dynamic_axes = {key: {0: "batch_size"} for key in input_names + output_names}

    patch_forward(model, input_names)

    torch.onnx.export(
        model,
        (args,),
        f,
        export_params=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    return ExperimentStatus.SUCCESS
