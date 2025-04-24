import sys

import torch.nn as nn

from sample_factory.utils.utils import log


def interleave(*args):
    return [val for pair in zip(*args) for val in pair]


def freeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def freeze_batch_norm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            module.trainable = False
            module.track_running_stats = False


def freeze_selected(step, cfg, model, models_frozen):
    for module_name, module_freeze in cfg.freeze.items():
        module_unfreeze = cfg.unfreeze.get(module_name, sys.maxsize)
        if step >= module_freeze and step <= module_unfreeze and not models_frozen[module_name]:
            freeze(getattr(model, module_name))
            log.debug(f"Frozen {module_name}.")
            models_frozen[module_name] = True

            if cfg.freeze_batch_norm:
                freeze_batch_norm(getattr(model, module_name))


def unfreeze(model):
    for name, param in model.named_parameters():
        param.requires_grad = True


def unfreeze_batch_norm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.train()
            module.trainable = True
            module.track_running_stats = True


def unfreeze_selected(step, cfg, model, models_frozen):
    for module_name, module_unfreeze in cfg.unfreeze.items():
        if step >= module_unfreeze and models_frozen[module_name]:
            unfreeze(getattr(model, module_name))
            log.debug(f"Unfrozen {module_name}.")
            models_frozen[module_name] = False

            if cfg.freeze_batch_norm:
                unfreeze_batch_norm(getattr(model, module_name))


def replace_batchnorm_with_layernorm(module):
    for i, child in enumerate(module.children()):
        prevchild = None
        for name, subchild in child.named_children():
            if prevchild is not None and isinstance(subchild, nn.BatchNorm2d):
                num_features = prevchild.output_shape[1:]
                new_child = nn.LayerNorm(num_features)
                setattr(child, name, new_child)
            prevchild = subchild

        replace_batchnorm_with_layernorm(child)


def inject_layernorm_before_activation(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Sequential):
            new_children = []
            for i, (sub_name, sub_child) in enumerate(child.named_children()):
                # Check if the next layer is an activation function and the current layer is not LayerNorm
                new_children.append(sub_child)
                if (
                    i + 1 < len(child)
                    and isinstance(child[i + 1], (nn.ELU, nn.ReLU, nn.Tanh))
                    and not isinstance(child[i], (nn.LayerNorm, nn.BatchNorm2d))
                ):
                    # Inject LayerNorm before activation function
                    num_features = sub_child.output_shape[1:]
                    new_layer = nn.LayerNorm(num_features)
                    new_children.append(new_layer)
            setattr(module, name, nn.Sequential(*new_children))
        else:
            inject_layernorm_before_activation(child)


def sequential_layernorm(module):
    # Initialize an empty list to hold the new layers
    new_layers = []

    # Iterate through each layer in the original module
    for i, layer in enumerate(module):
        # Add the original layer to the new layers list
        new_layers.append(layer)

        # Check if the current layer is a Linear layer
        if (
            i + 1 < len(module)
            and isinstance(module[i + 1], (nn.ELU, nn.ReLU, nn.Tanh))
            and not isinstance(module[i], nn.LayerNorm)
        ):
            # Inject LayerNorm before activation function
            num_features = layer.output_shape[1:]
            new_layer = nn.LayerNorm(num_features)
            new_layers.append(new_layer)

    # Use nn.Sequential to create a new module from the new layers list
    new_module = nn.Sequential(*new_layers)
    return new_module


def linear_layernorm(module):
    new_layers = []
    num_features = module.in_features
    new_layers.append(nn.LayerNorm([num_features]))
    new_layers.append(module)
    new_module = nn.Sequential(*new_layers)
    return new_module


def model_layernorm(model):
    replace_batchnorm_with_layernorm(model)
    inject_layernorm_before_activation(model)


def scale_width_critic(module, factor=2):
    for child_name, child in list(module.named_children()):
        if not child_name.startswith("critic"):
            continue

        for name, subchild in child.named_children():
            if isinstance(subchild, (nn.Conv1d, nn.Conv2d)):
                new_in_channels = int(subchild.in_channels * factor)
                new_out_channels = int(subchild.out_channels * factor)

                new_layer = subchild.__class__(
                    new_in_channels,
                    new_out_channels,
                    kernel_size=subchild.kernel_size,
                    bias=subchild.bias is not None,
                    stride=subchild.stride,
                    padding=subchild.padding,
                    dilation=subchild.dilation,
                )
                setattr(child, name, new_layer)

            elif isinstance(subchild, nn.Linear):
                new_in_features = int(subchild.in_features * factor)
                new_out_features = int(subchild.out_features * factor)

                new_layer = nn.Linear(new_in_features, new_out_features, bias=subchild.bias is not None)
                setattr(child, name, new_layer)

            elif isinstance(subchild, nn.BatchNorm2d):
                new_layer = nn.BatchNorm2d(int(subchild.num_features * factor))
                setattr(child, name, new_layer)

            elif isinstance(subchild, nn.LSTM):
                new_layer = nn.LSTM(
                    int(subchild.input_size * factor), int(subchild.hidden_size * factor), subchild.num_layers
                )
                setattr(child, name, new_layer)

        scale_width(child, factor=factor)


def scale_width(module, factor=2):
    for child in list(module.children()):
        for name, subchild in child.named_children():
            if isinstance(subchild, (nn.Conv1d, nn.Conv2d)):
                new_in_channels = int(subchild.in_channels * factor)
                new_out_channels = int(subchild.out_channels * factor)

                new_layer = subchild.__class__(
                    new_in_channels,
                    new_out_channels,
                    kernel_size=subchild.kernel_size,
                    bias=subchild.bias is not None,
                    stride=subchild.stride,
                    padding=subchild.padding,
                    dilation=subchild.dilation,
                )
                setattr(child, name, new_layer)

            elif isinstance(subchild, nn.Linear):
                new_in_features = int(subchild.in_features * factor)
                new_out_features = int(subchild.out_features * factor)

                new_layer = nn.Linear(new_in_features, new_out_features, bias=subchild.bias is not None)
                setattr(child, name, new_layer)

            elif isinstance(subchild, nn.BatchNorm2d):
                new_layer = nn.BatchNorm2d(int(subchild.num_features * factor))
                setattr(child, name, new_layer)

            elif isinstance(subchild, nn.LSTM):
                new_layer = nn.LSTM(
                    int(subchild.input_size * factor), int(subchild.hidden_size * factor), subchild.num_layers
                )
                setattr(child, name, new_layer)

        scale_width(child, factor=factor)


def downscale_input_layer(module, name, factor=2):
    cur_layer = getattr(module, name)

    if isinstance(cur_layer, (nn.Conv1d, nn.Conv2d)):
        new_layer = cur_layer.__class__(
            int(cur_layer.in_channels / factor),
            cur_layer.out_channels,
            kernel_size=cur_layer.kernel_size,
            bias=cur_layer.bias is not None,
            stride=cur_layer.stride,
            padding=cur_layer.padding,
            dilation=cur_layer.dilation,
        )
        setattr(module, name, new_layer)
    elif isinstance(cur_layer, nn.Linear):
        new_layer = nn.Linear(
            int(cur_layer.in_features / factor), cur_layer.out_features, bias=cur_layer.bias is not None
        )
        setattr(module, name, new_layer)


def downscale_output_layer(module, name, factor=2):
    cur_layer = getattr(module, name)

    if isinstance(cur_layer, (nn.Conv1d, nn.Conv2d)):
        new_layer = cur_layer.__class__(
            cur_layer.in_channels,
            int(cur_layer.out_channels / factor),
            kernel_size=cur_layer.kernel_size,
            bias=cur_layer.bias is not None,
            stride=cur_layer.stride,
            padding=cur_layer.padding,
            dilation=cur_layer.dilation,
        )
        setattr(module, name, new_layer)
    elif isinstance(cur_layer, nn.Linear):
        new_layer = nn.Linear(
            cur_layer.in_features, int(cur_layer.out_features / factor), bias=cur_layer.bias is not None
        )
        setattr(module, name, new_layer)


def reduce_input_layer(module, name, amount=121):
    cur_layer = getattr(module, name)

    if isinstance(cur_layer, (nn.Conv1d, nn.Conv2d)):
        new_layer = cur_layer.__class__(
            int(cur_layer.in_channels - amount),
            cur_layer.out_channels,
            kernel_size=cur_layer.kernel_size,
            bias=cur_layer.bias is not None,
            stride=cur_layer.stride,
            padding=cur_layer.padding,
            dilation=cur_layer.dilation,
        )
        setattr(module, name, new_layer)
    elif isinstance(cur_layer, nn.Linear):
        new_layer = nn.Linear(
            int(cur_layer.in_features - amount), cur_layer.out_features, bias=cur_layer.bias is not None
        )
        setattr(module, name, new_layer)


def reduce_output_layer(module, name, amount=121):
    cur_layer = getattr(module, name)

    if isinstance(cur_layer, (nn.Conv1d, nn.Conv2d)):
        new_layer = cur_layer.__class__(
            cur_layer.in_channels,
            int(cur_layer.out_channels - amount),
            kernel_size=cur_layer.kernel_size,
            bias=cur_layer.bias is not None,
            stride=cur_layer.stride,
            padding=cur_layer.padding,
            dilation=cur_layer.dilation,
        )
        setattr(module, name, new_layer)
    elif isinstance(cur_layer, nn.Linear):
        new_layer = nn.Linear(
            cur_layer.in_features, int(cur_layer.out_features - amount), bias=cur_layer.bias is not None
        )
        setattr(module, name, new_layer)
