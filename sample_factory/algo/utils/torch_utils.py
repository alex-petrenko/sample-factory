import torch


def init_torch_runtime(cfg):
    torch.multiprocessing.set_sharing_strategy('file_system')
    if cfg.device != 'gpu':
        return
    torch.backends.cudnn.benchmark = True


def inference_context(is_serial):
    if is_serial:
        # in serial mode we use the same tensors on sampler and learner
        return torch.no_grad()
    else:
        return torch.inference_mode()
