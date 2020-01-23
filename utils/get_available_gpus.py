import sys


def get_available_gpus_without_triggering_pytorch_cuda_initialization(envvars):
    import subprocess
    out = subprocess.run([sys.executable, '-m', 'utils.get_available_gpus'], capture_output=True, env=envvars)
    text_output = out.stdout.decode()
    from utils.utils import log
    log.debug('Queried available GPUs: %s', text_output)
    return text_output


def main():
    import torch
    device_count = torch.cuda.device_count()
    available_gpus = ','.join(str(g) for g in range(device_count))
    print(available_gpus)
    return 0


if __name__ == '__main__':
    sys.exit(main())
