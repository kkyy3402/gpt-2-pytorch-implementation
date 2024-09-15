# utils/device_setup.py

import torch

def get_device(config):
    use_cuda = config['device'].get('use_cuda', True)
    use_mps = config['device'].get('use_mps', True)
    
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    elif use_mps and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
