import torch

def get_device(verbose: bool = True) -> torch.device:
    """CUDA, MPS, CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name(device)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Device: {device}")

    return device