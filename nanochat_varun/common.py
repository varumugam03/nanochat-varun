import torch
from dataclasses import dataclass

def get_device():
    # CUDA --> MPS --> CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"

    print(f"Detected {device_type} device")
    return device_type


@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key-value heads (GQA)
    n_embd: int = 768

