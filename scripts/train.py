import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as f

from nanochat_varun.common import GPTConfig, get_device
from nanochat_varun.gpt import GPT
from nanochat_varun.data_loader import DataLoader

B = 8
T = 1024
max_steps = 100
learning_rate = 6e-4

device = get_device()
torch.set_float32_matmul_precision("high")

train_loader = DataLoader(B=B, T=T)

config = GPTConfig(sequence_len=T, vocab_size=50257, n_layer=8, n_head=8, n_kv_head=4, n_embd=256)
gpt = GPT(config)
gpt.to(device)

param_dict = {pn : p for pn, p in gpt.named_parameters() if p.requires_grad}
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

optim_groups = [
    {'params' : decay_params, 'weight_decay' : 0.1},
    {'params' : nodecay_params, 'weight_decay' : 0.0}
]

optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)

x = torch.randint(0, 50257, (B, T + 1))
y = x[:, 1:]
x = x[:, :-1]
x, y = x.to(device), y.to(device)
print(x.shape)
print(y.shape)

# training loop
for step in range(max_steps):
    t0 = time.time()

    # x, y = train_loader.next_batch()
    # x, y = x.to(device), y.to(device)

    with torch.autocast(device_type=device if device != 'mps' else 'cpu', dtype=torch.bfloat16):
        loss = gpt(x, targets=y)

    loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = (t1 - t0) * 1000 # ms
    tokens_per_sec = (B*T) / (t1 - t0)
    print(f"step {step:4d} | loss: {loss.item():.6f} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

print("training complete")