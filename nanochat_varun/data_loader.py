import torch
import tiktoken
import requests
import os
import numpy as np


# REALLY SHITTY DATA LOADER FOR TESTING PARTIAL KARPATHY GPT.PY

class DataLoader:
    def __init__(self, B, T, process_rank=0, num_processes=1):
        self.B = B
        self.T = T

        input_file_path = "input.txt"
        if not os.path.exists(input_file_path):
            data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            with open(input_file_path, "w") as f:
                f.write(requests.get(data_url).text)

        
        with open(input_file_path, "r") as f:
            data = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(data)
        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f" 1 epoch = {len(self.tokens) // self.B // self.T}")

        #state
        self.current_position = self.B * self.T * process_rank
        self.process_rank = process_rank
        self.num_processes = num_processes

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]

        x = (buf[:-1].view(B, T))
        y = (buf[1:]).view(B, T)

        self.current_position += B*T*self.num_processes

        if self.current_position + (B * T * self.num_processes)>= len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank

        return x, y
    