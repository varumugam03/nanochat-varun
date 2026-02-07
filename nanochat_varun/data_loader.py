from collections import deque
import torch
import pyarrow.parquet as pq

from nanochat_varun.dataset import list_parquet_files
from nanochat_varun.tokenizer import get_tokenizer

def tokenizing_data_loader_with_state_mps(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128, resume_state_dict=None):
    """
    Stream pretraining text from parquet files, tokenize , yield training batches
    """
    device = 'mps'
    assert split in ['train', 'val'], "split must be 'train' or 'val'"
    
    """
    TODO - add cuda distributed data parallelism support later - for now, just use single mps process
    """
    def document_batches(): # infinite generator
        parquet_paths = list_parquet_files()
        assert len(parquet_paths) > 0, "No parquet files found in data directory, did you run the dataset download script?"
        parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0 # use resume state dict if not None
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        first_pass = True
        pq_idx = resume_pq_idx
        while True: # multi epoch (so loop forever)
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths): # loop through all parquets
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                #need to resume rg_idx if resume state is a thing
                # parquet -> ~52 row groups -> ~1024 documents -> variable length documents
                rg_idx = resume_rg_idx if resume_rg_idx is not None and first_pass else 0 # check this logic later
                while rg_idx < pf.num_row_groups: # loop through all row groups
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column("text").to_pylist()
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i + tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += 1 # advance to next row group
                pq_idx += 1 # advance to next parquet
            first_pass = False
    batches = document_batches()

    # Now to tokenize batches and emit
    needed_tokens = B * T + 1
    #get the tokenizer
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token()
    #scratch buffer for tokenizing
    token_buffer = deque() # stream tokens in to right and pop from left
    while True:
        while len(token_buffer) < needed_tokens:
            batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(batch, prepend=bos_token, num_threads=tokenizer_threads)
            for token_list in token_lists:
                token_buffer.extend(token_list)
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]

        #TODO - add optimization to pin memory
        scratch_tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        inputs = scratch_tokens[:-1].view(B, T)
        targets = scratch_tokens[1:].view(B, T)  
        state_dict = {
            "pq_idx": pq_idx,
            "rg_idx": rg_idx,
        }
        yield inputs, targets, state_dict

if __name__ == "__main__":
    
    # little sanity test for data loader - removed print statements so useless - i think it works for the most part
    torch.set_printoptions(threshold=float('inf'))
    for i, (inputs, targets, state_dict) in enumerate(tokenizing_data_loader_with_state_mps(B=2, T=32, split="train")):
        if i == 2:
            break

    


                
