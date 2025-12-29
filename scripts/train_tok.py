import os
import time
import argparse
import torch
from tqdm import tqdm
from nanochat_varun.tokenizer import Tokenizer
from nanochat_varun.common import get_base_dir
from nanochat_varun.dataset import parquets_iter_batched


#parse command line arguments
parser = argparse.ArgumentParser(description="Train RustBPE Tokenizer")
parser.add_argument("--vocab-size", type=int, default=65536, help="Vocabulary size (default: 65536 which is 2^16)")
parser.add_argument("--max-chars", type=int, default=10_000_000_000, help="Maximum number of characters to use for training (default: 10B)")
parser.add_argument("--doc_cap", type=int, default=10_000, help="Maximum number of characters per document (default: 10K)")
args = parser.parse_args()
print(f"Training tokenizer with vocab size: {args.vocab_size}, max chars: {args.max_chars}, max characters per document: {args.doc_cap}")

#make the text_iterator
def text_iterator():
    """
    1) flatten batches into single iterator
    2) crop every document to doc_cap
    3) break when max_chars is reached
    """
    nchars = 0
    with tqdm(total=args.max_chars, desc="Processing documents") as pbar:
        for batch in parquets_iter_batched(split="train"):
            for doc in batch:
                doc_text = doc
                if len(doc_text) > args.doc_cap:
                    doc_text = doc_text[:args.doc_cap]
                nchars += len(doc_text)
                pbar.update(len(doc_text))
                yield doc_text
                if nchars >= args.max_chars:
                    print(f"Processed all documents. Hit max chars: {args.max_chars}")
                    return
        print(f"Not enough documents. Processed {nchars} characters")

text_iter = text_iterator()

#train the tokenizer

#check if the tokenizer already exists
tok_path = os.path.join(get_base_dir(), "tokenizer", "tokenizer.pkl")
if not os.path.exists(tok_path):
    t0 = time.time()
    tok = Tokenizer.train_tokenizer(text_iter, vocab_size=args.vocab_size)
    t1 = time.time()
    print(f"Training took {t1-t0:.2f} seconds")

    #save the tokenizer
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer/")
    tok.save(tokenizer_dir)
else:
    print(f"Tokenizer already exists at {tok_path}. Skipping training.")
    tok = Tokenizer.load(tok_path)

print(f"Total learned vocab size: {tok.get_vocab_size()}")

#Sanity check
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tok.encode(test_text)
decoded = tok.decode(encoded)
assert decoded == test_text

#visualize the tokenizer
# ids = tok.encode("""Hello world! This is a test. Numbers: 123, 4567, 89 Contractions: I'm, you're, it's Special chars: @#$%^&*() Unicode: 你好世界 🌍""")

# print(f"{'ID':<6} | {'Bytes':<15} | {'String (if safe)':<15}")
# print("-" * 40)

# for t_id in ids:
#     token_bytes = tok.enc.decode_single_token_bytes(t_id)
#     try:
#         token_str = token_bytes.decode('utf-8')
#     except UnicodeDecodeError:
#         token_str = "<partial bytes>"

#     print(f"{t_id:<6} | {str(token_bytes):<15} | {token_str:<15}")

vocab_size = tok.get_vocab_size()
special_set = set(tok.get_special_tokens())
token_strings = [tok.decode([token_id]) for token_id in range(vocab_size)]

token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] #python string representation of this token
    if token_str in special_set:
        token_bytes.append(0) # special characters don't count
    else:
        id_bytes = len(token_str.encode('utf-8')) # number of bytes that make up this token
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(get_base_dir(), "tokenizer", "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token bytes to {token_bytes_path}")

#stats to later be logged in report - not created yet

token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(torch.float32)

print("Token bytes stats:")
print(f"Min: {token_bytes_nonzero.min().item()}")
print(f"Max: {token_bytes_nonzero.max().item()}")
print(f"Mean: {token_bytes_nonzero.mean().item()}")
print(f"Median: {token_bytes_nonzero.median().item()}")
print(f"Std: {token_bytes_nonzero.std().item()}")


