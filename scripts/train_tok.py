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
                    print(f"Ran through all documents. Hit max chars: {args.max_chars}")
                    return
        print(f"Ran through all documents. Processed {nchars} characters")

text_iter = text_iterator()

#train the tokenizer
t0 = time.time()
tok = Tokenizer.train_tokenizer(text_iter, vocab_size=args.vocab_size)
t1 = time.time()
print(f"Training took {t1-t0:.2f} seconds")

#save the tokenizer
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tok.save(tokenizer_dir)

