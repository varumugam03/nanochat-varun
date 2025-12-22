import _rustbpe
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO)

def file_iterator(file_path):
    """
    Generator that streams the file line-by-line.
    We yield raw lines to ensure we don't strip important whitespace
    that the BPE tokenizer needs to learn (like newlines).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find {file_path}. Make sure you are running from the root directory.")

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line

def main():
    # 1. Initialize the tokenizer
    tokenizer = _rustbpe.Tokenizer()
    
    print(f"--- Starting Training on input.txt ---")
    
    # 2. Stream the file into Rust
    # buffer_size=1000: Rust will pull 1000 lines, release the GIL, 
    # and process them in parallel on all CPU cores.
    word_counts = tokenizer.train_from_iterator(
        file_iterator("/Users/varumugam/Desktop/Projects/nanochat-varun/input.txt"), 
        vocab_size=1000, 
        buffer_size=500 
    )
    
    print(f"\n--- Processing Complete ---")
    
    mergeable_ranks = tokenizer.get_mergeable_ranks()
    
    print(mergeable_ranks)
      
if __name__ == "__main__":
    main()