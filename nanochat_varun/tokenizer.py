import os
import pickle
import tiktoken
import _rustbpe

SPECIAL_TOKENS = [
    "<|bos|>", #every document begins with `beginning of sequence` token
]

class Tokenizer:
    def __init__(self, enc, bos_token):
        self.enc = enc

    @classmethod
    def train_tokenizer(cls, text_iter, vocab_size, buffer_size):
        tokenizer = _rustbpe.Tokenizer()
        #assert vocab_size - special tokens >= 256
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, "vocab_size without special tokens must be at least 256"
        #train the tokenizer
        tokenizer.train_from_iterator(text_iter, vocab_size_no_special, buffer_size)
        #get the mergeable ranks -> Vec<(Vec<u8>, u32)
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k) : v for k, v in mergeable_ranks_list}
        token_no_special_len = len(mergeable_ranks)
        #get the pattern
        pattern = tokenizer.get_pattern()
        #special tokens
        special_tokens = {k : token_no_special_len + i for i, k in enumerate(SPECIAL_TOKENS)}        
        #get the encoder from tiktoken
        enc = tiktoken.Encoding(
            name="nanochat_varun",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc, SPECIAL_TOKENS)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            enc = pickle.load(f)

        return cls(enc, SPECIAL_TOKENS[0])


    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.enc, f)

    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = self.encode_special(prepend)

        if isinstance(text, str):
            token_ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                token_ids.insert(0, prepend_id)
        elif isinstance(text, list):
            token_ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for token_ids_row in token_ids:
                    token_ids_row.insert(0, prepend_id)
        else:
            raise TypeError("text must be str or list[str]")
        return token_ids

    def decode(self, token_ids):
        return self.enc.decode(token_ids)

if __name__ == "__main__":

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
    
    # tokenizer = Tokenizer.train_tokenizer(file_iterator("input.txt"), vocab_size=50257, buffer_size=10000)
    # tokenizer.save("tokenizer.pkl")

    tokenizer = Tokenizer.load("tokenizer.pkl")



    ids = tokenizer.encode("Cash likes penises in his ass because he's a twink 🥰")

    # Assuming 'tokenizer' is your RustBPETokenizer instance
    # and 'ids' is the list of integers you outputted

    print(f"{'ID':<6} | {'Bytes':<15} | {'String (if safe)':<15}")
    print("-" * 40)

    for t_id in ids:
        # 1. Get raw bytes (Safe, Truth)
        token_bytes = tokenizer.enc.decode_single_token_bytes(t_id)
        # 2. Try to decode to string for readability (Unsafe)
        try:
            token_str = token_bytes.decode('utf-8')
        except UnicodeDecodeError:
            token_str = "<partial bytes>"

        print(f"{t_id:<6} | {str(token_bytes):<15} | {token_str:<15}")