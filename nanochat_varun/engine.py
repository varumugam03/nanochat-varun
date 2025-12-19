class KV_Cache:
    def __init__(self, num_layers, batch_size, num_heads, seq_len, head_dim):
        self.kv_cache_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim) # the 2 is for key and value
        self.kv_cache = None
        self.pos = 0

    def get_pos(self):
        return self.pos

    def insert(self, layer_idx, key, value):
        if self.kv_cache is None:
            self.kv_cache = torch.zeros(self.kv_cache_shape, dtype=key.dtype, device=key.device)
        
        # TODO : finish

    
        