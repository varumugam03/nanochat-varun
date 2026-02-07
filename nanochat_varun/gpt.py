import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat_varun.common import GPTConfig, get_device

def apply_rotary_embeddings(x, cos, sin):
    assert x.ndim == 4 # multiheaded attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split the last dimension into two --> head_size / 2
    y1 = x1 * cos - x2 * sin # RoPe rotation logic --> this is the rotation matrix multiplied by the input bc the frequencies are already computed
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1,y2], 3) #concat along the last dimension
    out = out.to(x.dtype) #keep the same input so no type casting
    return out

def norm(x):
    return F.rms_norm(x, (x.size(-1),)) #no learnable parameters b/c the following FC layer can just learn what it needs to

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.sequence_len = config.sequence_len
        self.n_head = config.n_head # query heads (GQA)
        self.n_kv_head = config.n_kv_head # key-value heads (GQA)
        self.n_embd = config.n_embd #embedding dimension
        self.head_dim = self.n_embd // self.n_head
        
        assert self.n_embd % self.n_head == 0, "Embedding dimension must be divisible by number of heads"
        assert self.n_kv_head <= self.n_head, "Number of key-value heads must be less than or equal to number of query heads"
        assert self.n_head % self.n_kv_head == 0, "Number of query heads must be divisible by number of key-value heads"

        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False) # dimensionality: [n_embd, n_embd] which is the same as n_head * n_head_dim
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False) # [n_embd, kv_heads * head_size] <-- this is because it could vary based on whether we're using GQA (MEMORY SAVING HERE!!)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False) # [n_embd, kv_heads * head_size] <-- this is because of GQA and you would only need to multiply the resulting Q*K which would be the GCD(n_heads, n_kv_heads)
        self.c_out = nn.Linear(self.n_embd, self.n_embd, bias=False) # this is just to mix all the results of the different heads together. Technically MLP could do it but we want Attention to be a self contained block.

    def forward(self, x, cos_sin):
        # x has shape [batch_size, sequence_len, n_embd] --> B,T,C
        B,T,C = x.size()

        #project input and get queries, keys, and values and reshape
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim) # --> (B, T, C) * (B, T, H * D) -> (B, T, C) : reshaped to (B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim) # --> (B, T, C) * (B, T, KVH * D) -> (B, T, ~C) : reshaped to (B, T, n_kv_head, head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim) # ""

        cos, sin = cos_sin
        '''only have to apply rotary embeddings to the query and key matrices because RoPe is designed specifically for the attention mechanism.
        its only applied to q and k because when you do the dot product of them the rotation is canceled out or rather it becomes some scaled version of the
        difference in angle. the value vector only represents the content of what is being attended to, not the actual attention'''
        q, k = apply_rotary_embeddings(q, cos, sin), apply_rotary_embeddings(k, cos, sin)
        '''norm applied before attention calculation because ... TODO: finish this'''
        q, k= norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # --> (B, n_head/kv_head, T, head_dim)


        # calculate attention scores
        enable_gqa = self.n_head != self.n_kv_head
        '''using the functional scaled dot product attention because it does the automatic "repeat_interleave" for GQA under the hood. It also uses 
           flashattention under the hood if supported by hardware. '''
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa) # y --> (B, kv_head, T, head_dim)

        y = y.transpose(1, 2).contiguous().view(B, T, -1) # --> (B, T, C)
        y = self.c_out(y) # --> (B, T, C)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.n_embd * 4)
        self.proj = nn.Linear(config.n_embd * 4, config.n_embd)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x).square() # performs same or better than Gelu and is faster and less expensive computation
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin):
        x = x + self.attn(norm(x), cos_sin) # pre-Norm and residual connections
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte" : nn.Embedding(config.vocab_size, config.n_embd),
            "h" : nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])
        }) 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        
        '''precomputing rotary embeddings for 10x max sequence length. Going to stick with initial Karpathy's hack for now. Will change this to 
           be grow dynamically with the input sequence length in the future
           
           Note: these embeddings are recalculated in init_weights() just so that when this model scales and its loaded with torch.device("meta") 
           we can still actually compute the embeddings on the device we want to run on. The redundant calculation is to make sure the model is ok to run
           without initializing weight for just testing purposes'''

        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        #zero out the classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        #zero out the proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.attn.c_out.weight)
            torch.nn.init.zeros_(block.mlp.proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin # TODO : search up if these were already defined by self.register_buffer in __init__
        #cast the embeddings to bfloat16
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)
        
    
    def _init_weights(self, module):
        #initializing weights based off Karpathy's method
        #TODO : comment about the intuition for these initializations
        if isinstance(module, nn.Linear):
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)


    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.transformer.wte.weight.device # get the device from the weights to keep pos embeddings on same device

        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        #stride time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)

        #calc rotation frequencies for each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # adding batch and n_head dimensions for broadcasting later
        
        return cos, sin

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()

        #change this for kv_cache later
        T0 = 0
        cos_sin = self.cos[:, T0:T], self.sin[:, T0:T]

        #forward transformer
        x = self.transformer.wte(idx) # --> (B, T, C)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin)
        x = norm(x)

        #forward the lm_head (compute logits)
        softcap = 15
        logits = self.lm_head(x)
        logits = logits.float() # switch to fp32 for logit softcap and loss calculation for numerical stability
        logits = softcap * torch.tanh(logits / softcap) # squash logits

        if targets is not None:
            #training
            # flattens the logits list into (B*T, V_size) and targets into (B*T)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference
            return logits







#Just thinking out loud here:

"""
Just for normal attention :

x = (T, C) - sequence length by embedding dimension

Q = (C, C) - we call C the hidden dimension size, so we can thinking of this as having C columns 
            asking C distinct question - its more subtle than that but that will do for now

when you do x @ Q = x_q ~ (T, C)

x @ Q takes every row of x and does a dot product with every column of Q

[x x x x x x x x]               [ x                      ]
[               ]               [ x                      ]
[               ]       X       [ x                      ]
[               ]               [ x                      ]
[               ]               [ x                      ]
                                [ x                      ]
                                [ x                      ]
                                [ x                      ]


this results in a matrix x_q ~ (T, C) which essentially is another matrix that basically says
how close is this token to the C questions. so one row in this resulting matrix consists of C
values that indicates how much this token cares about a particular "Question" or direction in
this dimensional space.

the same thing is done with the K matrix

where x @ K = x_k ~ (T, C). which essentially is another matrix that basically says how close
is this token to these C answers. so one row in this resulting matrix consists of C values that
indicate how much this token cares about this particular "Answer" or direction in this dimensional
space.

when you transpose x_k -> x_k^T. the meaning flips it now has dimensions ~ (C, T), where now each
row represents all the different answers, and if you take one row, then each value/column in that
row represents how much a particular token gives that answer

x_q @ x_k^T ~ (T, T) because you can think of the x_q matrix of a matrix of all the tokens and how
many questions each token asks, and you can think of the x_k^T matrix as a matrix of all the questions
and how many tokens closely answer that question


[x x x x x x x x]               [ x        ]                [x      ]
[               ]               [ x        ]                [       ]
[               ]       X       [ x        ]        =       [       ]
[               ]               [ x        ]                [       ]
[               ]               [ x        ]
                                [ x        ]
                                [ x        ]
                                [ x        ]

this essentially says how many questions does token 1 ask for example, and how many answers does token 1 have.
when you do the dot product it essentially says, how much should token 1 attend to this token (in this case itself)

now because its causal you mask out the upper triangle of this resulting matrix, resulting in a lower triangular
matrix of attetion scores.

you take this resulting matrix and apply dropout (if wanted) at random to get rid of overreliance on particular tokens
like bos or something.

apply softmax

then you multiply with the value matrix x_v -> (T, T) x (T, C) = (T, C). As you perform the dot products you will get
a weighted combination of each of the learned value embeddings for the tokens a particular token attends to.

a question i had is why you'd need a whole new learned value embedding for each attention block, could you not just use the
learned embedding for the token. I think the reason for this is because each attention block answers different questions
and because of that the context of the token changes with the question to be answered, so it's better to decouple
the context meaning from the tokens base meaning.

so the output of the scaled_dot_product_attention (disregarding batch size) is a matrix (T, C) which has the context
aware embeddings for each token. The feed forward layer then mixes all the information together to produce a
richer understanding of the sentence.

"""
