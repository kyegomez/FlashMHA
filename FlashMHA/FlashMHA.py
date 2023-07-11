import torch
from FlashMHA.attention import FlashAttention
# !pip install torch
# !pip install einops
from collections import namedtuple

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from einops import rearrange

EfficientAttentionConfig = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])



class FlashMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, dropout=0.0,
                 causal=False, device=None, dtype=None) -> None:
        assert batch_first
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0, "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.inner_attn = FlashAttention(dropout=dropout, causal=causal)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(self, query, key, value):
        qkv = self.Wqkv(query)
        q, k, v = rearrange(qkv, 'b s (three h d) -> three b s h d', three=3, h=self.num_heads, d=self.head_dim).unbind(dim=0)
        context = self.inner_attn(q, k, v)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)'))
