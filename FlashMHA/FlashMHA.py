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

import torch

# Example 1
flash_mha = FlashMHA(embed_dim=512, num_heads=8, dropout=0.1)
query = torch.randn(10, 32, 512)  # sequence length = 10, batch size = 32, embedding dimension = 512
key = torch.randn(10, 32, 512)
value = torch.randn(10, 32, 512)
output = flash_mha(query, key, value)
print(output[0].shape)  # should be [10, 32, 512]

# Example 2
flash_mha = FlashMHA(embed_dim=256, num_heads=4, dropout=0.0)
query = torch.randn(20, 16, 256)  # sequence length = 20, batch size = 16, embedding dimension = 256
key = torch.randn(20, 16, 256)
value = torch.randn(20, 16, 256)
output = flash_mha(query, key, value)
print(output[0].shape)  # should be [20, 16, 256]

# Example 3
flash_mha = FlashMHA(embed_dim=128, num_heads=2, dropout=0.2)
query = torch.randn(30, 64, 128)  # sequence length = 30, batch size = 64, embedding dimension = 128
key = torch.randn(30, 64, 128)
value = torch.randn(30, 64, 128)
output = flash_mha(query, key, value)
print(output[0].shape)  # should be [30, 64, 128]

import timeit
import matplotlib.pyplot as plt

# Initialize the model
flash_mha = FlashMHA(embed_dim=512, num_heads=8, bias=True, batch_first=True, dropout=0.0, causal=False)

# Define the sequence lengths for the benchmark
seq_lengths = [2000, 4000, 8000, 16000, 32000]

# Store the execution times
exec_times = []

for seq_len in seq_lengths:
    # Create input tensors
    query = torch.randn(10, seq_len, 512)
    key = torch.randn(10, seq_len, 512)
    value = torch.randn(10, seq_len, 512)

    # Measure the execution time
    start_time = timeit.default_timer()
    output = flash_mha(query, key, value)
    exec_time = timeit.default_timer() - start_time

    exec_times.append(exec_time)

# Plot the execution time against the sequence length
plt.plot(seq_lengths, exec_times)
plt.xlabel('Sequence Length')
plt.ylabel('Execution Time (s)')
plt.title('FlashMHA Benchmark')
plt.show()