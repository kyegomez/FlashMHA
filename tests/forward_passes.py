import torch
from FlashMHA import FlashMHA

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