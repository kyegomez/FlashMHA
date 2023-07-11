# FlashMHA
FlashMHA is a PyTorch implementation of the Flash Multi-Head Attention mechanism. It is designed to be efficient and flexible, allowing for both causal and non-causal attention. The implementation also includes support for the Flash Attention mechanism, which is a highly efficient attention mechanism designed for GPUs.

## Installation
You can install the FlashMHA library by cloning the GitHub repository and installing the required dependencies using `pip`:

```bash
git clone https://github.com/kyegomez/FlashMHA.git
cd FlashMHA
pip install -r requirements.txt
```

## Usage
Here is a basic example of how to use the FlashMHA module:

```python
import torch
from flash_mha import FlashMHA

# Initialize the model
flash_mha = FlashMHA(embed_dim=512, num_heads=8, bias=True, batch_first=True, dropout=0.0, causal=False)

# Create input tensors
query = torch.randn(10, 32, 512)
key = torch.randn(10, 32, 512)
value = torch.randn(10, 32, 512)

# Forward pass
output = flash_mha(query, key, value)
```

In this example, `query`, `key`, and `value` are input tensors with shape `(batch_size, sequence_length, embed_dim)`. The FlashMHA model applies the multi-head attention mechanism to these inputs and returns the output tensor.

## Documentation


### `FlashAttention`

FlashAttention is a PyTorch module that implements the Flash Attention mechanism, a highly efficient attention mechanism designed for GPUs. It provides a fast and flexible solution for attention computations in deep learning models.

## Parameters

- `causal` (bool, optional): If set to True, applies causal masking to the sequence. Default: False.
- `dropout` (float, optional): The dropout probability. Default: 0.
- `flash` (bool, optional): If set to True, enables the use of Flash Attention. Default: False.

## Inputs

- `q` (Tensor): The query tensor of shape (batch_size, num_heads, query_length, embed_dim).
- `k` (Tensor): The key tensor of shape (batch_size, num_heads, key_length, embed_dim).
- `v` (Tensor): The value tensor of shape (batch_size, num_heads, value_length, embed_dim).
- `mask` (Tensor, optional): An optional mask tensor of shape (batch_size, num_heads, query_length, key_length), used to mask out specific positions. Default: None.
- `attn_bias` (Tensor, optional): An optional additive bias tensor of shape (batch_size, num_heads, query_length, key_length), applied to the attention weights. Default: None.

## Outputs

- `output` (Tensor): The output tensor of shape (batch_size, query_length, embed_dim).

## `FlashMHA`

FlashMHA is a PyTorch module that implements the Flash Multi-Head Attention mechanism, which combines multiple FlashAttention layers. It is designed to be efficient and flexible, allowing for both causal and non-causal attention.

## Parameters

- `embed_dim` (int): The dimension of the input embedding.
- `num_heads` (int): The number of attention heads.
- `bias` (bool, optional): If set to False, the layers will not learn an additive bias. Default: True.
- `batch_first` (bool, optional): If True, then the input and output tensors are provided as (batch, seq, feature). Default: True.
- `dropout` (float, optional): The dropout probability. Default: 0.
- `causal` (bool, optional): If True, applies causal masking to the sequence. Default: False.
- `device` (torch.device, optional): The device to run the model on. Default: None.
- `dtype` (torch.dtype, optional): The data type to use for the model parameters. Default: None.

## Inputs

- `query` (Tensor): The query tensor of shape (batch_size, sequence_length, embed_dim).
- `key` (Tensor): The key tensor of shape (batch_size, sequence_length, embed_dim).
- `value` (Tensor): The value tensor of shape (batch_size, sequence_length, embed_dim).

## Outputs

- `output` (Tensor): The output tensor of shape (batch_size, sequence_length, embed_dim).

## License

FlashAttention and FlashMHA are open-source software, licensed under the MIT license. For more details, please refer to the [GitHub repository](https://github.com/kyegomez/FlashAttention).