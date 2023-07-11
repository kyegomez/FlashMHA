import timeit
import matplotlib.pyplot as plt
from FlashMHA import FlashAttention

# Initialize the model
flash_attention = FlashAttention(causal=False, dropout=0.0)

# Define the sequence lengths for the benchmark
seq_lengths = [2000, 4000, 8000, 16000, 32000]

# Store the execution times
exec_times = []

for seq_len in seq_lengths:
    #create input tensors
    query = torch.randn(10, 8, seq_len, 64) #added dimension for the attention heads
    key = torch.randn(10, 8, seq_len, 64) # added dimension for the number of attention heads
    value = torch.randn(10, 8, seq_len, 64) # added dimension for the number of attention heads

    #measure the execution time
    start_time = timeit.default_timer()
    output = flash_attention.forward(query, key, value)
    exec_time = timeit.default_timer() - start_time
    
    exec_times.append(exec_time)

# Plot the execution time against the sequence length
plt.plot(seq_lengths, exec_times)
plt.xlabel('Sequence Length')
plt.ylabel('Execution Time (s)')
plt.title('FlashAttention Benchmark')
plt.show()