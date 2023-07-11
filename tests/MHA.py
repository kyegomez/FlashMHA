import timeit
import matplotlib.pyplot as plt
from FlashMHA import FlashMHA

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