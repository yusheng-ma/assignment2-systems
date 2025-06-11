# benchmark_attention.py

import torch
import math
import itertools
from timeit import default_timer as timer
from cs336_basics.model import scaled_dot_product_attention

torch.manual_seed(42)

# Settings
batch_size = 8
d_models = [16, 32, 64, 128]
seq_lengths = [256, 1024, 4096, 8192, 16384]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def benchmark_attention(d_model, seq_len, attention_function):
    Q = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    K = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
    V = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

    # Warm-up (important for realistic timing)
    for _ in range(10):
        out = attention_function(Q, K, V)
        out.sum().backward()
        torch.cuda.synchronize()
        Q.grad.zero_(); K.grad.zero_(); V.grad.zero_()

    # Measure forward
    torch.cuda.reset_peak_memory_stats()
    start_time = timer()
    for _ in range(100):
        out = attention_function(Q, K, V)
        torch.cuda.synchronize()
    forward_time = timer() - start_time
    mem_used = torch.cuda.max_memory_allocated()

    # Measure backward
    start_time = timer()
    for _ in range(100):
        out = attention_function(Q, K, V)
        out.sum().backward()
        torch.cuda.synchronize()
        Q.grad.zero_(); K.grad.zero_(); V.grad.zero_()
    backward_time = timer() - start_time

    return forward_time, backward_time, mem_used

# Run benchmarks
results = []
for d_model, seq_len in itertools.product(d_models, seq_lengths):
    try:
        f_time, b_time, mem = benchmark_attention(d_model, seq_len, attention_function=scaled_dot_product_attention)
        print(f"d_model={d_model}, seq_len={seq_len} -> forward: {f_time:.2f}s, backward: {b_time:.2f}s, memory: {mem / 1e6:.2f} MB")
        results.append((d_model, seq_len, f_time, b_time, mem))
    except RuntimeError as e:
        print(f"d_model={d_model}, seq_len={seq_len} -> ERROR: {str(e)}")

# compiled benchmarks
print("[Compiled]")
for d_model, seq_len in itertools.product(d_models, seq_lengths):
    try:
        f_time, b_time, mem = benchmark_attention(d_model, seq_len, attention_function=torch.compile(scaled_dot_product_attention))
        print(f"d_model={d_model}, seq_len={seq_len} -> forward: {f_time:.2f}s, backward: {b_time:.2f}s, memory: {mem / 1e6:.2f} MB")
        results.append((d_model, seq_len, f_time, b_time, mem))
    except RuntimeError as e:
        print(f"d_model={d_model}, seq_len={seq_len} -> ERROR: {str(e)}")
