# flash_benchmarking.py

import torch
import math
import itertools
import einx
from triton.testing import do_bench
from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flashattention_triton import MyTritonFlashAttentionAutogradFunctionClass

torch.manual_seed(0)

# Settings
batch_size = 1
seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
d_models = [16, 32, 64, 128]
precisions = [torch.bfloat16, torch.float32]
device = torch.device("cuda")

# Define PyTorch baseline
def pytorch_attention(Q, K, V, is_causal):
    causal_mask = None
    if is_causal:
        *b, sequence_length, d_model = Q.shape
        seq = torch.arange(sequence_length, device=Q.device)
        qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
        kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b))
        causal_mask = qi >= kj  # (query, key)
    
    attn = scaled_dot_product_attention(
        Q, K, V, mask=causal_mask if causal_mask is not None else None
    )
    return attn

# Define Triton wrapper
def triton_attention(Q, K, V, is_causal):
    return MyTritonFlashAttentionAutogradFunctionClass.apply(Q, K, V, is_causal)

results = []

for dtype, d_model, seq_len in itertools.product(precisions, d_models, seq_lengths):
    try:
        Q = torch.randn(batch_size, seq_len, d_model, device=device, dtype=dtype, requires_grad=True)
        K = torch.randn_like(Q)
        V = torch.randn_like(Q)

        # Triton forward/backward timings
        def triton_fwd():
            return triton_attention(Q, K, V, is_causal=True)

        def triton_fwd_bwd():
            out = triton_attention(Q, K, V, is_causal=True)
            grad_out = torch.randn_like(out)
            out.backward(grad_out, retain_graph=True)

        triton_fwd_time = do_bench(triton_fwd)
        triton_bwd_time = do_bench(triton_fwd_bwd) - triton_fwd_time

        # PyTorch forward/backward timings
        def torch_fwd():
            return pytorch_attention(Q, K, V, is_causal=True)

        def torch_fwd_bwd():
            out = pytorch_attention(Q, K, V, is_causal=True)
            grad_out = torch.randn_like(out)
            out.backward(grad_out, retain_graph=True)

        torch_fwd_time = do_bench(torch_fwd)
        torch_bwd_time = do_bench(torch_fwd_bwd) - torch_fwd_time

        results.append({
            'dtype': str(dtype).split('.')[-1],
            'seq_len': seq_len,
            'd_model': d_model,
            'triton_fwd_ms': triton_fwd_time * 1e3,
            'triton_bwd_ms': triton_bwd_time * 1e3,
            'triton_total_ms': (triton_fwd_time + triton_bwd_time) * 1e3,
            'torch_fwd_ms': torch_fwd_time * 1e3,
            'torch_bwd_ms': torch_bwd_time * 1e3,
            'torch_total_ms': (torch_fwd_time + torch_bwd_time) * 1e3
        })

        print(f"Done: dtype={dtype}, seq_len={seq_len}, d_model={d_model}")

    except RuntimeError as e:
        print(f"ERROR: dtype={dtype}, seq_len={seq_len}, d_model={d_model}: {e}")
        continue

# Display results as a table
import pandas as pd

df = pd.DataFrame(results)
pd.set_option("display.max_rows", None)
print(df.to_string(index=False))
