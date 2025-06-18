import os
import timeit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.cuda.nvtx as nvtx
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.DDP import DDPIndividualParameters, DDPBucketed
import pandas as pd

# ==== MACRO CONFIG ====
MODEL_CONFIG = {
    "vocab_size": 10000,
    "context_length": 256,
    "d_model": 768,
    "d_ff": 3072,
    "num_layers": 12,
    "num_heads": 12,
    "rope_theta": 10000.0
}

BENCHMARK_CONFIG = {
    "batch_size": 32,
    "eval_steps": 10,
    "warmup_steps": 5,
    "lr": 1e-3
}
# ======================

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def generate_batch(vocab_size, batch_size, context_length, device):
    x = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return x, y

def create_model(device):
    model = BasicsTransformerLM(
        vocab_size=MODEL_CONFIG["vocab_size"],
        context_length=MODEL_CONFIG["context_length"],
        d_model=MODEL_CONFIG["d_model"],
        num_layers=MODEL_CONFIG["num_layers"],
        num_heads=MODEL_CONFIG["num_heads"],
        d_ff=MODEL_CONFIG["d_ff"],
        rope_theta=MODEL_CONFIG["rope_theta"]
    ).to(device)
    return model

def benchmark_naive_ddp(rank, world_size, results):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    model = create_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=BENCHMARK_CONFIG["lr"])

    x, y = generate_batch(MODEL_CONFIG["vocab_size"], BENCHMARK_CONFIG["batch_size"], MODEL_CONFIG["context_length"], device)

    num_steps = BENCHMARK_CONFIG["eval_steps"]
    warmup_steps = BENCHMARK_CONFIG["warmup_steps"]
    step_times = []
    comm_times = []

    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = cross_entropy(y_pred, y)
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
        optimizer.step()
        torch.cuda.synchronize()

    # Benchmark
    with nvtx.range("naive_ddp"):
        for step in range(num_steps):
            optimizer.zero_grad()
            start_step = timeit.default_timer()

            y_pred = model(x)
            loss = cross_entropy(y_pred, y)
            loss.backward()

            start_comm = timeit.default_timer()
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
            torch.cuda.synchronize()
            end_comm = timeit.default_timer()

            optimizer.step()
            torch.cuda.synchronize()
            end_step = timeit.default_timer()

            step_times.append(end_step - start_step)
            comm_times.append(end_comm - start_comm)

            if rank == 0:
                print(f"[Naive] Step {step}: total={step_times[-1]:.6f}s comm={comm_times[-1]:.6f}s")

    if rank == 0:
        results.append({
            "method": "Naive",
            "avg_step_time": sum(step_times)/num_steps,
            "avg_comm_time": sum(comm_times)/num_steps,
            "comm_ratio": (sum(comm_times)/sum(step_times)) * 100
        })

    cleanup()

def benchmark_minimal_ddp_flat(rank, world_size, results):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    model = create_model(device)
    optimizer = optim.AdamW(model.parameters(), lr=BENCHMARK_CONFIG["lr"])

    x, y = generate_batch(MODEL_CONFIG["vocab_size"], BENCHMARK_CONFIG["batch_size"], MODEL_CONFIG["context_length"], device)

    num_steps = BENCHMARK_CONFIG["eval_steps"]
    warmup_steps = BENCHMARK_CONFIG["warmup_steps"]
    step_times = []
    comm_times = []

    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = cross_entropy(y_pred, y)
        loss.backward()
        grads = [param.grad for param in model.parameters() if param.grad is not None]
        flat = _flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.AVG)
        synced_grads = _unflatten_dense_tensors(flat, grads)
        for grad, synced in zip(grads, synced_grads):
            grad.copy_(synced)
        optimizer.step()
        torch.cuda.synchronize()

    # Benchmark
    with nvtx.range("minimal_ddp_flat"):
        for step in range(num_steps):
            optimizer.zero_grad()
            start_step = timeit.default_timer()

            y_pred = model(x)
            loss = cross_entropy(y_pred, y)
            loss.backward()

            grads = [param.grad for param in model.parameters() if param.grad is not None]
            flat = _flatten_dense_tensors(grads)

            start_comm = timeit.default_timer()
            dist.all_reduce(flat, op=dist.ReduceOp.AVG)
            torch.cuda.synchronize()
            synced_grads = _unflatten_dense_tensors(flat, grads)
            for grad, synced in zip(grads, synced_grads):
                grad.copy_(synced)
            end_comm = timeit.default_timer()

            optimizer.step()
            torch.cuda.synchronize()
            end_step = timeit.default_timer()

            step_times.append(end_step - start_step)
            comm_times.append(end_comm - start_comm)

            if rank == 0:
                print(f"[Flat] Step {step}: total={step_times[-1]:.6f}s comm={comm_times[-1]:.6f}s")

    if rank == 0:
        results.append({
            "method": "Flat",
            "avg_step_time": sum(step_times)/num_steps,
            "avg_comm_time": sum(comm_times)/num_steps,
            "comm_ratio": (sum(comm_times)/sum(step_times)) * 100
        })

    cleanup()

def benchmark_individual_overlap_ddp(rank, world_size, results):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    model = create_model(device)
    model = DDPIndividualParameters(model)
    optimizer = optim.AdamW(model.parameters(), lr=BENCHMARK_CONFIG["lr"])

    x, y = generate_batch(MODEL_CONFIG["vocab_size"], BENCHMARK_CONFIG["batch_size"], MODEL_CONFIG["context_length"], device)

    num_steps = BENCHMARK_CONFIG["eval_steps"]
    warmup_steps = BENCHMARK_CONFIG["warmup_steps"]
    step_times = []
    comm_times = []

    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = cross_entropy(y_pred, y)
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()
        torch.cuda.synchronize()

    # Benchmark
    with nvtx.range("individual_overlap_ddp"):
        for step in range(num_steps):
            optimizer.zero_grad()
            start_step = timeit.default_timer()

            y_pred = model(x)
            loss = cross_entropy(y_pred, y)
            loss.backward()

            start_comm = timeit.default_timer()
            model.finish_gradient_synchronization()
            torch.cuda.synchronize()
            end_comm = timeit.default_timer()

            optimizer.step()
            torch.cuda.synchronize()
            end_step = timeit.default_timer()

            step_times.append(end_step - start_step)
            comm_times.append(end_comm - start_comm)

            if rank == 0:
                print(f"[IndividualOverlap] Step {step}: total={step_times[-1]:.6f}s comm={comm_times[-1]:.6f}s")

    if rank == 0:
        results.append({
            "method": "IndividualOverlap",
            "avg_step_time": sum(step_times)/num_steps,
            "avg_comm_time": sum(comm_times)/num_steps,
            "comm_ratio": (sum(comm_times)/sum(step_times)) * 100
        })

    cleanup()

def benchmark_bucketed_ddp(rank, world_size, results, bucket_size_mb):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    model = create_model(device)
    model = DDPBucketed(model, bucket_size_mb)
    optimizer = optim.AdamW(model.parameters(), lr=BENCHMARK_CONFIG["lr"])

    x, y = generate_batch(MODEL_CONFIG["vocab_size"], BENCHMARK_CONFIG["batch_size"], MODEL_CONFIG["context_length"], device)

    num_steps = BENCHMARK_CONFIG["eval_steps"]
    warmup_steps = BENCHMARK_CONFIG["warmup_steps"]
    step_times = []
    comm_times = []

    # Warmup
    for _ in range(warmup_steps):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = cross_entropy(y_pred, y)
        loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()
        torch.cuda.synchronize()

    # Benchmark
    tag = f"bucketed_ddp_{bucket_size_mb}MB"
    with nvtx.range(tag):
        for step in range(num_steps):
            optimizer.zero_grad()
            start_step = timeit.default_timer()

            y_pred = model(x)
            loss = cross_entropy(y_pred, y)
            loss.backward()

            start_comm = timeit.default_timer()
            model.finish_gradient_synchronization()
            torch.cuda.synchronize()
            end_comm = timeit.default_timer()

            optimizer.step()
            torch.cuda.synchronize()
            end_step = timeit.default_timer()

            step_times.append(end_step - start_step)
            comm_times.append(end_comm - start_comm)

            if rank == 0:
                print(f"[Bucketed {bucket_size_mb}MB] Step {step}: total={step_times[-1]:.6f}s comm={comm_times[-1]:.6f}s")

    if rank == 0:
        results.append({
            "method": f"Bucketed {bucket_size_mb}MB",
            "avg_step_time": sum(step_times)/num_steps,
            "avg_comm_time": sum(comm_times)/num_steps,
            "comm_ratio": (sum(comm_times)/sum(step_times)) * 100
        })

    cleanup()

def run_benchmark():
    world_size = 2
    manager = mp.Manager()
    results = manager.list()

    print("=== Running Naive DDP Benchmark ===")
    mp.spawn(benchmark_naive_ddp, args=(world_size, results), nprocs=world_size, join=True)

    print("\n=== Running Flat DDP Benchmark ===")
    mp.spawn(benchmark_minimal_ddp_flat, args=(world_size, results), nprocs=world_size, join=True)

    print("\n=== Running Individual Overlap DDP Benchmark ===")
    mp.spawn(benchmark_individual_overlap_ddp, args=(world_size, results), nprocs=world_size, join=True)

    for bucket_size_mb in [1, 10, 100, 1000]:
        print(f"\n=== Running Bucketed DDP Benchmark (bucket size {bucket_size_mb}MB) ===")
        mp.spawn(benchmark_bucketed_ddp, args=(world_size, results, bucket_size_mb), nprocs=world_size, join=True)

    df = pd.DataFrame(list(results))
    print("\n=== Benchmark Summary ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_benchmark()
