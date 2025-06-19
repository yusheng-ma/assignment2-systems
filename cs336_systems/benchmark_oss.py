import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import pandas as pd
import timeit

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_systems.OSS import OptimizerStateSharding
from cs336_systems.DDP import DDPIndividualParameters, DDPBucketed

MODEL_CONFIG = {
    "vocab_size": 10000,
    "context_length": 128,
    "d_model": 1024,
    "d_ff": 4096,
    "num_layers": 24,
    "num_heads": 16,
    "rope_theta": 10000.0
}

BENCHMARK_CONFIG = {
    "batch_size": 16,
    "eval_steps": 10,
    "warmup_steps": 5,
    "lr": 1e-3
}

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

def get_peak_memory(label):
    mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[{label}] Peak memory: {mem:.2f} GB")
    return mem

def benchmark_oss_speed_memory(rank, world_size, use_oss, results):
    try:
        setup(rank, world_size)
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)

        torch.cuda.reset_peak_memory_stats()
        model = create_model(device)
        model = DDPIndividualParameters(model)
        mem_after_init = get_peak_memory("After model initialization + warmup")

        if use_oss:
            optimizer = OptimizerStateSharding(model.parameters(), optim.AdamW, lr=BENCHMARK_CONFIG["lr"])
            label = "OSS"
        else:
            optimizer = optim.AdamW(model.parameters(), lr=BENCHMARK_CONFIG["lr"])
            label = "AdamW"

        x, y = generate_batch(MODEL_CONFIG["vocab_size"], BENCHMARK_CONFIG["batch_size"], MODEL_CONFIG["context_length"], device)

        # Warmup
        for i in range(BENCHMARK_CONFIG["warmup_steps"]):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = cross_entropy(y_pred, y)
            loss.backward()
            model.finish_gradient_synchronization()
            optimizer.step()
            torch.cuda.synchronize()
            if rank == 0:
                print(f"[{label}] Warmup step {i+1} done")

        step_times = []
        mem_before_list = []
        mem_after_list = []

        # Evaluation steps
        for step in range(BENCHMARK_CONFIG["eval_steps"]):
            torch.cuda.synchronize()
            start = timeit.default_timer()

            optimizer.zero_grad()
            y_pred = model(x)
            loss = cross_entropy(y_pred, y)
            loss.backward()
            
            model.finish_gradient_synchronization()

            mem_before = get_peak_memory(f"{label} Step {step+1} before optimizer step")
            torch.cuda.reset_peak_memory_stats()

            optimizer.step()
            torch.cuda.synchronize()

            end = timeit.default_timer()
            step_time = end - start
            step_times.append(step_time)

            mem_after = get_peak_memory(f"{label} Step {step+1} after optimizer step")
            torch.cuda.reset_peak_memory_stats()

            mem_before_list.append(mem_before)
            mem_after_list.append(mem_after)

            if rank == 0:
                print(f"[{label}] Eval step {step+1}: {step_time:.6f} s")

        if rank == 0:
            avg_time = sum(step_times) / len(step_times)
            avg_mem_before = sum(mem_before_list) / len(mem_before_list)
            avg_mem_after = sum(mem_after_list) / len(mem_after_list)

            results.append({
                "method": label,
                "avg_step_time_s": avg_time,
                "mem_after_init_GB": mem_after_init,
                "avg_mem_before_step_GB": avg_mem_before,
                "avg_mem_after_step_GB": avg_mem_after
            })

    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"[Rank {rank}] OOM error caught!")
            if rank == 0:
                results.append({
                    "method": label,
                    "avg_step_time_s": "OOM",
                    "mem_after_init_GB": "OOM",
                    "avg_mem_before_step_GB": "OOM",
                    "avg_mem_after_step_GB": "OOM"
                })
        else:
            raise
    finally:
        cleanup()

def run_speed_memory_benchmark():
    world_size = 2
    manager = mp.Manager()
    results = manager.list()

    print("\n=== Running AdamW benchmark ===")
    mp.spawn(benchmark_oss_speed_memory, args=(world_size, False, results), nprocs=world_size, join=True)

    print("\n=== Running OSS benchmark ===")
    mp.spawn(benchmark_oss_speed_memory, args=(world_size, True, results), nprocs=world_size, join=True)

    df = pd.DataFrame(list(results))
    print("\n=== Speed + Memory Summary ===")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_speed_memory_benchmark()
