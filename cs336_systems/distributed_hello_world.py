import os
import timeit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def benchmark_all_reduce(rank, world_size, backend, data_size_mb, device_type, results):
    try:
        setup(rank, world_size, backend)

        # 设备选择
        if device_type == 'cpu':
            device = torch.device("cpu")
        else:
            # 分配 GPU
            device_count = torch.cuda.device_count()
            if rank >= device_count:
                raise RuntimeError(f"Rank {rank} exceeds available GPUs {device_count}")
            device = torch.device(f"cuda:{rank % device_count}")

        # 数据张量
        num_elements = (data_size_mb * 1024 * 1024) // 4
        tensor = torch.ones(num_elements, dtype=torch.float32, device=device)

        # 预热
        for _ in range(5):
            dist.all_reduce(tensor, async_op=False)

        if device_type == 'gpu':
            torch.cuda.synchronize()

        # 计时
        start = timeit.default_timer()
        for _ in range(10):
            dist.all_reduce(tensor, async_op=False)
        if device_type == 'gpu':
            torch.cuda.synchronize()
        end = timeit.default_timer()

        avg_time = (end - start) / 10
        if rank == 0:
            results.append({
                'backend': backend,
                'device_type': device_type,
                'data_size_mb': data_size_mb,
                'world_size': world_size,
                'avg_time_sec': avg_time
            })

    except Exception as e:
        if rank == 0:
            print(f"[ERROR] backend={backend} device={device_type} size={data_size_mb}MB world_size={world_size} failed: {e}")
    finally:
        cleanup()

def run_benchmarks():
    configs = [
        {'backend': 'gloo', 'device_type': 'cpu'},
        {'backend': 'nccl', 'device_type': 'gpu'}
    ]
    data_sizes = [1, 10, 100, 1024]
    world_sizes = [2, 4, 6]

    results = mp.Manager().list()

    for cfg in configs:
        for ws in world_sizes:
            # 检查 NCCL 是否会超出 GPU 数量
            if cfg['backend'] == 'nccl' and ws > torch.cuda.device_count():
                print(f"Skipping NCCL with {ws} processes: only {torch.cuda.device_count()} GPUs available")
                continue

            for size in data_sizes:
                print(f"Running: backend={cfg['backend']} device={cfg['device_type']} size={size}MB world_size={ws}")
                mp.spawn(
                    benchmark_all_reduce,
                    args=(ws, cfg['backend'], size, cfg['device_type'], results),
                    nprocs=ws,
                    join=True
                )

    return list(results)

if __name__ == "__main__":
    final_results = run_benchmarks()
    import pandas as pd
    df = pd.DataFrame(final_results)
    print(df)
