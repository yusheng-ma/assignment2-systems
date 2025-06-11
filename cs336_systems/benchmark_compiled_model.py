import torch
import argparse
import numpy as np
from timeit import default_timer as timer
from omegaconf import OmegaConf
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config yaml file path")
    parser.add_argument("--override", nargs="*", default=[], help="override parameters")
    return parser.parse_args()


def load_cfg(config, override):
    cfg = OmegaConf.load(config)
    override_cfg = OmegaConf.from_dotlist(override)
    cfg = OmegaConf.merge(cfg, override_cfg)
    return cfg


def generate_batch(vocab_size, batch_size, context_length, device):
    size = (batch_size, context_length)
    return torch.randint(0, vocab_size, size=size, device=device), torch.randint(0, vocab_size, size=size, device=device)


def run_benchmark(model, optimizer, x, y, cfg, compiled=False):
    # warm-up
    for _ in range(cfg.benchmark.warmup_steps):
        torch.cuda.synchronize()
        y_pred = model(x)
        if not cfg.benchmark.forward_only:
            loss = cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        torch.cuda.synchronize()

    # eval
    forward_times = []
    backward_times = []
    optimize_times = []
    torch.cuda.reset_peak_memory_stats()

    for _ in range(cfg.benchmark.eval_steps):
        torch.cuda.synchronize()
        forward_start = timer()
        y_pred = model(x)
        torch.cuda.synchronize()
        forward_end = timer()
        forward_times.append(forward_end - forward_start)

        if not cfg.benchmark.forward_only:
            loss = cross_entropy(y_pred, y)
            backward_start = timer()
            loss.backward()
            torch.cuda.synchronize()
            backward_end = timer()
            backward_times.append(backward_end - backward_start)

            optimize_start = timer()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            optimize_end = timer()
            optimize_times.append(optimize_end - optimize_start)

    peak_mem = torch.cuda.max_memory_allocated() / 1e6
    print(f"{'[Compiled]' if compiled else '[Eager]'} Forward pass:  {np.mean(forward_times):.6f}s ± {np.std(forward_times):.6f}s")
    if not cfg.benchmark.forward_only:
        print(f"{'[Compiled]' if compiled else '[Eager]'} Backward pass: {np.mean(backward_times):.6f}s ± {np.std(backward_times):.6f}s")
        print(f"{'[Compiled]' if compiled else '[Eager]'} Optimize: {np.mean(optimize_times):.6f}s ± {np.std(optimize_times):.6f}s")
    print(f"{'[Compiled]' if compiled else '[Eager]'} Peak memory: {peak_mem:.2f} MB\n")


def main():
    args = parse_args()
    cfg = load_cfg(args.config, args.override)

    model = BasicsTransformerLM(
        vocab_size=cfg.model.vocab_size,
        context_length=cfg.model.context_length,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope_theta=cfg.model.rope_theta
    ).to(cfg.benchmark.device)

    optimizer = AdamW(
        params=model.parameters()
    )

    x, y = generate_batch(cfg.model.vocab_size, cfg.benchmark.batch_size, cfg.model.context_length, cfg.benchmark.device)

    print("Running eager benchmark...")
    run_benchmark(model, optimizer, x, y, cfg, compiled=False)

    try:
        print("Running compiled benchmark...")
        compiled_model = torch.compile(model)
        run_benchmark(compiled_model, optimizer, x, y, cfg, compiled=True)
    except RuntimeError as e:
        print(f"[Compiled] Benchmark failed: {e}")


if __name__ == "__main__":
    main()
