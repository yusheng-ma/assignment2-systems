import torch
import argparse
import numpy as np
from timeit import default_timer as timer
from contextlib import nullcontext
from omegaconf import OmegaConf
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy


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

    autocast_context = torch.autocast(device_type=cfg.benchmark.device, dtype=torch.float16) if cfg.benchmark.mixed_precision else nullcontext()

    # warmup
    for _ in range(cfg.benchmark.warmup_steps):
        torch.cuda.synchronize()

        with autocast_context:
            y_pred = model(x)
            
            if not cfg.benchmark.forward_only:
                loss = cross_entropy(y_pred, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        torch.cuda.synchronize()

    # eval
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    for _ in range(cfg.benchmark.eval_steps):
        torch.cuda.synchronize()

        with autocast_context:
        # measure forward
            y_pred = model(x)

            if not cfg.benchmark.forward_only:
                loss = cross_entropy(y_pred, y)

                # measure backward
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    
        torch.cuda.synchronize()

    output_filename_pre = f"outputs/small_cxt{cfg.model.context_length}"
    if cfg.benchmark.forward_only:
        output_filename_pre += "_forwardonly"
    if cfg.benchmark.mixed_precision:
        output_filename_pre += "_mixedprecision"

    torch.cuda.memory._dump_snapshot(f"{output_filename_pre}.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)

    print("Fininsh profiling")


if __name__ == "__main__":
    main()