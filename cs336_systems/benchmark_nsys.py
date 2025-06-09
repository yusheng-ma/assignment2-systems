import torch
import argparse
from omegaconf import OmegaConf
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
import torch.cuda.nvtx as nvtx


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
    
    x, y = generate_batch(cfg.model.vocab_size, cfg.benchmark.batch_size, cfg.model.context_length, cfg.benchmark.device)

    # warmup
    for _ in range(cfg.benchmark.warmup_steps):
        torch.cuda.synchronize()
        y_pred = model(x)
        
        if not cfg.benchmark.forward_only:
            loss = cross_entropy(y_pred, y)
            loss.backward()
        
        torch.cuda.synchronize()

    # eval
    for _ in range(cfg.benchmark.eval_steps):
        torch.cuda.synchronize()

        # measure forward
        with nvtx.range("forward"):
            y_pred = model(x)

        if not cfg.benchmark.forward_only:
            loss = cross_entropy(y_pred, y)

            # measure backward
            with nvtx.range("backward"):
                loss.backward()
            torch.cuda.synchronize()

    print(f"Finish nsys")


if __name__ == "__main__":
    main()