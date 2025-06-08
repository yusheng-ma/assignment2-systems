#!/bin/bash

overrides=(
    "model.d_model=768 model.d_ff=3072 model.num_layers=12 model.num_heads=12"
    "model.d_model=1024 model.d_ff=4096 model.num_layers=24 model.num_heads=16"
    "model.d_model=1280 model.d_ff=5120 model.num_layers=36 model.num_heads=20"
    "model.d_model=768 model.d_ff=3072 model.num_layers=12 model.num_heads=12 benchmark.warmup_steps=0"
    "model.d_model=1024 model.d_ff=4096 model.num_layers=24 model.num_heads=16 benchmark.warmup_steps=0"
    "model.d_model=1280 model.d_ff=5120 model.num_layers=36 model.num_heads=20 benchmark.warmup_steps=0"
    "model.d_model=768 model.d_ff=3072 model.num_layers=12 model.num_heads=12 benchmark.warmup_steps=2"
    "model.d_model=1024 model.d_ff=4096 model.num_layers=24 model.num_heads=16 benchmark.warmup_steps=2"
    "model.d_model=1280 model.d_ff=5120 model.num_layers=36 model.num_heads=20 benchmark.warmup_steps=2"
)

for override in "${overrides[@]}"; do
    echo "Running: $override"
    uv run ./cs336_systems/benchmark.py \
        --config ./cs336_systems/configs/base.yaml \
        --override $override
done