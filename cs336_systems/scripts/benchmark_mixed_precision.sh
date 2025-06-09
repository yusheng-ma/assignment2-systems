#!/bin/bash

small="model.d_model=768 model.d_ff=3072 model.num_layers=12 model.num_heads=12"
medium="model.d_model=1024 model.d_ff=4096 model.num_layers=24 model.num_heads=16"
large="model.d_model=1280 model.d_ff=5120 model.num_layers=36 model.num_heads=20"

overrides=(
    "$small"
    "$small benchmark.mixed_precision=True"
    "$medium"
    "$medium benchmark.mixed_precision=True"
    "$large"
    "$large benchmark.mixed_precision=True"
)

for override in "${overrides[@]}"; do
    echo "Running: $override"
    uv run ./cs336_systems/benchmark_mixed_precision.py \
        --config ./cs336_systems/configs/base.yaml \
        --override $override
done