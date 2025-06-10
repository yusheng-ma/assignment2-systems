#!/bin/bash

mkdir -p ./outputs

small="model.d_model=768 model.d_ff=3072 model.num_layers=12 model.num_heads=12"

overrides=(
    "$small model.context_length=128 benchmark.forward_only=True benchmark.mixed_precision=True"
    "$small model.context_length=256 benchmark.forward_only=True benchmark.mixed_precision=True"
    "$small model.context_length=512 benchmark.forward_only=True benchmark.mixed_precision=True"
    "$small model.context_length=128 benchmark.forward_only=True"
    "$small model.context_length=256 benchmark.forward_only=True"
    "$small model.context_length=512 benchmark.forward_only=True"
    "$small model.context_length=128 benchmark.mixed_precision=True"
    "$small model.context_length=256 benchmark.mixed_precision=True"
    "$small model.context_length=512 benchmark.mixed_precision=True"
    "$small model.context_length=128"
    "$small model.context_length=256"
    "$small model.context_length=512"
)

for override in "${overrides[@]}"; do
    echo "Running: $override"
    uv run ./cs336_systems/profile_memory.py \
        --config ./cs336_systems/configs/base.yaml \
        --override $override
done