#!/bin/bash

echo "Running benchmark with default settings"
uv run ./cs336_systems/benchmark_compiled_model.py \
    --config ./cs336_systems/configs/base.yaml


uv run ./cs336_systems/benchmark_compiled_model.py \
    --config ./cs336_systems/configs/base.yaml \
    --override \
    "benchmark.forward_only=True"

echo "Running benchmark with context length 512"
uv run ./cs336_systems/benchmark_compiled_model.py \
    --config ./cs336_systems/configs/base.yaml
    --override \
    "model.context_length=512"

uv run ./cs336_systems/benchmark_compiled_model.py \
    --config ./cs336_systems/configs/base.yaml \
    --override \
    "benchmark.forward_only=True" \
    "model.context_length=512"