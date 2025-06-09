#!/bin/bash

mkdir -p ./outputs

small="model.d_model=768 model.d_ff=3072 model.num_layers=12 model.num_heads=12"

overrides=(
    "$small"
)

i=1
for override in "${overrides[@]}"; do
    printf -v output_file "./outputs/result_softmax_%02d" "$i"

    echo "Running: $output_file.qdrep"
    uv run nsys profile -o "$output_file" python ./cs336_systems/benchmark_nsys_softmax.py \
        --config ./cs336_systems/configs/base.yaml \
        --override $override

    ((i++))
done
