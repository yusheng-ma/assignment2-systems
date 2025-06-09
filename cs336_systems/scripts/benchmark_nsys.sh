#!/bin/bash

mkdir -p ./outputs

small="model.d_model=768 model.d_ff=3072 model.num_layers=12 model.num_heads=12"
medium="model.d_model=1024 model.d_ff=4096 model.num_layers=24 model.num_heads=16"
large="model.d_model=1280 model.d_ff=5120 model.num_layers=36 model.num_heads=20"
xl="model.d_model=1600 model.d_ff=6400 model.num_layers=48 model.num_heads=25"
xxl="model.d_model=2560 model.d_ff=10240 model.num_layers=32 model.num_heads=32" # 2.7b

# only 1 2 3 5 not oom (i starts at 1)
overrides=(
    "$small model.context_length=128"
    "$small model.context_length=256"
    "$small model.context_length=512"
    "$small model.context_length=1024"
    "$medium model.context_length=128"
    "$medium model.context_length=256"
    "$medium model.context_length=512"
    "$medium model.context_length=1024"
    "$large model.context_length=128"
    "$large model.context_length=256"
    "$large model.context_length=512"
    "$large model.context_length=1024"
    "$xl model.context_length=128"
    "$xl model.context_length=256"
    "$xl model.context_length=512"
    "$xl model.context_length=1024"
    "$xxl model.context_length=128"
    "$xxl model.context_length=256"
    "$xxl model.context_length=512"
    "$xxl model.context_length=1024"
)

i=1
for override in "${overrides[@]}"; do
    printf -v output_file "./outputs/result_%02d" "$i"

    echo "Running: $output_file.qdrep"
    uv run nsys profile -o "$output_file" python ./cs336_systems/benchmark_nsys.py \
        --config ./cs336_systems/configs/base.yaml \
        --override $override

    ((i++))
done
