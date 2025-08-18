#!/bin/bash

models=(
    "unsloth/Qwen2.5-Coder-14B-Instruct-bnb-4bit"
    "meta-llama/CodeLlama-7b-Instruct-hf"
    "meta-llama/CodeLlama-13b-Instruct-hf"
)


for str in "${models[@]}"; do
    echo "Running evaluation for model: $str"
    nohup python3 eval_pipeline.py --model "$str" > output.log 2>&1 
    echo "Finished evaluation for model: $str"
done

echo "All models completed!"