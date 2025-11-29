# VayuBench Model Finetuning Guide

Quick reference for finetuning models on VayuBench.

## Prerequisites

```bash
# Install finetuning dependencies
pip install -r requirements-finetuning.txt
```

## Quick Start

### 1. Prepare Training Data

```bash
python prepare_finetuning_data.py
```

This creates:
- `train_data.json` (4,500 samples)
- `val_data.json` (500 samples)

### 2. Finetune Model

```bash
python finetune_model.py \
    --model_name "Qwen/Qwen2.5-Coder-3B-Instruct" \
    --output_dir "./finetuned_model" \
    --num_epochs 3
```

### 3. Generate Responses

```bash
python batch_generation.py \
    --model_name "./finetuned_model" \
    --questions_file questions.csv \
    --batch_size 10 \
    --num_samples 5
```

### 4. Evaluate Performance

```bash
python eval_pipeline.py \
    --model_name "finetuned_model" \
    --starts 0 \
    --ends 5000
```

### 5. Compare Results

```bash
python compare_results.py \
    --base_model "Qwen/Qwen2.5-Coder-3B-Instruct" \
    --finetuned_model "finetuned_model"
```

## Recommended Models

| Model | Size | Base exec@1 | Base pass@1 | Finetuning Potential |
|-------|------|:-----------:|:-----------:|---------------------|
| Qwen2.5-Coder-3B | 3B | 0.73 | 0.33 | High |
| Qwen2.5-Coder-1.5B | 1.5B | 0.47 | 0.08 | Very High |
| DeepSeek-Coder-6.7B | 6.7B | 0.77 | 0.48 | Medium |

## Memory Requirements

| Model | GPU Memory | Recommended GPU |
|-------|------------|-----------------|
| 1.5B | 8-12 GB | RTX 3060, RTX 4060 |
| 3B | 12-16 GB | RTX 3090, RTX 4090 |
| 6.7B | 16-24 GB | A40, A100 40GB |

## Customization

### Adjust Learning Rate

```bash
python finetune_model.py --learning_rate 1e-4  # More conservative
python finetune_model.py --learning_rate 5e-4  # More aggressive
```

### Adjust LoRA Rank

```bash
python finetune_model.py --lora_rank 8   # Fewer parameters
python finetune_model.py --lora_rank 32  # More parameters
```

### Reduce Memory Usage

```bash
python finetune_model.py \
    --batch_size 2 \
    --gradient_accumulation_steps 8
```

## Monitoring

Track training progress with TensorBoard:

```bash
tensorboard --logdir ./finetuned_model/runs
```

## Expected Improvements

After finetuning Qwen2.5-Coder-3B:

- exec@1: 0.73 → 0.82-0.88 (+12-20%)
- pass@1: 0.33 → 0.48-0.58 (+45-75%)
- pass@2: 0.47 → 0.62-0.72 (+32-53%)

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python finetune_model.py --batch_size 2
```

### Slow Training

Use smaller model or reduce epochs:
```bash
python finetune_model.py --num_epochs 2
```

### Poor Results

Try higher learning rate or more epochs:
```bash
python finetune_model.py --learning_rate 5e-4 --num_epochs 5
```

## Full Documentation

See [finetuning.qmd](finetuning.qmd) for comprehensive documentation.
