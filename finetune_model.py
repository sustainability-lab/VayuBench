"""
Finetune a small LLM on VayuBench using LoRA and 4-bit quantization.

This script uses PEFT (Parameter-Efficient Fine-Tuning) with LoRA adapters
to finetune models efficiently on consumer GPUs.
"""

import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset


def finetune_model(
    model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
    output_dir="./finetuned_model",
    train_data_file="train_data.json",
    val_data_file="val_data.json",
    max_seq_length=1024,
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.05
):
    """
    Finetune a model using LoRA.

    Args:
        model_name: HuggingFace model identifier
        output_dir: Directory to save finetuned model
        train_data_file: Path to training data JSON
        val_data_file: Path to validation data JSON
        max_seq_length: Maximum sequence length
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        gradient_accumulation_steps: Steps to accumulate gradients
        learning_rate: Learning rate
        lora_rank: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
    """
    print("=" * 60)
    print("VayuBench Model Finetuning")
    print("=" * 60)
    print(f"\nModel: {model_name}")
    print(f"Output directory: {output_dir}")
    print(f"Training data: {train_data_file}")
    print(f"Validation data: {val_data_file}")
    print(f"\nTraining configuration:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  - Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - LoRA rank: {lora_rank}")
    print(f"  - LoRA alpha: {lora_alpha}")
    print(f"  - LoRA dropout: {lora_dropout}")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("json", data_files={
        "train": train_data_file,
        "validation": val_data_file
    })
    print(f"  Train samples: {len(dataset['train']):,}")
    print(f"  Validation samples: {len(dataset['validation']):,}")

    # Quantization config for memory efficiency
    print("\nConfiguring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load model and tokenizer
    print(f"Loading model: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare model for k-bit training
    print("Preparing model for k-bit training...")
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration
    print(f"Configuring LoRA (rank={lora_rank}, alpha={lora_alpha})...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Apply LoRA
    print("Applying LoRA adapters...")
    model = get_peft_model(model, lora_config)

    print("\nTrainable parameters:")
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"  Total parameters: {all_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable %: {100 * trainable_params / all_params:.2f}%")

    # Format function for chat template
    def format_chat(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False
        )

    # Training arguments
    print("\nConfiguring training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        eval_steps=100,
        save_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="tensorboard"
    )

    # Initialize trainer
    print("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        formatting_func=format_chat,
        packing=False
    )

    # Start training
    print("\n" + "=" * 60)
    print("Starting finetuning...")
    print("=" * 60)
    print("\nMonitor progress with TensorBoard:")
    print(f"  tensorboard --logdir {output_dir}/runs")
    print()

    trainer.train()

    # Save final model
    print("\n" + "=" * 60)
    print("Training complete! Saving model...")
    print("=" * 60)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nModel saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Generate responses:")
    print(f"     python batch_generation.py --model_name {output_dir}")
    print("  2. Evaluate performance:")
    print(f"     python eval_pipeline.py --model_name finetuned_model")
    print("  3. Compare with base model:")
    print("     python compare_results.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finetune a model on VayuBench"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./finetuned_model",
        help="Directory to save finetuned model"
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="train_data.json",
        help="Path to training data JSON"
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default="val_data.json",
        help="Path to validation data JSON"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device batch size"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )

    args = parser.parse_args()

    finetune_model(
        model_name=args.model_name,
        output_dir=args.output_dir,
        train_data_file=args.train_data,
        val_data_file=args.val_data,
        max_seq_length=args.max_seq_length,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
