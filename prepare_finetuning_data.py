"""
Prepare VayuBench questions for finetuning.

This script converts the benchmark questions into a chat completion format
suitable for finetuning LLMs with the TRL library.
"""

import pandas as pd
import json
import argparse
from pathlib import Path


def prepare_finetuning_data(
    questions_file="questions.csv",
    system_prompt_file="system_prompt.txt",
    output_dir=".",
    train_split=0.9
):
    """
    Convert VayuBench questions to training format.

    Args:
        questions_file: Path to questions CSV
        system_prompt_file: Path to system prompt
        output_dir: Directory to save training files
        train_split: Fraction of data to use for training (rest is validation)
    """
    # Load benchmark questions
    print(f"Loading questions from {questions_file}...")
    questions_df = pd.read_csv(questions_file)
    print(f"Loaded {len(questions_df)} questions")

    # Load system prompt
    print(f"Loading system prompt from {system_prompt_file}...")
    with open(system_prompt_file, "r") as f:
        system_prompt = f.read().strip()

    # Prepare training data
    print("Converting to chat format...")
    training_data = []

    for idx, row in questions_df.iterrows():
        # Format as chat completion
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": row["question"]
                },
                {
                    "role": "assistant",
                    "content": row["canonical_solution"]
                }
            ]
        }
        training_data.append(conversation)

    # Split into train/validation
    split_idx = int(len(training_data) * train_split)
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]

    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    train_file = output_path / "train_data.json"
    val_file = output_path / "val_data.json"

    print(f"Saving training data to {train_file}...")
    with open(train_file, "w") as f:
        json.dump(train_data, f, indent=2)

    print(f"Saving validation data to {val_file}...")
    with open(val_file, "w") as f:
        json.dump(val_data, f, indent=2)

    print("\nData preparation complete!")
    print(f"Training samples: {len(train_data):,}")
    print(f"Validation samples: {len(val_data):,}")
    print(f"\nFiles created:")
    print(f"  - {train_file}")
    print(f"  - {val_file}")

    # Show example
    print("\nExample training sample:")
    print(json.dumps(train_data[0], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare VayuBench data for finetuning"
    )
    parser.add_argument(
        "--questions_file",
        type=str,
        default="questions.csv",
        help="Path to questions CSV file"
    )
    parser.add_argument(
        "--system_prompt_file",
        type=str,
        default="system_prompt.txt",
        help="Path to system prompt file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save output files"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.9,
        help="Fraction of data to use for training (default: 0.9)"
    )

    args = parser.parse_args()

    prepare_finetuning_data(
        questions_file=args.questions_file,
        system_prompt_file=args.system_prompt_file,
        output_dir=args.output_dir,
        train_split=args.train_split
    )
