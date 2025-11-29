"""
Compare evaluation results between base and finetuned models.

This script analyzes results from eval_pipeline.py and compares performance
metrics between different models.
"""

import json
import glob
import argparse
from pathlib import Path
from collections import defaultdict
import pandas as pd


def load_results(model_name, results_dir="results_chunk"):
    """
    Load all evaluation results for a model.

    Args:
        model_name: Name of model directory
        results_dir: Base directory containing results

    Returns:
        Dictionary with aggregated metrics
    """
    # Handle different path formats
    model_path = Path(results_dir) / model_name
    if not model_path.exists():
        # Try without results_dir prefix
        model_path = Path(model_name)
        if not model_path.exists():
            raise ValueError(f"Results not found for model: {model_name}")

    result_files = list(model_path.glob("**/*.json"))

    if not result_files:
        raise ValueError(f"No result files found in {model_path}")

    print(f"Loading results for {model_name}...")
    print(f"  Found {len(result_files)} result files")

    metrics = {
        "exec@1": [],
        "pass@1": [],
        "pass@2": [],
        "errors": defaultdict(int),
        "by_category": defaultdict(lambda: {
            "exec@1": [],
            "pass@1": [],
            "pass@2": [],
            "count": 0
        })
    }

    for file in result_files:
        try:
            with open(file) as f:
                result = json.load(f)

            # Overall metrics
            metrics["exec@1"].append(result.get("exec@1", 0))
            metrics["pass@1"].append(result.get("pass@1", 0))
            metrics["pass@2"].append(result.get("pass@2", 0))

            # Error types
            if result.get("error_type"):
                metrics["errors"][result["error_type"]] += 1

            # Category-specific metrics
            category = result.get("category", "unknown")
            metrics["by_category"][category]["exec@1"].append(
                result.get("exec@1", 0)
            )
            metrics["by_category"][category]["pass@1"].append(
                result.get("pass@1", 0)
            )
            metrics["by_category"][category]["pass@2"].append(
                result.get("pass@2", 0)
            )
            metrics["by_category"][category]["count"] += 1

        except Exception as e:
            print(f"  Warning: Could not load {file}: {e}")
            continue

    # Aggregate metrics
    aggregated = {
        "exec@1": sum(metrics["exec@1"]) / len(metrics["exec@1"]),
        "pass@1": sum(metrics["pass@1"]) / len(metrics["pass@1"]),
        "pass@2": sum(metrics["pass@2"]) / len(metrics["pass@2"]),
        "errors": dict(metrics["errors"]),
        "total_samples": len(metrics["exec@1"])
    }

    # Aggregate by category
    aggregated["by_category"] = {}
    for category, cat_metrics in metrics["by_category"].items():
        if cat_metrics["count"] > 0:
            aggregated["by_category"][category] = {
                "exec@1": sum(cat_metrics["exec@1"]) / len(cat_metrics["exec@1"]),
                "pass@1": sum(cat_metrics["pass@1"]) / len(cat_metrics["pass@1"]),
                "pass@2": sum(cat_metrics["pass@2"]) / len(cat_metrics["pass@2"]),
                "count": cat_metrics["count"]
            }

    return aggregated


def print_comparison(base_results, finetuned_results, base_name, finetuned_name):
    """Print formatted comparison of results."""

    print("\n" + "=" * 70)
    print("OVERALL PERFORMANCE COMPARISON")
    print("=" * 70)
    print(f"\nBase Model: {base_name}")
    print(f"Finetuned Model: {finetuned_name}")
    print(f"\nSamples evaluated:")
    print(f"  Base: {base_results['total_samples']:,}")
    print(f"  Finetuned: {finetuned_results['total_samples']:,}")

    print(f"\n{'Metric':<15} {'Base':>12} {'Finetuned':>12} {'Change':>12} {'Improvement':>12}")
    print("-" * 70)

    for metric in ["exec@1", "pass@1", "pass@2"]:
        base = base_results[metric]
        finetuned = finetuned_results[metric]
        change = finetuned - base
        improvement = ((finetuned - base) / base) * 100 if base > 0 else 0

        print(
            f"{metric:<15} "
            f"{base:>12.4f} "
            f"{finetuned:>12.4f} "
            f"{change:>+12.4f} "
            f"{improvement:>+11.1f}%"
        )

    # Error distribution
    print("\n" + "=" * 70)
    print("ERROR DISTRIBUTION")
    print("=" * 70)
    print(f"\n{'Error Type':<15} {'Base':>12} {'Finetuned':>12} {'Change':>12}")
    print("-" * 55)

    all_error_types = set(base_results["errors"].keys()) | set(finetuned_results["errors"].keys())
    for error_type in sorted(all_error_types):
        base = base_results["errors"].get(error_type, 0)
        finetuned = finetuned_results["errors"].get(error_type, 0)
        change = finetuned - base

        print(
            f"{error_type:<15} "
            f"{base:>12,} "
            f"{finetuned:>12,} "
            f"{change:>+12,}"
        )

    # Category-wise comparison
    print("\n" + "=" * 70)
    print("CATEGORY-WISE PERFORMANCE")
    print("=" * 70)

    all_categories = set(base_results["by_category"].keys()) | set(finetuned_results["by_category"].keys())

    for category in sorted(all_categories):
        if category not in base_results["by_category"] or category not in finetuned_results["by_category"]:
            continue

        base_cat = base_results["by_category"][category]
        finetuned_cat = finetuned_results["by_category"][category]

        print(f"\n{category.upper()} ({base_cat['count']} samples)")
        print("-" * 70)
        print(f"{'Metric':<15} {'Base':>12} {'Finetuned':>12} {'Change':>12} {'Improvement':>12}")
        print("-" * 70)

        for metric in ["exec@1", "pass@1", "pass@2"]:
            base = base_cat[metric]
            finetuned = finetuned_cat[metric]
            change = finetuned - base
            improvement = ((finetuned - base) / base) * 100 if base > 0 else 0

            print(
                f"{metric:<15} "
                f"{base:>12.4f} "
                f"{finetuned:>12.4f} "
                f"{change:>+12.4f} "
                f"{improvement:>+11.1f}%"
            )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    exec_improvement = ((finetuned_results["exec@1"] - base_results["exec@1"]) / base_results["exec@1"]) * 100
    pass_improvement = ((finetuned_results["pass@1"] - base_results["pass@1"]) / base_results["pass@1"]) * 100

    print(f"\nOverall exec@1 improvement: {exec_improvement:+.1f}%")
    print(f"Overall pass@1 improvement: {pass_improvement:+.1f}%")

    # Find categories with biggest improvements
    category_improvements = []
    for category in all_categories:
        if category in base_results["by_category"] and category in finetuned_results["by_category"]:
            base_pass = base_results["by_category"][category]["pass@1"]
            finetuned_pass = finetuned_results["by_category"][category]["pass@1"]
            improvement = ((finetuned_pass - base_pass) / base_pass) * 100 if base_pass > 0 else 0
            category_improvements.append((category, improvement))

    category_improvements.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 3 improved categories:")
    for i, (category, improvement) in enumerate(category_improvements[:3], 1):
        print(f"  {i}. {category}: {improvement:+.1f}%")

    print("\nBottom 3 categories:")
    for i, (category, improvement) in enumerate(category_improvements[-3:], 1):
        print(f"  {i}. {category}: {improvement:+.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Compare evaluation results between models"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name or path to results"
    )
    parser.add_argument(
        "--finetuned_model",
        type=str,
        required=True,
        help="Finetuned model name or path to results"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results_chunk",
        help="Base directory containing results"
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        help="Optional: Save comparison to CSV file"
    )

    args = parser.parse_args()

    # Load results
    try:
        base_results = load_results(args.base_model, args.results_dir)
        finetuned_results = load_results(args.finetuned_model, args.results_dir)
    except Exception as e:
        print(f"Error loading results: {e}")
        return

    # Print comparison
    print_comparison(
        base_results,
        finetuned_results,
        args.base_model,
        args.finetuned_model
    )

    # Save to CSV if requested
    if args.save_csv:
        data = []
        for metric in ["exec@1", "pass@1", "pass@2"]:
            data.append({
                "metric": metric,
                "base": base_results[metric],
                "finetuned": finetuned_results[metric],
                "improvement": ((finetuned_results[metric] - base_results[metric]) / base_results[metric]) * 100
            })

        df = pd.DataFrame(data)
        df.to_csv(args.save_csv, index=False)
        print(f"\nComparison saved to {args.save_csv}")


if __name__ == "__main__":
    main()
