#!/usr/bin/env python3
"""
Single model evaluation script.
Usage: python evaluate_single_model.py <model_name>
"""

import sys
import json
from pathlib import Path
from evaluate_models import ModelEvaluator
from eval import preprocess, preprocess_gj


def evaluate_single_model(model_name: str):
    """Evaluate a single model."""

    # Model configurations
    models = {
        "original": {
            "model_path": "models/original_gpt2_model",
            "tokenizer_path": "models/original_tokenizer"
        },
        "filtered_all": {
            "model_path": "models/filtered_all_gpt2_model",
            "tokenizer_path": "models/filtered_all_tokenizer"
        },
        "filtered_and": {
            "model_path": "models/filtered_and_gpt2_model",
            "tokenizer_path": "models/filtered_and_tokenizer"
        }
    }

    if model_name not in models:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {list(models.keys())}")
        return

    print(f"Evaluating {model_name} model...")

    # Load evaluation data
    print("Loading evaluation data...")
    alike_sentences = preprocess("evaluation/alike")
    unlike_sentences = preprocess("evaluation/unlike")
    gj_tests = preprocess_gj("evaluation/unlike_gj")

    print(f"Loaded {len(alike_sentences)} alike sentences")
    print(f"Loaded {len(unlike_sentences)} unlike sentences")
    print(f"Loaded {len(gj_tests)} grammaticality judgment tests")

    # Initialize evaluator
    config = models[model_name]
    evaluator = ModelEvaluator(
        model_path=config["model_path"],
        tokenizer_path=config["tokenizer_path"]
    )

    # Load model
    evaluator.load_model()

    # Evaluate on all datasets
    print("\nEvaluating on alike sentences...")
    alike_results = evaluator.evaluate_alike_unlike(alike_sentences)

    print("Evaluating on unlike sentences...")
    unlike_results = evaluator.evaluate_alike_unlike(unlike_sentences)

    print("Evaluating on grammaticality judgment tests...")
    gj_results = evaluator.evaluate_grammaticality_judgment(gj_tests)

    # Print results
    print(f"\n{'='*60}")
    print(f"Results for {model_name} model:")
    print(f"{'='*60}")

    print(f"\nAlike Sentences:")
    print(f"  Total sentences: {alike_results.get('total_sentences', 'N/A')}")
    print(f"  Valid scores: {alike_results.get('valid_scores', 'N/A')}")
    mean_ppl = alike_results.get('mean_perplexity', 'N/A')
    if isinstance(mean_ppl, (int, float)):
        print(f"  Mean perplexity: {mean_ppl:.2f}")
    else:
        print(f"  Mean perplexity: {mean_ppl}")
    std_ppl = alike_results.get('std_perplexity', 'N/A')
    if isinstance(std_ppl, (int, float)):
        print(f"  Std perplexity: {std_ppl:.2f}")
    else:
        print(f"  Std perplexity: {std_ppl}")

    print(f"\nUnlike Sentences:")
    print(f"  Total sentences: {unlike_results.get('total_sentences', 'N/A')}")
    print(f"  Valid scores: {unlike_results.get('valid_scores', 'N/A')}")
    mean_ppl = unlike_results.get('mean_perplexity', 'N/A')
    if isinstance(mean_ppl, (int, float)):
        print(f"  Mean perplexity: {mean_ppl:.2f}")
    else:
        print(f"  Mean perplexity: {mean_ppl}")
    std_ppl = unlike_results.get('std_perplexity', 'N/A')
    if isinstance(std_ppl, (int, float)):
        print(f"  Std perplexity: {std_ppl:.2f}")
    else:
        print(f"  Std perplexity: {std_ppl}")

    print(f"\nGrammaticality Judgment Tests:")
    print(f"  Total tests: {gj_results.get('total_tests', 'N/A')}")
    gj_acc = gj_results.get('overall_accuracy', 'N/A')
    if isinstance(gj_acc, (int, float)):
        print(f"  Overall accuracy: {gj_acc:.3f}")
    else:
        print(f"  Overall accuracy: {gj_acc}")
    print(
        f"  Correct predictions: {gj_results.get('correct_predictions', 'N/A')}")
    print(f"  Total predictions: {gj_results.get('total_predictions', 'N/A')}")

    # Print detailed test results
    print(f"\nDetailed Test Results:")
    print("-" * 80)
    for test_result in gj_results.get("test_results", []):
        test_name = test_result["test_purpose"]
        correct = test_result["correct_pairs"]
        total = test_result["total_pairs"]
        accuracy = correct / total if total > 0 else 0
        print(f"  {test_name:<30} {accuracy:.3f} ({correct}/{total})")

    # Save results
    results = {
        "model_name": model_name,
        "alike": alike_results,
        "unlike": unlike_results,
        "grammaticality_judgment": gj_results
    }

    output_file = f"evaluation_{model_name}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_single_model.py <model_name>")
        print("Available models: original, filtered_all, filtered_and")
        sys.exit(1)

    model_name = sys.argv[1]
    evaluate_single_model(model_name)
