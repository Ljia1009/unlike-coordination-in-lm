#!/usr/bin/env python3
"""
Model evaluation script for testing trained models against evaluation corpora.
Tests models on alike, unlike, and unlike_gj datasets.
"""

from eval import preprocess, preprocess_gj
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the evaluation directory to path to import our preprocessing functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ModelEvaluator:
    """Evaluator class for testing models on grammaticality judgment tasks."""

    def __init__(self, model_path: str, tokenizer_path: str = None):
        """
        Initialize the evaluator with a model and tokenizer.

        Args:
            model_path: Path to the model directory
            tokenizer_path: Path to tokenizer (if different from model_path)
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.model = None
        self.tokenizer = None
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model from {self.model_path}")
        print(f"Loading tokenizer from {self.tokenizer_path}")

        try:
            # Load tokenizer
            self.tokenizer = GPT2TokenizerFast.from_pretrained(
                self.tokenizer_path)

            # Set padding token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = GPT2LMHeadModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()

            print(f"Model loaded successfully on {self.device}")

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def get_sentence_score(self, sentence: str) -> float:
        """
        Get the perplexity score for a sentence (lower is better).

        Args:
            sentence: Input sentence

        Returns:
            Perplexity score (lower = more likely/grammatical)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Tokenize the sentence
        inputs = self.tokenizer(
            sentence, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get model outputs
            outputs = self.model(**inputs, labels=inputs["input_ids"])

            # Calculate perplexity
            loss = outputs.loss
            perplexity = torch.exp(loss).item()

        return perplexity

    def evaluate_alike_unlike(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Evaluate model on alike/unlike sentences.

        Args:
            sentences: List of sentences to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        scores = []

        for sentence in sentences:
            try:
                score = self.get_sentence_score(sentence)
                scores.append(score)
            except Exception as e:
                print(f"Error processing sentence '{sentence}': {e}")
                continue

        if not scores:
            return {"error": "No valid scores computed"}

        return {
            "total_sentences": len(sentences),
            "valid_scores": len(scores),
            "mean_perplexity": np.mean(scores),
            "std_perplexity": np.std(scores),
            "min_perplexity": np.min(scores),
            "max_perplexity": np.max(scores),
            "scores": scores
        }

    def evaluate_grammaticality_judgment(self, tests: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate model on grammaticality judgment tests.

        Args:
            tests: List of test dictionaries from preprocess_gj()

        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            "total_tests": len(tests),
            "test_results": [],
            "overall_accuracy": 0.0,
            "correct_predictions": 0,
            "total_predictions": 0
        }

        for test in tests:
            test_result = {
                "test_purpose": test["test_purpose"],
                "grammatical_sentences": test["grammatical_sentences"],
                "ungrammatical_sentences": test["ungrammatical_sentences"],
                "grammatical_scores": [],
                "ungrammatical_scores": [],
                "correct_pairs": 0,
                "total_pairs": 0
            }

            # Get scores for grammatical sentences
            for gram_sent in test["grammatical_sentences"]:
                try:
                    score = self.get_sentence_score(gram_sent)
                    test_result["grammatical_scores"].append(score)
                except Exception as e:
                    print(f"Error processing grammatical sentence: {e}")
                    continue

            # Get scores for ungrammatical sentences
            for ungram_sent in test["ungrammatical_sentences"]:
                try:
                    score = self.get_sentence_score(ungram_sent)
                    test_result["ungrammatical_scores"].append(score)
                except Exception as e:
                    print(f"Error processing ungrammatical sentence: {e}")
                    continue

            # Compare pairs (grammatical should have lower perplexity)
            gram_scores = test_result["grammatical_scores"]
            ungram_scores = test_result["ungrammatical_scores"]

            if gram_scores and ungram_scores:
                # For each grammatical sentence, check if it has lower perplexity than ungrammatical ones
                for gram_score in gram_scores:
                    for ungram_score in ungram_scores:
                        test_result["total_pairs"] += 1
                        if gram_score < ungram_score:  # Lower perplexity = more grammatical
                            test_result["correct_pairs"] += 1
                            results["correct_predictions"] += 1
                        results["total_predictions"] += 1

            results["test_results"].append(test_result)

        # Calculate overall accuracy
        if results["total_predictions"] > 0:
            results["overall_accuracy"] = results["correct_predictions"] / \
                results["total_predictions"]

        return results


def evaluate_all_models():
    """Evaluate all models in the models directory."""

    # Define model configurations
    models = [
        {
            "name": "original_gpt2",
            "model_path": "models/original_gpt2_model",
            "tokenizer_path": "models/original_tokenizer"
        },
        {
            "name": "filtered_all_gpt2",
            "model_path": "models/filtered_all_gpt2_model",
            "tokenizer_path": "models/filtered_all_tokenizer"
        },
        {
            "name": "filtered_and_gpt2",
            "model_path": "models/filtered_and_gpt2_model",
            "tokenizer_path": "models/filtered_and_tokenizer"
        }
    ]

    # Load evaluation data
    print("Loading evaluation data...")
    alike_sentences = preprocess("evaluation/alike")
    unlike_sentences = preprocess("evaluation/unlike")
    gj_tests = preprocess_gj("evaluation/unlike_gj")

    print(f"Loaded {len(alike_sentences)} alike sentences")
    print(f"Loaded {len(unlike_sentences)} unlike sentences")
    print(f"Loaded {len(gj_tests)} grammaticality judgment tests")

    # Evaluate each model
    all_results = {}

    for model_config in models:
        print(f"\n{'='*60}")
        print(f"Evaluating {model_config['name']}")
        print(f"{'='*60}")

        try:
            # Initialize evaluator
            evaluator = ModelEvaluator(
                model_path=model_config["model_path"],
                tokenizer_path=model_config["tokenizer_path"]
            )

            # Load model
            evaluator.load_model()

            # Evaluate on alike sentences
            print("\nEvaluating on alike sentences...")
            alike_results = evaluator.evaluate_alike_unlike(alike_sentences)

            # Evaluate on unlike sentences
            print("Evaluating on unlike sentences...")
            unlike_results = evaluator.evaluate_alike_unlike(unlike_sentences)

            # Evaluate on grammaticality judgment tests
            print("Evaluating on grammaticality judgment tests...")
            gj_results = evaluator.evaluate_grammaticality_judgment(gj_tests)

            # Store results
            all_results[model_config["name"]] = {
                "alike": alike_results,
                "unlike": unlike_results,
                "grammaticality_judgment": gj_results
            }

            # Print summary
            print(f"\nSummary for {model_config['name']}:")
            print(
                f"  Alike - Mean perplexity: {alike_results.get('mean_perplexity', 'N/A'):.2f}")
            print(
                f"  Unlike - Mean perplexity: {unlike_results.get('mean_perplexity', 'N/A'):.2f}")
            print(
                f"  GJ - Overall accuracy: {gj_results.get('overall_accuracy', 'N/A'):.3f}")

        except Exception as e:
            print(f"Error evaluating {model_config['name']}: {e}")
            all_results[model_config["name"]] = {"error": str(e)}

    # Save results
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to {output_file}")
    print(f"{'='*60}")

    # Print final comparison
    print("\nFinal Comparison:")
    print("-" * 80)
    print(f"{'Model':<20} {'Alike PPL':<12} {'Unlike PPL':<12} {'GJ Accuracy':<12}")
    print("-" * 80)

    for model_name, results in all_results.items():
        if "error" not in results:
            alike_ppl = results["alike"].get("mean_perplexity", "N/A")
            unlike_ppl = results["unlike"].get("mean_perplexity", "N/A")
            gj_acc = results["grammaticality_judgment"].get(
                "overall_accuracy", "N/A")

            if isinstance(alike_ppl, float):
                alike_ppl = f"{alike_ppl:.2f}"
            if isinstance(unlike_ppl, float):
                unlike_ppl = f"{unlike_ppl:.2f}"
            if isinstance(gj_acc, float):
                gj_acc = f"{gj_acc:.3f}"

            print(f"{model_name:<20} {alike_ppl:<12} {unlike_ppl:<12} {gj_acc:<12}")
        else:
            print(f"{model_name:<20} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")


if __name__ == "__main__":
    evaluate_all_models()
