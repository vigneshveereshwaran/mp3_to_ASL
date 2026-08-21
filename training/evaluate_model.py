"""
HearLink ASL — Model Evaluation
Comprehensive evaluation of the trained English → ASL Gloss model.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def evaluate_model(model_path: str, test_file: str = None,
                   task_prefix: str = "translate English to ASL gloss: ",
                   num_beams: int = 4, max_samples: int = None):
    """
    Evaluate a trained English → ASL Gloss model on the test set.

    Args:
        model_path: Path to saved model directory
        test_file: Path to test JSONL file
        task_prefix: Task prefix for T5 input
        num_beams: Beam search width
        max_samples: Max test samples (None = all)
    """
    import sacrebleu

    print("=" * 60)
    print("HearLink ASL — Model Evaluation")
    print("=" * 60)

    if test_file is None:
        test_file = str(PROJECT_ROOT / "datasets" / "data" / "test.jsonl")

    # Load model
    print(f"\nLoading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Device: {device}")

    # Load test data
    print(f"Loading test data from: {test_file}")
    test_pairs = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                test_pairs.append((data['en'], data['gloss']))

    if max_samples:
        test_pairs = test_pairs[:max_samples]

    print(f"Test samples: {len(test_pairs)}")

    # Run inference
    print("\nRunning inference...")
    predictions = []
    references = []

    with torch.no_grad():
        for i, (en, gloss) in enumerate(test_pairs):
            input_text = task_prefix + en
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_length=64,
                num_beams=num_beams,
                early_stopping=True,
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            predictions.append(decoded)
            references.append(gloss)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(test_pairs)}")

    # Compute metrics
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)

    # BLEU
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print(f"\nBLEU Score: {bleu.score:.2f}")
    print(f"  BLEU-1: {bleu.precisions[0]:.2f}")
    print(f"  BLEU-2: {bleu.precisions[1]:.2f}")
    print(f"  BLEU-3: {bleu.precisions[2]:.2f}")
    print(f"  BLEU-4: {bleu.precisions[3]:.2f}")

    # Exact Match
    exact = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    exact_rate = exact / len(predictions) * 100
    print(f"\nExact Match: {exact_rate:.2f}% ({exact}/{len(predictions)})")

    # Token-level accuracy
    total_tokens = 0
    correct_tokens = 0
    for pred, ref in zip(predictions, references):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        max_len = max(len(pred_tokens), len(ref_tokens))
        for j in range(max_len):
            total_tokens += 1
            if j < len(pred_tokens) and j < len(ref_tokens):
                if pred_tokens[j] == ref_tokens[j]:
                    correct_tokens += 1
    token_acc = correct_tokens / total_tokens * 100 if total_tokens else 0
    print(f"Token Accuracy: {token_acc:.2f}%")

    # Length statistics
    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]
    print(f"\nAvg prediction length: {np.mean(pred_lengths):.1f} tokens")
    print(f"Avg reference length:  {np.mean(ref_lengths):.1f} tokens")
    print(f"Length ratio: {np.mean(pred_lengths) / np.mean(ref_lengths):.3f}")

    # Sample predictions
    print("\n" + "=" * 60)
    print("Sample Predictions (first 10)")
    print("=" * 60)
    for i in range(min(10, len(predictions))):
        en, ref = test_pairs[i]
        pred = predictions[i]
        match = "✓" if pred.strip() == ref.strip() else "✗"
        print(f"\n  [{match}] English:    {en}")
        print(f"      Reference: {ref}")
        print(f"      Predicted: {pred}")

    # Save results
    results = {
        "bleu": bleu.score,
        "bleu_precisions": bleu.precisions,
        "exact_match": exact_rate,
        "token_accuracy": token_acc,
        "num_samples": len(predictions),
        "avg_pred_length": float(np.mean(pred_lengths)),
        "avg_ref_length": float(np.mean(ref_lengths)),
    }

    results_path = Path(model_path) / "eval_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ASL Gloss translation model")
    parser.add_argument("--model_path", default=str(PROJECT_ROOT / "app" / "models" / "best_gloss_model"),
                        help="Path to trained model directory")
    parser.add_argument("--test_file", default=None, help="Path to test JSONL file")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam search width")
    parser.add_argument("--max_samples", type=int, default=None, help="Max test samples")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.test_file, num_beams=args.num_beams,
                   max_samples=args.max_samples)
