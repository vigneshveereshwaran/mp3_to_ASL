"""
HearLink ASL — Model Export
Convert the fine-tuned T5 model to CTranslate2 format for fast inference.
"""

import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def export_to_ctranslate2(model_path: str, output_dir: str,
                          quantization: str = "int8"):
    """
    Convert a Hugging Face T5 model to CTranslate2 format.

    Args:
        model_path: Path to the trained HF model directory
        output_dir: Output directory for CTranslate2 model
        quantization: Quantization type ('int8', 'float16', 'float32')
    """
    import ctranslate2

    print("=" * 60)
    print("HearLink ASL — Model Export to CTranslate2")
    print("=" * 60)

    print(f"\nSource model: {model_path}")
    print(f"Output dir:   {output_dir}")
    print(f"Quantization: {quantization}")

    # Convert
    print("\nConverting model...")
    start = time.time()

    converter = ctranslate2.converters.TransformersConverter(model_path)
    converter.convert(output_dir, quantization=quantization, force=True)

    elapsed = time.time() - start
    print(f"Conversion complete in {elapsed:.1f}s")

    # Verify the output
    output_path = Path(output_dir)
    model_file = output_path / "model.bin"
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"Model size: {size_mb:.1f} MB")
    else:
        print("[warning] model.bin not found in output directory")

    return output_dir


def benchmark_inference(ct2_model_dir: str, model_path: str,
                        task_prefix: str = "translate English to ASL gloss: ",
                        n_iterations: int = 100):
    """
    Benchmark CTranslate2 inference latency.

    Args:
        ct2_model_dir: Path to CTranslate2 model directory
        model_path: Path to original HF model (for tokenizer)
        task_prefix: T5 task prefix
        n_iterations: Number of inference iterations for benchmarking
    """
    import ctranslate2
    from transformers import AutoTokenizer

    print("\n" + "=" * 60)
    print("Inference Benchmark")
    print("=" * 60)

    # Load model and tokenizer
    translator = ctranslate2.Translator(ct2_model_dir, compute_type="int8")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Test sentences of varying lengths
    test_sentences = [
        "hello",
        "how are you",
        "I want to learn sign language",
        "Yesterday I went to the store to buy some food",
        "She told me that she would come to the meeting tomorrow morning at nine o'clock",
    ]

    print(f"\nRunning {n_iterations} iterations per sentence...\n")

    for sentence in test_sentences:
        input_text = task_prefix + sentence
        input_tokens = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(input_text)
        )

        # Warmup
        for _ in range(5):
            translator.translate_batch([input_tokens], beam_size=4)

        # Benchmark
        latencies = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            results = translator.translate_batch([input_tokens], beam_size=4)
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)

        output_tokens = results[0].hypotheses[0]
        output_text = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(output_tokens),
            skip_special_tokens=True
        )

        avg_ms = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]

        words = len(sentence.split())
        print(f"  Input ({words} words): \"{sentence}\"")
        print(f"  Output: \"{output_text}\"")
        print(f"  Latency: avg={avg_ms:.1f}ms  p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms")
        meets_target = "✓" if avg_ms < 50 else "✗"
        print(f"  < 50ms target: {meets_target}")
        print()

    print("Benchmark complete!")


def main():
    """Main export pipeline."""
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Export trained model to CTranslate2")
    parser.add_argument("--model_path", default=str(PROJECT_ROOT / "app" / "models" / "best_gloss_model"),
                        help="Path to trained HF model")
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "app" / "models" / "ct2_gloss_model"),
                        help="Output directory for CT2 model")
    parser.add_argument("--quantization", default="int8",
                        choices=["int8", "float16", "float32"],
                        help="Quantization type")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference benchmark after export")
    parser.add_argument("--benchmark_iterations", type=int, default=100,
                        help="Number of benchmark iterations")
    args = parser.parse_args()

    # Export
    export_to_ctranslate2(args.model_path, args.output_dir, args.quantization)

    # Benchmark
    if args.benchmark:
        benchmark_inference(args.output_dir, args.model_path,
                            n_iterations=args.benchmark_iterations)


if __name__ == "__main__":
    main()
