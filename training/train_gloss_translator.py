"""
HearLink ASL — T5 English-to-ASL Gloss Translation Trainer
Fine-tunes a T5-small model on parallel English → ASL Gloss data.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ──────────────────────────────────────────────────
# Dataset Class
# ──────────────────────────────────────────────────
class GlossDataset(Dataset):
    """PyTorch Dataset for English → ASL Gloss translation pairs."""

    def __init__(self, file_path: str, tokenizer, max_input_len: int = 128,
                 max_target_len: int = 64, task_prefix: str = "translate English to ASL gloss: "):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len
        self.task_prefix = task_prefix
        self.pairs = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    self.pairs.append((data['en'], data['gloss']))

        print(f"  Loaded {len(self.pairs)} pairs from {file_path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        en, gloss = self.pairs[idx]

        # Tokenize input with task prefix
        input_text = self.task_prefix + en
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        # Tokenize target
        target_encoding = self.tokenizer(
            gloss,
            max_length=self.max_target_len,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        return {
            "input_ids": input_encoding["input_ids"],
            "attention_mask": input_encoding["attention_mask"],
            "labels": target_encoding["input_ids"],
        }


# ──────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────
def compute_metrics(eval_preds, tokenizer):
    """Compute BLEU score and exact match for evaluation."""
    import sacrebleu

    preds, labels = eval_preds

    # Decode predictions
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 with pad token id
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    # BLEU score
    bleu = sacrebleu.corpus_bleu(decoded_preds, [decoded_labels])

    # Exact match
    exact_matches = sum(1 for p, l in zip(decoded_preds, decoded_labels) if p == l)
    exact_match_rate = exact_matches / len(decoded_preds) if decoded_preds else 0

    return {
        "bleu": bleu.score,
        "exact_match": exact_match_rate * 100,
    }


# ──────────────────────────────────────────────────
# Training Pipeline
# ──────────────────────────────────────────────────
def load_config(config_path: str = None) -> dict:
    """Load training configuration from YAML file."""
    if config_path is None:
        config_path = PROJECT_ROOT / "training" / "configs" / "train_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def train(config_path: Optional[str] = None, resume_from: Optional[str] = None):
    """
    Main training function.

    Args:
        config_path: Path to training config YAML
        resume_from: Path to checkpoint to resume from
    """
    print("=" * 60)
    print("HearLink ASL — Model Training")
    print("=" * 60)

    # Load config
    config = load_config(config_path)
    model_name = config["model"]["name"]
    task_prefix = config["model"]["task_prefix"]

    print(f"\nModel: {model_name}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load tokenizer and model
    print(f"\nLoading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load datasets
    print("\nLoading datasets...")
    data_config = config["data"]
    train_dataset = GlossDataset(
        str(PROJECT_ROOT / data_config["train_file"]),
        tokenizer,
        max_input_len=data_config["max_input_length"],
        max_target_len=data_config["max_target_length"],
        task_prefix=task_prefix,
    )
    val_dataset = GlossDataset(
        str(PROJECT_ROOT / data_config["val_file"]),
        tokenizer,
        max_input_len=data_config["max_input_length"],
        max_target_len=data_config["max_target_length"],
        task_prefix=task_prefix,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=data_config["max_input_length"],
    )

    # Determine fp16 setting
    train_config = config["training"]
    use_fp16 = False
    if train_config.get("fp16") == "auto":
        use_fp16 = torch.cuda.is_available()
    elif train_config.get("fp16"):
        use_fp16 = True

    # Adjust batch size for CPU
    batch_size = train_config["batch_size"]
    if not torch.cuda.is_available():
        batch_size = min(batch_size, 4)
        print(f"  [info] Reduced batch size to {batch_size} for CPU training")

    # Training arguments
    output_dir = str(PROJECT_ROOT / train_config["output_dir"])
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_config["num_epochs"],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        warmup_steps=train_config["warmup_steps"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        fp16=use_fp16,
        eval_strategy="steps",
        eval_steps=train_config.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=train_config.get("eval_steps", 500),
        save_total_limit=train_config["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        predict_with_generate=True,
        generation_max_length=data_config["max_target_length"],
        logging_steps=train_config.get("logging_steps", 100),
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",  # Disable wandb/tensorboard
        seed=train_config.get("seed", 42),
        dataloader_num_workers=0,  # Windows compatibility
        remove_unused_columns=False,
    )

    # Create metric computation function with tokenizer closure
    def metric_fn(eval_preds):
        return compute_metrics(eval_preds, tokenizer)

    # Early stopping
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=train_config["early_stopping_patience"]
        )
    ]

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=metric_fn,
        callbacks=callbacks,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save best model
    best_model_dir = str(PROJECT_ROOT / "app" / "models" / "best_gloss_model")
    print(f"\nSaving best model to {best_model_dir}")
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # Sample predictions
    print("\nSample Predictions:")
    print("-" * 60)
    test_sentences = [
        "I want to learn sign language",
        "What is your name?",
        "The weather is beautiful today",
        "She doesn't understand me",
        "Yesterday I went to the hospital",
    ]

    model.eval()
    with torch.no_grad():
        for sent in test_sentences:
            input_text = task_prefix + sent
            inputs = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model.generate(**inputs, max_length=64, num_beams=4)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  EN:    {sent}")
            print(f"  GLOSS: {decoded}")
            print()

    print("[SUCCESS] Training complete!")
    return trainer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train English → ASL Gloss translator")
    parser.add_argument("--config", default=None, help="Path to training config YAML")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint path")
    args = parser.parse_args()

    train(config_path=args.config, resume_from=args.resume)
