"""Fine-tune FinBERT on financial sentiment datasets.

Uses TWO datasets for robust training:
1. nickmuchi/financial-classification (Financial PhraseBank, 5,057 sentences, 3-class)
2. zeroshot/twitter-financial-news-sentiment (11,931 financial tweets, 3-class)

Combined: ~17,000 labeled financial sentiment samples.

Output: data/models/finbert-finetuned/

Usage:
    python scripts/finetune_finbert.py [--epochs 3] [--batch-size 16] [--lr 2e-5]

VRAM: ~1-2GB. Can run on GPU alongside nothing else, or on CPU (slower).
Time: ~15 min on GPU, ~45 min on CPU.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from loguru import logger
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score


# Unified label map: negative=0, neutral=1, positive=2
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}
MODEL_NAME = "ProsusAI/finbert"
OUTPUT_DIR = "data/models/finbert-finetuned"


def compute_metrics(eval_pred):
    """Compute accuracy and F1 for evaluation."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1_weighted": f1}


def load_phrasebank():
    """Load Financial PhraseBank via nickmuchi/financial-classification.

    Labels: 0=negative, 1=neutral, 2=positive (matches our LABEL_MAP).
    """
    logger.info("Loading nickmuchi/financial-classification (Financial PhraseBank)...")
    ds = load_dataset("nickmuchi/financial-classification")

    texts = []
    labels = []
    for split in ds.keys():
        for sample in ds[split]:
            text = sample.get("text", "")
            label = sample.get("labels", -1)
            if text and label in (0, 1, 2):
                texts.append(text.strip())
                labels.append(label)

    logger.info(f"  PhraseBank: {len(texts)} samples")
    return texts, labels


def load_twitter_financial():
    """Load Twitter Financial News Sentiment dataset.

    Original labels: 0=Bearish, 1=Bullish, 2=Neutral
    Mapped to: 0=negative, 2=positive, 1=neutral
    """
    logger.info("Loading zeroshot/twitter-financial-news-sentiment...")
    try:
        ds = load_dataset("zeroshot/twitter-financial-news-sentiment")
    except Exception as e:
        logger.warning(f"Could not load twitter dataset: {e}")
        return [], []

    # Map: twitter 0 (bearish) → 0 (negative), 1 (bullish) → 2 (positive), 2 (neutral) → 1 (neutral)
    twitter_to_ours = {0: 0, 1: 2, 2: 1}

    texts = []
    labels = []
    for split in ds.keys():
        for sample in ds[split]:
            text = sample.get("text", "")
            label = sample.get("label", -1)
            if text and label in twitter_to_ours:
                texts.append(text.strip())
                labels.append(twitter_to_ours[label])

    logger.info(f"  Twitter Financial: {len(texts)} samples")
    return texts, labels


def main():
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT on financial sentiment")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    logger.info(f"Training on {device}")

    # Load both datasets
    pb_texts, pb_labels = load_phrasebank()
    tw_texts, tw_labels = load_twitter_financial()

    all_texts = pb_texts + tw_texts
    all_labels = pb_labels + tw_labels

    if len(all_texts) < 100:
        logger.error(f"Only {len(all_texts)} samples — not enough")
        return

    logger.info(f"Combined: {len(all_texts)} samples")
    from collections import Counter
    dist = Counter(all_labels)
    for label_id in sorted(dist.keys()):
        logger.info(f"  {ID2LABEL[label_id]:>10}: {dist[label_id]}")

    # Train/val split
    from sklearn.model_selection import train_test_split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        all_texts, all_labels, test_size=0.15, random_state=42, stratify=all_labels
    )
    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # Load tokenizer and model
    logger.info(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL_MAP,
        ignore_mismatched_sizes=True,
    )

    # Tokenize
    train_enc = tokenizer(train_texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    val_enc = tokenizer(val_texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")

    train_dataset = Dataset.from_dict({
        "input_ids": train_enc["input_ids"],
        "attention_mask": train_enc["attention_mask"],
        "labels": train_labels,
    })
    val_dataset = Dataset.from_dict({
        "input_ids": val_enc["input_ids"],
        "attention_mask": val_enc["attention_mask"],
        "labels": val_labels,
    })

    # Training arguments
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        logging_steps=50,
        save_total_limit=2,
        fp16=(device == "cuda"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info(f"Starting training: {args.epochs} epochs, batch_size={args.batch_size}, lr={args.lr}")
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate()
    logger.info(f"Final accuracy: {metrics['eval_accuracy']:.4f}")
    logger.info(f"Final F1 (weighted): {metrics['eval_f1_weighted']:.4f}")

    # Save
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    logger.info(f"Model saved to {output_path}")

    # Verify it loads
    from transformers import pipeline as hf_pipeline
    test_pipe = hf_pipeline(
        "sentiment-analysis",
        model=str(output_path),
        device=-1,
    )
    test_result = test_pipe("The company reported strong quarterly earnings growth")
    logger.info(f"Verification: '{test_result[0]['label']}' ({test_result[0]['score']:.3f})")

    print(f"\nDone! Fine-tuned FinBERT saved to: {output_path}")
    print(f"Accuracy: {metrics['eval_accuracy']:.4f}")
    print(f"F1 (weighted): {metrics['eval_f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
