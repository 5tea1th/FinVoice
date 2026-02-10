"""Fine-tune a Financial NER Model on FiNER-ORD dataset.

Uses gtfintechlab/finer-ord (80K+ tokens, 3 entity types: PER/LOC/ORG) to train
a token classification model for extracting financial entities.

Dataset: gtfintechlab/finer-ord (HuggingFace)
  - Flat token-level format: gold_token, gold_label, doc_idx, sent_idx
  - Labels: 0=O, 1=B-PER, 2=I-PER, 3=B-LOC, 4=I-LOC, 5=B-ORG, 6=I-ORG
  - 80,531 tokens, 3,262 sentences

Output: data/models/financial-ner/

Usage:
    python scripts/finetune_ner.py [--epochs 5] [--batch-size 16]
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))


# FiNER-ORD label scheme (BIO tagging)
FINER_ORD_LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-LOC", "I-LOC",
    "B-ORG", "I-ORG",
]

# Map FiNER-ORD types to our pipeline entity types
FINER_TO_FINVOICE = {
    "PER": "person_name",
    "LOC": "location",
    "ORG": "organization",
}

# Our target NER labels (BIO scheme) — includes financial entity types
# The model is trained on PER/LOC/ORG from FiNER-ORD
# Financial types (payment_amount, due_date, etc.) remain handled by regex+spaCy layer
TARGET_NER_LABELS = [
    "O",
    "B-person_name", "I-person_name",
    "B-location", "I-location",
    "B-organization", "I-organization",
]


def load_finer_dataset():
    """Load FiNER-ORD dataset and group into sentences."""
    from datasets import load_dataset

    logger.info("Loading gtfintechlab/finer-ord dataset...")
    ds = load_dataset("gtfintechlab/finer-ord")
    logger.info(f"Splits: {list(ds.keys())}")

    all_sentences = []  # list of (tokens_list, labels_list)

    for split_name in ds.keys():
        split = ds[split_name]
        logger.info(f"  {split_name}: {len(split)} tokens")

        # Group by (doc_idx, sent_idx)
        sentences = defaultdict(lambda: ([], []))
        for sample in split:
            key = (sample["doc_idx"], sample["sent_idx"])
            sentences[key][0].append(sample["gold_token"])
            sentences[key][1].append(sample["gold_label"])

        # Map FiNER-ORD labels to our labels
        # FiNER: 0=O, 1=B-PER, 2=I-PER, 3=B-LOC, 4=I-LOC, 5=B-ORG, 6=I-ORG
        # Ours:  0=O, 1=B-person_name, 2=I-person_name, 3=B-location, 4=I-location, 5=B-organization, 6=I-organization
        # Direct 1:1 mapping (same indices!)
        for key in sorted(sentences.keys()):
            tokens, labels = sentences[key]
            if tokens and labels and len(tokens) == len(labels):
                # Filter out sentences with None tokens
                if any(t is None for t in tokens):
                    clean_tokens = []
                    clean_labels = []
                    for t, l in zip(tokens, labels):
                        if t is not None:
                            clean_tokens.append(str(t))
                            clean_labels.append(l)
                    if clean_tokens:
                        all_sentences.append((clean_tokens, clean_labels))
                else:
                    all_sentences.append(([str(t) for t in tokens], labels))

        logger.info(f"  {split_name}: {len(sentences)} sentences extracted")

    logger.info(f"Total sentences: {len(all_sentences)}")

    # Show label distribution
    flat_labels = [l for _, labels in all_sentences for l in labels]
    dist = Counter(flat_labels)
    for label_id, count in sorted(dist.items()):
        logger.info(f"  {TARGET_NER_LABELS[label_id]:>20}: {count}")

    return all_sentences


def finetune_ner(
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    output_dir: str = "data/models/financial-ner",
):
    """Fine-tune a BERT model for financial NER."""
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification,
    )
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    # Load and prepare data
    all_sentences = load_finer_dataset()

    if len(all_sentences) < 100:
        logger.error(f"Only {len(all_sentences)} sentences — not enough")
        return None

    # Split into tokens and labels
    all_tokens = [s[0] for s in all_sentences]
    all_labels = [s[1] for s in all_sentences]

    # Train/val split
    train_tokens, val_tokens, train_labels, val_labels = train_test_split(
        all_tokens, all_labels, test_size=0.15, random_state=42
    )
    logger.info(f"Train: {len(train_tokens)} sentences, Val: {len(val_tokens)} sentences")

    # Load tokenizer and model (distilbert for speed)
    model_name = "distilbert-base-uncased"
    logger.info(f"Loading {model_name} for NER...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    id_to_label = {i: label for i, label in enumerate(TARGET_NER_LABELS)}
    label_to_id = {label: i for i, label in enumerate(TARGET_NER_LABELS)}

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(TARGET_NER_LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
    )

    # Tokenize with label alignment
    def tokenize_and_align(tokens_list, labels_list):
        tokenized = tokenizer(
            tokens_list,
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        aligned_labels = []
        for i, labels in enumerate(labels_list):
            word_ids = tokenized.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(labels[word_idx] if word_idx < len(labels) else 0)
                else:
                    label_ids.append(-100)  # sub-word tokens get -100
                previous_word_idx = word_idx
            aligned_labels.append(label_ids)
        tokenized["labels"] = aligned_labels
        return tokenized

    logger.info("Tokenizing...")
    train_enc = tokenize_and_align(train_tokens, train_labels)
    val_enc = tokenize_and_align(val_tokens, val_labels)

    train_dataset = Dataset.from_dict(train_enc)
    val_dataset = Dataset.from_dict(val_enc)

    # Metrics
    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)

        # Flatten, ignoring padding (-100)
        true_labels = []
        pred_labels = []
        for true_seq, pred_seq in zip(label_ids, preds):
            for true_val, pred_val in zip(true_seq, pred_seq):
                if true_val != -100:
                    true_labels.append(true_val)
                    pred_labels.append(pred_val)

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)
        # Entity-level (non-O) F1
        entity_true = [1 if t > 0 else 0 for t in true_labels]
        entity_pred = [1 if p > 0 else 0 for p in pred_labels]
        entity_f1 = f1_score(entity_true, entity_pred, zero_division=0)
        return {"accuracy": acc, "f1_weighted": f1, "entity_f1": entity_f1}

    # Training
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="entity_f1",
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        fp16=False,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    logger.info(f"Starting NER training: {epochs} epochs, batch_size={batch_size}")
    trainer.train()

    results = trainer.evaluate()
    logger.info(f"Final evaluation: {results}")

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(f"{output_dir}/label_map.json", "w") as f:
        json.dump(id_to_label, f, indent=2)

    logger.info(f"NER model saved to {output_dir}/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune financial NER on FiNER-ORD")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output", type=str, default="data/models/financial-ner")
    args = parser.parse_args()

    results = finetune_ner(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output,
    )

    if results:
        print(f"\nNER Training complete!")
        print(f"  Accuracy: {results.get('eval_accuracy', 0):.4f}")
        print(f"  Entity F1: {results.get('eval_entity_f1', 0):.4f}")
