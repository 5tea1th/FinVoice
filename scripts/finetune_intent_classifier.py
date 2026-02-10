"""Fine-tune FinBERT as Intent Classifier on real banking datasets.

Uses THREE public datasets for robust training:
1. skit-ai/skit-s2i (11,845 real Indian banking call utterances, 14 intents)
2. PolyAI/banking77 (13,083 customer service queries, 77 fine-grained intents)
3. bitext/Bitext-retail-banking-llm-chatbot-training-dataset (25,500 banking queries, 26 intents)

Combined: ~50,000+ labeled samples from real banking interactions.

This replaces the slow LLM intent classification for most utterances:
- FinBERT intent model: ~5ms/utterance on CPU
- LLM (Qwen3): ~5s/utterance on GPU

Output: data/models/finbert-intent/

Usage:
    python scripts/finetune_intent_classifier.py [--epochs 5] [--batch-size 16]
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))


# skit-s2i uses integer intent_class 0-13. Mapping based on actual sample inspection:
# 0=branch_info, 1=activate_card, 2=transaction_details, 3=card_dispatch,
# 4=outstanding_balance, 5=card_issue, 6=ifsc_code, 7=pin_generation,
# 8=fraud_report, 9=loan_inquiry, 10=balance_inquiry, 11=limit_change,
# 12=block_banking, 13=lost_card
SKIT_CLASS_TO_FINVOICE = {
    0: "info_request",       # branch_info
    1: "info_request",       # activate_card
    2: "info_request",       # transaction_details
    3: "info_request",       # card_dispatch_status
    4: "info_request",       # outstanding_balance
    5: "complaint",          # card_issue
    6: "info_request",       # ifsc_code
    7: "info_request",       # pin_generation
    8: "escalation",         # fraud_report
    9: "info_request",       # loan_inquiry
    10: "info_request",      # balance_inquiry
    11: "info_request",      # limit_change
    12: "escalation",        # block_banking (urgent action)
    13: "escalation",        # lost_card (urgent action)
}

# Map BANKING77's 77 fine-grained intents to our 13 coarse intents
BANKING77_TO_FINVOICE_INTENT = {
    # Info requests
    "activate_my_card": "info_request",
    "age_limit": "info_request",
    "apple_pay_or_google_pay": "info_request",
    "atm_support": "info_request",
    "automatic_top_up": "info_request",
    "balance_not_updated_after_bank_transfer": "info_request",
    "balance_not_updated_after_cheque_or_cash_deposit": "info_request",
    "beneficiary_not_allowed": "info_request",
    "card_about_to_expire": "info_request",
    "card_acceptance": "info_request",
    "card_arrival": "info_request",
    "card_delivery_estimate": "info_request",
    "card_linking": "info_request",
    "card_not_working": "info_request",
    "card_payment_not_recognised": "info_request",
    "card_swallowed": "info_request",
    "cash_withdrawal_charge": "info_request",
    "cash_withdrawal_not_recognised": "info_request",
    "change_pin": "info_request",
    "contactless_not_working": "info_request",
    "country_support": "info_request",
    "direct_debit_payment_not_recognised": "info_request",
    "exchange_charge": "info_request",
    "exchange_rate": "info_request",
    "exchange_via_app": "info_request",
    "extra_charge_on_statement": "info_request",
    "fiat_currency_support": "info_request",
    "get_disposable_virtual_card": "info_request",
    "get_physical_card": "info_request",
    "getting_spare_card": "info_request",
    "getting_virtual_card": "info_request",
    "order_physical_card": "info_request",
    "passcode_forgotten": "info_request",
    "pending_card_payment": "info_request",
    "pending_cash_withdrawal": "info_request",
    "pending_top_up": "info_request",
    "pending_transfer": "info_request",
    "pin_blocked": "info_request",
    "receiving_money": "info_request",
    "supported_cards_and_currencies": "info_request",
    "top_up_by_bank_transfer_charge": "info_request",
    "top_up_by_card_charge": "info_request",
    "top_up_by_cash_or_cheque": "info_request",
    "top_up_failed": "info_request",
    "top_up_limits": "info_request",
    "top_up_reverted": "info_request",
    "topping_up_by_card": "info_request",
    "transaction_charged_twice": "info_request",
    "transfer_fee_charged": "info_request",
    "transfer_into_account": "info_request",
    "transfer_not_received_by_recipient": "info_request",
    "transfer_timing": "info_request",
    "unable_to_verify_identity": "info_request",
    "verify_my_identity": "info_request",
    "verify_source_of_funds": "info_request",
    "verify_top_up": "info_request",
    "virtual_card_not_working": "info_request",
    "visa_or_mastercard": "info_request",
    "why_verify_identity": "info_request",
    # Complaints
    "card_payment_fee_charged": "complaint",
    "card_payment_wrong_exchange_rate": "complaint",
    "declined_card_payment": "complaint",
    "declined_cash_withdrawal": "complaint",
    "declined_transfer": "complaint",
    "wrong_amount_of_cash_received": "complaint",
    "wrong_exchange_rate_for_cash_withdrawal": "complaint",
    # Refusals / cancellations
    "cancel_transfer": "refusal",
    "terminate_account": "refusal",
    # Disputes
    "Refund_not_showing_up": "dispute",
    "reverted_card_payment?": "dispute",
    "request_refund": "dispute",
    # Lost / stolen → escalation
    "lost_or_stolen_card": "escalation",
    "lost_or_stolen_phone": "escalation",
    "compromised_card": "escalation",
    # Edit personal details → agreement/consent
    "edit_personal_details": "agreement",
}

# Map Bitext's 26 retail banking intents to our 13 FinVoice intents
# Actual intents from bitext/Bitext-retail-banking-llm-chatbot-training-dataset (25,545 samples)
BITEXT_TO_FINVOICE_INTENT = {
    # Info requests (routine queries about accounts/services)
    "activate_card": "info_request",
    "activate_card_international_usage": "info_request",
    "apply_for_loan": "info_request",
    "apply_for_mortgage": "info_request",
    "check_card_annual_fee": "info_request",
    "check_current_balance_on_card": "info_request",
    "check_fees": "info_request",
    "check_loan_payments": "info_request",
    "check_mortgage_payments": "info_request",
    "check_recent_transactions": "info_request",
    "create_account": "info_request",
    "find_ATM": "info_request",
    "find_branch": "info_request",
    "get_password": "info_request",
    "make_transfer": "info_request",
    "recover_swallowed_card": "info_request",
    "set_up_password": "info_request",
    # Refusals / cancellations
    "cancel_card": "refusal",
    "cancel_loan": "refusal",
    "cancel_mortgage": "refusal",
    "cancel_transfer": "refusal",
    "close_account": "refusal",
    # Escalation (security / urgent)
    "block_card": "escalation",
    "human_agent": "escalation",
    "customer_service": "escalation",
    # Disputes
    "dispute_ATM_withdrawal": "dispute",
}

# Our target labels (from CallIntent enum)
TARGET_LABELS = [
    "agreement", "refusal", "request_extension", "payment_promise",
    "complaint", "consent_given", "consent_denied", "info_request",
    "negotiation", "escalation", "dispute", "acknowledgment", "greeting",
]


def load_skit_dataset():
    """Load skit-ai/skit-s2i dataset, removing audio column to avoid torchcodec issues.

    Columns: template (text), intent_class (int 0-13), speaker_id (int)
    """
    from datasets import load_dataset

    logger.info("Loading skit-ai/skit-s2i dataset...")
    ds = load_dataset("skit-ai/skit-s2i")

    # Remove audio column to avoid torchcodec decoding errors
    for split in ds.keys():
        if "audio" in ds[split].column_names:
            ds[split] = ds[split].remove_columns("audio")

    logger.info(f"Dataset splits: {list(ds.keys())}")
    total = sum(len(ds[split]) for split in ds.keys())
    logger.info(f"skit-s2i total samples: {total}")
    return ds


def load_banking77_dataset():
    """Load PolyAI/banking77 dataset (13,083 samples, 77 intents)."""
    from datasets import load_dataset

    logger.info("Loading PolyAI/banking77 dataset...")
    try:
        ds = load_dataset("legacy-datasets/banking77")
        logger.info(f"BANKING77 splits: {list(ds.keys())}")
        total = sum(len(ds[split]) for split in ds.keys())
        logger.info(f"BANKING77 total samples: {total}")
        return ds
    except Exception as e:
        logger.warning(f"Could not load BANKING77 dataset: {e}")
        return None


def _extract_skit_samples(ds) -> tuple[list[str], list[int]]:
    """Extract text + labels from skit-s2i, mapped to our 13 intents."""
    label_to_id = {label: i for i, label in enumerate(TARGET_LABELS)}

    texts = []
    labels = []
    skipped = 0

    for split_name in ds.keys():
        for sample in ds[split_name]:
            text = sample.get("template", "")
            if not text or not text.strip():
                skipped += 1
                continue

            intent_class = sample.get("intent_class")
            mapped = SKIT_CLASS_TO_FINVOICE.get(intent_class)

            if mapped and mapped in label_to_id:
                texts.append(text.strip())
                labels.append(label_to_id[mapped])
            else:
                skipped += 1

    logger.info(f"skit-s2i: extracted {len(texts)} samples, skipped {skipped}")
    return texts, labels


def _extract_banking77_samples(ds) -> tuple[list[str], list[int]]:
    """Extract text + labels from BANKING77, mapped to our 13 intents."""
    label_to_id = {label: i for i, label in enumerate(TARGET_LABELS)}

    # BANKING77 uses integer labels — get the label names
    try:
        label_names = ds["train"].features["label"].names
    except Exception:
        label_names = None

    texts = []
    labels = []
    skipped = 0

    for split_name in ds.keys():
        for sample in ds[split_name]:
            text = sample.get("text", "")
            if not text or not text.strip():
                skipped += 1
                continue

            label_idx = sample.get("label")
            if label_names and isinstance(label_idx, int):
                intent_name = label_names[label_idx]
            else:
                intent_name = str(label_idx)

            # Map to our intent set
            mapped = BANKING77_TO_FINVOICE_INTENT.get(
                intent_name,
                BANKING77_TO_FINVOICE_INTENT.get(intent_name.lower(), None)
            )

            if mapped and mapped in label_to_id:
                texts.append(text.strip())
                labels.append(label_to_id[mapped])
            else:
                skipped += 1

    logger.info(f"BANKING77: extracted {len(texts)} samples, skipped {skipped}")
    return texts, labels


def load_bitext_dataset():
    """Load Bitext retail banking dataset (25,500 samples, 26 intents).

    Columns: flags, instruction, category, intent, response
    The 'instruction' column has customer utterances, 'intent' has the label.
    """
    from datasets import load_dataset

    logger.info("Loading bitext/Bitext-retail-banking-llm-chatbot-training-dataset...")
    try:
        ds = load_dataset(
            "bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
            streaming=True,
        )
        # Streaming dataset — materialize into a list
        samples = []
        for sample in ds["train"]:
            samples.append({
                "instruction": sample.get("instruction", ""),
                "intent": sample.get("intent", ""),
            })
        logger.info(f"Bitext: loaded {len(samples)} samples")
        return samples
    except Exception as e:
        logger.warning(f"Could not load Bitext dataset: {e}")
        return None


def _extract_bitext_samples(samples: list) -> tuple[list[str], list[int]]:
    """Extract text + labels from Bitext, mapped to our 13 intents."""
    label_to_id = {label: i for i, label in enumerate(TARGET_LABELS)}

    texts = []
    labels = []
    skipped = 0
    unmapped_intents = Counter()

    for sample in samples:
        text = sample.get("instruction", "").strip()
        intent = sample.get("intent", "").strip()

        if not text:
            skipped += 1
            continue

        # Try exact match, then lowercase match
        mapped = BITEXT_TO_FINVOICE_INTENT.get(
            intent,
            BITEXT_TO_FINVOICE_INTENT.get(intent.lower(), None)
        )

        if mapped and mapped in label_to_id:
            texts.append(text)
            labels.append(label_to_id[mapped])
        else:
            skipped += 1
            unmapped_intents[intent] += 1

    if unmapped_intents:
        top_unmapped = unmapped_intents.most_common(10)
        logger.info(f"Bitext unmapped intents (top 10): {top_unmapped}")

    logger.info(f"Bitext: extracted {len(texts)} samples, skipped {skipped}")
    return texts, labels


def prepare_training_data(skit_ds, banking77_ds=None, bitext_samples=None) -> tuple:
    """Convert datasets to text + label format for training."""
    # Extract from skit-s2i
    texts, labels = _extract_skit_samples(skit_ds)

    # Merge BANKING77 if available
    if banking77_ds is not None:
        b77_texts, b77_labels = _extract_banking77_samples(banking77_ds)
        texts.extend(b77_texts)
        labels.extend(b77_labels)
        logger.info(f"After BANKING77: {len(texts)} samples")

    # Merge Bitext if available
    if bitext_samples is not None:
        bt_texts, bt_labels = _extract_bitext_samples(bitext_samples)
        texts.extend(bt_texts)
        labels.extend(bt_labels)
        logger.info(f"After Bitext: {len(texts)} samples")

    logger.info(f"Combined total: {len(texts)} samples from all datasets")

    # Show label distribution
    dist = Counter(labels)
    for label_id, count in sorted(dist.items()):
        logger.info(f"  {TARGET_LABELS[label_id]:>20}: {count}")

    return texts, labels


def finetune(
    epochs: int = 5,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    output_dir: str = "data/models/finbert-intent",
):
    """Fine-tune ProsusAI/finbert for intent classification."""
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    # Load and prepare data from all datasets
    skit_ds = load_skit_dataset()
    banking77_ds = load_banking77_dataset()
    bitext_samples = load_bitext_dataset()
    texts, labels = prepare_training_data(skit_ds, banking77_ds=banking77_ds, bitext_samples=bitext_samples)

    if len(texts) < 100:
        logger.error(f"Only {len(texts)} training samples — not enough. Check dataset format.")
        return

    # Train/val split
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=42, stratify=labels
    )
    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # Load tokenizer and model
    model_name = "ProsusAI/finbert"
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(TARGET_LABELS),
        ignore_mismatched_sizes=True,
    )

    # Tokenize
    train_enc = tokenizer(
        train_texts, padding="max_length", truncation=True,
        max_length=128, return_tensors="pt",
    )
    val_enc = tokenizer(
        val_texts, padding="max_length", truncation=True,
        max_length=128, return_tensors="pt",
    )

    # Create HF datasets
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

    # Metrics
    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(label_ids, preds)
        f1 = f1_score(label_ids, preds, average="weighted")
        return {"accuracy": acc, "f1_weighted": f1}

    # Training arguments
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
        metric_for_best_model="f1_weighted",
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        fp16=False,  # CPU training — no FP16
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}")
    trainer.train()

    # Evaluate
    results = trainer.evaluate()
    logger.info(f"Final evaluation: {results}")

    # Save model + label mapping
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    label_map = {i: label for i, label in enumerate(TARGET_LABELS)}
    with open(f"{output_dir}/label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    logger.info(f"Model saved to {output_dir}/")
    logger.info(f"  Accuracy: {results.get('eval_accuracy', 0):.4f}")
    logger.info(f"  F1 (weighted): {results.get('eval_f1_weighted', 0):.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune FinBERT intent classifier on skit-s2i + BANKING77 + Bitext")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--output", type=str, default="data/models/finbert-intent",
                        help="Output directory for fine-tuned model")
    args = parser.parse_args()

    results = finetune(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output,
    )

    if results:
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"  Accuracy: {results.get('eval_accuracy', 0):.4f}")
        print(f"  F1 (weighted): {results.get('eval_f1_weighted', 0):.4f}")
        print(f"{'='*60}")
