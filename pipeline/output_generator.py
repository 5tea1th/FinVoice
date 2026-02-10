"""Stage 5: Output Generation — ML-trainable exports (CSV, Parquet, JSONL).

Exports processed CallRecord data to multiple formats:
- CSV:     Flat summary per call (for spreadsheets), entities, intents, compliance
- Parquet: Columnar ML-ready format (for pandas/spark/polars)
- JSONL:   One record per line (for HuggingFace datasets, training pipelines)
- Training pairs: (text, label) pairs for classifier fine-tuning

Accepts both CallRecord objects and raw dicts (from loaded JSON files).
"""

import json
from pathlib import Path
from loguru import logger

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not installed — CSV/Parquet export disabled")

from config.schemas import CallRecord


def _to_dict(record) -> dict:
    """Convert CallRecord or dict to dict."""
    if isinstance(record, CallRecord):
        return record.model_dump()
    return record


# ── Primary exports (CallRecord objects) ──


def export_to_parquet(call_records: list, output_path: str) -> str:
    """Export batch of call records as Parquet for ML training."""
    if not HAS_PANDAS:
        logger.error("pandas required for Parquet export")
        return ""
    rows = [_flatten_record(_to_dict(r)) for r in call_records]
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Parquet exported: {output_path} ({len(rows)} calls)")
    return output_path


def export_to_csv(call_records: list, output_path: str) -> str:
    """Export batch of call records as CSV."""
    if not HAS_PANDAS:
        logger.error("pandas required for CSV export")
        return ""
    rows = [_flatten_record(_to_dict(r)) for r in call_records]
    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"CSV exported: {output_path} ({len(rows)} calls)")
    return output_path


def export_to_jsonl(call_records: list, output_path: str) -> str:
    """Export call records as JSON Lines (one record per line)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in call_records:
            data = _to_dict(record)
            f.write(json.dumps(data, default=str) + "\n")
    logger.info(f"JSONL exported: {output_path} ({len(call_records)} records)")
    return output_path


# ── Detailed exports (per-entity, per-intent, etc.) ──


def export_entities_csv(call_records: list, output_path: str) -> str:
    """Export all financial entities to flat CSV (one row per entity)."""
    if not HAS_PANDAS:
        return ""

    rows = []
    for record in call_records:
        r = _to_dict(record)
        call_id = r.get("call_id")
        for e in r.get("financial_entities", []):
            rows.append({
                "call_id": call_id,
                "entity_type": e.get("entity_type"),
                "value": e.get("value"),
                "raw_text": e.get("raw_text"),
                "segment_id": e.get("segment_id"),
                "start_time": e.get("start_time"),
                "confidence": e.get("confidence"),
            })

    if not rows:
        return ""

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Entities CSV: {output_path} ({len(rows)} entities)")
    return output_path


def export_intents_csv(call_records: list, output_path: str) -> str:
    """Export all classified intents to flat CSV (one row per utterance)."""
    if not HAS_PANDAS:
        return ""

    rows = []
    for record in call_records:
        r = _to_dict(record)
        call_id = r.get("call_id")
        segments = r.get("transcript_segments", [])

        for intent in r.get("intents", []):
            seg_id = intent.get("segment_id", 0)
            seg_text = ""
            if seg_id < len(segments):
                seg_text = segments[seg_id].get("text", "")

            rows.append({
                "call_id": call_id,
                "segment_id": seg_id,
                "speaker": intent.get("speaker"),
                "intent": intent.get("intent"),
                "confidence": intent.get("confidence"),
                "text": seg_text,
            })

    if not rows:
        return ""

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Intents CSV: {output_path} ({len(rows)} intents)")
    return output_path


def export_compliance_csv(call_records: list, output_path: str) -> str:
    """Export all compliance check results to CSV."""
    if not HAS_PANDAS:
        return ""

    rows = []
    for record in call_records:
        r = _to_dict(record)
        call_id = r.get("call_id")
        for c in r.get("compliance_checks", []):
            rows.append({
                "call_id": call_id,
                "check_name": c.get("check_name"),
                "passed": c.get("passed"),
                "violation_type": c.get("violation_type"),
                "evidence_text": c.get("evidence_text"),
                "segment_id": c.get("segment_id"),
                "regulation": c.get("regulation"),
                "severity": c.get("severity"),
            })

    if not rows:
        return ""

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Compliance CSV: {output_path} ({len(rows)} checks)")
    return output_path


def export_training_pairs_jsonl(call_records: list, output_path: str) -> str:
    """Export (text, intent) pairs for intent classifier training.

    Each line: {"text": "utterance", "label": "intent_label", "speaker": "agent/customer"}
    Useful for fine-tuning FinBERT or other text classifiers.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    pairs = []
    for record in call_records:
        r = _to_dict(record)
        segments = r.get("transcript_segments", [])

        for intent in r.get("intents", []):
            seg_id = intent.get("segment_id", 0)
            if seg_id < len(segments):
                text = segments[seg_id].get("text", "").strip()
                if text:
                    pairs.append({
                        "text": text,
                        "label": intent.get("intent"),
                        "speaker": intent.get("speaker"),
                        "confidence": intent.get("confidence"),
                        "call_id": r.get("call_id"),
                    })

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    logger.info(f"Training pairs: {output_path} ({len(pairs)} pairs)")
    return output_path


def export_all(call_records: list, output_dir: str = "data/exports") -> dict:
    """Export all formats at once.

    Returns dict of {format: output_path} for all exported files.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    outputs = {}

    # Summary CSV
    path = export_to_csv(call_records, f"{output_dir}/call_summary.csv")
    if path:
        outputs["summary_csv"] = path

    # Entities CSV
    path = export_entities_csv(call_records, f"{output_dir}/entities.csv")
    if path:
        outputs["entities_csv"] = path

    # Intents CSV
    path = export_intents_csv(call_records, f"{output_dir}/intents.csv")
    if path:
        outputs["intents_csv"] = path

    # Compliance CSV
    path = export_compliance_csv(call_records, f"{output_dir}/compliance.csv")
    if path:
        outputs["compliance_csv"] = path

    # Parquet (full data)
    path = export_to_parquet(call_records, f"{output_dir}/calls.parquet")
    if path:
        outputs["parquet"] = path

    # JSONL (full records)
    path = export_to_jsonl(call_records, f"{output_dir}/calls.jsonl")
    if path:
        outputs["jsonl"] = path

    # Training pairs
    path = export_training_pairs_jsonl(call_records, f"{output_dir}/training_pairs.jsonl")
    if path:
        outputs["training_pairs"] = path

    logger.info(f"All exports complete: {len(outputs)} files in {output_dir}/")
    return outputs


# ── Helper to load from JSON files ──


def load_records_from_dir(results_dir: str = "data/processed") -> list[dict]:
    """Load all CallRecord JSON files from a directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return []

    records = []
    for f in sorted(results_path.glob("*_record.json")):
        try:
            with open(f) as fp:
                records.append(json.load(fp))
        except Exception as e:
            logger.warning(f"Failed to load {f}: {e}")

    logger.info(f"Loaded {len(records)} records from {results_dir}")
    return records


def _flatten_record(r: dict) -> dict:
    """Flatten a CallRecord dict to a single row for tabular export."""
    return {
        "call_id": r.get("call_id"),
        "audio_file": Path(r.get("audio_file", "")).name,
        "duration_sec": r.get("duration_seconds"),
        "language": r.get("detected_language", r.get("language")),
        "call_type": r.get("call_type"),
        "audio_quality_score": r.get("audio_quality_score"),
        "audio_quality_flag": r.get("audio_quality_flag"),
        "snr_db": r.get("snr_db"),
        "transcript_confidence": r.get("overall_transcript_confidence"),
        "num_speakers": r.get("num_speakers"),
        "agent_talk_pct": r.get("agent_talk_percentage"),
        "customer_talk_pct": r.get("customer_talk_percentage"),
        "num_intents": len(r.get("intents", [])),
        "num_entities": len(r.get("financial_entities", [])),
        "num_obligations": len(r.get("obligations", [])),
        "num_binding_obligations": len(
            [o for o in r.get("obligations", [])
             if (o.get("strength") if isinstance(o, dict) else o.strength.value) == "binding"]
        ),
        "num_compliance_violations": len(
            [c for c in r.get("compliance_checks", [])
             if not (c.get("passed") if isinstance(c, dict) else c.passed)]
        ),
        "num_fraud_signals": len(r.get("fraud_signals", [])),
        "pii_count": r.get("pii_count", 0),
        "num_toxicity_flags": len(r.get("toxicity_flags", [])),
        "tamper_risk": r.get("tamper_risk"),
        "compliance_score": r.get("compliance_score"),
        "risk_level": r.get("overall_risk_level") if isinstance(r.get("overall_risk_level"), str) else r.get("overall_risk_level", {}).get("value", "low") if isinstance(r.get("overall_risk_level"), dict) else str(r.get("overall_risk_level", "low")),
        "customer_emotion": r.get("customer_emotion_dominant"),
        "escalation_detected": r.get("escalation_detected"),
        "requires_review": r.get("requires_human_review"),
        "review_priority": r.get("review_priority"),
        "call_summary": (r.get("call_summary") or "")[:200],
    }
