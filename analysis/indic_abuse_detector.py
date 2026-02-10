"""Shared Indic Abuse Detection — MuRIL-based multilingual abuse classifier.

Uses `Hate-speech-CNERG/indic-abusive-allInOne-MuRIL` (Google MuRIL base) to
detect abusive, threatening, and toxic content in 10 Indian languages:
  Hindi, Hindi-English code-mixed, Tamil code-mixed, Bengali, Kannada,
  Malayalam, Marathi, Urdu, English

Shared across fraud detection (threat/coercion detection) and profanity
detection (abuse/toxicity). Loaded once, cached globally.

Runs on CPU (~480MB). ~15ms per sentence.
"""

from loguru import logger

_pipeline = None
HAS_MURIL = False

_MODEL_NAME = "Hate-speech-CNERG/indic-abusive-allInOne-MuRIL"


def _get_pipeline():
    """Load MuRIL abuse classification pipeline (lazy, cached)."""
    global _pipeline, HAS_MURIL

    if _pipeline is not None:
        return _pipeline

    try:
        from transformers import pipeline as hf_pipeline

        logger.info(f"Loading MuRIL abuse detection model: {_MODEL_NAME}...")
        _pipeline = hf_pipeline(
            "text-classification",
            model=_MODEL_NAME,
            device=-1,  # CPU
            truncation=True,
            max_length=512,
        )
        HAS_MURIL = True
        logger.info("MuRIL abuse model loaded (10 Indian languages, CPU)")
        return _pipeline
    except Exception as e:
        logger.warning(f"MuRIL abuse model not available: {e}")
        HAS_MURIL = False
        return None


def detect_abuse_batch(texts: list[str], threshold: float = 0.6) -> list[dict]:
    """Classify texts for abusive content using MuRIL.

    Args:
        texts: List of text strings to classify
        threshold: Minimum score to flag as abusive (default 0.6)

    Returns:
        List of dicts for flagged texts:
        [{"index": 0, "label": "abusive", "score": 0.92}, ...]
        Empty list items are omitted (only flagged texts returned).
    """
    pipe = _get_pipeline()
    if pipe is None:
        return []

    if not texts:
        return []

    # Truncate long texts
    truncated = [t[:512] if t else "" for t in texts]

    try:
        results = pipe(truncated, batch_size=32)
    except Exception as e:
        logger.warning(f"MuRIL batch prediction failed: {e}")
        return []

    flagged = []
    for i, result in enumerate(results):
        # MuRIL outputs labels like "abusive" / "not abusive" (or similar)
        label = result.get("label", "").lower()
        score = result.get("score", 0.0)

        # The model may use different label names depending on the version
        is_abusive = (
            "abusive" in label
            or "offensive" in label
            or "hate" in label
        ) and "not" not in label

        # If the label is "not abusive" with high confidence, skip
        if not is_abusive:
            # Check if the "not abusive" confidence is low (meaning it's uncertain)
            if score < (1.0 - threshold):
                # Low confidence in "not abusive" = possibly abusive
                flagged.append({
                    "index": i,
                    "label": "possibly_abusive",
                    "score": round(1.0 - score, 3),
                })
            continue

        if score >= threshold:
            flagged.append({
                "index": i,
                "label": label,
                "score": round(score, 3),
            })

    return flagged


def is_available() -> bool:
    """Check if MuRIL model can be loaded."""
    return _get_pipeline() is not None
