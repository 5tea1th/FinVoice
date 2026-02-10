"""Sentiment Analysis + Intent Classification — CPU pre-filter (multilingual).

Dual-model architecture for sentiment:
1. Fine-tuned FinBERT (ProsusAI/finbert) — best for English financial text
2. Pre-trained XLM-RoBERTa multilingual — for non-English (RU/PL/FR/DE/ES/PT + 90 more)

The pipeline auto-routes by detected language: English → FinBERT, non-English → XLM-R.

Intent classification uses fine-tuned FinBERT (English), with LLM fallback for
non-English text (where FinBERT confidence will naturally be low).
"""

import json
from pathlib import Path
from loguru import logger

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    import torch

    _finbert_model_name = "ProsusAI/finbert"
    _multilingual_model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"
    _sentiment_pipeline_en = None      # FinBERT for English
    _sentiment_pipeline_multi = None   # XLM-R for non-English
    _intent_pipeline = None
    _intent_label_map = None

    def _get_pipeline(lang: str = "en"):
        """Get sentiment pipeline, routing by language.

        English → fine-tuned FinBERT (strong financial domain performance)
        Non-English → pre-trained XLM-RoBERTa multilingual sentiment
        """
        global _sentiment_pipeline_en, _sentiment_pipeline_multi

        if lang == "en":
            if _sentiment_pipeline_en is None:
                # Check for fine-tuned FinBERT model first
                finetuned_path = Path("data/models/finbert-finetuned")
                if finetuned_path.exists() and (finetuned_path / "config.json").exists():
                    model_path = str(finetuned_path)
                    logger.info(f"Loading fine-tuned FinBERT from {model_path}...")
                else:
                    model_path = _finbert_model_name
                    logger.info(f"Loading base FinBERT ({model_path}) on CPU...")
                # Load model explicitly to CPU (bypasses accelerate meta tensor issue)
                _en_model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, low_cpu_mem_usage=False, device_map=None,
                )
                _en_model = _en_model.to("cpu")  # Force off meta device
                _en_tokenizer = AutoTokenizer.from_pretrained(model_path)
                _sentiment_pipeline_en = pipeline(
                    "sentiment-analysis",
                    model=_en_model,
                    tokenizer=_en_tokenizer,
                    device=-1,
                    truncation=True,
                    max_length=512,
                )
                logger.info("FinBERT loaded on CPU")
            return _sentiment_pipeline_en
        else:
            if _sentiment_pipeline_multi is None:
                logger.info(f"Loading multilingual sentiment model ({_multilingual_model_name}) on CPU...")
                _multi_model = AutoModelForSequenceClassification.from_pretrained(
                    _multilingual_model_name, low_cpu_mem_usage=False, device_map=None,
                )
                _multi_model = _multi_model.to("cpu")
                _multi_tokenizer = AutoTokenizer.from_pretrained(_multilingual_model_name)
                _sentiment_pipeline_multi = pipeline(
                    "sentiment-analysis",
                    model=_multi_model,
                    tokenizer=_multi_tokenizer,
                    device=-1,
                    truncation=True,
                    max_length=512,
                )
                logger.info(f"Multilingual sentiment model loaded on CPU (covers RU/PL/FR/DE/ES/PT + more)")
            return _sentiment_pipeline_multi

    def _get_intent_pipeline():
        """Load fine-tuned FinBERT intent classifier if available."""
        global _intent_pipeline, _intent_label_map
        if _intent_pipeline is not None:
            return _intent_pipeline, _intent_label_map

        intent_model_path = "data/models/finbert-intent"
        label_map_path = f"{intent_model_path}/label_map.json"

        if not Path(label_map_path).exists():
            return None, None

        try:
            with open(label_map_path) as f:
                _intent_label_map = json.load(f)

            logger.info(f"Loading FinBERT intent classifier from {intent_model_path}...")
            _intent_model = AutoModelForSequenceClassification.from_pretrained(
                intent_model_path, low_cpu_mem_usage=False, device_map=None,
            )
            _intent_model = _intent_model.to("cpu")
            _intent_tokenizer = AutoTokenizer.from_pretrained(intent_model_path)
            _intent_pipeline = pipeline(
                "text-classification",
                model=_intent_model,
                tokenizer=_intent_tokenizer,
                device=-1,
                truncation=True,
                max_length=128,
            )
            logger.info(f"FinBERT intent classifier loaded ({len(_intent_label_map)} classes)")
            return _intent_pipeline, _intent_label_map
        except Exception as e:
            logger.warning(f"Could not load intent classifier: {e}")
            return None, None

    HAS_FINBERT = True
except ImportError:
    HAS_FINBERT = False
    logger.warning("transformers not installed — sentiment analysis disabled")


# Intent subsets for pre-filtering
NEGATIVE_INTENTS = {"refusal", "complaint", "escalation", "consent_denied", "dispute"}
SKIP_INTENTS = {"acknowledgment", "greeting"}


def _normalize_label(label: str) -> str:
    """Normalize sentiment labels across different model outputs.

    FinBERT outputs: positive, negative, neutral
    XLM-R outputs: positive, negative, neutral (or Positive, Negative, Neutral)
    """
    return label.lower().strip()


def analyze_sentiment(text: str, lang: str = "en") -> dict:
    """Classify single text as positive/negative/neutral with score.

    Args:
        text: Input text
        lang: Language code (en, ru, pl, fr, de, es, pt). Non-English uses multilingual model.

    Returns:
        {"label": "positive"|"negative"|"neutral", "score": 0.0-1.0}
    """
    if not HAS_FINBERT:
        return {"label": "neutral", "score": 0.5}

    pipe = _get_pipeline(lang)
    result = pipe(text[:512], truncation=True, max_length=512)[0]
    return {"label": _normalize_label(result["label"]), "score": result["score"]}


def analyze_sentiment_batch(texts: list[str], lang: str = "en") -> list[dict]:
    """Batch sentiment analysis for efficiency, with per-item fallback."""
    if not HAS_FINBERT:
        return [{"label": "neutral", "score": 0.5} for _ in texts]

    pipe = _get_pipeline(lang)
    truncated = [t[:512] for t in texts]
    try:
        results = pipe(truncated, batch_size=32, truncation=True, max_length=512)
        return [{"label": _normalize_label(r["label"]), "score": r["score"]} for r in results]
    except Exception as e:
        logger.warning(f"Batch sentiment failed ({e}), falling back to per-item")
        results = []
        for t in truncated:
            try:
                r = pipe(t, truncation=True, max_length=512)[0]
                results.append({"label": _normalize_label(r["label"]), "score": r["score"]})
            except Exception:
                results.append({"label": "neutral", "score": 0.5})
        return results


def _is_agent_speaker(speaker: str) -> bool:
    """Check if speaker label indicates an agent."""
    s = speaker.lower()
    return s in ("agent", "speaker_00", "speaker 0", "spk_0")


def _is_customer_speaker(speaker: str) -> bool:
    """Check if speaker label indicates a customer."""
    s = speaker.lower()
    return s in ("customer", "speaker_01", "speaker 1", "spk_1")


def compute_sentiment_trajectories(segments: list, lang: str = "en") -> tuple[list[float], list[float]]:
    """Compute per-segment sentiment scores for customer and agent.

    Args:
        segments: List of transcript segments with 'text' and 'speaker' keys
        lang: Detected language code — routes to FinBERT (en) or XLM-R (non-en)

    Returns:
        (customer_trajectory, agent_trajectory) — lists of scores from -1 to +1
        Neutral sentiment returns a small value (not zero) so trajectories are non-flat.
    """
    if not HAS_FINBERT:
        return [0.0], [0.0]

    # Batch process all texts for efficiency
    texts = []
    speakers = []
    for seg in segments:
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "unknown")
        if text:
            texts.append(text)
            speakers.append(speaker)

    if not texts:
        return [0.0], [0.0]

    results = analyze_sentiment_batch(texts, lang=lang)

    customer_scores = []
    agent_scores = []

    for result, speaker in zip(results, speakers):
        # Convert to -1 to +1 scale (neutral maps to small positive, not zero)
        if result["label"] == "positive":
            score = result["score"]
        elif result["label"] == "negative":
            score = -result["score"]
        else:
            # Neutral: preserve as small but visible value (never exactly zero).
            # High-conf neutral (0.95) → 0.02, medium (0.80) → 0.06, low (0.50) → 0.15
            score = max(0.02, (1.0 - result["score"]) * 0.3)

        if _is_agent_speaker(speaker):
            agent_scores.append(round(score, 3))
        elif _is_customer_speaker(speaker):
            customer_scores.append(round(score, 3))
        else:
            # Unknown speaker — assign to both trajectories
            customer_scores.append(round(score, 3))
            agent_scores.append(round(score, 3))

    model_type = "FinBERT" if lang == "en" else f"XLM-R ({lang})"
    logger.info(
        f"Sentiment trajectories [{model_type}]: customer={len(customer_scores)} scores "
        f"(avg={sum(customer_scores)/max(len(customer_scores),1):.3f}), "
        f"agent={len(agent_scores)} scores "
        f"(avg={sum(agent_scores)/max(len(agent_scores),1):.3f})"
    )

    return customer_scores or [0.0], agent_scores or [0.0]


def classify_for_llm_routing(text: str, lang: str = "en") -> str:
    """Determine how to route an utterance for intent classification.

    Returns:
        "skip" — trivial utterance, classify as acknowledgment/greeting directly
        "negative_subset" — strongly negative, only consider 5 intents
        "full" — ambiguous, needs full 13-intent LLM classification
    """
    if not HAS_FINBERT:
        return "full"

    # Very short utterances are usually acknowledgments
    if len(text.split()) <= 3:
        lower = text.lower().strip().rstrip(".")
        from analysis.vocab_loader import get_trivial_phrases
        trivials = get_trivial_phrases(lang)
        if trivials and lower in trivials:
            return "skip"
        # For non-English without vocab, very short utterances (1-2 words) are likely acknowledgments
        if not trivials and lang != "en" and len(text.split()) <= 2:
            return "skip"

    try:
        result = analyze_sentiment(text, lang=lang)
    except Exception:
        return "full"  # FinBERT failed — let LLM handle it

    if result["label"] == "negative" and result["score"] > 0.85:
        return "negative_subset"
    elif result["label"] == "neutral" and result["score"] > 0.90:
        # High-confidence neutral — likely acknowledgment
        if len(text.split()) <= 5:
            return "skip"

    return "full"


def get_dominant_emotion(customer_trajectories: list[float]) -> str:
    """Determine dominant customer emotion from sentiment trajectory."""
    if not customer_trajectories:
        return "neutral"

    avg = sum(customer_trajectories) / len(customer_trajectories)
    if avg > 0.3:
        return "positive"
    elif avg < -0.5:
        return "angry"
    elif avg < -0.3:
        return "frustrated"
    else:
        return "neutral"


# ── FinBERT Intent Classification (when fine-tuned model is available) ──


def classify_intent(text: str, lang: str = "en") -> dict | None:
    """Classify intent using fine-tuned FinBERT intent model.

    FinBERT is English-only — returns None for non-English text (forces LLM fallback).

    Returns:
        {"intent": "payment_promise", "confidence": 0.92} or None if model not available
    """
    if not HAS_FINBERT:
        return None
    if lang != "en":
        return None  # FinBERT is English-only

    pipe, label_map = _get_intent_pipeline()
    if pipe is None:
        return None

    result = pipe(text[:128], truncation=True, max_length=128)[0]
    # Map LABEL_X to actual intent name
    label_key = result["label"]
    if label_key.startswith("LABEL_"):
        label_idx = label_key.replace("LABEL_", "")
        intent = label_map.get(label_idx, label_map.get(str(label_idx), label_key))
    else:
        intent = label_key

    return {"intent": intent, "confidence": round(result["score"], 3)}


def classify_intent_batch(texts: list[str]) -> list[dict | None]:
    """Batch intent classification for efficiency."""
    if not HAS_FINBERT:
        return [None] * len(texts)

    pipe, label_map = _get_intent_pipeline()
    if pipe is None:
        return [None] * len(texts)

    truncated = [t[:128] for t in texts]
    results = pipe(truncated, batch_size=32, truncation=True, max_length=128)

    classified = []
    for result in results:
        label_key = result["label"]
        if label_key.startswith("LABEL_"):
            label_idx = label_key.replace("LABEL_", "")
            intent = label_map.get(label_idx, label_map.get(str(label_idx), label_key))
        else:
            intent = label_key
        classified.append({"intent": intent, "confidence": round(result["score"], 3)})

    return classified


def has_intent_model() -> bool:
    """Check if the fine-tuned intent model is available."""
    return Path("data/models/finbert-intent/label_map.json").exists()
