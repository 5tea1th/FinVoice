"""Stage 4 Intelligence — CPU-based entity extraction (Layer 1) and FinBERT sentiment.

Layer 1 runs deterministic regex + spaCy extraction BEFORE the LLM.
This catches structured patterns (currency, dates, account numbers) without GPU cost.
The LLM (Layer 2) handles contextual extraction that regex can't do.
"""

import re
from loguru import logger
from config.schemas import FinancialEntity
from analysis.vocab_loader import get_currency_words, get_date_words

try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except (ImportError, OSError):
    _nlp = None
    HAS_SPACY = False
    logger.warning("spaCy not available — falling back to regex-only extraction")

# Fine-tuned financial NER model (distilbert on FiNER-ORD)
_ner_pipeline = None
_ner_label_map = None
HAS_FINETUNED_NER = False

try:
    from pathlib import Path
    import json as _json
    _ner_model_path = Path("data/models/financial-ner")
    _ner_label_map_path = _ner_model_path / "label_map.json"
    if _ner_model_path.exists() and _ner_label_map_path.exists():
        HAS_FINETUNED_NER = True
except Exception:
    pass


# ── CURRENCY PATTERNS (symbol-based, language-agnostic — always run) ──

SYMBOL_CURRENCY_PATTERNS = [
    # ₹ can appear anywhere; Rs/INR require word boundary before them so they don't match
    # inside phone numbers like "1037-498-400" or other alphanumeric strings.
    # Number part: plain digits OR comma-separated (Indian/Western format).
    # Magnitude group captures million/billion/thousand/mn/bn after the number.
    # Negative lookahead prevents matching numbers followed by more digits or date separators.
    (r"(?:₹|(?<![A-Za-z0-9])Rs\.?\s*|(?<![A-Za-z0-9])INR\s*)(\d+(?:,\d{2,3})*(?:\.\d{1,2})?)(?!\d|[-/])\s*(crore|lakh|thousand|million|billion|cr|mn|bn|[mbk]\b)?", "INR"),
    (r"(?:\$|(?<![A-Za-z0-9])USD\s*)(\d+(?:,\d{2,3})*(?:\.\d{1,2})?)(?!\d|[-/])\s*(million|billion|thousand|mn|bn|[mbk]\b)?", "USD"),
    (r"(?:€|(?<![A-Za-z0-9])EUR\s*)(\d+(?:,\d{2,3})*(?:\.\d{1,2})?)(?!\d|[-/])\s*(million|billion|thousand|mn|bn|[mbk]\b)?", "EUR"),
]

# Magnitude multipliers for currency patterns
_MAGNITUDE_MAP = {
    "million": 1_000_000, "mn": 1_000_000, "m": 1_000_000,
    "billion": 1_000_000_000, "bn": 1_000_000_000, "b": 1_000_000_000,
    "thousand": 1_000, "k": 1_000,
    "crore": 10_000_000, "cr": 10_000_000,
    "lakh": 100_000,
}

# ── DATE PATTERNS (numeric, language-agnostic — always run) ──

NUMERIC_DATE_PATTERNS = [
    r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b",
]


def _build_currency_patterns(lang: str = "en") -> list[tuple]:
    """Build currency patterns from vocab (symbol patterns + word-based patterns)."""
    patterns = list(SYMBOL_CURRENCY_PATTERNS)
    currency_words = get_currency_words(lang)
    for entry in currency_words:
        words = entry.get("words", [])
        currency = entry.get("currency", "INR")
        if words:
            word_alt = "|".join(re.escape(w) for w in words)
            if currency in ("INR_LAKH", "INR_CRORE", "INR_THOUSAND"):
                patterns.append((rf"(\d+(?:\.\d+)?)\s*(?:{word_alt})", currency))
            else:
                patterns.append((rf"(\d{{1,3}}(?:,\d{{2,3}})*)\s*(?:{word_alt})", currency))
    # For non-English, also include English word patterns (code-switching)
    if lang != "en":
        for entry in get_currency_words("en"):
            words = entry.get("words", [])
            currency = entry.get("currency", "INR")
            if words:
                word_alt = "|".join(re.escape(w) for w in words)
                if currency in ("INR_LAKH", "INR_CRORE", "INR_THOUSAND"):
                    patterns.append((rf"(\d+(?:\.\d+)?)\s*(?:{word_alt})", currency))
                else:
                    patterns.append((rf"(\d{{1,3}}(?:,\d{{2,3}})*)\s*(?:{word_alt})", currency))
    return patterns


def _build_date_patterns(lang: str = "en") -> list[str]:
    """Build date regex patterns from vocab."""
    patterns = list(NUMERIC_DATE_PATTERNS)
    date_words = get_date_words(lang)
    months = date_words.get("months", [])
    days = date_words.get("days", [])
    relative = date_words.get("relative", [])
    if months:
        month_alt = "|".join(re.escape(m) for m in months)
        patterns.append(rf"\b((?:{month_alt})\s+\d{{1,2}}(?:,?\s*\d{{4}})?)\b")
    if days:
        day_alt = "|".join(re.escape(d) for d in days)
        patterns.append(rf"\b(next\s+(?:{day_alt}))\b")
        patterns.append(rf"\b(this\s+(?:{day_alt}))\b")
    if relative:
        rel_alt = "|".join(re.escape(r) for r in relative)
        patterns.append(rf"\b((?:by\s+)?(?:{rel_alt}))\b")
    # For non-English, also add English date patterns
    if lang != "en":
        en_date = get_date_words("en")
        en_months = en_date.get("months", [])
        en_days = en_date.get("days", [])
        if en_months:
            month_alt = "|".join(re.escape(m) for m in en_months)
            patterns.append(rf"\b((?:{month_alt})\s+\d{{1,2}}(?:,?\s*\d{{4}})?)\b")
        if en_days:
            day_alt = "|".join(re.escape(d) for d in en_days)
            patterns.append(rf"\b(next\s+(?:{day_alt}))\b")
    return patterns

# Backward-compatible aliases (used by tests)
CURRENCY_PATTERNS = SYMBOL_CURRENCY_PATTERNS
DATE_PATTERNS = NUMERIC_DATE_PATTERNS

# ── ACCOUNT / REFERENCE PATTERNS ──

ACCOUNT_PATTERNS = [
    # 10-16 digit account numbers
    (r"\b(\d{10,16})\b", "account_number"),
    # PAN format: ABCDE1234F
    (r"\b([A-Z]{5}\d{4}[A-Z])\b", "pan_number"),
    # IFSC: ABCD0123456
    (r"\b([A-Z]{4}0\d{6})\b", "ifsc_code"),
    # Reference/ticket numbers (require at least one letter and one digit)
    (r"(?:ref(?:erence)?|ticket|case|complaint)\s*(?:#|no\.?|number)?\s*:?\s*([A-Z0-9\-]{6,20}(?=[A-Z])(?=.*\d))", "reference_number"),
]

# ── FINANCIAL TERM PATTERNS ──

FINANCIAL_PATTERNS = [
    (r"\b(\d+(?:\.\d+)?)\s*(?:%|percent|per\s*cent)", "interest_rate"),
    (r"\bemi\s*(?:of|is|was|amount)?\s*(?:₹|Rs\.?\s*)?(\d{1,3}(?:,\d{2,3})*)", "emi_amount"),
    (r"\bloan\s*(?:of|amount|is|was)?\s*(?:₹|Rs\.?\s*)?(\d{1,3}(?:,\d{2,3})*)", "loan_amount"),
    (r"\b(\d+)\s*(?:months?|yrs?|years?)\s*(?:tenure|term|period)", "tenure"),
]


def extract_entities_regex(segments: list, lang: str = "en") -> list[FinancialEntity]:
    """Layer 1: Deterministic entity extraction using regex patterns.

    Runs on CPU, instant. Catches structured patterns before LLM.
    Currency symbols (₹, $, €) and numeric formats work for any language.
    Word-based patterns use vocab files for the detected language.
    """
    entities = []
    currency_patterns = _build_currency_patterns(lang)
    date_patterns = _build_date_patterns(lang)

    for seg_idx, seg in enumerate(segments):
        text = seg.get("text", "")
        start_time = seg.get("start", 0.0)

        # Currency amounts
        for pattern, currency in currency_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                raw = match.group(0).strip()
                value_str = match.group(1).replace(",", "")

                if currency == "INR_LAKH":
                    value = float(value_str) * 100000
                    currency = "INR"
                elif currency == "INR_CRORE":
                    value = float(value_str) * 10000000
                    currency = "INR"
                elif currency == "INR_THOUSAND":
                    value = float(value_str) * 1000
                    currency = "INR"
                else:
                    value = float(value_str)
                    # Check for magnitude word (million, billion, etc.) in group 2
                    try:
                        magnitude = match.group(2)
                        if magnitude:
                            multiplier = _MAGNITUDE_MAP.get(magnitude.lower(), 1)
                            value *= multiplier
                    except (IndexError, AttributeError):
                        pass

                entities.append(FinancialEntity(
                    entity_type="payment_amount",
                    value=f"{value:.2f} {currency}",
                    raw_text=raw,
                    segment_id=seg_idx,
                    start_time=start_time,
                    confidence=0.95,
                ))

        # Dates
        for pattern in date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(FinancialEntity(
                    entity_type="due_date",
                    value=match.group(1),
                    raw_text=match.group(0),
                    segment_id=seg_idx,
                    start_time=start_time,
                    confidence=0.90,
                ))

        # Account/reference numbers
        for pattern, etype in ACCOUNT_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(FinancialEntity(
                    entity_type=etype,
                    value=match.group(1),
                    raw_text=match.group(0),
                    segment_id=seg_idx,
                    start_time=start_time,
                    confidence=0.90,
                ))

        # Financial terms (EMI, interest rate, etc.)
        for pattern, etype in FINANCIAL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entities.append(FinancialEntity(
                    entity_type=etype,
                    value=match.group(1).replace(",", ""),
                    raw_text=match.group(0),
                    segment_id=seg_idx,
                    start_time=start_time,
                    confidence=0.85,
                ))

    # Deduplicate by (entity_type, value, segment_id)
    seen = set()
    unique = []
    for e in entities:
        key = (e.entity_type, e.value, e.segment_id)
        if key not in seen:
            seen.add(key)
            unique.append(e)

    logger.info(f"Layer 1 entity extraction: {len(unique)} entities from regex")
    return unique


def extract_entities_spacy(segments: list) -> list[FinancialEntity]:
    """Extract named entities using spaCy NER (PERSON, ORG, MONEY, DATE, etc.)."""
    if not HAS_SPACY:
        return []

    entities = []
    for seg_idx, seg in enumerate(segments):
        text = seg.get("text", "")
        start_time = seg.get("start", 0.0)
        if not text.strip():
            continue

        doc = _nlp(text)
        for ent in doc.ents:
            if ent.label_ == "MONEY":
                etype = "payment_amount"
            elif ent.label_ == "DATE":
                etype = "due_date"
            elif ent.label_ == "ORG":
                etype = "organization"
            elif ent.label_ == "PERSON":
                etype = "person_name"
            elif ent.label_ == "CARDINAL" and any(c.isdigit() for c in ent.text):
                etype = "reference_number"
            else:
                continue

            entities.append(FinancialEntity(
                entity_type=etype,
                value=ent.text,
                raw_text=ent.text,
                segment_id=seg_idx,
                start_time=start_time,
                confidence=0.75,
            ))

    logger.info(f"spaCy NER: {len(entities)} entities extracted")
    return entities


def _get_ner_pipeline():
    """Load fine-tuned financial NER model (lazy, cached)."""
    global _ner_pipeline, _ner_label_map
    if _ner_pipeline is not None:
        return _ner_pipeline, _ner_label_map

    if not HAS_FINETUNED_NER:
        return None, None

    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline as hf_pipeline
        with open(_ner_label_map_path) as f:
            _ner_label_map = _json.load(f)

        # Explicit load + .to("cpu") to bypass accelerate meta tensor issue (PyTorch 2.8)
        _ner_model = AutoModelForTokenClassification.from_pretrained(
            str(_ner_model_path), low_cpu_mem_usage=False, device_map=None,
        )
        _ner_model = _ner_model.to("cpu")
        _ner_tokenizer = AutoTokenizer.from_pretrained(str(_ner_model_path))
        _ner_pipeline = hf_pipeline(
            "token-classification",
            model=_ner_model,
            tokenizer=_ner_tokenizer,
            device=-1,
            aggregation_strategy="simple",
            truncation=True,
            max_length=512,
        )
        logger.info(f"Financial NER model loaded from {_ner_model_path} ({len(_ner_label_map)} labels)")
        return _ner_pipeline, _ner_label_map
    except Exception as e:
        logger.warning(f"Could not load financial NER model: {e}")
        return None, None


# Map NER BIO labels to FinancialEntity types
_NER_TYPE_MAP = {
    "person_name": "person_name",
    "PER": "person_name",
    "location": "location",
    "LOC": "location",
    "organization": "organization",
    "ORG": "organization",
}


def extract_entities_finetuned_ner(segments: list) -> list[FinancialEntity]:
    """Extract entities using fine-tuned financial NER model (distilbert on FiNER-ORD).

    Detects person names, locations, and organizations with higher accuracy than spaCy
    on financial text (93.7% entity F1 on FiNER-ORD test set).
    """
    pipe, label_map = _get_ner_pipeline()
    if pipe is None:
        return []

    entities = []
    for seg_idx, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        start_time = seg.get("start", 0.0)
        if not text:
            continue

        try:
            results = pipe(text[:512], truncation=True, max_length=512)
            for ent in results:
                # Extract entity type from label (e.g., "B-person_name" → "person_name")
                label = ent.get("entity_group", ent.get("entity", ""))
                label_clean = label.replace("B-", "").replace("I-", "")
                etype = _NER_TYPE_MAP.get(label_clean)
                if not etype:
                    continue

                score = ent.get("score", 0.0)
                if score < 0.5:
                    continue

                entities.append(FinancialEntity(
                    entity_type=etype,
                    value=ent.get("word", "").strip(),
                    raw_text=ent.get("word", "").strip(),
                    segment_id=seg_idx,
                    start_time=start_time,
                    confidence=round(score, 3),
                ))
        except Exception as e:
            if seg_idx == 0:
                logger.warning(f"NER model failed on segment {seg_idx}: {e}")
            continue

    logger.info(f"Fine-tuned NER: {len(entities)} entities extracted")
    return entities


def extract_all_entities_layer1(segments: list, lang: str = "en") -> list[FinancialEntity]:
    """Run all Layer 1 (CPU) entity extraction and merge results.

    Priority: regex (highest confidence) > fine-tuned NER > spaCy (lowest).
    For non-English text, skips English-only models (NER, spaCy).
    """
    regex_entities = extract_entities_regex(segments, lang=lang)
    # Fine-tuned NER and spaCy are English-only — skip for non-English
    ner_entities = extract_entities_finetuned_ner(segments) if lang == "en" else []
    spacy_entities = extract_entities_spacy(segments) if lang == "en" else []

    # Merge: regex wins, then NER, then spaCy on conflicts
    seen_keys = set()
    merged = []

    for e in regex_entities:
        key = (e.entity_type, e.segment_id)
        seen_keys.add(key)
        merged.append(e)

    for e in ner_entities:
        key = (e.entity_type, e.segment_id)
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(e)

    for e in spacy_entities:
        key = (e.entity_type, e.segment_id)
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(e)

    # Validation: filter out noise entities
    validated = []
    for e in merged:
        raw = e.raw_text.strip() if e.raw_text else ""
        val = e.value.strip() if e.value else ""
        # Reject bare single digits as reference numbers (e.g., "2" from "press star then 2")
        if e.entity_type == "reference_number" and raw.isdigit() and int(raw) < 10:
            continue
        # Reject known acronyms misidentified as organizations
        if e.entity_type == "organization" and raw.upper() in (
            "OTP", "EMI", "KYC", "PAN", "UPI", "CEO", "CFO", "CTO", "COO",
            "SEC", "GDP", "IT", "HR", "ATM", "PIN", "SIP", "NPA", "API",
            "O&M", "EPS", "IPO", "ETF", "NAV", "AUM", "ROE", "ROI", "P&L",
        ):
            continue
        # Reject vague temporal references as due_date (not actionable)
        if e.entity_type in ("due_date", "date"):
            raw_lower = raw.lower()
            vague_dates = (
                "today", "tomorrow", "yesterday", "this year", "last year", "next year",
                "this quarter", "last quarter", "next quarter", "this month", "last month",
                "this week", "now", "recently", "currently", "the quarter", "the year",
                "this past quarter", "previous years", "prior year", "prior quarter",
                "winter season", "summer", "winter", "spring", "fall",
                "every day", "quarterly", "annually", "monthly", "weekly", "daily",
                "each year", "each quarter", "each month", "every year",
            )
            if raw_lower in vague_dates or raw_lower.endswith("ago") or raw_lower.isdigit():
                continue
        validated.append(e)

    logger.info(
        f"Layer 1 total: {len(validated)} entities "
        f"(regex={len(regex_entities)}, NER={len(ner_entities)}, spaCy={len(spacy_entities)}, filtered={len(merged)-len(validated)})"
    )
    return validated
