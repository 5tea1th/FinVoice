"""Tests for intent classification (FinBERT fine-tuned model)."""

import pytest
from pathlib import Path
from analysis.sentiment import (
    classify_intent,
    classify_intent_batch,
    has_intent_model,
    HAS_FINBERT,
)

pytestmark = pytest.mark.skipif(
    not HAS_FINBERT or not has_intent_model(),
    reason="transformers not installed or intent model not trained",
)

VALID_INTENTS = {
    "agreement", "refusal", "request_extension", "payment_promise",
    "complaint", "consent_given", "consent_denied", "info_request",
    "negotiation", "escalation", "dispute", "acknowledgment", "greeting",
}


def test_classify_intent_returns_dict():
    result = classify_intent("I want to check my account balance")
    assert result is not None
    assert "intent" in result
    assert "confidence" in result


def test_classify_intent_valid_label():
    result = classify_intent("What is the interest rate on my loan?")
    assert result["intent"] in VALID_INTENTS


def test_classify_intent_confidence_range():
    result = classify_intent("I will pay by Friday")
    assert 0.0 <= result["confidence"] <= 1.0


def test_classify_intent_batch():
    texts = [
        "I want to check my balance",
        "I refuse to pay this amount",
        "Can you transfer me to a manager?",
    ]
    results = classify_intent_batch(texts)
    assert len(results) == 3
    for r in results:
        assert r is not None
        assert r["intent"] in VALID_INTENTS


def test_info_request_detection():
    """Info requests should be classified correctly."""
    result = classify_intent("What is my outstanding balance?")
    assert result["intent"] == "info_request"


def test_complaint_returns_valid_intent():
    """Complaint-like text should return a valid intent (model quality may vary)."""
    result = classify_intent("This is unacceptable, I've been charged wrongly twice")
    assert result["intent"] in VALID_INTENTS
    assert result["confidence"] > 0.0


def test_escalation_returns_valid_intent():
    """Escalation-like text should return a valid intent (model quality may vary)."""
    result = classify_intent("I need to speak with your supervisor immediately")
    assert result["intent"] in VALID_INTENTS
    assert result["confidence"] > 0.0
