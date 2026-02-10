"""Tests for entity extraction (regex + spaCy + fine-tuned NER)."""

import pytest
from analysis.intelligence import (
    extract_entities_regex,
    extract_entities_spacy,
    extract_entities_finetuned_ner,
    extract_all_entities_layer1,
    CURRENCY_PATTERNS,
    DATE_PATTERNS,
    ACCOUNT_PATTERNS,
    HAS_SPACY,
    HAS_FINETUNED_NER,
)
from config.schemas import FinancialEntity


# ── Regex Entity Extraction ──

class TestRegexExtraction:
    def test_inr_currency(self):
        segments = [{"text": "Your EMI of ₹5,000 is due tomorrow", "start": 0.0}]
        entities = extract_entities_regex(segments)
        amounts = [e for e in entities if e.entity_type == "payment_amount"]
        assert len(amounts) >= 1
        assert "5000" in amounts[0].value

    def test_usd_currency(self):
        segments = [{"text": "The charge is $250.00", "start": 0.0}]
        entities = extract_entities_regex(segments)
        amounts = [e for e in entities if e.entity_type == "payment_amount"]
        assert len(amounts) >= 1
        assert "USD" in amounts[0].value

    def test_lakh_amount(self):
        segments = [{"text": "Your loan amount is 5 lakh rupees", "start": 0.0}]
        entities = extract_entities_regex(segments)
        amounts = [e for e in entities if e.entity_type == "payment_amount"]
        assert len(amounts) >= 1
        assert "500000" in amounts[0].value

    def test_date_extraction(self):
        segments = [{"text": "Your payment is due on January 15, 2024", "start": 0.0}]
        entities = extract_entities_regex(segments)
        dates = [e for e in entities if e.entity_type == "due_date"]
        assert len(dates) >= 1

    def test_pan_number(self):
        segments = [{"text": "My PAN number is ABCDE1234F", "start": 0.0}]
        entities = extract_entities_regex(segments)
        pans = [e for e in entities if e.entity_type == "pan_number"]
        assert len(pans) >= 1
        assert pans[0].value == "ABCDE1234F"

    def test_ifsc_code(self):
        segments = [{"text": "IFSC code is HDFC0001234", "start": 0.0}]
        entities = extract_entities_regex(segments)
        ifscs = [e for e in entities if e.entity_type == "ifsc_code"]
        assert len(ifscs) >= 1

    def test_interest_rate(self):
        segments = [{"text": "The interest rate is 12.5 percent", "start": 0.0}]
        entities = extract_entities_regex(segments)
        rates = [e for e in entities if e.entity_type == "interest_rate"]
        assert len(rates) >= 1

    def test_emi_amount(self):
        segments = [{"text": "Your EMI of ₹15,000 is pending", "start": 0.0}]
        entities = extract_entities_regex(segments)
        emis = [e for e in entities if e.entity_type == "emi_amount"]
        assert len(emis) >= 1

    def test_empty_segments(self):
        entities = extract_entities_regex([])
        assert entities == []

    def test_no_entities(self):
        segments = [{"text": "Hello, how are you?", "start": 0.0}]
        entities = extract_entities_regex(segments)
        assert len(entities) == 0

    def test_deduplication(self):
        segments = [{"text": "Pay ₹5000, that is ₹5000", "start": 0.0}]
        entities = extract_entities_regex(segments)
        amounts = [e for e in entities if e.entity_type == "payment_amount"]
        # Should deduplicate same (type, value, segment_id)
        values = [e.value for e in amounts]
        # At least one, but check uniqueness
        assert len(set(values)) == len(values)

    def test_entity_has_correct_fields(self):
        segments = [{"text": "₹10,000 due on January 15", "start": 5.5}]
        entities = extract_entities_regex(segments)
        for e in entities:
            assert isinstance(e, FinancialEntity)
            assert e.segment_id == 0
            assert e.start_time == 5.5
            assert 0 <= e.confidence <= 1


# ── spaCy NER ──

@pytest.mark.skipif(not HAS_SPACY, reason="spaCy not installed")
class TestSpacyExtraction:
    def test_person_detection(self):
        segments = [{"text": "Mr. John Smith called about his account", "start": 0.0}]
        entities = extract_entities_spacy(segments)
        persons = [e for e in entities if e.entity_type == "person_name"]
        assert len(persons) >= 1

    def test_org_detection(self):
        segments = [{"text": "Goldman Sachs reported quarterly earnings", "start": 0.0}]
        entities = extract_entities_spacy(segments)
        orgs = [e for e in entities if e.entity_type == "organization"]
        assert len(orgs) >= 1


# ── Fine-tuned NER ──

@pytest.mark.skipif(not HAS_FINETUNED_NER, reason="Fine-tuned NER model not available")
class TestFinetunedNER:
    def test_person_detection(self):
        segments = [{"text": "John Smith called about the loan", "start": 0.0}]
        entities = extract_entities_finetuned_ner(segments)
        persons = [e for e in entities if e.entity_type == "person_name"]
        assert len(persons) >= 1

    def test_org_detection(self):
        segments = [{"text": "Goldman Sachs announced new investments", "start": 0.0}]
        entities = extract_entities_finetuned_ner(segments)
        orgs = [e for e in entities if e.entity_type == "organization"]
        assert len(orgs) >= 1

    def test_confidence_above_threshold(self):
        segments = [{"text": "Contact Jane Doe at HDFC Bank", "start": 0.0}]
        entities = extract_entities_finetuned_ner(segments)
        for e in entities:
            assert e.confidence >= 0.5


# ── Merged Layer 1 ──

class TestLayer1Merged:
    def test_merges_all_sources(self):
        segments = [
            {"text": "Mr. John Smith owes ₹50,000 to HDFC Bank", "start": 0.0},
        ]
        entities = extract_all_entities_layer1(segments)
        types = {e.entity_type for e in entities}
        assert "payment_amount" in types  # from regex

    def test_regex_takes_priority(self):
        """Regex entities should not be overridden by lower-priority sources."""
        segments = [{"text": "Pay ₹10,000 by January 15, 2024", "start": 0.0}]
        entities = extract_all_entities_layer1(segments)
        amounts = [e for e in entities if e.entity_type == "payment_amount"]
        if amounts:
            # Regex entities have confidence 0.95
            assert amounts[0].confidence >= 0.85
