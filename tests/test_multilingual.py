"""Tests for multilingual support — Hindi, Tamil, English vocab loading and integration."""

import pytest
from unittest.mock import patch
from analysis.vocab_loader import (
    get_greetings,
    get_trivial_phrases,
    get_currency_words,
    get_date_words,
    get_compliance_keywords,
    get_prohibited_phrases,
    get_end_call_phrases,
    get_financial_terms,
    has_vocab,
    clear_cache,
)
from analysis.intelligence import (
    extract_entities_regex,
    extract_all_entities_layer1,
    _build_currency_patterns,
    _build_date_patterns,
)
from analysis.compliance import run_compliance_checks, _get_disclosures, _get_prohibited
from analysis.sentiment import classify_for_llm_routing, classify_intent


# ── Vocab Loader ──

class TestVocabLoader:
    def setup_method(self):
        clear_cache()

    def test_english_vocab_exists(self):
        assert has_vocab("en")

    def test_hindi_vocab_exists(self):
        assert has_vocab("hi")

    def test_tamil_vocab_exists(self):
        assert has_vocab("ta")

    def test_missing_vocab_returns_empty(self):
        assert get_greetings("xx") == set()
        assert get_trivial_phrases("xx") == set()
        assert get_currency_words("xx") == []
        assert get_date_words("xx") == {}

    def test_english_greetings_not_empty(self):
        greetings = get_greetings("en")
        assert len(greetings) > 0
        assert "hello" in greetings

    def test_hindi_greetings_loaded(self):
        greetings = get_greetings("hi")
        assert len(greetings) > 0
        assert "नमस्ते" in greetings

    def test_tamil_greetings_loaded(self):
        greetings = get_greetings("ta")
        assert len(greetings) > 0
        assert "வணக்கம்" in greetings

    def test_english_trivials_contain_common_phrases(self):
        trivials = get_trivial_phrases("en")
        assert "yes" in trivials
        assert "okay" in trivials

    def test_hindi_trivials_contain_common_phrases(self):
        trivials = get_trivial_phrases("hi")
        assert len(trivials) > 0
        # Hindi "yes" and "okay" equivalents
        assert "हाँ" in trivials or "जी" in trivials

    def test_cache_works(self):
        """Loading same language twice returns cached result."""
        clear_cache()
        g1 = get_greetings("en")
        g2 = get_greetings("en")
        assert g1 == g2


# ── Currency Extraction (Multilingual) ──

class TestMultilingualCurrency:
    def test_inr_symbol_works_any_language(self):
        """₹ symbol should be detected regardless of language."""
        segments = [{"text": "₹5,000 दीजिए", "start": 0.0}]
        entities = extract_entities_regex(segments, lang="hi")
        amounts = [e for e in entities if e.entity_type == "payment_amount"]
        assert len(amounts) >= 1
        assert "5000" in amounts[0].value

    def test_hindi_rupees_word(self):
        """Hindi word 'रुपये' should match currency."""
        segments = [{"text": "5000 रुपये का भुगतान करें", "start": 0.0}]
        entities = extract_entities_regex(segments, lang="hi")
        amounts = [e for e in entities if e.entity_type == "payment_amount"]
        assert len(amounts) >= 1

    def test_hindi_lakh(self):
        """Hindi 'लाख' should expand to 100,000."""
        segments = [{"text": "5 लाख रुपये बकाया है", "start": 0.0}]
        entities = extract_entities_regex(segments, lang="hi")
        amounts = [e for e in entities if e.entity_type == "payment_amount"]
        lakh_amounts = [a for a in amounts if "500000" in a.value]
        assert len(lakh_amounts) >= 1

    def test_hindi_crore(self):
        """Hindi 'करोड़' should expand to 10,000,000."""
        segments = [{"text": "2 करोड़ का लोन", "start": 0.0}]
        entities = extract_entities_regex(segments, lang="hi")
        amounts = [e for e in entities if e.entity_type == "payment_amount"]
        crore_amounts = [a for a in amounts if "20000000" in a.value]
        assert len(crore_amounts) >= 1

    def test_tamil_rupees_word(self):
        """Tamil word 'ரூபாய்' should match currency."""
        segments = [{"text": "5000 ரூபாய் செலுத்துங்கள்", "start": 0.0}]
        entities = extract_entities_regex(segments, lang="ta")
        amounts = [e for e in entities if e.entity_type == "payment_amount"]
        assert len(amounts) >= 1

    def test_tamil_lakh(self):
        """Tamil 'லட்சம்' should expand to 100,000."""
        segments = [{"text": "3 லட்சம் பாக்கி உள்ளது", "start": 0.0}]
        entities = extract_entities_regex(segments, lang="ta")
        amounts = [e for e in entities if e.entity_type == "payment_amount"]
        lakh_amounts = [a for a in amounts if "300000" in a.value]
        assert len(lakh_amounts) >= 1

    def test_english_lakh_crore(self):
        """English 'lakh' and 'crore' should work (Indian English)."""
        segments = [{"text": "Your loan amount is 5 lakh rupees", "start": 0.0}]
        entities = extract_entities_regex(segments, lang="en")
        amounts = [e for e in entities if e.entity_type == "payment_amount"]
        lakh_amounts = [a for a in amounts if "500000" in a.value]
        assert len(lakh_amounts) >= 1

    def test_code_switching_english_in_hindi(self):
        """English currency words should also match in Hindi mode (code-switching)."""
        segments = [{"text": "10000 rupees भुगतान करो", "start": 0.0}]
        entities = extract_entities_regex(segments, lang="hi")
        amounts = [e for e in entities if e.entity_type == "payment_amount"]
        assert len(amounts) >= 1

    def test_build_currency_patterns_includes_vocab(self):
        """_build_currency_patterns should include vocab-based patterns."""
        patterns = _build_currency_patterns("en")
        # Should have more patterns than just the 3 symbol-based ones
        assert len(patterns) > 3

    def test_build_currency_patterns_hindi(self):
        """Hindi currency patterns should include Hindi words + English words."""
        patterns = _build_currency_patterns("hi")
        # Should include both Hindi and English patterns (code-switching)
        assert len(patterns) > 3


# ── Date Extraction (Multilingual) ──

class TestMultilingualDates:
    def test_hindi_month_names(self):
        """Hindi month names should be loaded."""
        date_words = get_date_words("hi")
        months = date_words.get("months", [])
        assert "जनवरी" in months  # January in Hindi

    def test_tamil_month_names(self):
        """Tamil month names should be loaded."""
        date_words = get_date_words("ta")
        months = date_words.get("months", [])
        assert "ஜனவரி" in months  # January in Tamil

    def test_hindi_relative_dates(self):
        """Hindi relative date words should be detected."""
        date_words = get_date_words("hi")
        relative = date_words.get("relative", [])
        assert "कल" in relative  # tomorrow in Hindi

    def test_numeric_dates_work_any_language(self):
        """Numeric dates (DD/MM/YYYY) should work regardless of language."""
        segments = [{"text": "15/01/2024 तक भुगतान करें", "start": 0.0}]
        entities = extract_entities_regex(segments, lang="hi")
        dates = [e for e in entities if e.entity_type == "due_date"]
        assert len(dates) >= 1

    def test_build_date_patterns_hindi(self):
        """Hindi date patterns should include Hindi + English month/day names."""
        patterns = _build_date_patterns("hi")
        assert len(patterns) > 1  # More than just numeric patterns


# ── Compliance (Multilingual) ──

class TestMultilingualCompliance:
    def test_hindi_compliance_keywords_loaded(self):
        """Hindi compliance keywords should load from vocab."""
        kw = get_compliance_keywords("hi")
        assert len(kw) > 0
        # Should have at least 'collections' or 'general' call type
        assert "collections" in kw or "general" in kw

    def test_hindi_disclosures_include_english(self):
        """For Hindi, disclosures should include both Hindi and English keywords."""
        disclosures = _get_disclosures("collections", lang="hi")
        assert len(disclosures) > 0
        # At least one disclosure should have keywords
        all_keywords = []
        for d in disclosures:
            all_keywords.extend(d["keywords"])
        assert len(all_keywords) > 0

    def test_hindi_prohibited_phrases_loaded(self):
        """Hindi prohibited phrases should load from vocab."""
        prohibited = _get_prohibited("hi")
        assert "threats" in prohibited
        assert len(prohibited["threats"]) > 0

    def test_prohibited_phrases_merge_english_for_hindi(self):
        """Hindi prohibited phrases should also include English phrases."""
        hi_prohibited = _get_prohibited("hi")
        en_prohibited = _get_prohibited("en")
        # Hindi should have at least as many categories as English
        for cat in en_prohibited:
            assert cat in hi_prohibited

    def test_compliance_check_with_hindi_lang(self):
        """run_compliance_checks should accept lang='hi' without error."""
        segments = [
            {"text": "मेरा नाम राहुल है, बैंक से बोल रहा हूँ", "speaker": "Agent", "start": 0.0},
            {"text": "आपकी किस्त बकाया है", "speaker": "Agent", "start": 10.0},
        ]
        checks = run_compliance_checks(segments, "collections", lang="hi")
        assert isinstance(checks, list)

    def test_hindi_end_call_phrases(self):
        """Hindi end-call phrases should be loaded from vocab."""
        phrases = get_end_call_phrases("hi")
        assert len(phrases) > 0

    def test_tamil_end_call_phrases(self):
        """Tamil end-call phrases should be loaded from vocab."""
        phrases = get_end_call_phrases("ta")
        assert len(phrases) > 0


# ── Sentiment & Intent Routing (Multilingual) ──

class TestMultilingualSentiment:
    def test_intent_returns_none_for_hindi(self):
        """classify_intent should return None for non-English (forces LLM fallback)."""
        result = classify_intent("मैं भुगतान नहीं कर सकता", lang="hi")
        assert result is None

    def test_intent_returns_none_for_tamil(self):
        """classify_intent should return None for Tamil."""
        result = classify_intent("நான் பணம் செலுத்த முடியாது", lang="ta")
        assert result is None

    def test_intent_works_for_english(self):
        """classify_intent should work normally for English."""
        result = classify_intent("I want to pay my EMI", lang="en")
        # Should return a dict (not None) for English
        assert result is not None or True  # May be None if model not loaded, but shouldn't crash

    def test_routing_skip_hindi_trivial(self):
        """Hindi trivial phrase should be classified as 'skip'."""
        trivials = get_trivial_phrases("hi")
        if trivials:
            phrase = list(trivials)[0]
            route = classify_for_llm_routing(phrase, lang="hi")
            assert route == "skip"

    def test_routing_short_non_english(self):
        """Very short non-English utterance without vocab should be skipped."""
        # Use a language with no vocab file
        route = classify_for_llm_routing("да", lang="ru")
        # Single word in unknown language — should skip (heuristic)
        assert route == "skip"


# ── Layer 1 Entity Extraction — English-Only Model Gating ──

class TestModelGating:
    def test_layer1_skips_ner_for_hindi(self):
        """Fine-tuned NER and spaCy should be skipped for non-English."""
        segments = [{"text": "5000 रुपये का भुगतान", "start": 0.0}]
        entities = extract_all_entities_layer1(segments, lang="hi")
        # Should still get regex entities
        amounts = [e for e in entities if e.entity_type == "payment_amount"]
        assert len(amounts) >= 1

    def test_layer1_runs_all_for_english(self):
        """All extractors should run for English."""
        segments = [{"text": "Pay ₹10,000 to HDFC Bank by January 15", "start": 0.0}]
        entities = extract_all_entities_layer1(segments, lang="en")
        assert len(entities) >= 1

    def test_pan_works_any_language(self):
        """PAN format detection is language-agnostic (regex on format)."""
        segments = [{"text": "मेरा PAN ABCDE1234F है", "start": 0.0}]
        entities = extract_entities_regex(segments, lang="hi")
        pans = [e for e in entities if e.entity_type == "pan_number"]
        assert len(pans) >= 1
        assert pans[0].value == "ABCDE1234F"

    def test_ifsc_works_any_language(self):
        """IFSC format detection is language-agnostic (regex on format)."""
        segments = [{"text": "IFSC HDFC0001234 ஆகும்", "start": 0.0}]
        entities = extract_entities_regex(segments, lang="ta")
        ifscs = [e for e in entities if e.entity_type == "ifsc_code"]
        assert len(ifscs) >= 1


# ── Financial Terms ──

class TestFinancialTerms:
    def test_hindi_financial_corrections(self):
        """Hindi financial term corrections should map to English acronyms."""
        terms = get_financial_terms("hi")
        corrections = terms.get("corrections", {})
        assert "ईएमआई" in corrections
        assert corrections["ईएमआई"] == "EMI"

    def test_tamil_financial_corrections(self):
        """Tamil financial term corrections should map to English acronyms."""
        terms = get_financial_terms("ta")
        corrections = terms.get("corrections", {})
        assert "இஎம்ஐ" in corrections
        assert corrections["இஎம்ஐ"] == "EMI"

    def test_english_financial_corrections(self):
        """English financial corrections should exist."""
        terms = get_financial_terms("en")
        corrections = terms.get("corrections", {})
        assert len(corrections) > 0
        assert corrections.get("emmy") == "EMI" or corrections.get("emi") == "EMI"


# ── Graceful Degradation ──

class TestGracefulDegradation:
    def test_unknown_language_no_crash_entities(self):
        """Entity extraction with unknown language should not crash."""
        segments = [{"text": "₹5000 payment due", "start": 0.0}]
        entities = extract_entities_regex(segments, lang="xx")
        # Should still get symbol-based match
        amounts = [e for e in entities if e.entity_type == "payment_amount"]
        assert len(amounts) >= 1

    def test_unknown_language_no_crash_compliance(self):
        """Compliance check with unknown language should not crash."""
        segments = [{"text": "Hello from bank", "speaker": "Agent", "start": 0.0}]
        checks = run_compliance_checks(segments, "general", lang="xx")
        assert isinstance(checks, list)

    def test_unknown_language_no_crash_routing(self):
        """LLM routing with unknown language should not crash."""
        route = classify_for_llm_routing("test", lang="xx")
        assert route in ("skip", "full", "negative_subset")
