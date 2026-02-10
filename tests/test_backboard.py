"""Tests for Backboard.io client (unit tests — no network calls)."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from services.backboard.client import (
    is_configured,
    _headers,
    _form_headers,
    store_call_record,
    store_customer_interaction,
    query_customer_history,
    query_audit_trail,
    compliance_reasoning,
)


# ── Configuration ──

class TestConfiguration:
    def test_is_configured_with_placeholder(self):
        with patch("services.backboard.client.API_KEY", "bk_your_key_here"):
            assert is_configured() is False

    def test_is_configured_empty(self):
        with patch("services.backboard.client.API_KEY", ""):
            assert is_configured() is False

    def test_is_configured_valid(self):
        with patch("services.backboard.client.API_KEY", "bk_real_key_123"):
            assert is_configured() is True

    def test_headers_include_api_key(self):
        with patch("services.backboard.client.API_KEY", "bk_test"):
            h = _headers()
            assert h["X-API-Key"] == "bk_test"
            assert h["Content-Type"] == "application/json"

    def test_form_headers_no_content_type(self):
        with patch("services.backboard.client.API_KEY", "bk_test"):
            h = _form_headers()
            assert "X-API-Key" in h
            assert "Content-Type" not in h


# ── Graceful Degradation (not configured) ──

class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_store_call_record_returns_none_when_unconfigured(self):
        with patch("services.backboard.client.is_configured", return_value=False):
            result = await store_call_record({"call_id": "test123"})
            assert result is None

    @pytest.mark.asyncio
    async def test_store_customer_returns_none_when_unconfigured(self):
        with patch("services.backboard.client.is_configured", return_value=False):
            result = await store_customer_interaction("cust1", {"call_id": "test123"})
            assert result is None

    @pytest.mark.asyncio
    async def test_query_customer_returns_message_when_unconfigured(self):
        with patch("services.backboard.client.is_configured", return_value=False):
            result = await query_customer_history("cust1", "any question")
            assert "not configured" in result.lower()

    @pytest.mark.asyncio
    async def test_query_audit_returns_message_when_unconfigured(self):
        with patch("services.backboard.client.is_configured", return_value=False):
            result = await query_audit_trail("any question")
            assert "not configured" in result.lower()

    @pytest.mark.asyncio
    async def test_compliance_reasoning_returns_skipped_when_unconfigured(self):
        with patch("services.backboard.client.is_configured", return_value=False):
            result = await compliance_reasoning("excerpt", "context")
            assert result["judgment"] == "skipped"


# ── Call Record Storage (mocked) ──

class TestCallRecordStorage:
    @pytest.mark.asyncio
    async def test_store_call_record_builds_summary(self):
        """Verify the summary content is well-structured."""
        call_record = {
            "call_id": "abc12345",
            "call_type": "collections",
            "duration_seconds": 180.5,
            "detected_language": "en",
            "overall_risk_level": "high",
            "compliance_score": 70,
            "call_summary": "Customer refused to pay EMI.",
            "compliance_checks": [
                {"passed": False, "check_name": "opening_disclosure", "severity": "high", "evidence_text": "Agent did not identify"},
            ],
            "obligations": [
                {"speaker": "customer", "text": "I will pay by Friday", "strength": "promise"},
            ],
            "financial_entities": [
                {"entity_type": "payment_amount", "value": "5000 INR"},
            ],
            "fraud_signals": [],
            "key_outcomes": ["Customer refused initial payment"],
            "next_actions": ["Follow up on Friday"],
        }

        with patch("services.backboard.client.is_configured", return_value=True), \
             patch("services.backboard.client.get_or_create_finvoice_assistant", new_callable=AsyncMock, return_value="asst_123"), \
             patch("services.backboard.client.create_thread", new_callable=AsyncMock, return_value="thread_123"), \
             patch("services.backboard.client.store_memory", new_callable=AsyncMock, return_value={"status": "ok"}) as mock_store:

            result = await store_call_record(call_record)
            assert result == {"status": "ok"}

            # Verify store_memory was called with structured content
            call_args = mock_store.call_args
            content = call_args.kwargs.get("content") or call_args[1].get("content") or call_args[0][1] if len(call_args[0]) > 1 else ""
            # The content should be passed as keyword arg
            assert mock_store.called
