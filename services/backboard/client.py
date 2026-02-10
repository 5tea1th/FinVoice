"""Backboard.io client — persistent memory, audit trails, and cloud LLM for FinVoice.

Provides 5 capabilities:
1. Cloud LLM fallback (GPT-4o for complex compliance reasoning)
2. Persistent customer memory across calls
3. RAG over transcripts + regulations
4. Embedding generation (offload from local GPU)
5. Immutable audit trail per call

API docs: https://app.backboard.io/docs
Auth: X-API-Key header with bk_ prefixed key
"""

import os
import json
import asyncio
import time
from typing import Optional
from loguru import logger

import httpx


API_BASE_URL = os.getenv("BACKBOARD_API_URL", "https://app.backboard.io/api")
API_KEY = os.getenv("BACKBOARD_API_KEY", "")

# Cache assistant/thread IDs to avoid recreating them
_assistant_cache: dict[str, str] = {}  # name -> assistant_id
_thread_cache: dict[str, str] = {}     # key -> thread_id


def _headers() -> dict:
    return {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY,
    }


def _form_headers() -> dict:
    """Headers for form-data endpoints (no Content-Type — httpx sets it)."""
    return {"X-API-Key": API_KEY}


def is_configured() -> bool:
    """Check if Backboard API key is set (not placeholder)."""
    return bool(API_KEY) and API_KEY != "bk_your_key_here"


# ── Assistant Management ──

async def create_assistant(
    name: str,
    description: str,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o",
) -> str:
    """Create a Backboard assistant. Returns assistant_id.

    Assistants define the LLM provider and memory scope.
    All threads under one assistant share the same memory pool.
    """
    if name in _assistant_cache:
        return _assistant_cache[name]

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{API_BASE_URL}/assistants",
            headers=_headers(),
            json={
                "name": name,
                "description": description,
                "llm_provider": llm_provider,
                "llm_model_name": llm_model,
                "tools": [],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        assistant_id = data["assistant_id"]
        _assistant_cache[name] = assistant_id
        logger.info(f"Backboard assistant created: {name} -> {assistant_id}")
        return assistant_id


# ── Thread Management ──

async def create_thread(assistant_id: str, cache_key: str | None = None) -> str:
    """Create a thread under an assistant. Returns thread_id.

    Each thread is an isolated conversation.
    Use cache_key to reuse threads (e.g., per customer).
    """
    if cache_key and cache_key in _thread_cache:
        return _thread_cache[cache_key]

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{API_BASE_URL}/assistants/{assistant_id}/threads",
            headers=_headers(),
            json={},
        )
        resp.raise_for_status()
        data = resp.json()
        thread_id = data["thread_id"]

        if cache_key:
            _thread_cache[cache_key] = thread_id

        return thread_id


# ── Message Operations ──

async def store_memory(
    thread_id: str,
    content: str,
    metadata: dict | None = None,
) -> dict:
    """Store information in Backboard memory WITHOUT triggering LLM response.

    Use this to ingest call data, transcripts, analysis results.
    Memory is automatically indexed for future retrieval.
    """
    form_data = {
        "content": content,
        "stream": "false",
        "memory": "auto",
        "send_to_llm": "false",
    }
    if metadata:
        form_data["metadata"] = json.dumps(metadata)

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{API_BASE_URL}/threads/{thread_id}/messages",
            headers=_form_headers(),
            data=form_data,
        )
        resp.raise_for_status()
        return resp.json()


async def query_with_memory(
    thread_id: str,
    question: str,
    llm_provider: str | None = None,
    model_name: str | None = None,
) -> dict:
    """Ask a question that uses Backboard's retrieved memories for context.

    The LLM receives relevant memories from all threads under the same assistant,
    then generates a response grounded in that context.
    """
    form_data = {
        "content": question,
        "stream": "false",
        "memory": "auto",
        "send_to_llm": "true",
    }
    if llm_provider:
        form_data["llm_provider"] = llm_provider
    if model_name:
        form_data["model_name"] = model_name

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{API_BASE_URL}/threads/{thread_id}/messages",
            headers=_form_headers(),
            data=form_data,
        )
        resp.raise_for_status()
        data = resp.json()
        return {
            "content": data.get("content", ""),
            "retrieved_memories": data.get("retrieved_memories", []),
            "memory_operation_id": data.get("memory_operation_id"),
        }


async def wait_for_memory_operation(
    operation_id: str,
    timeout_seconds: int = 60,
) -> dict | None:
    """Poll a memory operation until it completes."""
    start = time.time()
    async with httpx.AsyncClient(timeout=10.0) as client:
        while True:
            resp = await client.get(
                f"{API_BASE_URL}/assistants/memories/operations/{operation_id}",
                headers=_headers(),
            )
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status", "").upper()
                if status in ("COMPLETED", "ERROR"):
                    return data

            if time.time() - start > timeout_seconds:
                logger.warning(f"Memory operation {operation_id} timed out after {timeout_seconds}s")
                return None

            await asyncio.sleep(0.5)


# ── High-Level FinVoice Integration ──

async def get_or_create_finvoice_assistant() -> str:
    """Get or create the main FinVoice assistant on Backboard."""
    return await create_assistant(
        name="FinVoice Call Intelligence",
        description=(
            "Financial call analysis assistant with persistent memory. "
            "Analyzes collections, KYC, onboarding, complaint, and consent calls. "
            "Tracks customer interactions across calls, detects compliance violations, "
            "and maintains immutable audit trails. "
            "Has expertise in RBI Fair Practice Code, SEBI KYC Guidelines, "
            "RBI Digital Lending Guidelines, and Indian banking regulations."
        ),
        llm_provider="openai",
        llm_model="gpt-4o",
    )


async def store_call_record(call_record: dict) -> dict | None:
    """Store a processed CallRecord in Backboard for audit trail + memory.

    Creates a structured summary that Backboard can index and retrieve.
    Each call becomes a searchable memory entry.
    """
    if not is_configured():
        return None

    try:
        assistant_id = await get_or_create_finvoice_assistant()
        thread_id = await create_thread(assistant_id, cache_key="audit_trail")

        # Build structured summary for memory
        violations = [
            c for c in call_record.get("compliance_checks", [])
            if not c.get("passed", True)
        ]
        obligations = call_record.get("obligations", [])
        entities = call_record.get("financial_entities", [])

        summary_parts = [
            f"CALL RECORD — ID: {call_record['call_id']}",
            f"Type: {call_record.get('call_type', 'general')}",
            f"Duration: {call_record.get('duration_seconds', 0):.0f}s",
            f"Language: {call_record.get('detected_language', 'en')}",
            f"Risk: {call_record.get('overall_risk_level', 'low')}",
            f"Compliance Score: {call_record.get('compliance_score', 100)}/100",
            f"Summary: {call_record.get('call_summary', 'N/A')}",
        ]

        if violations:
            summary_parts.append(
                f"VIOLATIONS ({len(violations)}): "
                + "; ".join(
                    f"{v.get('check_name', '?')} [{v.get('severity', '?')}] — {v.get('evidence_text', '')[:100]}"
                    for v in violations
                )
            )

        if obligations:
            summary_parts.append(
                f"OBLIGATIONS ({len(obligations)}): "
                + "; ".join(
                    f"{o.get('speaker', '?')}: {o.get('text', '')[:100]} [{o.get('strength', '?')}]"
                    for o in obligations
                )
            )

        if entities:
            entity_summary = ", ".join(
                f"{e.get('entity_type', '?')}={e.get('value', '?')}"
                for e in entities[:10]
            )
            summary_parts.append(f"ENTITIES: {entity_summary}")

        if call_record.get("fraud_signals"):
            fraud_summary = "; ".join(
                f"{f.get('signal_type', '?')} (conf={f.get('confidence', 0):.2f})"
                for f in call_record["fraud_signals"]
            )
            summary_parts.append(f"FRAUD SIGNALS: {fraud_summary}")

        outcomes = call_record.get("key_outcomes", [])
        if outcomes:
            summary_parts.append(f"OUTCOMES: {'; '.join(outcomes)}")

        next_actions = call_record.get("next_actions", [])
        if next_actions:
            summary_parts.append(f"NEXT ACTIONS: {'; '.join(next_actions)}")

        content = "\n".join(summary_parts)

        result = await store_memory(
            thread_id=thread_id,
            content=content,
            metadata={
                "call_id": call_record["call_id"],
                "call_type": call_record.get("call_type", "general"),
                "risk_level": call_record.get("overall_risk_level", "low"),
                "compliance_score": call_record.get("compliance_score", 100),
            },
        )

        logger.info(f"Backboard: call {call_record['call_id']} stored in audit trail")
        return result

    except Exception as e:
        logger.warning(f"Backboard store_call_record failed: {e}")
        return None


async def store_customer_interaction(
    customer_id: str,
    call_record: dict,
) -> dict | None:
    """Store call data under a customer-specific thread for cross-call tracking.

    Enables queries like: "What payment promises has this customer made?"
    """
    if not is_configured():
        return None

    try:
        assistant_id = await get_or_create_finvoice_assistant()
        thread_id = await create_thread(
            assistant_id, cache_key=f"customer_{customer_id}"
        )

        # Build customer-focused summary
        obligations = call_record.get("obligations", [])
        customer_obligations = [
            o for o in obligations if o.get("speaker", "").lower() == "customer"
        ]

        parts = [
            f"CUSTOMER CALL — {call_record.get('call_type', 'general')} call",
            f"Call ID: {call_record['call_id']}",
            f"Customer emotion: {call_record.get('customer_emotion_dominant', 'neutral')}",
            f"Escalation: {'Yes' if call_record.get('escalation_detected') else 'No'}",
        ]

        if customer_obligations:
            for o in customer_obligations:
                parts.append(
                    f"Customer {o.get('obligation_type', 'statement')}: "
                    f"\"{o.get('text', '')}\" "
                    f"[strength: {o.get('strength', 'unknown')}]"
                    + (f" amount: {o['amount']}" if o.get("amount") else "")
                    + (f" date: {o['date_referenced']}" if o.get("date_referenced") else "")
                )

        if call_record.get("call_summary"):
            parts.append(f"Summary: {call_record['call_summary']}")

        content = "\n".join(parts)

        result = await store_memory(
            thread_id=thread_id,
            content=content,
            metadata={
                "customer_id": customer_id,
                "call_id": call_record["call_id"],
            },
        )

        logger.info(f"Backboard: customer {customer_id} interaction stored")
        return result

    except Exception as e:
        logger.warning(f"Backboard store_customer_interaction failed: {e}")
        return None


async def query_customer_history(
    customer_id: str,
    question: str,
) -> str:
    """Query across all calls for a specific customer using Backboard memory.

    Examples:
        "What payment promises has this customer made?"
        "Were there any compliance violations in previous calls?"
        "Has this customer escalated before?"
    """
    if not is_configured():
        return "Backboard not configured — set BACKBOARD_API_KEY in .env"

    try:
        assistant_id = await get_or_create_finvoice_assistant()

        # Use the customer's existing thread if it exists, otherwise create a query thread
        thread_id = await create_thread(
            assistant_id, cache_key=f"customer_{customer_id}_query"
        )

        result = await query_with_memory(
            thread_id=thread_id,
            question=f"Regarding customer {customer_id}: {question}",
        )

        return result.get("content", "No response from Backboard")

    except Exception as e:
        logger.warning(f"Backboard customer query failed: {e}")
        return f"Query failed: {e}"


async def query_audit_trail(question: str) -> str:
    """Query across ALL processed calls using Backboard's memory.

    Examples:
        "How many calls had compliance violations this week?"
        "Show me all calls where the agent used threatening language"
        "What are the most common fraud signals?"
    """
    if not is_configured():
        return "Backboard not configured — set BACKBOARD_API_KEY in .env"

    try:
        assistant_id = await get_or_create_finvoice_assistant()
        thread_id = await create_thread(assistant_id, cache_key="audit_query")

        result = await query_with_memory(
            thread_id=thread_id,
            question=question,
        )

        return result.get("content", "No response from Backboard")

    except Exception as e:
        logger.warning(f"Backboard audit query failed: {e}")
        return f"Query failed: {e}"


async def compliance_reasoning(
    transcript_excerpt: str,
    context: str,
    regulation: str = "RBI Fair Practice Code",
) -> dict:
    """Tier 2 compliance: Use cloud LLM (GPT-4o) for nuanced compliance judgments.

    For cases that keyword matching can't handle:
    - "We may have to take further steps" — veiled threat or legitimate warning?
    - "Your account will be affected" — misleading or factual?
    - Implied coercion through tone/repetition
    """
    if not is_configured():
        return {"judgment": "skipped", "reason": "Backboard not configured"}

    try:
        assistant_id = await get_or_create_finvoice_assistant()
        thread_id = await create_thread(assistant_id, cache_key="compliance_reasoning")

        question = (
            f"COMPLIANCE ANALYSIS REQUEST\n\n"
            f"Regulation: {regulation}\n"
            f"Context: {context}\n\n"
            f"Transcript excerpt:\n\"{transcript_excerpt}\"\n\n"
            f"Question: Does this excerpt violate the {regulation}? "
            f"Provide your judgment (violation/no_violation/borderline), "
            f"the specific section of the regulation that applies, "
            f"and your reasoning."
        )

        result = await query_with_memory(
            thread_id=thread_id,
            question=question,
        )

        return {
            "judgment": result.get("content", ""),
            "regulation": regulation,
            "excerpt": transcript_excerpt,
            "memories_used": len(result.get("retrieved_memories", [])),
        }

    except Exception as e:
        logger.warning(f"Backboard compliance reasoning failed: {e}")
        return {"judgment": "error", "reason": str(e)}


# ── Sync Wrappers (for non-async callers like the orchestrator) ──

def store_call_record_sync(call_record: dict) -> dict | None:
    """Synchronous wrapper for store_call_record."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in an async context — schedule as task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run, store_call_record(call_record)
                ).result(timeout=30)
        else:
            return asyncio.run(store_call_record(call_record))
    except Exception as e:
        logger.warning(f"Backboard sync store failed: {e}")
        return None


def store_customer_interaction_sync(customer_id: str, call_record: dict) -> dict | None:
    """Synchronous wrapper for store_customer_interaction."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(
                    asyncio.run, store_customer_interaction(customer_id, call_record)
                ).result(timeout=30)
        else:
            return asyncio.run(store_customer_interaction(customer_id, call_record))
    except Exception as e:
        logger.warning(f"Backboard sync customer store failed: {e}")
        return None
