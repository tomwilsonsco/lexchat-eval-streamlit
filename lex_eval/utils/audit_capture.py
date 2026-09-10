"""
audit_capture.py

Runs a single question through the LexChat /api/system/chat SSE endpoint
and returns a structured result dict.

This is the core capture layer.  The LexChat server (commit da3070d and
later) emits a single structured ``audit`` SSE event carrying the full
request trace (delegations → tools → API calls, with raw and summarised
results).  This module streams the endpoint, stashes the ``audit`` event,
and derives the flat result dict from it.

Produces a dict with these keys:

    actual_output, retrieval_context, tools_called, research_output,
    research_mode, case_law_context, tool_sequence, fallback_used,
    summarisation_output, summarisation_used, is_error, error_message,
    chat_mode, provider, total_cost_usd, total_ms, max_turns_halted,
    react_turns_max, local_cache_hits, memo_hits, reformatted,
    audit_schema_version, audit_json
"""

from __future__ import annotations

import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VERBOSE_TRUNCATE_CHARS = 500


def _trunc(s: str, n: int = VERBOSE_TRUNCATE_CHARS) -> str:
    """Truncate a string for display in verbose logs."""
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + f"…[truncated {len(s) - n} chars]"


def _vlog(f, msg: str) -> None:
    """Write a line to the verbose log file if *f* is not None."""
    if f is not None:
        f.write(msg + "\n")


def _tool_result_json(tool: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a tool's own JSON result out of an audit tool entry.

    ``raw_result`` is what the tool returned; ``final_result`` is the same
    thing after any summarisation or appended guidance, so raw is preferred.
    Returns None if neither parses as a JSON object.
    """
    for key in ("raw_result", "final_result"):
        value = tool.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def derive_retrieval(
    audit: Dict[str, Any],
) -> tuple[List[str], List[Dict[str, Any]], bool]:
    """Derive (retrieval_context, case_law_context, fallback_used) from an audit event.

    Split out of :func:`audit_capture` so a stored audit event can be re-read
    without gathering the response again (``db.py --backfill-case-law``).
    """
    retrieval_context: List[str] = []
    case_law_context: List[Dict[str, Any]] = []
    fallback_used = False

    for d in audit.get("delegations", []):
        for t in d.get("tools", []):
            tool_name = t.get("name", "")

            # Case law tools read from the tool's own result, not from
            # api_calls. The National Archives returns Atom XML/LegalDocML,
            # which LexChat parses into JSON inside the tool; the api_call
            # response it records is only the first 300 chars of that raw XML
            # (executor.py, "preview"), so there is nothing structured there.
            if tool_name == "search_case_law":
                result = _tool_result_json(t)
                results = result.get("results", []) if result else []
                for r in results:
                    if not isinstance(r, dict):
                        continue
                    case_law_context.append(
                        {
                            "title": r.get("title", ""),
                            "ncn": r.get("ncn", ""),
                            "court": r.get("court", ""),
                            "date": r.get("date", ""),
                            "url": r.get("url", ""),
                        }
                    )
                    parts = [
                        p for p in [r.get("ncn"), r.get("court"), r.get("date")] if p
                    ]
                    title = r.get("title", "")
                    retrieval_context.append(
                        f"{title} ({' | '.join(parts)})" if parts else title
                    )
                continue

            if tool_name == "get_case_law_text":
                result = _tool_result_json(t)
                text = result.get("text", "") if result else ""
                if text:
                    ncn = result.get("ncn", "")
                    header = " ".join(p for p in [result.get("title", ""), ncn] if p)
                    retrieval_context.append(f"{header}: {text}" if header else text)
                continue

            for call in t.get("api_calls", []):
                resp = call.get("response", {})

                if tool_name == "get_legislation_text":
                    fallback_used = True
                    if isinstance(resp, dict) and resp.get("full_text"):
                        retrieval_context.append(resp["full_text"])

                elif tool_name == "search_legislation_sections":
                    if isinstance(resp, list):
                        sections = resp
                    elif isinstance(resp, dict):
                        sections = resp.get("sections") or resp.get("results") or []
                    else:
                        sections = []
                    for sec in sections:
                        if not isinstance(sec, dict):
                            continue
                        content = (
                            sec.get("content")
                            or sec.get("text")
                            or sec.get("excerpt")
                            or ""
                        )
                        sec_title = sec.get("title") or sec.get("section_title") or ""
                        if content:
                            retrieval_context.append(
                                f"{sec_title}: {content}" if sec_title else content
                            )

                elif tool_name == "search_legislation":
                    results = resp.get("results", []) if isinstance(resp, dict) else []
                    for r in results:
                        if not isinstance(r, dict):
                            continue
                        title = r.get("title", "")
                        year = str(r.get("year", "")) if r.get("year") else ""
                        status = r.get("status", "")
                        parts = [p for p in [year, status] if p]
                        retrieval_context.append(
                            f"{title} ({', '.join(parts)})" if parts else title
                        )

    # deduplicate retrieval_context preserving order
    retrieval_context = list(dict.fromkeys(retrieval_context))

    # the same judgment usually comes back from several searches, keep one entry
    _seen_cases: set = set()
    _deduped_cases: List[Dict[str, Any]] = []
    for case in case_law_context:
        key = (case.get("ncn", ""), case.get("url", ""))
        if key in _seen_cases:
            continue
        _seen_cases.add(key)
        _deduped_cases.append(case)
    case_law_context = _deduped_cases

    return retrieval_context, case_law_context, fallback_used


def audit_capture(
    client,
    question: str,
    model_name: str,
    research_mode: str = "legislation_only",
    chat_mode: str = "research",
    deep_research_plan: Optional[dict] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    verbose_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Stream the SSE endpoint for *question* and return a structured result dict.

    Parameters
    ----------
    client:
        Authenticated ``httpx.Client`` from :func:`get_authenticated_client`.
    question:
        The natural-language legal question to evaluate.
    model_name:
        The LLM model name as returned by ``/api/models``.
    research_mode:
        ``legislation_only``, ``case_law_only``, or ``legislation_and_case_law``.
    chat_mode:
        ``research``, ``conversational``, or ``deep_research``.
    deep_research_plan:
        Optional plan dict obtained from ``POST /api/research/plan``.  When
        set, it is added to the chat payload so the server runs the
        pre-computed plan instead of planning inline.
    on_event:
        Optional callback invoked for every raw SSE ``data`` dict *after*
        the event has been processed.  Used by ``--debug-events``.
    verbose_log_path:
        Optional path to write a verbose human-readable audit log for this
        capture call.  When ``None`` (default), no verbose log is written.

    Returns
    -------
    dict
        Keys: ``actual_output``, ``retrieval_context``, ``tools_called``,
        ``research_output``, ``research_mode``, ``case_law_context``,
        ``tool_sequence``, ``fallback_used``, ``summarisation_output``,
        ``summarisation_used``, ``is_error``, ``error_message``,
        ``chat_mode``, ``provider``, ``total_cost_usd``, ``total_ms``,
        ``max_turns_halted``, ``react_turns_max``, ``local_cache_hits``,
        ``memo_hits``, ``reformatted``, ``audit_schema_version``,
        ``audit_json``.
    """

    # --- Verbose log file setup ------------------------------------------------
    _vf = open(verbose_log_path, "w", encoding="utf-8") if verbose_log_path else None
    _started_ts = datetime.now(timezone.utc)

    _vlog(_vf, "=== AUDIT CAPTURE ===")
    _vlog(_vf, f'question:      "{question}"')
    _vlog(_vf, f"model:         {model_name}")
    _vlog(_vf, f"mode:          {research_mode}")
    _vlog(_vf, f"chat_mode:     {chat_mode}")
    _vlog(_vf, f"started:       {_started_ts.isoformat()}")
    _vlog(_vf, f"log_path:      {verbose_log_path}")
    _vlog(_vf, "=====================")
    _vlog(_vf, "")

    if deep_research_plan is not None:
        _vlog(_vf, "=== DEEP RESEARCH PLAN (from POST /api/research/plan) ===")
        _vlog(_vf, json.dumps(deep_research_plan, indent=2, default=str))
        _vlog(_vf, "==========================================================")
        _vlog(_vf, "")

    # ------------------------------------------------------------------
    # Request payload
    # ------------------------------------------------------------------
    chat_payload: Dict[str, Any] = {
        "messages": [{"role": "user", "content": question}],
        "model": model_name,
        "research_mode": research_mode,
        "chat_mode": chat_mode,
    }
    if deep_research_plan is not None:
        chat_payload["deep_research_plan"] = deep_research_plan

    # ------------------------------------------------------------------
    # Per-call mutable state
    # ------------------------------------------------------------------
    actual_output: str = ""
    _audit_event: Optional[Dict[str, Any]] = None
    is_error: bool = False
    error_message: str = ""

    try:
        # A per-request timeout overrides the client's, so this is the value
        # that actually governs the stream. It is a read timeout: how long the
        # stream may go silent, not how long the response may take. Deep
        # Research can pause longer than the 300s default while a single tool
        # call runs, so keep it in step with LEXCHAT_TIMEOUT.
        stream_timeout = float(os.getenv("LEXCHAT_TIMEOUT", "300"))
        with client.stream(
            "POST", "/api/system/chat", json=chat_payload, timeout=stream_timeout
        ) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                decoded = line
                try:
                    if isinstance(decoded, bytes):
                        decoded = decoded.decode("utf-8")
                except Exception:
                    continue

                if not decoded.startswith("data: "):
                    continue

                data_str = decoded[6:]
                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse JSON: %s", data_str[:200])
                    continue

                event_type = data.get("type", "")

                # ----------------------------------------------------------
                # audit: structured request trace (canonical source)
                # ----------------------------------------------------------
                if event_type == "audit":
                    _audit_event = data

                # ----------------------------------------------------------
                # token: streaming token from final LLM response
                # ----------------------------------------------------------
                elif event_type == "token":
                    actual_output += data.get("content", "")

                # ----------------------------------------------------------
                # result: final complete message
                # ----------------------------------------------------------
                elif event_type == "result":
                    message = data.get("message", {})
                    if isinstance(message, dict):
                        content = message.get("content", "")
                        if content:
                            actual_output = content
                    elif isinstance(message, str) and message:
                        actual_output = message

                # ----------------------------------------------------------
                # error
                # ----------------------------------------------------------
                elif event_type == "error":
                    err_msg = data.get("error", "Unknown error")
                    logger.warning("Stream error: %s", err_msg)

                # queue and timing events carry no data the eval needs;
                # they fall through and are silently ignored.

                # ----------------------------------------------------------
                # Post-event callback (--debug-events)
                # ----------------------------------------------------------
                if on_event:
                    on_event(data)

    except Exception as exc:
        _vlog(_vf, f"ERROR: audit_capture raised exception: {exc}")
        logger.error("audit_capture failed: %s", exc)
        is_error = True
        error_message = str(exc)

    # ------------------------------------------------------------------
    # Guards: the audit event is mandatory
    # ------------------------------------------------------------------
    if _audit_event is None and not is_error:
        _vlog(_vf, "ERROR: No 'audit' event received from the stream.")
        if _vf:
            _vf.close()
        raise RuntimeError(
            "No 'audit' event received from /api/system/chat. "
            "The LexChat server predates commit da3070d and does not emit "
            "structured audit traces. Update LexChat to continue."
        )

    if _audit_event is not None:
        if _audit_event.get("schema_version", 0) != 1:
            _vlog(
                _vf,
                f"ERROR: Unsupported audit schema_version "
                f"{_audit_event.get('schema_version')!r}, expected 1.",
            )
            if _vf:
                _vf.close()
            raise RuntimeError(
                f"Unsupported audit schema_version "
                f"{_audit_event.get('schema_version')!r}, expected 1. "
                "Update lex_eval to match the new LexChat schema."
            )

        # The server emits the audit event on BOTH the success path and the
        # exception path (system.py), setting audit["error"] on failures. A
        # failed run is still an eval data point, record it as an error row.
        if _audit_event.get("error"):
            is_error = True
            error_message = _audit_event["error"]

        # Delegation-level errors: if ALL delegations failed (each has an error
        # and no report), the research completely failed even though the
        # top-level audit["error"] is null.  This catches the case where the
        # worker tool execution failed internally before making any API calls
        # (e.g. the sniffer callback bug in LexChat ≤9e76339), producing an
        # apology message that would otherwise be stored as valid eval data.
        _delegations = _audit_event.get("delegations") or []
        if (
            not is_error
            and _delegations
            and all(d.get("error") for d in _delegations)
            and not any(d.get("report") for d in _delegations)
        ):
            is_error = True
            _first_err = next(d["error"] for d in _delegations if d.get("error"))
            error_message = f"All delegations failed, first error: {_first_err}"

    # ------------------------------------------------------------------
    # Derive the return dict from the audit event
    # ------------------------------------------------------------------
    audit = _audit_event or {}

    # actual_output, prefer the structured field, fall back to token stream
    actual_output = audit.get("answer") or actual_output
    if not isinstance(actual_output, str):
        actual_output = str(actual_output) if actual_output else ""

    # research_output, join all delegation reports
    research_output = "\n\n".join(
        d.get("report", "") for d in audit.get("delegations", []) if d.get("report")
    )

    # tools_called and tool_sequence, built together to preserve ordering
    tools_called: List[Dict[str, Any]] = []
    tool_sequence: List[str] = []

    for d in audit.get("delegations", []):
        # Manager-level delegation entry
        tools_called.append(
            {
                "name": "delegate_research",
                "input_parameters": {"brief": d.get("brief", "")},
                "output": d.get("report", ""),
            }
        )
        tool_sequence.append("delegate_research")

        # Worker-level tool entries
        for t in d.get("tools", []):
            tool_name = t.get("name") or ""
            if not tool_name:
                logger.warning(
                    "audit event: tool entry missing 'name' field, "
                    "using 'unknown' placeholder"
                )
                tool_name = "unknown"
            tools_called.append(
                {
                    "name": f"Worker: {tool_name}",
                    "input_parameters": t.get("args", {}),
                    "output": t.get("final_result", ""),
                }
            )
            tool_sequence.append(f"Worker: {tool_name}")

    retrieval_context, case_law_context, fallback_used = derive_retrieval(audit)

    # summarisation_used and summarisation_output
    summarisation_used = any(
        t.get("summarised", False)
        for d in audit.get("delegations", [])
        for t in d.get("tools", [])
    )
    summarisation_output = [
        t.get("final_result", "")
        for d in audit.get("delegations", [])
        for t in d.get("tools", [])
        if t.get("summarised") and t.get("final_result")
    ] or None  # None = SQL NULL when no summarisation occurred

    # research_mode, prefer the server's resolved mode
    research_mode_out = audit.get("research_mode") or research_mode

    # New fields
    chat_mode_out = audit.get("chat_mode", chat_mode)
    provider = audit.get("provider")
    timings = audit.get("timings") or {}
    total_cost_usd = timings.get("total_cost_usd")
    total_ms = timings.get("total_ms")
    # How many research steps the server cut short at its ReAct turn cap, and
    # the highest turn count any step reached. A halted step returns no report,
    # so its findings are missing from the answer through no fault of the
    # model. Without these the only trace is the words "Research halted" inside
    # research_output, which the models phrase inconsistently.
    max_turns_halted = timings.get("max_turns_halted")
    react_turns_max = timings.get("react_turns_max")
    local_cache_hits = sum(
        1
        for d in audit.get("delegations", [])
        for t in d.get("tools", [])
        if t.get("local_cache_hit")
    )
    memo_hits = sum(
        1
        for d in audit.get("delegations", [])
        for t in d.get("tools", [])
        if t.get("memo_hit")
    )
    reformatted = any(d.get("reformatted", False) for d in audit.get("delegations", []))
    audit_schema_version = audit.get("schema_version")

    # ------------------------------------------------------------------
    # Verbose log: audit event + final state
    # ------------------------------------------------------------------
    if _audit_event is not None:
        _vlog(_vf, "=== AUDIT EVENT ===")
        _vlog(_vf, json.dumps(_audit_event, indent=2, default=str))
        _vlog(_vf, "==================")
        _vlog(_vf, "")

    _ended_ts = datetime.now(timezone.utc)
    _vlog(_vf, "=== FINAL STATE ===")
    _vlog(_vf, f"actual_output:          {len(actual_output)} chars")
    _vlog(
        _vf,
        (
            f"research_output:        {len(research_output)} chars"
            if research_output
            else "research_output:        (empty)"
        ),
    )
    _vlog(_vf, f"retrieval_context:      {len(retrieval_context)} items")
    _vlog(_vf, f"tools_called:           {[t['name'] for t in tools_called]}")
    _vlog(_vf, f"tool_sequence:          {list(tool_sequence)}")
    _vlog(
        _vf,
        f"summarisation_output:   {len(summarisation_output) if summarisation_output else 0} item(s)"
        + (
            f"  (total {sum(len(s) for s in summarisation_output)} chars)"
            if summarisation_output
            else ""
        ),
    )
    _vlog(_vf, f"summarisation_used:     {summarisation_used}")
    _vlog(_vf, f"fallback_used:          {fallback_used}")
    _vlog(_vf, f"case_law_context:       {len(case_law_context)} items")
    _vlog(_vf, f"chat_mode:              {chat_mode_out}")
    _vlog(_vf, f"provider:               {provider}")
    _vlog(_vf, f"total_cost_usd:         {total_cost_usd}")
    _vlog(_vf, f"total_ms:               {total_ms}")
    _vlog(_vf, f"max_turns_halted:       {max_turns_halted}")
    _vlog(_vf, f"react_turns_max:        {react_turns_max}")
    _vlog(_vf, f"local_cache_hits:       {local_cache_hits}")
    _vlog(_vf, f"memo_hits:              {memo_hits}")
    _vlog(_vf, f"reformatted:            {reformatted}")
    _vlog(_vf, f"audit_schema_version:   {audit_schema_version}")
    _vlog(_vf, f"is_error:               {is_error}")
    if error_message:
        _vlog(_vf, f"error_message:          {error_message}")
    _vlog(_vf, f"ended:                  {_ended_ts.isoformat()}")
    _vlog(
        _vf, f"duration_s:             {(_ended_ts - _started_ts).total_seconds():.1f}"
    )
    _vlog(_vf, "==================")

    if _vf:
        _vf.close()

    return {
        # existing keys, unchanged types and values
        "actual_output": actual_output,
        "retrieval_context": retrieval_context,
        "tools_called": tools_called,
        "research_output": research_output,
        "research_mode": research_mode_out,
        "case_law_context": case_law_context,
        "tool_sequence": tool_sequence,
        "fallback_used": fallback_used,
        "summarisation_output": summarisation_output,
        "summarisation_used": summarisation_used,
        "is_error": is_error,
        "error_message": error_message,
        # new keys
        "chat_mode": chat_mode_out,
        "provider": provider,
        "total_cost_usd": total_cost_usd,
        "total_ms": total_ms,
        "max_turns_halted": max_turns_halted,
        "react_turns_max": react_turns_max,
        "local_cache_hits": local_cache_hits,
        "memo_hits": memo_hits,
        "reformatted": reformatted,
        "audit_schema_version": audit_schema_version,
        "audit_json": json.dumps(audit, default=str) if audit else None,
    }
