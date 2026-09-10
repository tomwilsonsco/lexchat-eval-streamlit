"""
Helpers for loading captured responses (from DuckDB) and converting them into
LLMTestCase objects for evaluation.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..testcase import LLMTestCase, ToolCall

from .db import load_records as _db_load_records
from .db import consistency_group_key

DATA_DIR = Path(__file__).parent.parent / "data"


def load_records(
    filepath: Optional[Path] = None,
    read_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load all records from the DuckDB responses database.

    Each record is a flat dict:
        {question_id, question, llm_name, timestamp,
         actual_output, retrieval_context, tools_called}

    Error rows are excluded.

    Pass ``read_only=True`` at pytest collection time, since multiple
    pytest-xdist workers may call this concurrently (see
    ``db.load_records``).
    """
    records = _db_load_records(path=filepath, read_only=read_only)
    selected = os.getenv("LEX_EVAL_RESPONSE_IDS")
    if selected is not None:
        ids = set(json.loads(selected))
        records = [r for r in records if r["response_id"] in ids]
    return records


def record_to_test_case(record: Dict[str, Any]) -> LLMTestCase:
    """
    Convert a flat record dict into an LLMTestCase.

    Handles the serialisation format produced by
    ``gather_responses.serialize_test_case`` (Pydantic model_dump).
    """
    tools_called = []
    for tool_dict in record.get("tools_called", []):
        tools_called.append(
            ToolCall(
                name=tool_dict.get("name", ""),
                input_parameters=tool_dict.get("input_parameters")
                or tool_dict.get("inputParameters", {}),
                output=tool_dict.get("output", ""),
            )
        )

    return LLMTestCase(
        input=record.get("question", ""),
        actual_output=record.get("actual_output", ""),
        retrieval_context=record.get("retrieval_context", []),
        tools_called=tools_called,
    )


_WORKER_TOOL_PREFIX = "Worker: "


def agent_visible_context(record: Dict[str, Any]) -> List[str]:
    """
    Return the legal text the research agent actually had to work from.

    LexChat summarises a tool result before handing it back to the research
    agent when the result is large (see LexChat's ``run_worker_tool``). When
    that happens the agent never sees the full Act, only the summary, so
    ``retrieval_context`` (built from the raw API responses) is not what the
    agent worked from and must not be used to judge its report.

    Returns the agent's own tool results when the run summarised anything,
    and ``retrieval_context`` unchanged when it did not.
    """
    if not record.get("summarisation_used"):
        return record.get("retrieval_context") or []

    seen: List[str] = []
    for tool in record.get("tools_called") or []:
        if not str(tool.get("name", "")).startswith(_WORKER_TOOL_PREFIX):
            continue
        output = str(tool.get("output") or "").strip()
        if output:
            seen.append(output)

    # Fall back rather than hand the judge nothing, so a capture gap reads as
    # a capture gap rather than as an agent that invented everything.
    return list(dict.fromkeys(seen)) or (record.get("retrieval_context") or [])


def group_by_question(
    records: Optional[List[Dict[str, Any]]] = None,
    filepath: Optional[Path] = None,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Group records by ``question_id``.

    Useful for consistency testing across LLMs or across repeated runs
    of the same question.
    """
    if records is None:
        records = load_records(filepath)

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        qid = r["question_id"]
        grouped.setdefault(qid, []).append(r)
    return grouped


def group_by_question_llm_and_mode(
    records: Optional[List[Dict[str, Any]]] = None,
    filepath: Optional[Path] = None,
    read_only: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group records by (question_id, llm_name, chat_mode), see
    :func:`consistency_group_key`.

    Useful for testing repeatability of the same LLM answering the same
    question the same way, when ``gather_responses.py`` is run with
    ``--experiment-id`` to join the same condition.

    Pass ``read_only=True`` at pytest collection time (see ``load_records``).
    """
    if records is None:
        records = load_records(filepath=filepath, read_only=read_only)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        grouped.setdefault(consistency_group_key(r), []).append(r)
    return grouped


def record_id(record: Dict[str, Any]) -> str:
    """Return a short pytest-friendly identifier for a record."""
    return (
        f"Q{record['question_id']}_{record['llm_name']}_response{record['response_id']}"
    )
