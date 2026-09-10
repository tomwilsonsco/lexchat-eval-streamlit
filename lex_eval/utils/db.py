"""
DuckDB storage layer for LexChat evaluation responses and metric results.

A single file-based database (responses.db) stores both tables so the
Streamlit dashboard only needs one file.  Top-level fields are stored as
proper columns; complex nested structures are stored as JSON columns.

Schema
------
responses
    id                SEQUENCE primary key
    question_id       INTEGER
    question          TEXT
    llm_name          TEXT
    timestamp         TEXT
    actual_output     TEXT        (empty string if not captured)
    retrieval_context JSON       (list of context strings; includes legislation section text and case law references)
    tools_called      JSON        (list of tool-call dicts)
    is_error          BOOLEAN     (True when the capture failed)
    error_message     TEXT        (error description, NULL on success)
    research_mode     TEXT        (legislation_only | case_law_only | legislation_and_case_law)
    case_law_context  JSON       (list of {title, ncn, court, date, url} dicts from search_case_law)
    tool_sequence     JSON       (ordered list of worker tool names called, e.g. [search_legislation, search_legislation_sections])
    fallback_used     BOOLEAN     (True when get_legislation_text was invoked)
    summarisation_llm TEXT        (model used for summarisation; equals llm_name when no separate model is configured)
    chat_mode         TEXT        (research | conversational | deep_research)
    max_turns_halted  INTEGER     (research steps the server cut short at its ReAct turn cap;
                                   >0 means at least one step returned no report)
    react_turns_max   INTEGER     (highest ReAct turn count any step reached in this run)
    research_plan     JSON        (deep_research only: the plan from POST /api/research/plan, NULL otherwise)
    needs_clarification    BOOLEAN (True when POST /api/research/plan asked a clarifying question instead
                                    of proposing a plan; a valid outcome, distinct from is_error)
    clarification_question TEXT   (the clarifying question asked, NULL unless needs_clarification)
    attempts          INTEGER     (capture attempts this row took; 1 means the first try was kept,
                                   NULL for rows gathered before the column existed)

eval_<metric>
    One table per metric (e.g. eval_tool_usage, eval_response_groundedness),
    table name derived from the metric's test_name. Uniform schema across all
    of them; judge_llm/judge_tokens are NULL for non-judge metrics.

    id           SEQUENCE primary key
    response_id  INTEGER (FK to responses.id)
    llm_name     TEXT
    question_id  INTEGER
    question     TEXT
    score        DOUBLE
    threshold    DOUBLE
    passed       BOOLEAN
    reason       TEXT
    error        TEXT
    tools_used   JSON    (list of tool name strings, or null)
    run_at       TEXT    (ISO timestamp this metric was evaluated)
    judge_llm    TEXT    (model that actually answered; AI-judge metrics only)
    judge_tokens INTEGER (total tokens for the judge call(s); AI-judge metrics only)
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_DB = DATA_DIR / "responses.db"

_CREATE_TABLE = """
CREATE SEQUENCE IF NOT EXISTS responses_id_seq START 1;

CREATE TABLE IF NOT EXISTS responses (
    id                INTEGER DEFAULT nextval('responses_id_seq') PRIMARY KEY,
    question_id       INTEGER  NOT NULL,
    question          TEXT     NOT NULL,
    llm_name          TEXT     NOT NULL,
    timestamp         TEXT     NOT NULL,
    actual_output     TEXT     NOT NULL DEFAULT '',
    retrieval_context JSON,
    tools_called      JSON,
    research_output   TEXT     NOT NULL DEFAULT '',
    is_error          BOOLEAN  NOT NULL DEFAULT FALSE,
    error_message     TEXT,
    research_mode     TEXT     NOT NULL DEFAULT 'legislation_only',
    case_law_context  JSON,
    tool_sequence     JSON,
    fallback_used     BOOLEAN  NOT NULL DEFAULT FALSE,
    summarisation_output JSON,
    summarisation_used BOOLEAN DEFAULT FALSE,
    summarisation_llm  TEXT,
    chat_mode         TEXT     NOT NULL DEFAULT 'research',
    provider          TEXT,
    total_cost_usd    DOUBLE,
    total_ms          INTEGER,
    max_turns_halted  INTEGER,
    react_turns_max   INTEGER,
    reformatted       BOOLEAN  NOT NULL DEFAULT FALSE,
    local_cache_hits  INTEGER  NOT NULL DEFAULT 0,
    memo_hits         INTEGER  NOT NULL DEFAULT 0,
    audit_schema_version INTEGER,
    audit_json        JSON,
    research_plan     JSON,
    needs_clarification BOOLEAN NOT NULL DEFAULT FALSE,
    clarification_question TEXT,
    attempts          INTEGER
);
"""

# Columns added after the initial schema; applied to existing databases via init_db.
# NOTE: Do NOT use IF NOT EXISTS here. DuckDB's ADD COLUMN IF NOT EXISTS silently
# resets all existing row values to the column DEFAULT instead of raising an error.
# Without IF NOT EXISTS, DuckDB raises CatalogException when the column already
# exists, which the exception handler in init_db catches and skips. This preserves
# existing data.
#
# NOTE: DuckDB does not support ADD COLUMN with NOT NULL constraints, it raises
# ParserException ("Adding columns with constraints not yet supported"). DEFAULT
# alone is fine, but we omit it here for consistency. Columns are added without
# constraints; the application code provides defaults via dict.get() with fallback
# values, and load_records() applies fallbacks when reading.
_MIGRATE_RESPONSES = [
    "ALTER TABLE responses ADD COLUMN research_mode TEXT",
    "ALTER TABLE responses ADD COLUMN case_law_context JSON",
    "ALTER TABLE responses ADD COLUMN tool_sequence JSON",
    "ALTER TABLE responses ADD COLUMN fallback_used BOOLEAN",
    "ALTER TABLE responses ADD COLUMN summarisation_output JSON",
    "ALTER TABLE responses ADD COLUMN summarisation_used BOOLEAN",
    "ALTER TABLE responses ADD COLUMN summarisation_llm TEXT",
    # --- audit event migration (LexChat commit da3070d) ---
    "ALTER TABLE responses ADD COLUMN chat_mode TEXT",
    "ALTER TABLE responses ADD COLUMN provider TEXT",
    "ALTER TABLE responses ADD COLUMN total_cost_usd DOUBLE",
    "ALTER TABLE responses ADD COLUMN total_ms INTEGER",
    "ALTER TABLE responses ADD COLUMN reformatted BOOLEAN",
    "ALTER TABLE responses ADD COLUMN local_cache_hits INTEGER",
    "ALTER TABLE responses ADD COLUMN memo_hits INTEGER",
    "ALTER TABLE responses ADD COLUMN audit_schema_version INTEGER",
    "ALTER TABLE responses ADD COLUMN audit_json JSON",
    # --- deep_research plan capture (POST /api/research/plan) ---
    "ALTER TABLE responses ADD COLUMN research_plan JSON",
    # --- research steps cut short at the server's ReAct turn cap ---
    "ALTER TABLE responses ADD COLUMN max_turns_halted INTEGER",
    "ALTER TABLE responses ADD COLUMN react_turns_max INTEGER",
    # --- deep_research clarification path (distinct outcome, not an error) ---
    "ALTER TABLE responses ADD COLUMN needs_clarification BOOLEAN",
    "ALTER TABLE responses ADD COLUMN clarification_question TEXT",
    # --- capture attempts, so a retried question is not read as a clean pass ---
    "ALTER TABLE responses ADD COLUMN attempts INTEGER",
]

_INSERT_RESPONSE = """
INSERT INTO responses (
    question_id, question, llm_name, timestamp, actual_output,
    retrieval_context, tools_called, research_output, is_error, error_message,
    research_mode, case_law_context, tool_sequence, fallback_used,
    summarisation_output, summarisation_used, summarisation_llm,
    chat_mode, provider, total_cost_usd, total_ms, max_turns_halted,
    react_turns_max, reformatted,
    local_cache_hits, memo_hits, audit_schema_version, audit_json, research_plan,
    needs_clarification, clarification_question, attempts
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Same columns as _INSERT_RESPONSE plus an explicit id, for copying rows
# (e.g. make_deploy_db) where the source id must be preserved rather than
# reassigned from the destination's own sequence.
_INSERT_RESPONSE_WITH_ID = """
INSERT INTO responses (
    id, question_id, question, llm_name, timestamp, actual_output,
    retrieval_context, tools_called, research_output, is_error, error_message,
    research_mode, case_law_context, tool_sequence, fallback_used,
    summarisation_output, summarisation_used, summarisation_llm,
    chat_mode, provider, total_cost_usd, total_ms, max_turns_halted,
    react_turns_max, reformatted,
    local_cache_hits, memo_hits, audit_schema_version, audit_json, research_plan,
    needs_clarification, clarification_question, attempts
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def get_connection(
    path: Path = DEFAULT_DB, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection, creating the file if it doesn't exist.

    Pass ``read_only=True`` for read-only access. DuckDB's single-file format
    allows multiple concurrent read-only connections but only one read-write
    connection at a time, read-only mode is required for callers that may run
    alongside other processes reading the same file (e.g. pytest-xdist workers
    collecting tests in parallel).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(path), read_only=read_only)


def init_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Create the responses table and sequence if they don't already exist.

    Also applies column migrations so existing databases gain new fields.
    """
    conn.execute(_CREATE_TABLE)
    for stmt in _MIGRATE_RESPONSES:
        try:
            conn.execute(stmt)
        except duckdb.CatalogException:
            # Column already exists, roll back the aborted statement so the
            # connection remains usable, then skip.
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            col_hint = (
                stmt.split("ADD COLUMN")[-1].strip() if "ADD COLUMN" in stmt else stmt
            )
            logger.debug("Migration skipped (column may already exist): %s", col_hint)
        except Exception:
            # Unexpected migration failure (syntax error, type mismatch, etc.).
            # Roll back and surface it at WARNING so real failures aren't
            # silently hidden at the default INFO log level.
            try:
                conn.execute("ROLLBACK")
            except Exception:
                pass
            col_hint = (
                stmt.split("ADD COLUMN")[-1].strip() if "ADD COLUMN" in stmt else stmt
            )
            logger.warning("Migration failed for column: %s", col_hint, exc_info=True)


def clear_responses(conn: duckdb.DuckDBPyConnection) -> None:
    """Delete all rows from the responses table."""
    _delete_eval_rows(
        conn, [r[0] for r in conn.execute("SELECT id FROM responses").fetchall()]
    )
    conn.execute("DELETE FROM responses")


def insert_response(conn: duckdb.DuckDBPyConnection, record: Dict[str, Any]) -> int:
    """
    Insert one record into the responses table.

    *record* is the flat dict produced by ``gather_responses.process_combination``
    with top-level keys: ``actual_output``, ``retrieval_context``, ``tools_called``.
    An ``error`` key signals a failed capture.
    """
    is_error = "error" in record

    inserted = conn.execute(
        _INSERT_RESPONSE.rstrip().rstrip(";") + " RETURNING id",
        [
            record["question_id"],
            record["question"],
            record["llm_name"],
            record["timestamp"],
            record.get("actual_output", "") if not is_error else "",
            json.dumps(record.get("retrieval_context") or []),
            json.dumps(record.get("tools_called") or []),
            record.get("research_output", "") if not is_error else "",
            is_error,
            record.get("error") if is_error else None,
            record.get("research_mode", "legislation_only"),
            json.dumps(record.get("case_law_context") or []),
            json.dumps(record.get("tool_sequence") or []),
            record.get("fallback_used", False),
            # Pass None through as SQL NULL; only JSON-encode when there is real content.
            (
                json.dumps(record["summarisation_output"])
                if record.get("summarisation_output") is not None
                else None
            ),
            record.get("summarisation_used", False),
            record.get("summarisation_llm") or None,
            record.get("chat_mode", "research"),
            record.get("provider") or None,
            record.get("total_cost_usd") or None,
            record.get("total_ms") or None,
            # 0 is meaningful (no step halted), so keep it rather than
            # collapsing it to NULL the way the cost/timing fields do.
            record.get("max_turns_halted"),
            record.get("react_turns_max"),
            record.get("reformatted", False),
            record.get("local_cache_hits", 0),
            record.get("memo_hits", 0),
            record.get("audit_schema_version") or None,
            record.get("audit_json") or None,
            (
                json.dumps(record["research_plan"])
                if record.get("research_plan") is not None
                else None
            ),
            bool(record.get("needs_clarification", False)),
            record.get("clarification_question") or None,
            record.get("attempts"),
        ],
    )
    return inserted.fetchone()[0]


def load_records(
    path: Optional[Path] = None,
    include_errors: bool = False,
    read_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load responses from the database and return them as flat record dicts::

        {question_id, question, llm_name, timestamp,
         actual_output, retrieval_context, tools_called, is_error, ...}

    Error rows are excluded unless *include_errors* is True. ``is_error`` and
    ``error_message`` are returned either way, so a caller that opts in can
    tell them apart.

    Pass ``read_only=True`` when this may run concurrently with other readers
    of the same file (e.g. pytest-xdist workers collecting tests in
    parallel). Read-only connections can't run schema migrations, so callers
    that pass it are responsible for having already migrated the schema via a
    prior read-write connection (``run_evals.py`` does this once, up front).
    """
    path = path or DEFAULT_DB
    if not path.exists():
        return []

    conn = get_connection(path, read_only=read_only)
    try:
        if not read_only:
            # Ensure the schema is migrated (adds new columns to existing DBs)
            init_db(conn)
        where = "" if include_errors else "WHERE NOT is_error"
        rows = conn.execute(f"""
            SELECT id, question_id, question, llm_name, timestamp,
                   actual_output, retrieval_context, tools_called, research_output,
                   research_mode, case_law_context, tool_sequence, fallback_used,
                   summarisation_output, summarisation_used, summarisation_llm,
                   chat_mode, provider, total_cost_usd, total_ms,
                   max_turns_halted, react_turns_max, reformatted,
                   local_cache_hits, memo_hits, audit_schema_version, audit_json,
                   research_plan, needs_clarification, clarification_question,
                   attempts, is_error, error_message
            FROM responses
            {where}
            ORDER BY id
            """).fetchall()
        from lex_eval.utils.versioning import response_history

        history = response_history(conn)
    finally:
        conn.close()

    records = []
    for (
        response_id,
        qid,
        question,
        llm_name,
        timestamp,
        actual_output,
        retrieval_context_json,
        tools_called_json,
        research_output,
        research_mode,
        case_law_context_json,
        tool_sequence_json,
        fallback_used,
        summarisation_output_json,
        summarisation_used,
        summarisation_llm,
        chat_mode,
        provider,
        total_cost_usd,
        total_ms,
        max_turns_halted,
        react_turns_max,
        reformatted,
        local_cache_hits,
        memo_hits,
        audit_schema_version,
        audit_json,
        research_plan_json,
        needs_clarification,
        clarification_question,
        attempts,
        is_error,
        error_message,
    ) in rows:
        retrieval_context = (
            json.loads(retrieval_context_json) if retrieval_context_json else []
        )
        tools_called = json.loads(tools_called_json) if tools_called_json else []
        case_law_context = (
            json.loads(case_law_context_json) if case_law_context_json else []
        )
        tool_sequence = json.loads(tool_sequence_json) if tool_sequence_json else []
        summarisation_output = (
            json.loads(summarisation_output_json) if summarisation_output_json else []
        )
        research_plan = json.loads(research_plan_json) if research_plan_json else None
        records.append(
            {
                "response_id": response_id,
                # Always carried, even though error rows are excluded by
                # default: reports/attribution.py cannot report a terminal
                # failure it is never shown.
                "is_error": bool(is_error),
                "error_message": error_message or "",
                "question_id": qid,
                "question": question,
                "llm_name": llm_name,
                "timestamp": timestamp,
                "actual_output": actual_output,
                "retrieval_context": retrieval_context,
                "tools_called": tools_called,
                "research_output": research_output or "",
                "research_mode": research_mode or "legislation_only",
                "case_law_context": case_law_context,
                "tool_sequence": tool_sequence,
                "fallback_used": bool(fallback_used),
                "summarisation_output": summarisation_output,
                "summarisation_used": bool(summarisation_used),
                "summarisation_llm": summarisation_llm or "",
                "chat_mode": chat_mode or "research",
                "provider": provider,
                "total_cost_usd": total_cost_usd,
                "total_ms": total_ms,
                "max_turns_halted": max_turns_halted,
                "react_turns_max": react_turns_max,
                "reformatted": bool(reformatted),
                "local_cache_hits": local_cache_hits or 0,
                "memo_hits": memo_hits or 0,
                "audit_schema_version": audit_schema_version,
                "audit_json": audit_json,
                "research_plan": research_plan,
                "needs_clarification": bool(needs_clarification),
                "clarification_question": clarification_question,
                "attempts": attempts,
            }
        )
    for record in records:
        record.update(history.get(record["response_id"], {}))
    return records


def consistency_group_key(record: Dict[str, Any]) -> str:
    """
    Return the ``'Q{question_id}_{llm_name}_{chat_mode}'`` key used to decide
    which responses are repeat runs of each other.

    ``chat_mode`` is part of the key because consistency asks "did the same
    question, asked the same way twice, get the same answer". A deep research
    answer and an ordinary research answer are not repeat runs of each other,
    and comparing them measures the difference between the two modes rather
    than the model's repeatability.

    Both the test IDs in ``tests/eval/test_consistency.py`` and the deselect
    IDs in ``run_evals.py`` are built from this, so they cannot drift apart.
    """
    from lex_eval.utils.versioning import fingerprint

    condition = fingerprint(
        {
            "question": record.get("question"),
            "research_mode": record.get("research_mode", "legislation_only"),
            "experiment": record.get("experiment_id"),
        }
    )[:12]
    return (
        f"Q{record['question_id']}_{record['llm_name']}_"
        f"{record.get('chat_mode') or 'research'}_{condition}"
    )


def group_by_question_llm_and_mode(
    path: Optional[Path] = None,
    read_only: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return records grouped by :func:`consistency_group_key`.

    Excludes error rows.
    """
    records = load_records(path, read_only=read_only)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        grouped.setdefault(consistency_group_key(r), []).append(r)
    return grouped


def _eval_rows_for(
    conn: duckdb.DuckDBPyConnection, response_ids: List[int]
) -> Dict[str, int]:
    """How many rows in each eval_<metric> table point at *response_ids*.

    Only tables that exist and actually have rows are in the result, keyed by
    table name.
    """
    from lex_eval.run_evals import METRIC_FILES

    if not response_ids:
        return {}

    placeholders = ",".join("?" for _ in response_ids)
    counts: Dict[str, int] = {}
    for metric in METRIC_FILES:
        if not _eval_table_exists(conn, metric):
            continue
        table = _eval_table_name(metric)
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE response_id IN ({placeholders})",
            response_ids,
        ).fetchone()[0]
        if count:
            counts[table] = count
    return counts


def _delete_eval_rows(
    conn: duckdb.DuckDBPyConnection, response_ids: List[int]
) -> Dict[str, int]:
    """Delete every eval_<metric> row pointing at *response_ids*.

    Deleting a response without this leaves eval rows behind whose response_id
    no longer resolves, so every caller that removes a response must call this
    first. Returns the same {table: rows deleted} shape as _eval_rows_for.
    """
    if not response_ids:
        return {}
    counts = _eval_rows_for(conn, response_ids)
    placeholders = ",".join("?" for _ in response_ids)
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    if "response_runs" in tables:
        conn.execute(
            f"DELETE FROM response_runs WHERE response_id IN ({placeholders})",
            response_ids,
        )
    for table in counts:
        if "eval_versions" in tables:
            conn.execute(
                f"DELETE FROM eval_versions WHERE metric=? AND eval_id IN (SELECT id FROM {table} WHERE response_id IN ({placeholders}))",
                [table.removeprefix("eval_"), *response_ids],
            )
        conn.execute(
            f"DELETE FROM {table} WHERE response_id IN ({placeholders})", response_ids
        )
    return counts


def clean_incomplete_responses(
    path: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    """
    Delete rows where:
    - actual_output is empty or whitespace-only,
    - the row is an error (is_error = TRUE), OR
    - no context was captured (retrieval_context is '[]' or NULL).

    A ``needs_clarification`` row (a Deep Research plan that asked the user a
    clarifying question instead of running) is a valid outcome, not an
    incomplete or error one, even though it also has empty actual_output and
    retrieval_context, so it's excluded from all three conditions above.

    Args:
        path:    Path to the database file. Defaults to DEFAULT_DB.
        dry_run: If True, print what would be deleted without deleting.

    Returns:
        Number of rows deleted (or that would be deleted in dry_run mode).
    """
    path = path or DEFAULT_DB
    conn = get_connection(path)
    try:
        where = (
            "NOT COALESCE(needs_clarification, FALSE) AND (TRIM(actual_output) = '' "
            "OR is_error OR retrieval_context = '[]' OR retrieval_context IS NULL)"
        )
        rows = conn.execute(f"""
            SELECT id, question_id, llm_name, is_error, retrieval_context
            FROM responses
            WHERE {where}
            ORDER BY question_id, llm_name
            """).fetchall()
        count = len(rows)
        ids = [row[0] for row in rows]

        if dry_run:
            print(f"Dry run, {count} row(s) would be deleted:")
            for row in rows:
                rid, qid, llm, is_err, ctx = row
                if is_err:
                    tag = "error"
                elif ctx in ("[]", None):
                    tag = "no context"
                else:
                    tag = "empty output"
                print(f"  id={rid}  Q{qid}  {llm}  [{tag}]")
            for table, n in _eval_rows_for(conn, ids).items():
                print(f"  {table}: {n} row(s) would be deleted with them")
        else:
            eval_deleted = _delete_eval_rows(conn, ids)
            conn.execute(f"DELETE FROM responses WHERE {where}")
            conn.commit()
            print(f"Deleted {count} incomplete/error/no-context row(s).")
            for table, n in eval_deleted.items():
                print(f"  {table}: {n} row(s)")
    finally:
        conn.close()

    return count


def list_responses(path: Optional[Path] = None) -> None:
    """Print id, llm_name, and timestamp for every row in the responses table."""
    path = path or DEFAULT_DB
    if not path.exists():
        print("Database not found:", path)
        return

    conn = get_connection(path, read_only=True)
    try:
        rows = conn.execute(
            "SELECT id, llm_name, timestamp FROM responses ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    print(f"{'id':>5}  {'llm_name':<35}  timestamp")
    print("-" * 70)
    for rid, llm_name, timestamp in rows:
        print(f"{rid:>5}  {llm_name:<35}  {timestamp}")
    print(f"\n{len(rows)} response(s)")


def delete_response(response_id: int, path: Optional[Path] = None) -> Dict[str, int]:
    """
    Delete *response_id* from the responses table and from every eval_<metric>
    table (only those that actually exist) that has rows for it.

    Returns a dict of {"responses": <0 or 1>, "eval_<metric>": <rows deleted>, ...}
    covering only the tables a row was actually deleted from.
    """
    path = path or DEFAULT_DB
    conn = get_connection(path)
    try:
        deleted = _delete_eval_rows(conn, [response_id])

        count = conn.execute(
            "SELECT COUNT(*) FROM responses WHERE id = ?", [response_id]
        ).fetchone()[0]
        if count:
            conn.execute("DELETE FROM responses WHERE id = ?", [response_id])
            deleted["responses"] = count

        conn.commit()
    finally:
        conn.close()

    if "responses" not in deleted:
        print(f"No response with id={response_id} found.")
    else:
        print(f"Deleted response id={response_id}:")
        for table, count in deleted.items():
            print(f"  {table}: {count} row(s)")

    return deleted


def completeness_report(path: Optional[Path] = None) -> None:
    """Print complete (non-empty) responses per question, LLM, and chat mode.

    Grouped the same way :func:`consistency_group_key` groups, so the "ok"
    column answers the question the consistency metric actually asks. A deep
    research answer and an ordinary research answer to the same question are
    not repeat runs of each other, so counting them together would report a
    pair as ready when consistency still cannot score it.
    """
    path = path or DEFAULT_DB
    if not path.exists():
        print("Database not found:", path)
        return

    # This function only reads, so it takes no write lock of its own. DuckDB
    # still refuses to open the file at all while a gather holds its lock, so
    # this does not make the report runnable mid-gather.
    conn = get_connection(path, read_only=True)
    try:
        rows = conn.execute("""
            SELECT
                question_id,
                llm_name,
                COALESCE(chat_mode, 'research') AS mode,
                COUNT(*) AS total_runs, -- Total runs for this Q/LLM/mode group
                SUM(CASE WHEN TRIM(actual_output) != '' AND NOT is_error THEN 1 ELSE 0 END) AS complete_runs,
                SUM(CASE WHEN TRIM(actual_output) != '' AND NOT is_error THEN LENGTH(actual_output) ELSE 0 END) AS total_actual_output_chars,
                SUM(
                    CASE
                        WHEN TRIM(actual_output) != '' AND NOT is_error
                        THEN COALESCE(LENGTH(LIST_AGGR(JSON_EXTRACT_STRING(retrieval_context, '$[*]'), 'string_agg')), 0)
                        ELSE 0
                    END
                ) AS total_retrieval_context_chars
            FROM responses
            GROUP BY question_id, llm_name, mode
            ORDER BY mode, question_id, llm_name
            """).fetchall()
    finally:
        conn.close()

    print(
        f"{'Q':>3}  {'LLM':<28}  {'mode':<14}  {'total':>5}  {'comp':>4}  "
        f"{'out_chars':>9}  {'ctx_chars':>9}  {'ok':>4}"
    )
    print("-" * 95)
    for qid, llm, mode, total, complete, out_chars, ctx_chars in rows:
        ok = "YES" if complete >= 2 else "NO "
        print(
            f"{qid:>3}  {llm:<28}  {mode:<14}  {total:>5}  {complete:>4}  "
            f"{out_chars:>9}  {ctx_chars:>9}  {ok}"
        )

    print(f"\n{'mode':<16} {'groups':>6}  {'ready':>5}")
    for mode in sorted({r[2] for r in rows}):
        in_mode = [r for r in rows if r[2] == mode]
        ready = sum(1 for r in in_mode if r[4] >= 2)
        flag = "" if ready == len(in_mode) else "   <- consistency cannot score these"
        print(f"{mode:<16} {len(in_mode):>6}  {ready:>5}{flag}")

    total_groups = len(rows)
    total_ready = sum(1 for r in rows if r[4] >= 2)
    print(
        f"\n{total_ready}/{total_groups} (question, LLM, mode) groups have "
        f">= 2 complete responses"
    )

    _print_halt_summary(path)


def backfill_halt_columns(path: Optional[Path] = None) -> int:
    """Fill max_turns_halted/react_turns_max from audit_json for older rows.

    Runs captured before these columns existed still hold the numbers inside
    the stored audit event, so they can be recovered without gathering again.
    Only rows where the column is NULL and the audit event has the value are
    touched, so this is safe to run repeatedly.

    Returns the number of rows updated.
    """
    path = path or DEFAULT_DB
    if not path.exists():
        print("Database not found:", path)
        return 0

    conn = get_connection(path)
    try:
        init_db(conn)
        before = conn.execute(
            "SELECT COUNT(*) FROM responses WHERE max_turns_halted IS NULL"
        ).fetchone()[0]
        conn.execute("""
            UPDATE responses SET
                max_turns_halted = CAST(
                    json_extract(audit_json, '$.timings.max_turns_halted') AS INTEGER),
                react_turns_max = CAST(
                    json_extract(audit_json, '$.timings.react_turns_max') AS INTEGER)
            WHERE audit_json IS NOT NULL
              AND max_turns_halted IS NULL
              AND json_extract(audit_json, '$.timings.max_turns_halted') IS NOT NULL
            """)
        conn.commit()
        after = conn.execute(
            "SELECT COUNT(*) FROM responses WHERE max_turns_halted IS NULL"
        ).fetchone()[0]
    finally:
        conn.close()

    filled = before - after
    print(f"Backfilled {filled} row(s) from audit_json; {after} still unset.")
    return filled


def backfill_case_law_context(path: Optional[Path] = None) -> int:
    """Re-derive retrieval_context and case_law_context from stored audit_json.

    Case law tool results were read from the wrong place until Sept 2026, so
    responses gathered before then recorded no retrieved case law even when the
    Worker found real judgments. The stored audit event holds everything needed,
    so those rows can be repaired without gathering them again.

    Only rows whose re-derived values differ are written, so this is safe to run
    repeatedly. Returns the number of rows updated.
    """
    from .audit_capture import derive_retrieval

    path = path or DEFAULT_DB
    if not path.exists():
        print("Database not found:", path)
        return 0

    conn = get_connection(path)
    try:
        init_db(conn)
        rows = conn.execute(
            "SELECT id, audit_json, retrieval_context, case_law_context "
            "FROM responses WHERE audit_json IS NOT NULL"
        ).fetchall()

        updated = 0
        for row_id, audit_json, old_retrieval, old_case_law in rows:
            try:
                audit = json.loads(audit_json)
            except (ValueError, TypeError):
                logger.warning("response %s: audit_json is not valid JSON", row_id)
                continue
            retrieval, case_law, _ = derive_retrieval(audit)
            new_retrieval = json.dumps(retrieval)
            new_case_law = json.dumps(case_law)
            if new_retrieval == (old_retrieval or "") and new_case_law == (
                old_case_law or ""
            ):
                continue
            conn.execute(
                "UPDATE responses SET retrieval_context = ?, case_law_context = ? "
                "WHERE id = ?",
                [new_retrieval, new_case_law, row_id],
            )
            updated += 1
        conn.commit()
    finally:
        conn.close()

    print(f"Re-derived retrieval context on {updated} of {len(rows)} row(s).")
    return updated


def backfill_measured_column(path: Optional[Path] = None) -> int:
    """Set measured = FALSE on eval rows written before the column existed.

    A metric that cannot score a response, e.g. a deep-research-only metric
    handed a conversational one, still writes a row so the gap is visible, with
    score 0.0 because the column is NOT NULL. That 0.0 is not a verdict. Before
    `measured` existed the only marker was the wording of `reason`, which just
    one consumer knew to check, so any other query averaged those zeros in.

    Matches on the same reason prefixes and is safe to run repeatedly.

    Returns the number of rows updated.
    """
    path = path or DEFAULT_DB
    if not path.exists():
        print("Database not found:", path)
        return 0

    conn = get_connection(path)
    try:
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_name LIKE 'eval\\_%' ESCAPE '\\'"
            ).fetchall()
        ]
        clause = " OR ".join(["reason LIKE ?"] * len(_NOT_MEASURED_PREFIXES))
        params = [p + "%" for p in _NOT_MEASURED_PREFIXES]
        total = 0
        for table in tables:
            cols = [
                r[0]
                for r in conn.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = ?",
                    [table],
                ).fetchall()
            ]
            if "reason" not in cols:
                continue
            if "measured" not in cols:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "
                    "measured BOOLEAN DEFAULT TRUE"
                )
                conn.execute(
                    f"UPDATE {table} SET measured = TRUE WHERE measured IS NULL"
                )
            n = conn.execute(
                f"SELECT COUNT(*) FROM {table} WHERE measured AND ({clause})", params
            ).fetchone()[0]
            if n:
                conn.execute(
                    f"UPDATE {table} SET measured = FALSE WHERE measured AND ({clause})",
                    params,
                )
                print(f"  {table}: {n} row(s) marked not measured")
                total += n
        conn.commit()
    finally:
        conn.close()

    print(f"Marked {total} row(s) as not measured.")
    return total


def _print_halt_summary(path: Path) -> None:
    """Print which runs had a research step cut short at the ReAct turn cap.

    A halted step returns no report, so its findings are missing from the
    answer no matter how well the model performed. Worth seeing next to the
    completeness counts, since it looks like a model failure otherwise.
    """
    conn = get_connection(path, read_only=True)
    try:
        rows = conn.execute("""
            SELECT question_id, llm_name, max_turns_halted, react_turns_max
            FROM responses
            WHERE max_turns_halted > 0
            ORDER BY question_id, llm_name
            """).fetchall()
    except duckdb.Error:
        # Column absent: database predates the migration and has nothing to say.
        return
    finally:
        conn.close()

    if not rows:
        print("No research steps were halted at the turn cap.")
        return

    print(f"\n{len(rows)} run(s) had a research step halted at the turn cap:")
    for qid, llm, halted, turns_max in rows:
        print(f"  Q{qid:<3} {llm:<35} {halted} step(s) halted, max turns {turns_max}")


# ----------------------------
# EVAL — one table per metric

_METRIC_NAME_RE = re.compile(r"^[a-z_]+$")

_EVAL_COLUMNS = (
    "response_id, llm_name, question_id, question, score, threshold, "
    "passed, reason, error, tools_used, run_at, judge_llm, judge_tokens, measured, "
    "reference_sha256, reference_mode"
)

# Reasons a metric writes when it could not measure a response at all, e.g. a
# deep-research-only metric handed a conversational one. Rows like these carry
# score 0.0 because the column is NOT NULL, and that 0.0 is not a verdict: it
# must never reach a mean. `measured` is the column that says so. This tuple is
# only used to set it on rows written before the column existed, via
# backfill_measured_column; new rows are told directly by the metric.
_NOT_MEASURED_PREFIXES = (
    "Not measured:",
    "Judge error:",
    "Output too short",
    "No retrieval context captured",
    "No research output captured",
    "No reference outputs provided.",
    "No 'delegate_research' tool call found;",
    "No reference answer for this question;",
    "No reference statements for this question;",
    "No reference answer citations to compare against;",
    "No research plan for this record;",
    "Not deep_research;",
    "Not applicable in conversational mode;",
)


def reason_is_not_measured(reason: Optional[str]) -> bool:
    """Whether *reason* is one a metric writes when it could not score at all."""
    return bool(reason) and reason.startswith(_NOT_MEASURED_PREFIXES)


def _eval_table_name(metric: str) -> str:
    """Return the ``eval_<metric>`` table name for *metric*.

    *metric* is always drawn from an internal registry
    (``run_evals.py::METRIC_FILES``), never from user input, but this is
    validated anyway since it is interpolated directly into SQL identifiers.
    """
    if not _METRIC_NAME_RE.match(metric):
        raise ValueError(f"invalid metric name: {metric!r}")
    return f"eval_{metric}"


def _eval_table_exists(conn: duckdb.DuckDBPyConnection, metric: str) -> bool:
    """Whether eval_<metric> already exists on *conn*, without creating it.

    Used by make_deploy_db to read from the source database, which must
    never be modified, so it cannot call init_eval_table (CREATE ... IF NOT
    EXISTS still counts as a write) just to check.
    """
    table = _eval_table_name(metric)
    row = conn.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
        [table],
    ).fetchone()
    return bool(row and row[0])


def init_eval_table(conn: duckdb.DuckDBPyConnection, metric: str) -> None:
    """Create the eval_<metric> table and sequence if they don't already exist."""
    table = _eval_table_name(metric)
    conn.execute(f"""
        CREATE SEQUENCE IF NOT EXISTS {table}_id_seq START 1;

        CREATE TABLE IF NOT EXISTS {table} (
            id           INTEGER DEFAULT nextval('{table}_id_seq') PRIMARY KEY,
            response_id  INTEGER NOT NULL,
            llm_name     TEXT    NOT NULL,
            question_id  INTEGER NOT NULL,
            question     TEXT    NOT NULL,
            score        DOUBLE  NOT NULL,
            threshold    DOUBLE  NOT NULL,
            passed       BOOLEAN NOT NULL,
            reason       TEXT,
            error        TEXT,
            tools_used   JSON,
            run_at       TEXT    NOT NULL,
            judge_llm    TEXT,
            judge_tokens INTEGER,
            measured     BOOLEAN NOT NULL DEFAULT TRUE,
            reference_sha256 TEXT,
            reference_mode   TEXT
        );
    """)
    # Tables created before `measured` existed. DuckDB does not allow a NOT NULL
    # constraint on ADD COLUMN, so the default carries it for existing rows and
    # every insert supplies the value explicitly.
    conn.execute(
        f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS measured BOOLEAN DEFAULT TRUE"
    )
    # Which reference answer a row was scored against, NULL for the metrics
    # that use none and for rows written before the columns existed. A NULL
    # never matches a current reference, so those rows read as out of date and
    # are scored again rather than being trusted.
    conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS reference_sha256 TEXT")
    conn.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS reference_mode TEXT")


def insert_eval_result(
    conn: duckdb.DuckDBPyConnection, metric: str, record: Dict[str, Any]
) -> None:
    """Insert one eval result record into the eval_<metric> table."""
    table = _eval_table_name(metric)
    inserted = conn.execute(
        f"INSERT INTO {table} ({_EVAL_COLUMNS}) VALUES ("
        + ", ".join(["?"] * len(_EVAL_COLUMNS.split(", ")))
        + ") RETURNING id",
        [
            record["response_id"],
            record["llm_name"],
            int(record["question_id"]),
            record["question"],
            float(record["score"]),
            float(record["threshold"]),
            bool(record["passed"]),
            record.get("reason") or None,
            record.get("error") or None,
            json.dumps(record.get("tools_used")),
            datetime.now(timezone.utc).isoformat(),
            record.get("judge_llm") or None,
            record.get("judge_tokens"),
            bool(record.get("measured", True)),
            record.get("reference_sha256") or None,
            record.get("reference_mode") or None,
        ],
    )
    eval_id = inserted.fetchone()[0]
    if record.get("scoring_run_id"):
        conn.execute(
            "INSERT INTO eval_versions VALUES (?, ?, ?, ?, ?)",
            [
                metric,
                eval_id,
                record["scoring_run_id"],
                record["metric_version"],
                json.dumps(record.get("details")),
            ],
        )


def clear_eval_results(
    conn: duckdb.DuckDBPyConnection, metric: str, llm: Optional[str] = None
) -> None:
    """Delete rows from the eval_<metric> table.

    Deletes only rows for *llm* if given, otherwise every row in the table.

    *llm* matches by substring, not exact equality, the same way run_evals.py
    selects which tests to re-run via pytest's ``-k`` (a substring match
    against the test id, which embeds llm_name). Exact matching here would
    clear only "gpt-4"'s rows while ``-k gpt-4`` reruns "gpt-4o" too,
    leaving gpt-4o with duplicate eval rows after --overwrite.
    """
    table = _eval_table_name(metric)
    if llm:
        rows = conn.execute(f"SELECT DISTINCT llm_name FROM {table}").fetchall()
        matched = [name for (name,) in rows if name and llm in name]
        if matched:
            placeholders = ", ".join("?" for _ in matched)
            conn.execute(
                f"DELETE FROM {table} WHERE llm_name IN ({placeholders})", matched
            )
    else:
        conn.execute(f"DELETE FROM {table}")
    if conn.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='eval_versions'"
    ).fetchone()[0]:
        conn.execute(
            f"DELETE FROM eval_versions WHERE metric=? AND eval_id NOT IN (SELECT id FROM {table})",
            [metric],
        )


def covered_response_ids(
    conn: duckdb.DuckDBPyConnection,
    metric: str,
    reference_versions: Optional[Dict[int, tuple]] = None,
) -> set:
    """Return the set of response_ids already scored for *metric*.

    Pass *reference_versions*, question_id -> (reference_sha256,
    reference_mode), for a metric scored against the reference answers. A row
    calculated against a reference that has since changed, or against a draft
    that has since been signed off, does not count as covering its response:
    the score is out of date and has to be taken again.
    """
    init_eval_table(conn, metric)
    table = _eval_table_name(metric)
    if reference_versions is None:
        rows = conn.execute(f"SELECT DISTINCT response_id FROM {table}").fetchall()
        return {r[0] for r in rows}

    rows = conn.execute(
        f"SELECT response_id, question_id, reference_sha256, reference_mode "
        f"FROM {table}"
    ).fetchall()
    return {
        response_id
        for response_id, question_id, sha, mode in rows
        if reference_versions.get(question_id, (None, None)) == (sha, mode)
    }


def clear_outdated_eval_results(
    conn: duckdb.DuckDBPyConnection, metric: str, reference_versions: Dict[int, tuple]
) -> int:
    """Delete rows scored against a reference version that is no longer current.

    Returns how many were deleted. Replacing them rather than leaving them is
    what stops the dashboard averaging a score taken against a corrected answer
    together with one taken against the answer that replaced it.
    """
    init_eval_table(conn, metric)
    table = _eval_table_name(metric)
    rows = conn.execute(
        f"SELECT id, question_id, reference_sha256, reference_mode FROM {table}"
    ).fetchall()
    outdated = [
        row_id
        for row_id, question_id, sha, mode in rows
        if reference_versions.get(question_id, (None, None)) != (sha, mode)
    ]
    if outdated:
        placeholders = ", ".join("?" for _ in outdated)
        conn.execute(f"DELETE FROM {table} WHERE id IN ({placeholders})", outdated)
    return len(outdated)


def load_eval_results(
    path: Optional[Path] = None,
    metric: Optional[str] = None,
    read_only: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load all rows from the eval_<metric> table as a list of dicts.

    Pass ``read_only=True`` when this may run concurrently with other readers
    of the same file (see ``load_records`` for why).
    """
    if not metric:
        raise ValueError("metric is required")
    path = path or DEFAULT_DB
    if not path.exists():
        return []

    table = _eval_table_name(metric)
    conn = get_connection(path, read_only=read_only)
    try:
        if not read_only:
            init_eval_table(conn, metric)
        try:
            rows = conn.execute(
                f"SELECT {_EVAL_COLUMNS} FROM {table} ORDER BY id"
            ).fetchall()
        except duckdb.CatalogException:
            # Read-only connection against a metric that has never been run.
            rows = []
    finally:
        conn.close()

    results = []
    for (
        response_id,
        llm_name,
        question_id,
        question,
        score,
        threshold,
        passed,
        reason,
        error,
        tools_used_json,
        run_at,
        judge_llm,
        judge_tokens,
        measured,
        reference_sha256,
        reference_mode,
    ) in rows:
        results.append(
            {
                "response_id": response_id,
                "llm_name": llm_name,
                "question_id": question_id,
                "question": question,
                "score": score,
                "threshold": threshold,
                "passed": passed,
                "reason": reason or "",
                "error": error or "",
                "tools_used": (
                    json.loads(tools_used_json)
                    if tools_used_json and tools_used_json != "null"
                    else None
                ),
                "run_at": run_at,
                "judge_llm": judge_llm,
                "judge_tokens": judge_tokens,
                "measured": True if measured is None else bool(measured),
                "reference_sha256": reference_sha256,
                "reference_mode": reference_mode,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


def compact_db(path: Optional[Path] = None) -> Path:
    """
    Rewrite the database file to reclaim space left behind by deletes and
    ``--overwrite`` runs.

    DuckDB never shrinks its file on disk: blocks freed by a DELETE or a
    dropped/recreated table are kept around for internal reuse, not returned
    to the OS, so the file only ever grows. This copies every table into a
    fresh file (which contains only live data, none of the old free blocks)
    and replaces the original with it.

    Args:
        path: Path to the database to compact (default: ``data/responses.db``).

    Returns:
        The (unchanged) path of the compacted database.
    """
    path = path or DEFAULT_DB
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}")

    before = path.stat().st_size / 1024 / 1024
    tmp_path = path.with_suffix(".compact.db")
    if tmp_path.exists():
        tmp_path.unlink()

    conn = get_connection(path)
    try:
        source_db = conn.execute("SELECT current_database()").fetchone()[0]
        conn.execute(f"ATTACH '{tmp_path}' AS compacted")
        conn.execute(f"COPY FROM DATABASE {source_db} TO compacted")
        conn.execute("DETACH compacted")
    finally:
        conn.close()

    tmp_path.replace(path)
    after = path.stat().st_size / 1024 / 1024
    print(f"Compacted {path}\n  Before: {before:.1f} MB\n  After : {after:.1f} MB")
    return path


# ---------------------------------------------------------------------------
# Deploy copy
# ---------------------------------------------------------------------------

_DEPLOY_CONTEXT_CHARS = 2_000  # per context item


def _slim_audit(audit_json: Optional[str]) -> Optional[str]:
    """Drop the retrieved text from a stored audit trace.

    The dashboard reads an audit for its searches and steps panel only, which
    shows each delegation's title and report, each tool's name, arguments and
    error, and how many items a search returned. It never shows the retrieved
    text itself, and that text is most of the file: the LEX API responses
    (``api_calls``) and the pre-summarisation tool results (``raw_result``).
    Both go, along with ``final_result``, whose text the ``tools_called``
    column already carries and is what the log actually renders. Each search's
    item count is kept in ``result_count`` in place of
    the text it was counted from, and any error the text carried folded into
    the tool's own ``error`` so nothing is lost from the panel.

    A copy slimmed this way can no longer re-derive retrieval context via
    ``backfill_case_law_context``, which reads ``api_calls``. Run that against
    ``responses.db``, which keeps the full trace.
    """
    if not audit_json:
        return audit_json
    try:
        audit = json.loads(audit_json)
    except (ValueError, TypeError):
        logger.warning("audit_json is not valid JSON; copied unchanged")
        return audit_json

    from lex_eval.reports.diagnostics import _search_outcome

    for delegation in audit.get("delegations", []):
        for tool in delegation.get("tools", []):
            name = tool.get("name", "")
            if name.startswith("search_"):
                count, error = _search_outcome(tool, name)
                tool["result_count"] = count
                if error:
                    tool["error"] = error
            tool.pop("raw_result", None)
            tool.pop("api_calls", None)
            tool.pop("final_result", None)
    return json.dumps(audit)


# Every field of a reference answer that anything downstream still reads: the
# statements the dashboard shows, the review that decides whether a sign-off
# holds, and the inputs to reference_fingerprint so a stored fingerprint can
# still be recomputed. A reference's own tools_called and retrieval_context,
# which are ~93% of each record, are read by nothing.
_DEPLOY_REFERENCE_FIELDS = (
    "question_id",
    "question",
    "research_mode",
    "statements",
    "review",
    "reference_sha256",
    "final_answer",
    "sources_retrieved",
    "cases_retrieved",
)


def _slim_reference(record: Dict[str, Any]) -> Dict[str, Any]:
    """One reference answer reduced to the fields the dashboard still needs."""
    return {k: record[k] for k in _DEPLOY_REFERENCE_FIELDS if k in record}


def _slim_scoring_config(config_json: str) -> str:
    """Drop the research evidence from a scoring run's reference snapshot.

    Every scoring run embeds the whole reference manifest, so the same
    evidence is stored once per run. The dashboard reads only each snapshot's
    statements and fingerprint.
    """
    try:
        config = json.loads(config_json)
    except (ValueError, TypeError):
        return config_json
    references = config.get("references")
    if isinstance(references, dict):
        config["references"] = {
            qid: _slim_reference(record) if isinstance(record, dict) else record
            for qid, record in references.items()
        }
    return json.dumps(config)


def make_deploy_db(
    source_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Write a deploy copy of the database as Parquet, one file per table.

    ``retrieval_context`` is trimmed to ``_DEPLOY_CONTEXT_CHARS`` per item, and
    the retrieved text nothing displays is dropped from ``audit_json`` and from
    each scoring run's reference snapshot. Everything the dashboard reads is
    copied verbatim, and the source database is never modified.

    Parquet rather than DuckDB because DuckDB stores large text uncompressed:
    the same data is about nine times smaller this way, which is what makes the
    copy small enough to commit. ``reports/data.py`` reads either.

    Args:
        source_path: Path to the source DB (default: ``data/responses.db``).
        output_path: Destination directory (default: ``data/deploy/``).

    Returns:
        The path of the written deploy directory.
    """
    source_path = source_path or DEFAULT_DB
    output_path = output_path or (DATA_DIR / "deploy")

    if not source_path.exists():
        raise FileNotFoundError(f"Source database not found: {source_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    for stale in output_path.glob("*.parquet"):
        stale.unlink()

    # Built in a scratch DuckDB so the existing insert paths are reused, then
    # exported. Nothing keeps the scratch file.
    scratch = tempfile.TemporaryDirectory()
    src = get_connection(source_path, read_only=True)
    dst = get_connection(Path(scratch.name) / "build.db")
    try:
        # Recreate schema in the destination
        init_db(dst)

        # Copy responses with trimmed retrieval_context. Source ids are
        # preserved (not re-assigned from dst's own sequence): eval tables
        # are copied below with their original response_id, so a gap in src
        # ids (from a deleted response) must not shift every later row's id,
        # or every eval row after the gap would point at the wrong response.
        rows = src.execute(
            "SELECT id, question_id, question, llm_name, timestamp, actual_output, "
            "retrieval_context, tools_called, research_output, is_error, error_message, "
            "research_mode, case_law_context, tool_sequence, fallback_used, "
            "summarisation_output, summarisation_used, summarisation_llm, "
            "chat_mode, provider, total_cost_usd, total_ms, "
            "max_turns_halted, react_turns_max, reformatted, "
            "local_cache_hits, memo_hits, audit_schema_version, audit_json, research_plan, "
            "needs_clarification, clarification_question, attempts "
            "FROM responses ORDER BY id"
        ).fetchall()

        trimmed_count = 0
        audit_saved = 0
        max_id = 0
        for row in rows:
            (
                response_id,
                question_id,
                question,
                llm_name,
                timestamp,
                actual_output,
                ctx_json,
                tools_json,
                research_output,
                is_error,
                error_message,
                research_mode,
                case_law_context_json,
                tool_sequence_json,
                fallback_used,
                summarisation_output_json,
                summarisation_used,
                summarisation_llm,
                chat_mode,
                provider,
                total_cost_usd,
                total_ms,
                max_turns_halted,
                react_turns_max,
                reformatted,
                local_cache_hits,
                memo_hits,
                audit_schema_version,
                audit_json,
                research_plan_json,
                needs_clarification,
                clarification_question,
                attempts,
            ) = row

            ctx: list = json.loads(ctx_json) if ctx_json else []
            trimmed = [item[:_DEPLOY_CONTEXT_CHARS] for item in ctx]
            if trimmed != ctx:
                trimmed_count += 1

            slim_audit = _slim_audit(audit_json)
            audit_saved += len(audit_json or "") - len(slim_audit or "")

            max_id = max(max_id, response_id)
            dst.execute(
                _INSERT_RESPONSE_WITH_ID,
                [
                    response_id,
                    question_id,
                    question,
                    llm_name,
                    timestamp,
                    actual_output,
                    json.dumps(trimmed),
                    tools_json,
                    research_output,
                    is_error,
                    error_message,
                    research_mode or "legislation_only",
                    (
                        case_law_context_json
                        if case_law_context_json is not None
                        else "[]"
                    ),
                    tool_sequence_json if tool_sequence_json is not None else "[]",
                    bool(fallback_used),
                    summarisation_output_json,
                    bool(summarisation_used),
                    summarisation_llm,
                    chat_mode or "research",
                    provider,
                    total_cost_usd,
                    total_ms,
                    max_turns_halted,
                    react_turns_max,
                    bool(reformatted),
                    local_cache_hits or 0,
                    memo_hits or 0,
                    audit_schema_version,
                    slim_audit,
                    research_plan_json,
                    bool(needs_clarification),
                    clarification_question,
                    attempts,
                ],
            )

        # Keep dst's sequence past the highest id just inserted, so any
        # future direct insert into the deploy DB (outside this function)
        # won't collide with a preserved source id. DuckDB has no ALTER
        # SEQUENCE RESTART, so the sequence is fast-forwarded by consuming it.
        if max_id:
            dst.execute("SELECT nextval('responses_id_seq') FROM range(?)", [max_id])

        # Copy each per-metric eval table verbatim. Tables are only ever
        # created on dst, never src: src must never be modified (see this
        # function's docstring), and a metric that was never run on src
        # simply has nothing to copy, not an error.
        from lex_eval.run_evals import METRIC_FILES

        eval_row_count = 0
        for metric in METRIC_FILES:
            init_eval_table(dst, metric)
            if not _eval_table_exists(src, metric):
                continue
            table = _eval_table_name(metric)
            eval_rows = src.execute(
                f"SELECT id, {_EVAL_COLUMNS} FROM {table} ORDER BY id"
            ).fetchall()
            placeholders = ", ".join(["?"] * (1 + len(_EVAL_COLUMNS.split(", "))))
            for er in eval_rows:
                dst.execute(
                    f"INSERT INTO {table} (id, {_EVAL_COLUMNS}) VALUES ({placeholders})",
                    list(er),
                )
            if eval_rows:
                dst.execute(
                    f"SELECT nextval('{table}_id_seq') FROM range(?)",
                    [max(r[0] for r in eval_rows)],
                )
            eval_row_count += len(eval_rows)

        from lex_eval.utils.versioning import init_history

        init_history(dst)
        source_tables = {r[0] for r in src.execute("SHOW TABLES").fetchall()}
        for table in (
            "experiments",
            "gather_runs",
            "response_runs",
            "scoring_runs",
            "eval_versions",
        ):
            if table in source_tables:
                rows = src.execute(f"SELECT * FROM {table}").fetchall()
                if table == "scoring_runs":
                    # config carries a full reference snapshot per run, and the
                    # runs hold identical copies of it.
                    rows = [(*row[:-1], _slim_scoring_config(row[-1])) for row in rows]
                if rows:
                    placeholders = ", ".join("?" for _ in rows[0])
                    dst.executemany(
                        f"INSERT INTO {table} VALUES ({placeholders})", rows
                    )
        dst.execute("CHECKPOINT")

        # One Parquet file per table, named for it, which is what
        # reports/data.py::_connect turns back into views.
        written = 0
        for (name,) in dst.execute("SHOW TABLES").fetchall():
            target = str(output_path / f"{name}.parquet").replace("'", "''")
            dst.execute(
                f"COPY {name} TO '{target}' "
                "(FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 9)"
            )
            written += 1
    finally:
        src.close()
        dst.close()
        scratch.cleanup()

    before = source_path.stat().st_size / 1024 / 1024
    after = sum(f.stat().st_size for f in output_path.glob("*.parquet")) / 1024 / 1024
    print(
        f"Deploy data written to {output_path}\n"
        f"  Source : {before:.1f} MB\n"
        f"  Deploy : {after:.1f} MB across {written} Parquet file(s) "
        f"({trimmed_count} response row(s) trimmed, "
        f"{audit_saved / 1024 / 1024:.1f} MB of audit text dropped, "
        f"{eval_row_count} eval result row(s) copied)"
    )
    return output_path


if __name__ == "__main__":
    import argparse as _argparse

    _parser = _argparse.ArgumentParser(
        description="DuckDB responses database utilities"
    )
    _parser.add_argument(
        "--clean", action="store_true", help="Delete incomplete/error responses"
    )
    _parser.add_argument(
        "--dry-run", action="store_true", help="Preview what --clean would delete"
    )
    _parser.add_argument(
        "--deploy-db",
        metavar="OUTPUT_DIR",
        help="Write the deploy data as Parquet, one file per table (default: data/deploy/)",
        nargs="?",
        const="",  # sentinel: use default path
    )
    _parser.add_argument(
        "--list",
        action="store_true",
        help="List response id, llm_name, and timestamp for every response",
    )
    _parser.add_argument(
        "--backfill-halts",
        action="store_true",
        help="Fill max_turns_halted/react_turns_max from stored audit_json",
    )
    _parser.add_argument(
        "--backfill-measured",
        action="store_true",
        help="Mark pre-existing eval rows that carry a 'not measured' reason, so "
        "their placeholder score of 0.0 is kept out of every mean",
    )
    _parser.add_argument(
        "--backfill-case-law",
        action="store_true",
        help="Re-derive retrieval_context and case_law_context from stored "
        "audit_json, repairing responses gathered before case law tool results "
        "were read correctly",
    )
    _parser.add_argument(
        "--delete-response",
        metavar="ID",
        type=int,
        help="Delete a response by id from responses and every eval_<metric> table",
    )
    _parser.add_argument(
        "--compact",
        action="store_true",
        help="Rewrite the database file to reclaim space left by deletes/--overwrite runs",
    )
    _args = _parser.parse_args()

    if _args.compact:
        compact_db()
    elif _args.deploy_db is not None:
        _out = Path(_args.deploy_db) if _args.deploy_db else None
        make_deploy_db(output_path=_out)
    elif _args.list:
        list_responses()
    elif _args.backfill_halts:
        backfill_halt_columns()
        completeness_report()
    elif _args.backfill_measured:
        backfill_measured_column()
    elif _args.backfill_case_law:
        backfill_case_law_context()
    elif _args.delete_response is not None:
        delete_response(_args.delete_response)
    elif _args.clean or _args.dry_run:
        clean_incomplete_responses(dry_run=_args.dry_run)
        if not _args.dry_run:
            completeness_report()
    else:
        completeness_report()
