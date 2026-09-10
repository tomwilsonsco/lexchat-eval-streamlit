"""Read-only dashboard data and summaries of stored verdicts."""

import json
import re
from collections import defaultdict
from pathlib import Path

import duckdb

from lex_eval.utils.db import reason_is_not_measured
from lex_eval.utils.applicability import (
    exclusion,
    reason_is_not_applicable,
    scope_note,
)
from lex_eval.utils.versioning import response_history

# A Parquet file's name is the table it holds, and it is interpolated into a
# CREATE VIEW, so only plain identifiers are accepted.
_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _connect(path: Path):
    """Open a results database: a DuckDB file, or a directory of Parquet.

    The deploy copy is Parquet, one file per table. Reading it through views
    means every query below is the same for both, so there is one dashboard
    rather than one per storage format.
    """
    if not path.is_dir():
        return duckdb.connect(str(path), read_only=True)
    conn = duckdb.connect()
    for file in sorted(path.glob("*.parquet")):
        if not _TABLE_NAME_RE.match(file.stem):
            continue
        location = str(file).replace("'", "''")
        conn.execute(
            f"CREATE VIEW {file.stem} AS SELECT * FROM read_parquet('{location}')"
        )
    return conn


def read_database(path: Path, metrics: list[tuple]) -> tuple[list[dict], list[dict]]:
    """Read old and current databases without running migrations."""
    if not path.exists():
        return [], []
    with _connect(path) as conn:
        tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
        if "responses" not in tables:
            return [], []
        responses = _rows(conn, "responses")
        for rec in responses:
            rec["response_id"] = rec.pop("id")
        history = response_history(conn)
        experiments = (
            {r["id"]: r for r in _rows(conn, "experiments")}
            if "experiments" in tables
            else {}
        )
        scoring = (
            {r["id"]: r for r in _rows(conn, "scoring_runs")}
            if "scoring_runs" in tables
            else {}
        )
        versions = {}
        if "eval_versions" in tables:
            for metric, eid, run, version, details in conn.execute(
                "SELECT * FROM eval_versions"
            ).fetchall():
                versions[(metric, eid)] = dict(
                    scoring_run_id=run,
                    metric_version=version,
                    details=json.loads(details) if details else None,
                    scoring_config=scoring.get(run, {}).get("config", {}),
                    scoring_label=scoring.get(run, {}).get("label", run),
                )
        for rec in responses:
            rec.update(history.get(rec["response_id"], {}))
            rec["experiment"] = experiments.get(rec.get("experiment_id"), {})
        by_id = {r["response_id"]: r for r in responses}
        results = []
        for key, name, _ in metrics:
            if f"eval_{key}" not in tables:
                continue
            for row in _rows(conn, f"eval_{key}"):
                row.update(versions.get((key, row["id"]), {}))
                rec = by_id.get(row["response_id"], {})
                row.update(
                    experiment_id=rec.get("experiment_id"),
                    test_name=key,
                    metric_name=name,
                    chat_mode=rec.get("chat_mode", "unknown"),
                    research_mode=rec.get("research_mode", "unknown"),
                )
                results.append(row)
    return responses, results


def _rows(conn, table: str) -> list[dict]:
    cursor = conn.execute(f'SELECT * FROM "{table}" ORDER BY id')
    columns = [c[0] for c in cursor.description]
    json_columns = {c[0] for c in cursor.description if str(c[1]) == "JSON"}
    rows = []
    for values in cursor.fetchall():
        row = dict(zip(columns, values, strict=True))
        for col in json_columns:
            if isinstance(row[col], str):
                row[col] = json.loads(row[col])
        rows.append(row)
    return rows


def measured(row: dict) -> bool:
    return (
        bool(row["measured"])
        if row.get("measured") is not None
        else not reason_is_not_measured(row.get("reason") or "")
    )


def latest_results(rows: list[dict]) -> list[dict]:
    """One scoring row per response and metric, with deterministic tie breaking."""
    selected = {}
    for row in rows:
        key = (row["response_id"], row["test_name"])
        order = (row.get("run_at") or "", row.get("id") or 0)
        previous = selected.get(key)
        if previous is None or order > (
            previous.get("run_at") or "",
            previous.get("id") or 0,
        ):
            selected[key] = row
    return list(selected.values())


def aggregate_metrics(rows: list[dict]) -> list[dict]:
    """Preserve verdicts, including contradiction vetoes and mixed repeats."""
    grouped = defaultdict(list)
    for row in latest_results(rows):
        grouped[row["metric_name"]].append(row)
    aggregates = []
    for name, values in grouped.items():
        scored = [r for r in values if measured(r)]
        unmeasured = [r for r in values if not measured(r)]
        signatures = {
            (
                r.get("metric_version"),
                r.get("reference_sha256"),
                r.get("reference_mode"),
                r.get("threshold"),
            )
            for r in scored
        }
        judges = {r["judge_llm"] for r in scored if r.get("judge_llm")}
        incompatible = len(signatures) > 1 or len(judges) > 1
        if incompatible:
            unmeasured = [
                *unmeasured,
                *[
                    {
                        **r,
                        "measured": False,
                        "not_comparable": True,
                        "reason": "Incompatible scoring versions; select one scoring run.",
                    }
                    for r in scored
                ],
            ]
            scored = []
        scores = [r["score"] for r in scored]
        passes = sum(bool(r["passed"]) for r in scored)
        state = (
            "Not measured"
            if not scored
            else (
                "Stable pass"
                if passes == len(scored)
                else "Stable fail" if not passes else "Mixed"
            )
        )
        if incompatible:
            state = "Not comparable"
        elif len(scored) == 1:
            state = "Pass (one measurement)" if passes else "Fail (one measurement)"
        aggregates.append(
            {
                **values[0],
                "metric_name": name,
                "score": sum(scores) / len(scores) if scores else 0.0,
                "min_score": min(scores, default=0.0),
                "max_score": max(scores, default=0.0),
                "passed": bool(scored) and passes == len(scored),
                "scored": bool(scored),
                "state": state,
                "pass_count": passes,
                "measured_count": len(scored),
                "n_runs": len(values),
                "raw_results": values,
                "not_scored_count": len(unmeasured),
                "not_scored_reasons": [
                    r.get("reason") or "Not measured" for r in unmeasured
                ],
                "not_scored_rows": unmeasured,
            }
        )
    return aggregates


def outcome(rec: dict) -> str:
    if rec.get("is_error"):
        return "Error"
    if rec.get("needs_clarification"):
        return "Clarification"
    return "Answer" if (rec.get("actual_output") or "").strip() else "No answer"


def outcome_counts(records: list[dict]) -> dict[str, int]:
    return {
        "Attempts": len(records),
        **{
            label: sum(outcome(r) == label for r in records)
            for label in ("Answer", "Clarification", "Error", "No answer")
        },
        "Turn-cap flags": sum((r.get("max_turns_halted") or 0) > 0 for r in records),
        "Reformatted": sum(bool(r.get("reformatted")) for r in records),
    }


def question_groups(records: list[dict]) -> dict[tuple, list[dict]]:
    groups = defaultdict(list)
    for rec in records:
        key = (
            rec["question_id"],
            rec["llm_name"],
            rec.get("chat_mode", "unknown"),
            rec.get("research_mode", "unknown"),
            rec["question"],
            rec.get("experiment_id") or "Legacy",
        )
        groups[key].append(rec)
    return dict(groups)


def coverage(records: list[dict], rows: list[dict], metrics: list[tuple]) -> tuple:
    """Which checks have a stored result for each question of one experiment.

    Returns one row per question, counting the responses each check scored,
    and the checks still missing a result for a response they cover. A check
    writes a row even when it cannot measure a response, so a missing row
    means the check has not been run. Checks that do not cover a question's
    runs at all read "n/a" and are never reported as missing.
    """
    grouped = defaultdict(list)
    for rec in records:
        grouped[
            (rec["question_id"], rec.get("chat_mode"), rec.get("research_mode"))
        ].append(rec)
    stored = defaultdict(set)
    for row in rows:
        stored[row["test_name"]].add(row["response_id"])
    table = []
    for key in sorted(grouped):
        group = grouped[key]
        ids = {r["response_id"] for r in group}
        table.append(
            {
                "Question": f"Q{key[0]}",
                "Chat mode": key[1],
                "Research mode": key[2],
                "Responses": len(group),
                "Answered": sum(outcome(r) == "Answer" for r in group),
                # Every response in a group shares the question's modes, so a
                # check either covers all of them or none.
                **{
                    name: (
                        "n/a"
                        if exclusion(metric, group[0])
                        else str(len(ids & stored[metric]))
                    )
                    for metric, name, _ in metrics
                },
            }
        )
    pending = []
    for metric, _name, _tip in metrics:
        covered = {r["response_id"] for r in records if not exclusion(metric, r)}
        if covered - stored[metric]:
            pending.append(metric)
    return table, pending


def run_numbers(records: list[dict]) -> dict:
    """Response id to run number for one question group, oldest run first.

    The run number is a property of the group, not of a metric's result list,
    so a metric that scored only the second attempt cannot rename it Run 1.
    """
    ordered = sorted(
        records, key=lambda r: (r.get("timestamp") or "", r["response_id"])
    )
    return {rec["response_id"]: index for index, rec in enumerate(ordered, 1)}


# The four states a row with no verdict can be in. They are different things to
# a reviewer: an expected exclusion needs no action, a failed measurement does.
NOT_APPLICABLE = "Not applicable"
NOT_MEASURED = "Not measured"
NOT_COMPARABLE = "Not comparable"
NO_STORED_RESULT = "No stored result"


def unmeasured_state(row: dict) -> str:
    """Which non-verdict state a stored row is in. Display only, no rescoring."""
    if row.get("not_comparable"):
        return NOT_COMPARABLE
    if row.get("not_applicable") or reason_is_not_applicable(row.get("reason")):
        return NOT_APPLICABLE
    return NOT_MEASURED


_COHORT_SIZE = re.compile(r"across (\d+) responses")


def consistency_cohort(rows: list[dict]) -> dict | None:
    """One repeat-similarity comparison shared by every response it covers.

    Consistency stores one row per response, all carrying the same comparison,
    so the rows are the comparison rather than one verdict each. Returns None
    unless the stored rows themselves say so: same score, same threshold, same
    verdict, and a recorded cohort size matching how many rows there are.
    Identical scores alone are not enough to merge two unrelated comparisons.
    """
    if len(rows) < 2:
        return None
    found = [_COHORT_SIZE.search(r.get("reason") or "") for r in rows]
    if any(match is None for match in found):
        return None
    if {int(match.group(1)) for match in found} != {len(rows)}:
        return None
    if len({(r["score"], r["threshold"], bool(r["passed"])) for r in rows}) != 1:
        return None
    return {
        "response_ids": sorted(r["response_id"] for r in rows),
        "score": rows[0]["score"],
        "threshold": rows[0]["threshold"],
        "passed": bool(rows[0]["passed"]),
    }


def apply_scope(rows: list[dict]) -> list[dict]:
    """Label unsupported historical checks without rewriting stored scores."""
    result = []
    for row in rows:
        reason = exclusion(row["test_name"], row)
        stored_reason = row.get("reason") or ""
        if (
            not reason
            and row["test_name"] == "report_integration"
            and stored_reason.startswith("No step both retrieved")
        ):
            reason = "Not measured: no reportable legislation findings to check for integration."
        if (
            not reason
            and row["test_name"] == "claim_support"
            and stored_reason.startswith("All ")
            and "claim(s) are claims of absence" in stored_reason
        ):
            reason = "Not measured: only absence claims were selected; none could be traced to a passage."
        if (
            not reason
            and row["test_name"] == "tool_usage"
            and row.get("research_mode") == "case_law_only"
            and not row.get("metric_version")
        ):
            reason = "Not measured: legacy Tool Usage used legislation rules; rescore with the current check."
        if reason:
            row = {
                **row,
                "measured": False,
                "not_applicable": reason_is_not_applicable(reason),
                "stored_reason": row.get("reason"),
                "reason": reason,
            }
        result.append({**row, "scope_note": scope_note(row["test_name"], row)})
    return result


def question_metadata(record: dict) -> dict:
    snapshot = record.get("experiment", {}).get("config", {}).get("questions", [])
    if snapshot:
        candidates = snapshot
        source = "Experiment snapshot"
    else:
        candidates = []
        for name in ("questions.json", "questions_new.json"):
            path = Path(__file__).resolve().parents[1] / "data" / name
            if path.exists():
                candidates.extend(json.loads(path.read_text()))
        source = "Current question file; historical metadata unavailable"
    for q in candidates:
        if (
            q["id"] == record["question_id"]
            and q["question"] == record["question"]
            and q.get("research_mode", "legislation_only")
            == record.get("research_mode")
        ):
            return {**q, "metadata_source": source}
    return {"metadata_source": "No matching question metadata"}


def current_reference_rows(rows, references):
    from lex_eval.reference.store import reference_version

    output = []
    for row in rows:
        if row["test_name"] in {
            "citation_agreement",
            "reference_answer_agreement",
            "plan_coverage",
        } and measured(row):
            ref = references.get(row["question_id"])
            if not ref or reference_version(ref) != (
                row.get("reference_sha256"),
                row.get("reference_mode"),
            ):
                row = {
                    **row,
                    "measured": False,
                    "stored_reason": row.get("reason"),
                    "reason": "Not measured in the current-reference view: this score used a different or unknown reference version.",
                }
        output.append(row)
    return output
