"""Append-only experiment and scoring history beside the existing results."""

import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def revision():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


# The only utils modules the eval tests reach. The metrics import nothing from
# utils/, so the rest of it is storage and gather plumbing that cannot change a
# score; including it would invalidate every stored score, and force a paid
# judge rerun, for an edit that moves no number.
SCORING_UTILS = ("applicability.py", "collector.py", "judge.py", "test_helpers.py")


def source_version():
    """Fingerprint scoring code, including uncommitted changes."""
    paths = [ROOT / "lex_eval/testcase.py", ROOT / "lex_eval/reference/store.py"]
    paths.extend(ROOT / "lex_eval/utils" / name for name in SCORING_UTILS)
    for folder in ("metrics", "tests/eval"):
        paths.extend(sorted((ROOT / "lex_eval" / folder).glob("*.py")))
    return fingerprint(
        {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        }
    )


def capture_version():
    paths = [
        "gather_responses.py",
        "utils/audit_capture.py",
        "utils/lexchat_client.py",
        "utils/get_llm.py",
    ]
    return fingerprint(
        {
            name: hashlib.sha256((ROOT / "lex_eval" / name).read_bytes()).hexdigest()
            for name in paths
        }
    )


def judge_config():
    from lex_eval.utils import judge

    return {
        key: getattr(judge, f"OPENROUTER_JUDGE_{key.upper()}")
        for key in (
            "model",
            "temperature",
            "max_tokens",
            "reasoning_effort",
            "fallback_model",
        )
    }


def init_history(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id TEXT PRIMARY KEY, label TEXT NOT NULL, created_at TEXT NOT NULL,
            config JSON NOT NULL, config_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS gather_runs (
            id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, started_at TEXT NOT NULL,
            finished_at TEXT, status TEXT NOT NULL, question_ids JSON NOT NULL
        );
        CREATE TABLE IF NOT EXISTS response_runs (
            response_id INTEGER PRIMARY KEY, gather_run_id TEXT NOT NULL,
            experiment_id TEXT NOT NULL, question_hash TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS scoring_runs (
            id TEXT PRIMARY KEY, label TEXT NOT NULL, started_at TEXT NOT NULL,
            finished_at TEXT, status TEXT NOT NULL, config JSON NOT NULL
        );
        CREATE TABLE IF NOT EXISTS eval_versions (
            metric TEXT NOT NULL, eval_id INTEGER NOT NULL, scoring_run_id TEXT NOT NULL,
            metric_version TEXT NOT NULL, details JSON,
            PRIMARY KEY(metric, eval_id)
        );
    """)


def start_gather(conn, *, label, config, question_ids, experiment_id=None):
    """Joining an experiment requires the same captured condition."""
    init_history(conn)
    digest = fingerprint(config)
    if experiment_id:
        row = conn.execute(
            "SELECT config_hash FROM experiments WHERE id=?", [experiment_id]
        ).fetchone()
        if not row or row[0] != digest:
            raise ValueError(
                "Experiment not found or configuration changed. Start a new experiment."
            )
    else:
        experiment_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO experiments VALUES (?, ?, ?, ?, ?)",
            [experiment_id, label, now(), json.dumps(config), digest],
        )
    run_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO gather_runs VALUES (?, ?, ?, NULL, 'running', ?)",
        [run_id, experiment_id, now(), json.dumps(question_ids)],
    )
    return experiment_id, run_id


def link_response(conn, response_id, run_id, experiment_id, question):
    conn.execute(
        "INSERT INTO response_runs VALUES (?, ?, ?, ?)",
        [response_id, run_id, experiment_id, fingerprint(question)],
    )


def finish_run(conn, table, run_id, status):
    if table not in {"gather_runs", "scoring_runs"}:
        raise ValueError("Unknown run table")
    conn.execute(
        f"UPDATE {table} SET finished_at=?, status=? WHERE id=?",
        [now(), status, run_id],
    )


def scoring_config(metrics, response_ids):
    from lex_eval.reference.store import load_reference_answers

    return {
        "source_version": source_version(),
        "revision": revision(),
        "judge": judge_config(),
        "metrics": list(metrics),
        "response_ids": sorted(response_ids),
        "references": load_reference_answers(),
    }


def start_scoring(conn, config, label="Evaluation"):
    init_history(conn)
    run_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO scoring_runs VALUES (?, ?, ?, NULL, 'running', ?)",
        [run_id, label, now(), json.dumps(config)],
    )
    return run_id


def metric_version(config, metric):
    data = {
        "source": config["source_version"],
        "metric": metric,
        "judge": config["judge"],
    }
    if metric == "consistency":
        data["response_ids"] = config.get("response_ids", [])
    return fingerprint(data)


def response_history(conn):
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    if "response_runs" not in tables:
        return {}
    return {
        rid: dict(experiment_id=exp, gather_run_id=run, question_hash=question)
        for rid, run, exp, question in conn.execute(
            "SELECT * FROM response_runs"
        ).fetchall()
    }


def compatible_ids(conn, metric, version, references=None):
    """Only compatible scores cover a response; older scores remain stored."""
    tables = {r[0] for r in conn.execute("SHOW TABLES").fetchall()}
    if "eval_versions" not in tables or f"eval_{metric}" not in tables:
        return set()
    rows = conn.execute(
        f"""SELECT e.response_id, e.question_id, e.reference_sha256, e.reference_mode
        FROM eval_{metric} e JOIN eval_versions v ON v.eval_id=e.id AND v.metric=?
        WHERE v.metric_version=?""",
        [metric, version],
    ).fetchall()
    return {
        rid
        for rid, qid, sha, mode in rows
        if references is None or references.get(qid, (None, None)) == (sha, mode)
    }
