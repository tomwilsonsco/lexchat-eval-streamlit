#!/usr/bin/env python3
"""
Run LexChat evaluations using pytest.

This script wraps pytest so individual metrics (or all of them) can be
launched from the command line with sensible defaults and optional filters.
Each metric writes results to its own table (``eval_<metric>``) in the shared
DuckDB database (data/responses.db).

By default, existing results are preserved: a response that already has a
result in a metric's table is skipped for that metric. Use ``--overwrite`` to
clear a metric's table and re-run everything, or ``--append`` to re-run
everything without clearing, so new rows accumulate alongside old ones
(useful for checking a metric's determinism).

Tests run in parallel via pytest-xdist (``EVAL_WORKERS`` in lex_eval/.env,
default 4). This mainly speeds up the AI-judge metrics (response_groundedness,
claim_support, reference_answer_agreement), which are otherwise a long serial
chain of blocking OpenRouter calls. Use ``--workers 1`` to disable and run
single-process.

Examples
--------
Run everything (skipping already-completed metrics):
    python lex_eval/run_evals.py

Run a single metric (requires OPENROUTER_API_KEY for judge metrics):
    python lex_eval/run_evals.py --metrics response_groundedness

Run several metrics at once:
    python lex_eval/run_evals.py --metrics citation_grounding citation_domain

Force re-run (clear and replace existing results):
    python lex_eval/run_evals.py --metrics response_groundedness --overwrite

Re-run without clearing, to check a metric's determinism:
    python lex_eval/run_evals.py --metrics claim_support --append

Run only tool-usage checks (fast, no LLM judge needed):
    python lex_eval/run_evals.py --metrics tool_usage

Verbose output:
    python lex_eval/run_evals.py -v
"""

import json
import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

load_dotenv(dotenv_path=Path(__file__).parent / ".env")

TESTS_DIR = Path(__file__).parent / "tests" / "eval"

# Every metric, keyed by its test_name (the same string used everywhere it
# matters: attach_metric's test_name, the eval_<metric> table name, and the
# pytest function name test_<metric>), mapped to the file it lives in.
METRIC_FILES = {
    "tool_usage": "test_tool_usage.py",
    "response_groundedness": "test_groundedness.py",
    "claim_support": "test_groundedness.py",
    "consistency": "test_consistency.py",
    "mandatory_structure": "test_structure.py",
    "citation_passthrough": "test_structure.py",
    "citation_grounding": "test_structure.py",
    "citation_read": "test_structure.py",
    "citation_domain": "test_structure.py",
    "genuine_gap": "test_structure.py",
    "step_completion": "test_structure.py",
    "report_integration": "test_structure.py",
    "citation_agreement": "test_reference.py",
    "reference_answer_agreement": "test_reference.py",
    "plan_coverage": "test_reference.py",
}

_DEFAULT_WORKERS = 4


def _default_workers() -> int:
    """Resolve the pytest-xdist worker count: EVAL_WORKERS env var, else 4."""
    raw = os.getenv("EVAL_WORKERS")
    if raw is None or raw.strip() == "":
        return _DEFAULT_WORKERS
    try:
        return int(raw)
    except ValueError:
        print(
            f"⚠️  EVAL_WORKERS={raw!r} is not a valid integer; "
            f"falling back to {_DEFAULT_WORKERS}"
        )
        return _DEFAULT_WORKERS


# The metrics that score against a reference answer. Their stored results are
# only valid for the reference version they were calculated from.
REFERENCE_METRICS = {
    "citation_agreement",
    "reference_answer_agreement",
    "plan_coverage",
}


def _group_by_file(metrics: list[str]) -> dict[str, list[str]]:
    """Group *metrics* by their test file, preserving METRIC_FILES order."""
    grouped: dict[str, list[str]] = {}
    for metric in metrics:
        grouped.setdefault(METRIC_FILES[metric], []).append(metric)
    return grouped


def _deselect_args(covered: dict[str, set], test_file: str, records=None) -> list[str]:
    """
    Build pytest ``--deselect`` arguments for test IDs that already have a
    result for that *specific* response, one metric's covered response_ids
    at a time.

    *covered* maps metric -> set of response_ids already scored for it.
    Returns an empty list if nothing is covered.
    """
    if not any(covered.values()):
        return []

    # Ordinary metrics use unique response IDs; consistency also numbers
    # the peers within its question and experiment group.
    # test_consistency.py is the one exception: _same_model_cases() assigns
    # its own explicit ids of the form "{group_key}_run{n+1}" rather than
    # letting pytest auto-number them, so its deselect ids must match that
    # instead, or they silently fail to match anything and consistency rows
    # never get deselected. Its group key is built by consistency_group_key,
    # the same function the test itself uses, so the two cannot drift apart.
    from lex_eval.utils.test_helpers import (
        consistency_group_key,
        load_records,
        record_id,
    )

    def _numbered(keys: list[str]) -> list[int]:
        """Occurrence index of each key within the list, counting from 0."""
        counts: dict[str, int] = {}
        out: list[int] = []
        for k in keys:
            n = counts.get(k, 0)
            out.append(n)
            counts[k] = n + 1
        return out

    if records is None:
        records = load_records(read_only=True)
    base_ids = [record_id(r) for r in records]
    # Consistency groups (and numbers within) by mode as well, so it needs its
    # own key and its own run numbering.
    cons_ids = [consistency_group_key(r) for r in records]
    cons_occurrences = _numbered(cons_ids)

    deselect_args: list[str] = []
    for record, bid, cons_id, cons_n in zip(
        records, base_ids, cons_ids, cons_occurrences, strict=True
    ):
        response_id = record.get("response_id")
        for metric, response_ids in covered.items():
            if response_id in response_ids:
                if metric == "consistency":
                    test_id = f"{cons_id}_run{cons_n + 1}"
                else:
                    test_id = bid
                deselect_args.extend(
                    [
                        "--deselect",
                        f"lex_eval/tests/eval/{test_file}::test_{metric}[{test_id}]",
                    ]
                )
    return deselect_args


def run_evals(
    metrics: list[str] | None = None,
    markers: str | None = None,
    verbose: bool = False,
    overwrite: bool = False,
    append: bool = False,
    extra_args: list[str] | None = None,
    llm: str | None = None,
    workers: int | None = None,
    experiment: str | None = None,
    label: str = "Evaluation",
    dry_run: bool = False,
) -> int:
    """
    Launch pytest against the requested metrics (all of them if *metrics* is
    None).

    Returns the pytest exit code (0 = all passed).
    """
    requested = metrics or list(METRIC_FILES.keys())
    grouped = _group_by_file(requested)
    overall_rc = 0

    from lex_eval.utils.db import (
        DEFAULT_DB,
        clear_eval_results,
        get_connection,
        init_db,
        init_eval_table,
    )
    from lex_eval.reference.store import current_reference_versions

    # Scored against the reference answers, so a row is only still valid while
    # the reference it was scored against is unchanged.
    reference_versions = current_reference_versions()

    from lex_eval.utils.versioning import (
        scoring_config,
        start_scoring,
        finish_run,
        compatible_ids,
        metric_version,
    )
    from lex_eval.utils.db import load_records

    # load_records below is read-only, and a read-only connection cannot run
    # schema migrations, so bring an older database up to date first.
    if DEFAULT_DB.exists():
        conn = get_connection(DEFAULT_DB)
        try:
            init_db(conn)
            conn.commit()
        finally:
            conn.close()

    source_records = load_records(DEFAULT_DB, read_only=True)
    if experiment:
        source_records = [
            r for r in source_records if r.get("experiment_id") == experiment
        ]
    if llm:
        source_records = [r for r in source_records if llm in r["llm_name"]]
    if not source_records:
        print("No responses match the selected experiment/model.")
        return 5
    config = scoring_config(requested, {r["response_id"] for r in source_records})
    if dry_run:
        conn = get_connection(DEFAULT_DB, read_only=True)
        try:
            for metric in requested:
                versions = reference_versions if metric in REFERENCE_METRICS else None
                covered = (
                    set()
                    if append or overwrite
                    else compatible_ids(
                        conn, metric, metric_version(config, metric), versions
                    )
                )
                pending = set(config["response_ids"]) - covered
                print(
                    f"{metric}: up to {len(pending)} response evaluations; {len(set(config['response_ids']) & covered)} already compatible"
                )
            print(
                "Preview only. No database changes or judge calls. Applicability gates may reduce these counts."
            )
        finally:
            conn.close()
        return 0

    conn = get_connection(DEFAULT_DB)
    try:
        scoring_run_id = start_scoring(conn, config, label)
    finally:
        conn.close()
    worker_env = dict(
        os.environ,
        LEX_EVAL_SCORING_RUN_ID=scoring_run_id,
        LEX_EVAL_RESPONSE_IDS=json.dumps(config["response_ids"]),
    )
    print(f"Scoring run: {scoring_run_id}")

    run_status = "interrupted"
    try:
        for test_file, file_metrics in grouped.items():
            conn = get_connection(DEFAULT_DB)
            covered: dict[str, set] = {}
            try:
                # Migrate the responses schema and each requested metric's table
                # here, up front, in this single read-write connection. Eval test
                # modules load records/results via read-only connections (safe
                # under parallel pytest-xdist workers) and skip migration
                # themselves, so it must happen once before pytest starts.
                init_db(conn)
                for metric in file_metrics:
                    init_eval_table(conn, metric)
                    versions = (
                        reference_versions if metric in REFERENCE_METRICS else None
                    )
                    if overwrite:
                        clear_eval_results(conn, metric, llm=llm)
                    elif not append:
                        covered[metric] = compatible_ids(
                            conn, metric, metric_version(config, metric), versions
                        )
                conn.commit()
            finally:
                conn.close()

            cmd: list[str] = [
                sys.executable,
                "-m",
                "pytest",
                str(TESTS_DIR / test_file),
            ]

            if markers:
                cmd.extend(["-m", markers])

            # Filter to just the requested metrics' functions (only needed when
            # not every metric in this file was requested), anded with an LLM
            # filter if given.
            all_file_metrics = [m for m, f in METRIC_FILES.items() if f == test_file]
            keyword_parts = []
            if set(file_metrics) != set(all_file_metrics):
                fn_expr = " or ".join(f"test_{m}" for m in file_metrics)
                keyword_parts.append(
                    f"({fn_expr})" if len(file_metrics) > 1 else fn_expr
                )
            if llm:
                keyword_parts.append(llm)
            if keyword_parts:
                cmd.extend(["-k", " and ".join(keyword_parts)])

            # skip logic: deselect tests that already have results
            deselect: list[str] = []
            if not overwrite and not append:
                deselect = _deselect_args(covered, test_file, source_records)
                if deselect:
                    cmd.extend(deselect)
                    n_skipped = deselect.count("--deselect")
                    print(
                        f"ℹ️  {test_file}: skipping {n_skipped} test(s) with existing "
                        f"results (use --overwrite or --append to force)"
                    )

            # parallelise via pytest-xdist unless disabled (--workers 1); applied
            # uniformly across metrics so any future AI-judge metric benefits with
            # no extra wiring, and fast/offline metrics just pay a small
            # worker-startup cost
            n_workers = workers if workers is not None else _default_workers()
            if n_workers != 1:
                cmd.extend(["-n", str(n_workers)])

            # display
            cmd.extend(["-v" if verbose else "-q", "--tb=short"])

            # pass-through args
            if extra_args:
                cmd.extend(extra_args)

            print(f"\n{'='*60}")
            print(f"Running: {', '.join(file_metrics)}")
            print(f"{'='*60}")
            print(f"Command: {' '.join(cmd)}\n")

            result = subprocess.run(cmd, env=worker_env)
            rc = result.returncode

            # pytest exits 5 (NO_TESTS_COLLECTED) when every test was deselected,
            # which happens whenever a metric is already fully covered, the
            # documented default behaviour, not a failure. Only normalize it when
            # we know that's why: --deselect args were present in this exact
            # invocation. A rc 5 with no deselect args (e.g. a typo'd --llm
            # matching nothing) is a genuine collection problem and still
            # surfaces.
            if rc == 5 and deselect:
                print(
                    f"ℹ️  {test_file}: nothing new to run, all requested responses already covered"
                )
                rc = 0

            if rc > overall_rc:
                overall_rc = rc

        if overall_rc in (0, 1):
            print(
                "\n📊 Results written to data/responses.db (one eval_<metric> table per metric)"
                "\n   View dashboard: streamlit run lex_eval/reports/streamlit_report.py"
            )
        run_status = "finished" if overall_rc in (0, 1) else "incomplete"
    finally:
        conn = get_connection(DEFAULT_DB)
        try:
            finish_run(conn, "scoring_runs", scoring_run_id, run_status)
        finally:
            conn.close()

    return overall_rc


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LexChat evaluations (pytest)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Metrics:
  tool_usage                   Tools were invoked correctly (fast, offline)
  response_groundedness        Final response grounded in research output (needs OPENROUTER_API_KEY)
  claim_support                Research claims traceable to retrieved text (needs OPENROUTER_API_KEY)
  consistency                  Same-model repeatability (fast, cosine similarity)
  mandatory_structure          Worker output has required headings (fast, offline)
  citation_passthrough         Worker citations reach the final response (fast, offline)
  citation_grounding           Cited Acts were actually retrieved (fast, offline)
  citation_read                Cited Acts had their text read, not just their title seen (fast, offline)
  citation_domain               Citations point to legislation.gov.uk (fast, offline)
  genuine_gap                  Failed retrieval is disclosed, not glossed over (fast, offline)
  step_completion               Every step's own retrieval reached its own report, deep research only (fast, offline)
  report_integration            Every step's finding reached the final answer, deep research only (needs OPENROUTER_API_KEY)
  citation_agreement           Cites what the reference answer cites (fast, offline)
  reference_answer_agreement   States the reference answer's key points (needs OPENROUTER_API_KEY)
  plan_coverage                Deep research plan sets out to cover key points (needs OPENROUTER_API_KEY)

Results:
  Each metric writes to its own eval_<metric> table in data/responses.db.

  By default, a response already scored for a metric is skipped for it.
  Use --overwrite to clear and re-run, or --append to re-run without
  clearing (accumulates extra rows; useful for determinism checks).
  Use --llm to restrict evaluation to a single model.

Dashboard:
  Launch the Streamlit dashboard at any time:
    streamlit run lex_eval/reports/streamlit_report.py
""",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(METRIC_FILES.keys()),
        metavar="METRIC",
        help="Run only these metrics instead of all of them",
    )
    parser.add_argument(
        "-m",
        "--markers",
        help="Pytest marker expression (e.g. 'not slow')",
    )
    parser.add_argument(
        "--llm",
        metavar="LLM_NAME",
        help="Only evaluate this LLM (e.g. 'gpt-oss:120b-cloud')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Clear existing results for the selected metrics and re-run everything",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help=(
            "Re-run the selected metrics against every response without "
            "clearing or skipping, so new rows accumulate alongside existing "
            "ones. For troubleshooting/testing a metric's determinism."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Parallel pytest-xdist workers for AI-judge calls (default: "
            "EVAL_WORKERS env var, or 4). Use --workers 1 to disable "
            "parallelism, e.g. for easier-to-read debugging output."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "extra",
        nargs="*",
        help="Additional arguments passed through to pytest",
    )

    parser.add_argument(
        "--experiment", help="Only score responses belonging to this experiment ID"
    )
    parser.add_argument("--label", default="Evaluation", help="Scoring run label")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview pending evaluations without writes or judge calls",
    )
    args = parser.parse_args()

    if args.overwrite and args.append:
        parser.error("--overwrite and --append are mutually exclusive")

    if args.overwrite and args.experiment:
        parser.error(
            "--overwrite cannot be combined with --experiment; use --append to "
            "rescore an experiment while preserving historical results"
        )

    return run_evals(
        metrics=args.metrics,
        markers=args.markers,
        verbose=args.verbose,
        overwrite=args.overwrite,
        append=args.append,
        extra_args=args.extra,
        llm=args.llm,
        workers=args.workers,
        experiment=args.experiment,
        label=args.label,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    sys.exit(main())
