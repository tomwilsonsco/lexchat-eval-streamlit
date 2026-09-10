"""A review pack as a document a reviewer can read, from the same data as the JSON.

Pure functions, no Streamlit and no database, so the layout can be tested
offline. Nothing here recomputes a score: every verdict, reason and passage is
the one the dashboard was showing when the pack was built.
"""

from __future__ import annotations


def _section(title: str) -> str:
    return f"\n## {title}\n"


def _responses(pack: dict) -> str:
    lines = [_section("Responses")]
    for rec in pack.get("responses", []):
        lines.append(f"### {rec.get('run_label') or rec.get('response_id')}\n")
        facts = [
            f"- Response id: {rec.get('response_id')}",
            f"- Captured: {rec.get('timestamp') or 'unknown'}",
            f"- Outcome: {rec.get('outcome') or 'unknown'}",
        ]
        if rec.get("research_limit_reached"):
            facts.append("- Research limit reached: a step stopped at the turn limit")
        if rec.get("error_message"):
            facts.append(f"- Error: {rec['error_message']}")
        lines.append("\n".join(facts) + "\n")
        lines.append("**Answer to the user**\n")
        lines.append((rec.get("actual_output") or "_(no output captured)_") + "\n")
    return "\n".join(lines)


def _checks(pack: dict) -> str:
    """Failures and gaps first, then the checks that passed."""
    lines = [_section("Checks")]
    checks = pack.get("checks", [])
    failed = [
        c
        for c in checks
        if c.get("state", "").startswith(("Stable fail", "Fail", "Mixed"))
    ]
    gaps = [c for c in checks if not c.get("scored", True)]
    passed = [c for c in checks if c not in failed and c not in gaps]
    for heading, subset in (
        ("Failed", failed),
        ("No verdict", gaps),
        ("Passed", passed),
    ):
        if not subset:
            continue
        lines.append(f"### {heading}\n")
        for check in subset:
            lines.append(f"**{check['metric_name']}** ({check.get('state', '')})\n")
            for result in check.get("results", []):
                label = (
                    result.get("run_label") or f"Response {result.get('response_id')}"
                )
                score = (
                    f"score {result['score']:.3f}"
                    if result.get("scored") and result.get("score") is not None
                    else "no score"
                )
                lines.append(
                    f"- {label}: {result.get('status', '')}, {score}. "
                    f"{result.get('reason') or ''}".rstrip()
                )
                if result.get("provenance"):
                    lines.append(f"  - Scored by: {result['provenance']}")
            lines.append("")
    return "\n".join(lines)


def _reference(pack: dict) -> str:
    statements = pack.get("current_reference_statements") or []
    if not statements:
        return ""
    lines = [_section("Reference answer key statements (current version)")]
    lines.append(
        f"Fingerprint: {pack.get('current_reference_sha256') or 'unknown'}. "
        "A draft reference records agreement with its author, not legal correctness.\n"
    )
    lines.extend(f"{i}. {text}" for i, text in enumerate(statements, 1))
    return "\n".join(lines) + "\n"


def _notes(pack: dict) -> str:
    notes = pack.get("review") or {}
    lines = [_section("Reviewer notes")]
    for field, label in (
        ("observed_problem", "Observed problem"),
        ("evidence_passage", "Evidence passage"),
        ("reviewer", "Reviewer"),
    ):
        lines.append(f"**{label}**\n\n{notes.get(field) or '_(not recorded)_'}\n")
    return "\n".join(lines)


def review_markdown(pack: dict) -> str:
    """Render a review pack as Markdown, in the order a reviewer reads it."""
    question = pack.get("question") or {}
    selection = pack.get("selection") or {}
    header = [
        f"# Review: Q{question.get('id', '')} {question.get('question', '')}".rstrip(),
        "",
        "\n".join(f"- {name}: {value}" for name, value in selection.items()),
        "",
        f"Question metadata source: {question.get('metadata_source', 'unknown')}.",
    ]
    if question.get("known_gap"):
        header += ["", f"Previously recorded failure: {question['known_gap']}"]
    return "\n".join(
        [
            "\n".join(header),
            _notes(pack),
            _checks(pack),
            _responses(pack),
            _reference(pack),
        ]
    )
