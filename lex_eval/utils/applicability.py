"""Explicit source coverage for the existing checks."""

LEGISLATION_CHECKS = {
    "citation_grounding",
    "citation_read",
    "step_completion",
    "report_integration",
}

# Checks that only mean anything for a deep research run, which is a property
# of the run rather than of the reason a metric happened to write.
DEEP_RESEARCH_CHECKS = {"step_completion", "report_integration", "plan_coverage"}


def exclusion(metric: str, record: dict) -> str | None:
    mode = record.get("research_mode", "legislation_only")
    # Only for a mode we actually recorded: an unknown chat mode is not
    # evidence that the check did not apply.
    if metric in DEEP_RESEARCH_CHECKS and record.get("chat_mode") in {
        "research",
        "conversational",
    }:
        return "Not applicable: this check covers deep research runs only."
    if metric == "genuine_gap" and mode != "legislation_only":
        return "Not measured: Genuine Gap checks legislation-only research."
    if metric in LEGISLATION_CHECKS and mode == "case_law_only":
        return "Not measured: this check covers legislation, not judgment evidence."
    return None


def scope_note(metric: str, record: dict) -> str | None:
    if record.get(
        "research_mode"
    ) == "legislation_and_case_law" and metric in LEGISLATION_CHECKS | {
        "tool_usage",
        "citation_agreement",
    }:
        return "Legislation only; case-law behavior is not checked by this result."
    return None


# A stored reason that begins with one of these says the check was never
# expected to run for this response, which is different from a check that
# should have run and could not. The dashboard shows the two differently, so
# the wording lives here rather than being re-guessed in the UI.
NOT_APPLICABLE_PREFIXES = (
    "Not applicable",
    "Not deep_research",
    "Not measured: Genuine Gap checks legislation-only research.",
    "Not measured: this check covers legislation, not judgment evidence.",
)


def reason_is_not_applicable(reason: str | None) -> bool:
    """Whether *reason* says the check does not apply, rather than could not run."""
    return bool(reason) and reason.startswith(NOT_APPLICABLE_PREFIXES)
