from __future__ import annotations

import hashlib
import html
import json
import sys
from collections import Counter
from pathlib import Path

import streamlit as st

# Ensure the repo root is importable when Streamlit launches this file directly.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lex_eval.reference.store import (
    ANSWERS_DIR,
    MANIFEST_NAME,
    effective_verified,
    load_reference_answers,
    reference_version,
)
from lex_eval.reports.attribution import caveat, worst_attribution
from lex_eval.reports.comparison import (
    PASS_FREQUENCY_CHANGE,
    change_counts,
    cohort_key,
    compare,
    matched_entries,
    metric_summary,
    shared_cohorts,
)
from lex_eval.reports.diagnostics import searches, search_summary, plan_steps
from lex_eval.reports.data import (
    NOT_MEASURED,
    coverage,
    apply_scope,
    consistency_cohort,
    question_metadata,
    current_reference_rows,
    aggregate_metrics,
    measured,
    read_database,
    latest_results,
    outcome,
    outcome_counts,
    question_groups,
    run_numbers,
    unmeasured_state,
)
from lex_eval.reports.review_export import review_markdown
from lex_eval.utils.db import (
    DEFAULT_DB,
    reason_is_not_measured,
)

script_dir = Path(__file__).parent
data_dir = script_dir.parent / "data"

# Where results are read from: the working database when there is one, and
# otherwise the Parquet directory the deploy build writes. A deployed copy
# ships only the Parquet, so it has to resolve without being told.
RESPONSES_DB = DEFAULT_DB if DEFAULT_DB.exists() else data_dir / "deploy"


DEFAULT_CHAT_MODE = "research"


def _results_mtime(path: Path) -> int:
    """Cache key for the results, whichever form they take.

    A Parquet directory's own mtime does not move when a file inside it is
    rewritten, so the newest file decides instead.
    """
    if path.is_dir():
        return max(
            (f.stat().st_mtime_ns for f in path.glob("*.parquet")),
            default=0,
        )
    return path.stat().st_mtime_ns


@st.cache_data
def load_dashboard(db_path: str, mtime: float):
    """Cache by path and modification time; never migrate the source database."""
    return read_database(Path(db_path), METRICS)


# Single source of truth for every metric this dashboard displays: its
# eval_<key> table, its display name, and its tooltip, in display order.
# Keys here must match run_evals.py::METRIC_FILES.
#
# Grouped, in this order: 1) deterministic metrics that run in research or
# deep research mode, 2) AI-judge metrics that run in research or deep
# research mode, 3) deep-research-only deterministic metrics, 4)
# deep-research-only AI-judge metrics. The "(Deep research only)" suffix on
# groups 3-4's display names is what shows that scope on the title bar
# wherever the metric name is rendered.
METRICS: list[tuple[str, str, str]] = [
    # 1. Deterministic, research or deep research
    (
        "tool_usage",
        "Tool Usage",
        "Are all of delegate research, search legislation and search legislation sections used, in the correct order (search legislation - search legislation sections - get legislation text if needed), and does the Worker stick to that order rather than looping back to an earlier step later in the same run?",
    ),
    (
        "mandatory_structure",
        "Research Output Structure",
        "Does the worker agent return the findings to the manager with the requested headers. Not measured in conversational mode, where the worker is told not to use those headers.",
    ),
    (
        "citation_passthrough",
        "Reference Links",
        "Are all reference links found by the researcher included in the final answer given to the user.",
    ),
    (
        "citation_grounding",
        "Citation Grounding",
        "Does every Act cited in the researcher's report correspond to legislation the run's own tool calls actually retrieved, rather than one invented by the model.",
    ),
    (
        "citation_read",
        "Citation Read",
        "Did the researcher actually read every Act it cites? An Act whose text was pulled counts as read, one that only appeared as a title in a search results list does not. Catches a report making claims about a real, correctly linked source it never opened.",
    ),
    (
        "citation_domain",
        "Citation Domain",
        "Does every citation link in the researcher's report point to a domain the Worker is permitted to cite. That is legislation.gov.uk for legislation only mode, caselaw.nationalarchives.gov.uk for case law only mode, and both for the hybrid mode, matching what each Worker prompt asks for.",
    ),
    (
        "genuine_gap",
        "Genuine Gap",
        "When retrieval found no usable legislation text, does the researcher's report say so plainly instead of answering with unsupported confidence. In research mode the wording is set by the prompt, so a paraphrase scores half. In conversational mode no wording is mandated, so a plain statement of the gap scores full.",
    ),
    (
        "consistency",
        "Consistency (Cosine)",
        "Compare the answers provided when the same question is asked multiple times in the same chat mode, using TF cosine similarity. Research and deep research answers are never compared against each other, and a mode with only one stored run is not scored. Any legislation section cited in one answer but not the other is listed in the detail, but does not decide pass or fail.",
    ),
    (
        "citation_agreement",
        "Citation Agreement",
        "Of the legislation provisions the hand written reference answer cites, how many does the response cite too. No AI judge, it compares the two lists of legislation.gov.uk links. For each Act the reference answer relied on that the response does not cite, the reason says whether any search turned it up, so a search that missed the law reads differently from an answer that had the law and left it out.",
    ),
    # 2. AI judge, research or deep research
    (
        "reference_answer_agreement",
        "Reference Answer Agreement",
        "AI as a judge metric: How many of the question's key statements the response also makes, at most 5 of them. The statements are written once alongside the hand written reference answer and stored with it, so the judge labels a fixed list rather than picking the points afresh on every run. A second judge call looks for contradictions and nothing else, which is what catches a long answer that makes a point correctly in one section and then undoes it in another. A statement the response contradicts fails the metric outright, since a confidently wrong statement of law is worse than a missing one. The reference answers are unverified drafts, so read a flagged contradiction as a prompt to compare the two texts.",
    ),
    (
        "response_groundedness",
        "Response Groundedness",
        "AI as a judge metric: Is the final answer to the user grounded in the research worker's summary. A near-unmodified copy is accepted automatically with no AI judge involved. Anything reworded enough to matter goes to the judge, which fails it on any unsupported claim or meaningful misrepresentation and passes only trivial wording differences. There is no partial credit, so the average is a pass rate.",
    ),
    (
        "claim_support",
        "Claim Support",
        "AI as a judge metric: What share of the report's verifiable legal claims are backed by text the researcher actually read? Claims whose truth depends on the absence of a provision are reported separately because absence generally cannot be established from retrieved excerpts or summaries.",
    ),
    # 3. Deterministic, deep research only
    (
        "step_completion",
        "Step Completion (Deep research only)",
        "Deep research only. Did every step of the approved research plan carry its own retrieved legal text into its own report, rather than a step that retrieved text and then reported nothing (for example, hitting a tool-call budget limit mid-step).",
    ),
    # 4. AI judge, deep research only
    (
        "report_integration",
        "Report Integration (Deep research only)",
        "Deep research only, AI as a judge metric: For every step that reported a real, cited finding of its own, does the final answer reflect that finding, rather than dropping it when the Manager condenses several step reports into one response. A step with nothing of its own to check (empty or uncited retrieval) is not scored here; that is Step Completion's and Genuine Gap's question.",
    ),
    (
        "plan_coverage",
        "Plan Coverage (Deep research only)",
        "Deep research only, AI as a judge metric: Does the approved research plan set out to cover the question's key statements, before any research happens. Reuses the same fixed statement list as Reference Answer Agreement rather than a separately authored golden plan.",
    ),
]

# order shown in streamlit
METRIC_DISPLAY_ORDER: list[str] = [name for _key, name, _tooltip in METRICS]

# hover over tips on app summary tables
METRIC_TOOLTIPS: dict[str, str] = {name: tip for _key, name, tip in METRICS}


# What puts a question in the "needs attention" set. Written once because
# the summary column and the checkbox that filters on it are the same rule.
NEEDS_ATTENTION_HELP = (
    "A question needs attention when a metric failed or could not be scored, "
    "when no metric ran at all, or when a run ended without an answer or hit "
    "the turn limit."
)

# Hover over help for the column headers of every table on the dashboard, one
# dict per table. They are kept separate because the same header means
# different things in different tables: "Error" counts failed attempts in the
# outcomes table and holds one tool call's error message in the searches table.
# Headings for the outcomes table. "Response received" says only that final
# answer text was captured; it is not a judgement that the research finished or
# that the text answers the question. The stored outcome values are unchanged,
# so anything else reading them keeps working.
OUTCOME_COLUMNS: dict[str, str] = {
    "Attempts": "Every captured run of this question, including repeats and runs that failed.",
    "Response received": "Runs that returned final answer text. It does not mean the research finished, or that the text answers the question.",
    "Clarification requested": "Runs where LexChat asked the user a clarifying question instead of researching.",
    "Error": "Runs that failed with an error and produced no answer.",
    "No response text": "Runs that ended with no error, no clarifying question and no answer text.",
    "Research limit reached": "Runs where at least one research step was cut short at the server's turn limit, so that step returned no findings.",
    "Reformatted": "Runs where the worker's report missed the required headings and LexChat asked the model to rewrite it once.",
}

# Stored outcome name to the heading a reviewer sees, applied at the display
# boundary only.
OUTCOME_DISPLAY: dict[str, str] = {
    "Answer": "Response received",
    "Clarification": "Clarification requested",
    "No answer": "No response text",
    "Turn-cap flags": "Research limit reached",
}


def _display_counts(counts: dict) -> dict:
    """Rename stored outcome keys to their headings, keeping the order."""
    return {OUTCOME_DISPLAY.get(key, key): value for key, value in counts.items()}


def _outcome_label(rec: dict) -> str:
    """What happened on this run, as a reviewer should read it.

    "Response received" means final text was captured, nothing more, so a reply
    that asks the user to narrow the question is not counted as completed
    research. The research limit is shown beside the response it belongs to
    rather than only in the question's counts.
    """
    label = OUTCOME_DISPLAY.get(outcome(rec), outcome(rec))
    if rec.get("max_turns_halted"):
        label += " · Research limit reached"
    return label


# The question table is a queue of what to look at next, so what needs
# attention comes before the configuration every row repeats.
QUESTION_COLUMNS: dict[str, str] = {
    "Question": "The question id and the start of the question text.",
    "Failed checks": "The checks that failed for this question, by name. The first three, with a count of any others; the question's detail lists them all.",
    "Measurement gaps": "Checks that produced no verdict: how many do not apply to this question, and how many could not be measured.",
    "Execution warnings": "Runs that errored, asked for clarification, returned no text, or reached the research limit.",
    "Responses": OUTCOME_COLUMNS["Attempts"],
    "Reference agreement": "Reference Answer Agreement only: how many runs matched the reference answer's key statements, out of the runs it could score.",
    "Needs attention": NEEDS_ATTENTION_HELP,
    "Model": "The model LexChat had active when these runs were gathered.",
    "Chat mode": "The LexChat mode used: research, deep research or conversational.",
    "Research mode": "Which sources the question expects: legislation only, case law only, or both.",
    "Experiment": "The label of the gather run these attempts belong to. Only runs in the same experiment are compared.",
    "Type": "From the question set. A regression question reproduces a past failure, a positive control is one LexChat is expected to get right.",
}

# Dropped from the table unless the reviewer asks for them: every row in one
# selection usually repeats them, and the selected question's header says them.
CONFIGURATION_COLUMNS = ("Model", "Chat mode", "Research mode", "Experiment", "Type")

# Fixed so the whole queue fits a 1440px window without horizontal scrolling.
QUESTION_COLUMN_WIDTHS: dict[str, str] = {
    "Question": "medium",
    "Failed checks": "medium",
    "Measurement gaps": "small",
    "Execution warnings": "small",
    "Responses": "small",
    "Reference agreement": "small",
    "Needs attention": "small",
}

SEARCH_COLUMNS: dict[str, str] = {
    "Response": "Which attempt made this search. One question can have several.",
    "Step": "Which research step made the call. Deep research numbers its steps, the other modes have one.",
    "Tool": "The search tool called, for example search_legislation.",
    "Arguments": "The arguments the tool was called with, including the search terms.",
    "Outcome": "Results returned, Empty, Error, or Unknown / incomplete when the result was cut short or blocked by the tool call budget.",
    "Returned": "How many items came back. This counts items, not relevance, and is blank when the count is unknown.",
    "Cache reused": "True when the result came from a cache rather than a fresh API call.",
    "Error": "The error the tool call returned, if any.",
    "Later nonempty search in this step": "True when this search returned nothing but a repeat of the same tool later in the step did return results.",
}

SEARCH_SUMMARY_COLUMNS: dict[str, str] = {
    "Response": SEARCH_COLUMNS["Response"],
    "Tool": SEARCH_COLUMNS["Tool"],
    "Outcome": SEARCH_COLUMNS["Outcome"],
    "Searches": "How many calls to this tool on this response ended this way.",
    "Items returned": "Total items those calls returned. Blank where no count is available, which is every Error and Unknown / incomplete row.",
}

COMPARISON_SUMMARY_COLUMNS: dict[str, str] = {
    "Matched questions and modes": "How many question and mode combinations appear in both experiments. Only these are compared.",
    "Baseline only": "Combinations gathered in the baseline experiment but not the candidate.",
    "Candidate only": "Combinations gathered in the candidate experiment but not the baseline.",
}

EXPERIMENT_INFO_COLUMNS: dict[str, str] = {
    "Question": "The question id, as gathered in this experiment.",
    "Chat mode": QUESTION_COLUMNS["Chat mode"],
    "Research mode": QUESTION_COLUMNS["Research mode"],
    "Responses": "How many responses this experiment gathered for the question.",
    "Answered": "How many of those responses came back with answer text.",
}

COMPARISON_CHANGE_COUNT_COLUMNS: dict[str, str] = {
    "Checks": "How many checks the table below reports, which is every check with a stored result on either side.",
    "More passes": "Checks the candidate passed more often than the baseline, over the runs both sides measured.",
    "Fewer passes": "Checks the candidate passed less often than the baseline.",
    "Same pass frequency": "Checks that passed equally often on both sides.",
    "Not measured": "Checks with nothing measured on one or both sides, for example a deep research check on conversational runs.",
    "Not comparable": "Checks where no shared question could be compared, because the two sides used different scoring versions or different judges.",
}

COMPARISON_METRIC_COLUMNS: dict[str, str] = {
    "Check": "The check these totals are for.",
    "Baseline": "Baseline totals over the questions both experiments answered: measured passes out of measured runs, the mean of those scores, and how many runs the check could not measure.",
    "Candidate": "The same totals for the candidate experiment.",
    PASS_FREQUENCY_CHANGE: "How often the candidate passed compared with the baseline, over the runs both sides measured. This is not the movement in the mean score, which is shown in the two columns before it and can go the other way. Runs that could not be measured are not part of either.",
    "Questions compared": "How many shared questions are in these totals.",
    "Not compared": "Shared questions left out because the two sides used different scoring versions or different judges. They are in no total here.",
}

COMPARISON_OUTCOME_COLUMNS: dict[str, str] = {
    "Experiment": "Which side of the comparison the counts on this row come from.",
    **OUTCOME_COLUMNS,
}

def _column_help(tooltips: dict[str, str], widths: dict | None = None) -> dict:
    """Turn a column name to description mapping into Streamlit column config.

    *widths* pins the columns that must stay on screen at 1440px, so a long
    list of failed check names cannot push the execution warnings out of view.
    """
    return {
        name: st.column_config.Column(name, help=text, width=(widths or {}).get(name))
        for name, text in tooltips.items()
    }


def _is_scored(result: dict) -> bool:
    """Whether this row's score is a verdict, and so belongs in a mean.

    A metric that could not score a response still writes a row, so the gap
    stays visible, carrying score 0.0 because the column is NOT NULL. The
    `measured` column is what marks those. Rows written before that column
    existed fall back to the reason wording, which is the rule the column was
    filled from (see db.backfill_measured_column), so a stale deploy.db still
    aggregates correctly.
    """
    if "measured" in result:
        return bool(result["measured"])
    return not reason_is_not_measured(result.get("reason") or "")


def _metric_sort_key(metric: dict) -> int:
    name = metric["metric_name"]
    try:
        return METRIC_DISPLAY_ORDER.index(name)
    except ValueError:
        return len(METRIC_DISPLAY_ORDER)


def _aggregate_metrics(results: list[dict]) -> list[dict]:
    return sorted(aggregate_metrics(results), key=_metric_sort_key)


# Status wording and colour come from the stored verdict, never from where the
# number sits on a scale: a high score can still be a stored failure. Streamlit
# colour tokens are used rather than fixed hex so both themes stay readable.
_STATE_MARKUP: dict[str, str] = {
    "Stable pass": ":green[**Passed**]",
    "Pass (one measurement)": ":green[**Passed**]",
    "Stable fail": ":red[**Failed**]",
    "Fail (one measurement)": ":red[**Failed**]",
    "Mixed": ":orange[**Mixed**]",
    "Not comparable": ":orange[**Not comparable**]",
}


def _identity(records: list[dict]) -> dict:
    """Response id to the one label that identifies it everywhere on the page.

    Built once per question group and reused by every metric, the answers, the
    comparison and the export, so "Run 2" always means the same response. A
    metric that scored only one of two attempts cannot renumber the other.
    """
    numbers = run_numbers(records)
    return {
        rec["response_id"]: {
            "run": numbers[rec["response_id"]],
            "label": (
                f"Response {rec['response_id']} · Run {numbers[rec['response_id']]}"
                f" of {len(numbers)} · {(rec.get('timestamp') or '')[:19]}"
            ),
            "short": f"Response {rec['response_id']} (run {numbers[rec['response_id']]})",
        }
        for rec in records
    }


def _response_heading(response_id, identity: dict | None) -> str:
    entry = (identity or {}).get(response_id)
    return entry["label"] if entry else f"Response {response_id}"


def _reason_block(text: str) -> None:
    """Stored reason text, shown as written.

    Kept inside an HTML block so Markdown in a judge's wording is not
    interpreted, and given no fixed colours so it reads in either theme.
    """
    body = html.escape(text or "").replace("\n", "<br>")
    st.markdown(
        f'<div style="font-size:0.88em;opacity:0.85;margin:2px 0 6px 0;">{body}</div>',
        unsafe_allow_html=True,
    )


def _gap_counts(m: dict) -> Counter:
    """How many responses sit in each non-verdict state for this metric."""
    return Counter(unmeasured_state(row) for row in m.get("not_scored_rows", []))


def _gap_states(metrics: list[dict]) -> Counter:
    """How many checks sit in each non-verdict state across a question.

    One entry per check, not per response, so a check that could not be
    measured on either repeat is one gap rather than two.
    """
    gaps = Counter()
    for m in metrics:
        for state in set(_gap_counts(m)):
            gaps[state] += 1
    return gaps


def _gap_phrase(gaps: Counter) -> str:
    """A Counter of states as "1 not applicable, 2 not measured"."""
    return ", ".join(
        f"{count} {state.lower()}" for state, count in sorted(gaps.items())
    )


def _headline_state(m: dict) -> str:
    """The state to show for a whole check.

    For a check with no verdict this is the state its rows are actually in, so
    an expected exclusion is never announced as a measurement that did not
    happen. The row label and the exported report say the same thing.
    """
    if m.get("scored", True):
        return m["state"]
    states = sorted(_gap_counts(m))
    return states[0] if len(states) == 1 else m.get("state", NOT_MEASURED)


def _provenance_bits(r: dict) -> list[str]:
    """Which stored scoring event produced this row, in the order it reads.

    The same wording is shown beside the score and written into the exported
    report, so a reader given only the report can still say which scoring run
    and which judge produced a finding.
    """
    bits = [r.get("test_name") or ""]
    if r.get("scoring_run_id"):
        bits.append(f"scoring run {r['scoring_run_id']}")
        bits.append(f"code {r.get('metric_version', 'unknown')[:12]}")
    else:
        bits.append("legacy scorer version unknown")
    if not _reference_is_current(r):
        bits.append("reference answer changed since, re-run to update")
    if r.get("run_at"):
        bits.append(f"scored {r['run_at'][:19]}")
    if r.get("judge_llm"):
        bits.append(f"judge: {r['judge_llm']}")
        if r.get("judge_tokens"):
            bits.append(f"{r['judge_tokens']} tokens")
    return [bit for bit in bits if bit]


def _display_rows(m: dict, identity: dict | None) -> list[dict]:
    """This check's stored rows, in the question's run order, as displayed.

    A row in a group that could not be compared is not a pass, whatever its own
    stored verdict says. aggregate_metrics keeps a corrected copy of every row
    it excluded, so use that wherever it has one.
    """
    identity = identity or {}
    rows = sorted(
        m.get("raw_results", []),
        key=lambda r: identity.get(r["response_id"], {}).get("run", 0),
    )
    corrected = {r["response_id"]: r for r in m.get("not_scored_rows", [])}
    return [corrected.get(r["response_id"], r) for r in rows]


def _check_results(m: dict, identity: dict) -> list[dict]:
    """One export entry per stored result, in the question's own run order.

    A shared repeat-similarity comparison becomes one entry naming every
    response in it, the same way the screen shows it, so the document cannot
    read as one verdict per response.
    """
    rows = _display_rows(m, identity)
    if m["test_name"] == "consistency":
        cohort = consistency_cohort([r for r in rows if measured(r)])
        if cohort is not None:
            names = ", ".join(
                _identity_short(rid, identity) for rid in cohort["response_ids"]
            )
            return [
                {
                    "response_id": cohort["response_ids"],
                    "run_label": names,
                    "scored": True,
                    "status": "Passed" if cohort["passed"] else "Failed",
                    "score": cohort["score"],
                    "provenance": " · ".join(_provenance_bits(rows[0])),
                    "reason": (
                        "One answer-text similarity comparison shared by these "
                        f"responses, threshold {cohort['threshold']:.2f}. It "
                        "measures how similar the wording is, not whether the "
                        "answers agree on the law."
                    ),
                }
            ]
    return [
        {
            "response_id": row["response_id"],
            "run_label": identity.get(row["response_id"], {}).get("label"),
            "scored": _is_scored(row),
            "status": (
                ("Passed" if row["passed"] else "Failed")
                if _is_scored(row)
                else unmeasured_state(row)
            ),
            "score": row.get("score"),
            "provenance": " · ".join(_provenance_bits(row)),
            "reason": row.get("reason"),
        }
        for row in rows
    ]


def _metric_row_label(m: dict) -> str:
    """One-line summary of a metric, used as its expander label.

    Carries everything the old summary table's row carried, so opening the row
    is the only step between seeing a score and reading why it came out that
    way. The words come first and the colour matches them, so a colour scan and
    a careful read say the same thing.
    """
    name = m["metric_name"].strip()
    gaps = _gap_counts(m)

    if not m.get("scored", True):
        states = sorted(gaps)
        detail = (
            f"{gaps[states[0]]} response(s)" if len(states) == 1 else _gap_phrase(gaps)
        )
        return (
            f"**{name}** &nbsp; :gray[**{_headline_state(m)}**] "
            f"&nbsp; :gray[no score · {detail}]"
        )

    # One shared comparison is one result, however many rows store it.
    cohort = (
        consistency_cohort([r for r in m.get("raw_results", []) if measured(r)])
        if m["test_name"] == "consistency"
        else None
    )
    if cohort is not None:
        status = ":green[**Passed**]" if cohort["passed"] else ":red[**Failed**]"
        return (
            f"**{name}** &nbsp; {status} &nbsp; :gray[one comparison across "
            f"{len(cohort['response_ids'])} responses · similarity "
            f"{cohort['score']:.3f} · threshold {cohort['threshold']:.2f}]"
        )

    status = _STATE_MARKUP.get(m["state"], f":gray[**{m['state']}**]")
    label = (
        f"**{name}** &nbsp; {status} &nbsp; "
        f":gray[{m['pass_count']} of {m['measured_count']} measured responses passed"
        f" · score {m['score']:.3f} · threshold {m['threshold']:.2f}]"
    )

    # Only worth the space when the runs actually disagreed.
    if "min_score" in m and m["min_score"] != m["max_score"]:
        label += f" &nbsp; :gray[runs {m['min_score']:.3f} to {m['max_score']:.3f}]"

    if gaps:
        label += f" &nbsp; :gray[{_gap_phrase(gaps)}]"

    return label


def _is_failure(m: dict) -> bool:
    """A scored metric that did not meet its threshold.

    Not-scored metrics are excluded: a judge error is not a failure.
    """
    return m.get("scored", True) and not m["passed"]


def _render_metric_rows(
    metrics: list[dict],
    identity: dict | None = None,
    records: dict | None = None,
    failures_only: bool = False,
) -> None:
    """One expander per metric: the label is the summary row, the body is that
    metric's per-run detail.

    Failed metrics open by default, passed ones stay shut, so the reasons you
    need are on screen and the rest is one line each.
    """
    shown = [m for m in metrics if _is_failure(m)] if failures_only else metrics
    for m in shown:
        scored = m.get("scored", True)
        with st.expander(
            _metric_row_label(m),
            expanded=scored and not m["passed"],
        ):
            tooltip = METRIC_TOOLTIPS.get(m["metric_name"].strip(), "")
            if tooltip:
                st.caption(tooltip)
            _render_metric_body(m, identity, records)


def _render_consistency_summary(m: dict, identity: dict | None) -> bool:
    """Show one repeat-similarity comparison once, naming the responses in it.

    Consistency writes a row per response, all carrying the same comparison, so
    two rows are one observation rather than two verdicts. Rendered as one
    summary only when the stored rows say they belong to the same comparison;
    otherwise the caller falls back to showing them as stored.
    """
    cohort = consistency_cohort([r for r in m.get("raw_results", []) if measured(r)])
    if cohort is None:
        return False
    names = ", ".join(_identity_short(rid, identity) for rid in cohort["response_ids"])
    status = ":green[**Passed**]" if cohort["passed"] else ":red[**Failed**]"
    st.markdown(
        f"Answer-text similarity across {names}: **{cohort['score']:.3f}** "
        f"&nbsp; {status} &nbsp; :gray[threshold {cohort['threshold']:.2f}]"
    )
    st.caption(
        "One comparison shared by these responses, not one verdict each. It "
        "measures how similar the wording is, not whether the answers agree on "
        "the law."
    )
    return True


def _identity_short(response_id, identity: dict | None) -> str:
    entry = (identity or {}).get(response_id)
    return entry["short"] if entry else f"response {response_id}"


def _render_metric_body(
    m: dict,
    identity: dict | None = None,
    records: dict | None = None,
    evidence_for_passes: bool = False,
) -> None:
    """Per-response detail for one metric, in the question's own run order."""
    identity = identity or {}
    rows = _display_rows(m, identity)

    summarised = m["test_name"] == "consistency" and _render_consistency_summary(
        m, identity
    )

    n = m.get("n_runs", len(rows))
    if n > 1 and not summarised:
        st.caption(
            f"{m['measured_count']} of {n} responses measured; "
            "one selected score per response"
        )

    # A metric with no row at all for a response is not a pass for it.
    scored_ids = {r["response_id"] for r in rows}
    missing = [rid for rid in identity if rid not in scored_ids]
    if missing:
        names = ", ".join(_identity_short(rid, identity) for rid in sorted(missing))
        st.markdown(f":gray[**No stored result** for {names}.]")

    # A reason shared by several responses is said once here, and left off
    # those responses' own rows. A reason only one response has stays on it.
    gaps = Counter(
        (unmeasured_state(row), row.get("reason") or NOT_MEASURED)
        for row in m.get("not_scored_rows", [])
    )
    repeated = {pair: count for pair, count in gaps.items() if count > 1}
    for (state, reason), count in repeated.items():
        st.markdown(
            f":gray[**{state}** &nbsp; {count} responses, excluded from any "
            "mean and not a quality verdict:]"
        )
        _reason_block(reason)

    def covered(row: dict) -> bool:
        pair = (unmeasured_state(row), row.get("reason") or NOT_MEASURED)
        return not _is_scored(row) and pair in repeated

    if summarised:
        with st.expander("Individual stored rows for this comparison"):
            for raw in rows:
                _render_single_eval_result(raw, identity, records, evidence=False)
        return

    for raw in rows:
        _render_single_eval_result(
            raw,
            identity,
            records,
            show_reason=not covered(raw),
            evidence_for_passes=evidence_for_passes,
        )


# Readable headings for stored evidence fields. A field name is not always
# enough on its own: "Statements" does not say whose statements they are,
# which is the first thing a reviewer needs to know.
_DETAIL_HEADINGS: dict[str, str] = {
    "statements": "Reference answer statements",
    "points": "Reference answer statements",
    "claims": "Claims checked against the retrieved text",
}

# How a stored verdict reads on the page.
_VERDICT_WORDS: dict[str, str] = {
    "stated": "Stated",
    "contradicted": "Contradicted",
    "missing": "Not stated",
    "supported": "Supported",
    "unsupported": "Unsupported",
}

# Fields the verdict list already shows, so they are not repeated below it.
_VERDICT_FIELDS = frozenset(
    {"statements", "points", "claims", "contradicted_indexes", "contradiction_findings"}
)


def _verdict_items(details: dict) -> list[tuple[str, str, str]]:
    """One (verdict, statement or claim, quote from the response) per item.

    Reference Answer Agreement stores the statement text and the judge's
    verdicts as two parallel lists, so neither says much alone: the statements
    do not show how the response did, and the verdicts are bare indexes.
    Claim Support already stores the two together.
    """
    items = details.get("points") or details.get("claims")
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        return []
    statements = details.get("statements") or []
    contradicted = set(details.get("contradicted_indexes") or [])
    # The contradiction judge call looks for nothing else, so its quote is the
    # evidence for any contradiction the labelling call did not itself flag.
    quotes = {
        f["index"]: f["quote"]
        for f in details.get("contradiction_findings") or []
        if isinstance(f, dict) and f.get("contradicted") and f.get("quote")
    }
    rows = []
    for position, item in enumerate(items, start=1):
        index = item.get("index", position)
        label = "contradicted" if index in contradicted else item.get("label", "")
        text = item.get("claim") or (
            statements[index - 1] if 0 < index <= len(statements) else ""
        )
        quote = quotes.get(index) or item.get("quote", "")
        rows.append((_VERDICT_WORDS.get(label, label or "No verdict"), text, quote))
    return rows


def _render_details(details) -> None:
    """Stored per-claim evidence as prose, with the raw record still available."""
    if isinstance(details, dict):
        verdicts = _verdict_items(details)
        if verdicts:
            key = "points" if details.get("points") else "claims"
            st.markdown(f"**{_DETAIL_HEADINGS[key]}**")
            for position, (verdict, text, quote) in enumerate(verdicts, start=1):
                _reason_block(f"{position}. [{verdict}] {text}")
                if quote:
                    _reason_block(f"\u2003Response said: \u201c{quote}\u201d")
        for field, value in details.items():
            if verdicts and field in _VERDICT_FIELDS:
                continue
            heading = _DETAIL_HEADINGS.get(field, field.replace("_", " ").capitalize())
            if isinstance(value, str) and value:
                st.markdown(f"**{heading}**")
                _reason_block(value)
            elif isinstance(value, list) and all(isinstance(v, str) for v in value):
                st.markdown(f"**{heading}**")
                for item in value:
                    _reason_block(f"- {item}")
    with st.expander("Raw stored evidence"):
        st.json(details, expanded=False)


def _render_evidence(r: dict, rec: dict, heading: str) -> None:
    """The failure reason beside the answer it is about, for one response.

    The answer shown is the one this row scored, so a reason about response 80
    can never be read against response 72's text. Nothing here is recomputed:
    it is the stored reason, the stored evidence if the row has any, and the
    captured text.
    """
    with st.expander(f"Inspect evidence · {heading}"):
        st.caption("The stored reason for this score, beside this response's own text.")
        _reason_block(r.get("reason") or "")
        details = r.get("details")
        if details:
            _render_details(details)
        else:
            st.caption(
                "This stored score has no passage-level evidence, so the whole "
                "captured answer is shown for manual inspection."
            )
        answer_tab, worker_tab = st.tabs(
            ["Answer to the user", "Worker research report"]
        )
        with answer_tab:
            _captured_text(rec.get("actual_output") or "_(no output captured)_")
        with worker_tab:
            st.caption(
                "The Worker's report to the Manager. This is not the answer the "
                "user saw."
            )
            _captured_text(rec.get("research_output") or "_(no report captured)_")


def _render_single_eval_result(
    r: dict,
    identity: dict | None = None,
    records: dict | None = None,
    evidence: bool = True,
    show_reason: bool = True,
    evidence_for_passes: bool = False,
) -> None:
    """One raw eval result entry, headed by the response it belongs to."""
    heading = _response_heading(r.get("response_id", "unknown"), identity)
    passing = _is_scored(r) and bool(r["passed"])

    if _is_scored(r):
        status = ":green[**Passed**]" if r["passed"] else ":red[**Failed**]"
        threshold = r.get("threshold")
        detail = f"score {r['score']:.3f}" + (
            f" · threshold {threshold:.2f}" if threshold is not None else ""
        )
    else:
        status = f":gray[**{unmeasured_state(r)}**]"
        detail = "no score"

    with st.container(border=True):
        st.markdown(f"{status} &nbsp; {heading} &nbsp; :gray[{detail}]")
        if show_reason:
            _reason_block(r.get("reason") or "")

        if r.get("scope_note"):
            st.caption(r["scope_note"])
        rec = (records or {}).get(r.get("response_id"))
        # Question review focuses evidence on failures. Comparison also opens
        # passing answers so reviewers can investigate improvements. A passing row's answer
        # is still in "Response to user", the comparison and the research log,
        # and repeating every answer and Worker report once per check made the
        # page many times larger than the reading it supports.
        if evidence and rec is not None and (not passing or evidence_for_passes):
            _render_evidence(r, rec, heading)
        elif r.get("details"):
            _render_details(r["details"])
        if r.get("reference_sha256"):
            refs = r.get("scoring_config", {}).get("references", {})
            reference = refs.get(str(r["question_id"])) or refs.get(r["question_id"])
            if reference:
                with st.expander("Reference used for this score"):
                    st.caption(
                        f"Reference status: {r.get('reference_mode', 'unknown')}; "
                        f"fingerprint: {r['reference_sha256']}"
                    )
                    for statement in reference.get("statements", []):
                        st.write(statement)

        st.caption(" · ".join(_provenance_bits(r)))

        tools = r.get("tools_used")
        if tools:
            st.caption(f"Tools used: {', '.join(tools)}")

        if r.get("error"):
            st.error(r["error"])


def _strip_worker_prefix(name: str) -> str:
    """Remove the 'Worker: ' prefix from tool names for display."""
    return name.removeprefix("Worker: ")


# chat_mode -> (icon, label) used for tab labels and the execution context badge.
_CHAT_MODE_DISPLAY = {
    "research": ("🔎", "research"),
    "deep_research": ("🧭", "deep_research"),
    "conversational": ("💬", "conversational"),
}


def _chat_mode_badge(chat_mode: str) -> str:
    icon, label = _CHAT_MODE_DISPLAY.get(chat_mode, ("❔", chat_mode or "unknown"))
    return f"{icon} {label}"


def _log_section(title: str) -> None:
    """A label for one of the log's own sections.

    Deliberately unlike a Markdown heading: a captured answer carries its own
    # and ## headings, which render larger than anything this page could write,
    so a heading here would sit below the content it introduces.
    """
    st.markdown(
        f'<div style="border-left:3px solid #58a6ff;padding:2px 10px;'
        f"margin:18px 0 6px 0;font-size:0.75em;font-weight:700;"
        f'letter-spacing:0.08em;text-transform:uppercase;color:#58a6ff;">'
        f"{html.escape(title)}</div>",
        unsafe_allow_html=True,
    )


# Below this many characters an answer is shown at its natural height. A short
# clarification request in a fixed 420px box used to fill the screen while the
# answer it should be compared with sat below the fold.
_NATURAL_HEIGHT_CHARS = 1200


def _captured_text(text: str, height: int = 420) -> None:
    """Model output in a bordered box, scrolling only once it is long.

    The border marks where captured text starts and stops. A long answer is
    bounded so the next section stays on screen; a short one takes the room it
    needs and no more.
    """
    if len(text or "") <= _NATURAL_HEIGHT_CHARS:
        with st.container(border=True):
            st.markdown(text)
        return
    with st.container(border=True, height=height):
        st.markdown(text)


def _render_chat_interaction(records: list[dict], identity: dict | None = None) -> None:
    """
    raw chat interaction(s) for an LLM/question pair.
    A row/record in responses.db is one run.

    Tabs are labelled from the question group's own run numbering, so a tab
    called Run 2 is the same response the metrics above call Run 2.
    """
    if not records:
        st.info("No response records found in responses.db for this combination.")
        return

    identity = identity or _identity(records)
    run_tabs = st.tabs(
        [
            f"Run {identity.get(r['response_id'], {}).get('run', i + 1)}  "
            f"Response {r['response_id']}  "
            f"{_chat_mode_badge(r.get('chat_mode', 'research'))}  "
            f"({r['timestamp'][:19]})"
            for i, r in enumerate(records)
        ]
    )

    for tab, rec in zip(run_tabs, records, strict=True):
        with tab:
            st.caption(
                f"{_response_heading(rec['response_id'], identity)} · {_outcome_label(rec)}"
            )
            if rec.get("needs_clarification"):
                st.info(rec.get("clarification_question") or "Clarification requested")
            if rec.get("is_error"):
                st.error(rec.get("error_message") or "Request failed")
            if rec.get("max_turns_halted"):
                st.warning(
                    "Research limit reached: a research step stopped at the "
                    "tool-call limit, so it returned no findings."
                )
            st.caption(
                f"Reformatted: {bool(rec.get('reformatted'))}; request attempts: {rec.get('attempts', 'unknown')}"
            )
            # --- Execution Metadata ---
            _log_section("Execution context")
            cols = st.columns(5)
            with cols[0]:
                chat_mode = rec.get("chat_mode", "research")
                st.markdown(f"**Chat mode:** {_chat_mode_badge(chat_mode)}")
            with cols[1]:
                st.markdown(f"**Research Mode:** `{rec.get('research_mode', 'N/A')}`")
            with cols[2]:
                fallback = rec.get("fallback_used", False)
                st.markdown(f"**Fallback Used:** {'Yes' if fallback else 'No'}")
            with cols[3]:
                summarisation = rec.get("summarisation_used", False)
                summ_llm = rec.get("summarisation_llm", "")
                main_llm = rec.get("llm_name", "")
                if summarisation and summ_llm and summ_llm != main_llm:
                    st.markdown("**Summarisation:** Yes")
                    st.caption(f"Model: `{summ_llm}`")
                elif summarisation:
                    st.markdown("**Summarisation:** Yes *(main model)*")
                else:
                    st.markdown("**Summarisation:** No")
            with cols[4]:
                tool_seq = rec.get("tool_sequence") or []
                st.markdown(f"**Tool Sequence:** `{len(tool_seq)}` steps")
                if tool_seq:
                    # The opening of the sequence only. Printing all of a long
                    # one here fills the screen in a narrow column, and the
                    # Tools called section below lists every call in order.
                    display_seq = [_strip_worker_prefix(t) for t in tool_seq[:6]]
                    preview = " → ".join(display_seq)
                    if len(tool_seq) > 6:
                        preview += f" → and {len(tool_seq) - 6} more"
                    st.caption(preview)

            # --- Deep Research Plan ---
            research_plan = rec.get("research_plan")
            if chat_mode == "deep_research" and research_plan:
                with st.expander("📋 Deep Research Plan (as presented to the user)"):
                    steps = (
                        research_plan.get("steps")
                        if isinstance(research_plan, dict)
                        else None
                    )
                    if steps:
                        for i, step in enumerate(steps):
                            if isinstance(step, dict):
                                title = step.get("title") or f"Step {i + 1}"
                                detail = (
                                    step.get("detail")
                                    or step.get("brief")
                                    or step.get("description")
                                )
                                st.markdown(f"**{i + 1}. {title}**")
                                if detail:
                                    st.caption(detail)
                            else:
                                st.markdown(f"**{i + 1}.** {step}")
                    else:
                        st.json(research_plan, expanded=False)

            # --- LLM Answer ---
            _log_section("Answer to the user")
            actual = rec.get("actual_output", "")
            if actual:
                _captured_text(actual)
            else:
                st.caption("_(no output captured)_")

            # --- Research Output ---
            research_output = rec.get("research_output", "")
            if research_output:
                _log_section("Research output (worker findings)")
                _captured_text(research_output)

            # --- Summarisation Output ---
            summarisation_output: list = rec.get("summarisation_output") or []
            summarisation_used = rec.get("summarisation_used", False)
            if summarisation_output:
                _log_section(
                    f"Summarised context ({len(summarisation_output)} passages)"
                )
                # A long list of identical rows fills screens; one scrolling
                # box keeps the section after it within reach.
                with st.container(height=320):
                    for i, summary_text in enumerate(summarisation_output):
                        with st.expander(
                            f"Summarised Passage {i + 1}", expanded=i == 0
                        ):
                            st.markdown(summary_text)
            elif summarisation_used:
                st.info("Summarisation was used but no output was captured.")

            # --- Tools Called (sorted by tool_sequence start order) ---
            tools_called: list[dict] = [
                t
                for t in (rec.get("tools_called") or [])
                if t.get("name") != "Research Agent"
            ]
            if tools_called:
                _log_section(f"Tools called ({len(tools_called)})")
                st.caption(
                    "Captured order. The step view above preserves each delegation boundary."
                )
                # A long list of identical rows fills screens; one scrolling
                # box keeps the section after it within reach.
                with st.container(height=360):
                    for index, tool in enumerate(tools_called, 1):
                        name = _strip_worker_prefix(tool.get("name", "unknown"))
                        with st.expander(f"{index}. {name}"):
                            params = (
                                tool.get("input_parameters")
                                or tool.get("inputParameters")
                                or {}
                            )
                            st.caption("Tool arguments")
                            st.json(params, expanded=True)
                            output = tool.get("output")
                            if isinstance(output, str):
                                try:
                                    output = json.loads(output)
                                except (ValueError, TypeError):
                                    pass
                            if isinstance(output, (dict, list)):
                                st.json(output, expanded=False)
                            elif output:
                                st.code(str(output), language="text")
                            else:
                                st.caption(
                                    "No output captured; this is not proof of an empty search."
                                )
            else:
                st.caption("No tool calls captured.")

            # --- Case Law Context ---
            case_law_ctx: list[dict] = rec.get("case_law_context") or []
            if case_law_ctx:
                _log_section(f"Case law context ({len(case_law_ctx)} items)")
                for i, case_data in enumerate(case_law_ctx):
                    title = case_data.get("title", "Unknown Title")
                    ncn = case_data.get("ncn", "")
                    court = case_data.get("court", "")
                    date = case_data.get("date", "")
                    url = case_data.get("url", "")

                    meta_parts = [p for p in [ncn, court, date] if p]
                    meta_str = f" ({' | '.join(meta_parts)})" if meta_parts else ""
                    st.markdown(f"**{i + 1}. {title}{meta_str}**")
                    if url:
                        st.markdown(f"   🔗 [Link to judgment]({url})")

            # --- Retrieved Context ---
            contexts: list[str] = rec.get("retrieval_context") or []
            if contexts:
                _log_section(f"Retrieved context ({len(contexts)} items)")
                # A long list of identical rows fills screens; one scrolling
                # box keeps the section after it within reach.
                with st.container(height=360):
                    for i, ctx in enumerate(contexts):
                        st.markdown(f"**Context {i + 1}**")
                        st.code(ctx, language="text")
            else:
                st.caption("No retrieval context captured.")

            _log_section("Full record metadata")
            st.json(
                {
                    "response_id": rec.get("response_id"),
                    "experiment_id": rec.get("experiment_id"),
                    "gather_run_id": rec.get("gather_run_id"),
                    "timestamp": rec.get("timestamp"),
                    "llm_name": rec.get("llm_name"),
                    "summarisation_llm": rec.get("summarisation_llm", ""),
                    "question_id": rec.get("question_id"),
                    "research_mode": rec.get("research_mode"),
                    "chat_mode": rec.get("chat_mode"),
                    "research_plan": rec.get("research_plan"),
                    "fallback_used": rec.get("fallback_used"),
                    "summarisation_used": rec.get("summarisation_used"),
                    "tool_sequence": rec.get("tool_sequence", []),
                },
                expanded=False,
            )


# Colour and wording for the attribution flag, one entry per verdict from
# reports/attribution.py. Grey for "cannot say", which is a gap in the
# reference answers rather than a finding about the response.
_ATTRIBUTION_STYLE = {
    "tech": ("#d1242f", "Run did not finish"),
    "search": ("#bf8700", "The search did not find the law"),
    "model": ("#0969da", "Found the law, did not cite it"),
    # Not the pass green used elsewhere: this says no step lost any law, which
    # is narrower than the response being good, and the detail says so.
    "no_law_lost": ("#1a7f37", "No law lost"),
    "not_attributable": ("#6e7781", "Cannot say"),
}


@st.cache_data
def _reference_answers(mtime: float = 0.0) -> dict:
    """Reference answers keyed by question_id.

    Signed and unsigned, matching tests/eval/test_reference.py: excluding
    drafts would leave every question unattributable until sign off.

    ``_mtime`` busts the cache when the manifest is rewritten, the same way
    ``load_eval_results`` and ``load_responses`` do for the database.
    """
    return load_reference_answers()


def _reference_is_current(r: dict) -> bool:
    """Whether a stored metric row was scored against the reference in use now.

    A row scored against an answer that has since been corrected, or against a
    draft since signed off, is not comparable with the attribution shown beside
    it, so it is labelled rather than quietly averaged in.
    """
    if not r.get("reference_sha256"):
        return True
    reference = _reference_answers(_reference_manifest_mtime()).get(r["question_id"])
    if not reference:
        return False
    return reference_version(reference) == (
        r["reference_sha256"],
        r.get("reference_mode"),
    )


def _reference_manifest_mtime() -> float:
    path = ANSWERS_DIR / MANIFEST_NAME
    return path.stat().st_mtime if path.exists() else 0.0


def _render_attribution_flag(response_records: list[dict]) -> None:
    """One line saying which step lost the law, for one response.

    Always rendered when there are records to judge. An absent flag used to
    mean "no step lost any law" and was read as "nothing to report", so that
    case now has its own label rather than being silence. No score is shown
    here; the question's metric rows above hold those.
    """
    verdict = worst_attribution(
        response_records, _reference_answers(_reference_manifest_mtime())
    )
    if verdict is None:
        return

    colour, label = _ATTRIBUTION_STYLE.get(verdict["stage"], ("#6e7781", "Unclear"))
    ids = ", ".join(verdict["ids"][:4])
    if len(verdict["ids"]) > 4:
        ids += f" and {len(verdict['ids']) - 4} more"
    detail = html.escape(verdict["detail"] + (f": {ids}" if ids else ""))
    st.markdown(
        f'<div style="border-left:3px solid {colour};padding:4px 10px;'
        f'margin:0 0 10px 0;font-size:0.9em;">'
        f'<span style="color:{colour};font-weight:600;">{label}</span>'
        f'<span style="opacity:0.8;"> &nbsp; {detail}</span></div>',
        unsafe_allow_html=True,
    )
    note = caveat(verdict["stage"])
    if note:
        st.caption(note)


def _reference_status_line() -> str:
    """How many reference answers a lawyer has signed off, and how many are drafts.

    The three reference metrics score against both, so the reader needs to know
    how much of what they are looking at is measured against lawyer-approved
    law and how much against a draft.
    """
    references = _reference_answers(_reference_manifest_mtime())
    if not references:
        return "No reference answers yet, so the reference metrics score nothing."
    verified = sum(1 for r in references.values() if effective_verified(r))
    drafts = len(references) - verified
    return (
        f"Reference answers in use: {verified} signed off by a lawyer, "
        f"{drafts} unverified draft(s). A draft's scores measure agreement with "
        "its author, not legal correctness."
    )


def _render_outcomes(records):
    counts = _display_counts(outcome_counts(records))
    st.dataframe(
        [counts],
        hide_index=True,
        width="stretch",
        column_config=_column_help(OUTCOME_COLUMNS),
    )
    st.caption(
        "Outcomes count all attempts. A response received is captured text, not "
        "a judgement that the research finished. Research-limit and reformat "
        "flags may overlap with a received response or an error."
    )


STAGES = {
    "Answer coverage": {"reference_answer_agreement", "citation_agreement"},
    "Planning and search": {"plan_coverage"},
    "Worker evidence": {
        "claim_support",
        "citation_grounding",
        "citation_read",
        "citation_domain",
        "genuine_gap",
        "tool_usage",
        "step_completion",
    },
    "Final synthesis": {
        "response_groundedness",
        "report_integration",
        "citation_passthrough",
        "mandatory_structure",
    },
    "Repeat similarity": {"consistency"},
}


# The question text is trimmed hard so the columns that say what to look at
# next stay inside a 1440px window without horizontal scrolling.
_QUESTION_TEXT_CHARS = 70


# Short forms for the queue table's narrow column. The full wording is in the
# column tooltip and beside each response in the question's detail.
_WARNING_SHORT = {
    "Error": "error",
    "Clarification": "clarification",
    "No answer": "no text",
    "Turn-cap flags": "limit reached",
}


def _execution_warnings(counts: dict) -> str:
    """Runs that did not simply answer, as a short phrase for the queue table."""
    return ", ".join(
        f"{counts[stored]} {short}"
        for stored, short in _WARNING_SHORT.items()
        if counts[stored]
    )


def _question_summary(key, records, rows):
    qid, model, chat, research, question, experiment = key
    ids = {r["response_id"] for r in records}
    metrics = _aggregate_metrics([r for r in rows if r["response_id"] in ids])
    agreement = next(
        (m for m in metrics if m["test_name"] == "reference_answer_agreement"), None
    )
    counts = outcome_counts(records)
    metadata = question_metadata(records[0])
    failed = [m["metric_name"].strip() for m in metrics if _is_failure(m)]
    # The count comes first so it survives a narrow column, then as many names
    # as fit. The question's detail lists all of them.
    listed = ", ".join(failed[:3]) + (
        f", and {len(failed) - 3} more" if len(failed) > 3 else ""
    )
    listed = f"{len(failed)}: {listed}" if failed else ""
    gaps = _gap_states(metrics)
    return {
        "Question": f"Q{qid}: {question[:_QUESTION_TEXT_CHARS]}",
        "Failed checks": listed,
        "Measurement gaps": _gap_phrase(gaps)
        or ("no checks stored" if not metrics else ""),
        "Execution warnings": _execution_warnings(counts),
        "Responses": counts["Attempts"],
        "Reference agreement": (
            f"{agreement['pass_count']}/{agreement['measured_count']} measured"
            if agreement
            else "Not scored"
        ),
        "Needs attention": any(
            _is_failure(m) or m.get("not_scored_count") for m in metrics
        )
        or any(outcome(r) != "Answer" or r.get("max_turns_halted") for r in records)
        or not metrics,
        "Model": model,
        "Chat mode": chat,
        "Research mode": research,
        "Experiment": records[0].get("experiment", {}).get("label", experiment),
        "Type": metadata.get("test_type", ""),
    }


# Notes a reviewer typed, kept for the session and keyed by the whole question
# group, so returning to a question brings its own notes back and never another
# question's. Streamlit drops a widget's value as soon as the widget stops being
# drawn, which is what leaving the question does, so the text cannot live in the
# widget alone.
_NOTES_STATE = "review_notes"

REVIEW_FIELDS: tuple[tuple[str, str, str], ...] = (
    (
        "observed_problem",
        "Observed problem",
        "What you found wrong with the answer, in your own words.",
    ),
    (
        "evidence_passage",
        "Evidence passage",
        "The passage from the answer that shows it.",
    ),
    ("reviewer", "Reviewer", "Who reviewed this."),
)


def _review_notes(key) -> dict:
    """Draw the note fields for one question group and return what they hold.

    Kept in this browser session only. Nothing here is written to the database
    or sent anywhere; the downloads below are what carries a note onwards.
    """
    store = st.session_state.setdefault(_NOTES_STATE, {})
    saved = store.setdefault(key, {})
    group = hashlib.sha1(repr(key).encode()).hexdigest()[:12]

    def remember(field: str, widget_key: str) -> None:
        saved[field] = st.session_state[widget_key]

    notes = {}
    for field, label, help_text in REVIEW_FIELDS:
        widget_key = f"review_{field}_{group}"
        notes[field] = st.text_area(
            label,
            value=saved.get(field, ""),
            key=widget_key,
            help=help_text,
            on_change=remember,
            args=(field, widget_key),
        )
    st.caption(
        "Notes stay in this browser session and go into both downloads. They "
        "are not written to the database."
    )
    return notes


def _stage_of(test_name: str) -> str:
    """Which detail-page group a check is rendered under."""
    for title, keys in STAGES.items():
        if test_name in keys:
            return title
    return "Other checks"


def _stage_anchor(title: str) -> str:
    """The link target Streamlit gives this stage's heading."""
    return title.lower().replace(" ", "-")


def _stage_heading(title: str) -> None:
    """A stage heading, whose Streamlit anchor the failure summary links to."""
    st.markdown(f"#### {title}")


def _render_failure_summary(metrics: list[dict]) -> None:
    """What failed on this question, and where on the page to read about it."""
    failed = [m for m in metrics if _is_failure(m)]
    gaps = _gap_states(metrics)
    if not failed and not gaps:
        st.markdown(":green[**No failed checks and no measurement gaps.**]")
        return
    if failed:
        named = ", ".join(
            f"{m['metric_name'].strip()} (in "
            f"[{_stage_of(m['test_name'])}](#{_stage_anchor(_stage_of(m['test_name']))}))"
            for m in failed
        )
        st.markdown(f":red[**Failed checks:**] {named}")
    if gaps:
        st.markdown(f":gray[**Checks with no verdict:** {_gap_phrase(gaps)}]")
    st.caption("Failed checks are open below; the rest stay closed.")


def _response_state_line(response_id, metrics: list[dict]) -> str:
    """This response's own check results, in one line.

    A shared repeat-similarity comparison is left out: it belongs to the
    responses together, and the Repeat similarity section says so.
    """
    failed, gaps, passed = [], 0, 0
    for m in metrics:
        if m["test_name"] == "consistency" and consistency_cohort(
            [r for r in m.get("raw_results", []) if measured(r)]
        ):
            continue
        row = next(
            (r for r in _display_rows(m, None) if r["response_id"] == response_id),
            None,
        )
        if row is None:
            continue
        if not _is_scored(row):
            gaps += 1
        elif row["passed"]:
            passed += 1
        else:
            failed.append(m["metric_name"].strip())
    parts = [f":red[Failed: {', '.join(failed)}]"] if failed else []
    parts.append(f"Passed: {passed}")
    if gaps:
        parts.append(f"No verdict: {gaps}")
    return " · ".join(parts)


def _render_response_comparison(
    group: list[dict], identity: dict, metrics: list[dict]
) -> None:
    """Two of this question's answers next to each other, chosen by the reader.

    Repeat attempts are read against one another far more often than they are
    read alone, and stacking them vertically put the second one below the fold.
    Nothing is scored here: these are the stored answers as captured.
    """
    with st.expander(f"Compare two responses ({len(group)} captured)"):
        st.caption(
            "These are repeat attempts as captured. A later date does not make "
            "an answer an improvement, and unknown deployment conditions still "
            "apply to legacy responses."
        )
        by_id = {rec["response_id"]: rec for rec in group}
        ordered = sorted(by_id, key=lambda rid: identity[rid]["run"])
        left, right = st.columns(2)
        first = left.selectbox(
            "Left",
            ordered,
            index=0,
            format_func=lambda rid: identity[rid]["label"],
            key=f"compare_left_{group[0]['question_id']}",
        )
        second = right.selectbox(
            "Right",
            ordered,
            index=1 if len(ordered) > 1 else 0,
            format_func=lambda rid: identity[rid]["label"],
            key=f"compare_right_{group[0]['question_id']}",
        )
        side_by_side = st.toggle(
            "Side by side",
            value=True,
            key=f"compare_side_{group[0]['question_id']}",
            help="Turn off for one answer above the other on a narrow screen.",
        )
        panels = st.columns(2) if side_by_side else [st.container(), st.container()]
        for panel, rid in zip(panels, (first, second), strict=True):
            with panel:
                rec = by_id[rid]
                st.markdown(f"**{identity[rid]['label']}**")
                st.caption(_outcome_label(rec))
                st.markdown(_response_state_line(rid, metrics))
                _captured_text(rec.get("actual_output") or "_(no output captured)_")


def _question_detail(key, group, rows, scoring_label: str = "Latest stored"):
    st.subheader(f"Q{key[0]}: {key[4]}")
    metadata = question_metadata(group[0])
    st.caption(
        f"{key[1]} · {key[2]} · {key[3]} · experiment "
        f"{group[0].get('experiment', {}).get('label', key[5])}"
    )
    st.caption(metadata["metadata_source"])
    if metadata.get("eval_observation"):
        st.caption(str(metadata["eval_observation"]))
    identity = _identity(group)
    records_by_id = {rec["response_id"]: rec for rec in group}
    ids = set(records_by_id)
    selected_rows = [r for r in rows if r["response_id"] in ids]
    metrics = _aggregate_metrics(selected_rows)
    _render_failure_summary(metrics)
    _render_outcomes(group)
    # Above the known failure and the metric rows: a reviewer checking whether a
    # recorded failure came back needs the answer beside the description of it,
    # not several screens below. Both start closed so the page still opens on
    # the scores.
    with st.expander("Response to user"):
        # One panel per run, all closed when there are several: an answer runs
        # to pages, and the point of this section is comparing the runs, not
        # scrolling through the first to reach the second.
        for rec in sorted(group, key=lambda r: identity[r["response_id"]]["run"]):
            with st.expander(
                f"{identity[rec['response_id']]['label']} · {_outcome_label(rec)}",
                expanded=len(group) == 1,
            ):
                if rec.get("needs_clarification"):
                    st.info(
                        rec.get("clarification_question") or "Clarification requested"
                    )
                if rec.get("is_error"):
                    st.error(rec.get("error_message") or "Request failed")
                if rec.get("max_turns_halted"):
                    st.warning(
                        "Research limit reached: a research step stopped at the "
                        "tool-call limit, so it returned no findings."
                    )
                _captured_text(
                    rec.get("actual_output") or "_(no output captured)_", height=500
                )
    if len(group) > 1:
        _render_response_comparison(group, identity, metrics)
    with st.expander("Full research log"):
        _render_chat_interaction(group, identity)
    # Same heading level as the metric groups below, and before them: what this
    # question was recorded as failing is what those scores are checking for.
    if metadata.get("known_gap"):
        st.markdown("#### Previous known failure check")
        st.write(metadata["known_gap"])
    for title, keys in STAGES.items():
        subset = [m for m in metrics if m["test_name"] in keys]
        if subset:
            _stage_heading(title)
            _render_metric_rows(subset, identity, records_by_id)
    if key[3] != "case_law_only":
        with st.expander(
            "Did this run find the legislation the reference answer relies on?"
        ):
            st.caption(
                "For each Act the reference answer cites, this says where that Act was "
                "lost: no search in the run found it, or a search found it and the "
                "answer did not cite it. It reads the run's own stored tool calls and "
                "answer, against the reference answer as it stands now, not the version "
                "an older stored score was measured against. Acts only, so it says "
                "nothing about sections, judgments, or whether the answer is legally "
                "correct."
            )
            for rec in group:
                st.caption(_response_heading(rec["response_id"], identity))
                _render_attribution_flag([rec])
    search_rows = [r for rec in group for r in searches(rec)]
    with st.expander("Searches and deep-research steps"):
        st.caption(
            "Counts describe returned items, not relevance. A later nonempty search is evidence of further results, not proof that the question was resolved."
        )
        if search_rows:
            st.dataframe(
                search_summary(search_rows),
                hide_index=True,
                width="stretch",
                column_config=_column_help(SEARCH_SUMMARY_COLUMNS),
            )
            with st.expander(f"Every search call ({len(search_rows)})"):
                st.dataframe(
                    search_rows,
                    hide_index=True,
                    width="stretch",
                    column_config=_column_help(SEARCH_COLUMNS),
                )
        else:
            st.info("No captured search facts for these attempts.")
        for rec in group:
            steps = plan_steps(rec)
            approved = len((rec.get("research_plan") or {}).get("steps", []))
            st.caption(
                f"{_response_heading(rec['response_id'], identity)}: "
                f"{approved} approved steps, {len(steps)} captured delegations"
            )
            for step in steps:
                with st.expander(
                    f"Response {rec['response_id']}, step {step['Step']}: {step['Title']}"
                ):
                    st.caption(
                        f"Tools: {step['Tools']}; reformatted: {step['Reformatted']}; error: {step['Error'] or 'none'}"
                    )
                    st.markdown(step["Report"] or "No report")
    with st.expander("Export review evidence"):
        st.caption(
            "Includes answers, Worker reports, selected verdicts, question metadata and current reference statements. Draft agreement is not legal correctness."
        )
        notes = _review_notes(key)
        notes["metric_caught_it"] = None
        reference = _reference_answers(_reference_manifest_mtime()).get(key[0], {})
        pack = {
            "question": metadata,
            "selection": {
                "Model": key[1],
                "Chat mode": key[2],
                "Research mode": key[3],
                "Experiment": group[0].get("experiment", {}).get("label", key[5]),
                "Scoring selection": scoring_label,
            },
            "current_reference_statements": reference.get("statements", []),
            "current_reference_sha256": reference.get("reference_sha256"),
            "scored_reference_snapshots": {
                r["reference_sha256"]: r.get("scoring_config", {})
                .get("references", {})
                .get(str(key[0]))
                for r in selected_rows
                if r.get("reference_sha256") and r.get("scoring_config")
            },
            "responses": [
                {
                    **{
                        k: rec.get(k)
                        for k in (
                            "response_id",
                            "question",
                            "timestamp",
                            "actual_output",
                            "research_output",
                            "experiment_id",
                            "gather_run_id",
                            "needs_clarification",
                            "error_message",
                        )
                    },
                    "run_label": identity[rec["response_id"]]["label"],
                    "outcome": _outcome_label(rec),
                    "research_limit_reached": bool(rec.get("max_turns_halted")),
                }
                for rec in sorted(
                    group, key=lambda r: identity[r["response_id"]]["run"]
                )
            ],
            "checks": [
                {
                    "metric_name": m["metric_name"].strip(),
                    "state": _headline_state(m),
                    "scored": m.get("scored", True),
                    "threshold": m.get("threshold"),
                    "results": _check_results(m, identity),
                }
                for m in metrics
            ],
            "metrics": [
                {k: v for k, v in r.items() if k != "scoring_config"}
                for r in selected_rows
            ],
            "searches": search_rows,
            "review": notes,
        }
        json_column, markdown_column = st.columns(2)
        json_column.download_button(
            "Download review pack",
            json.dumps(pack, ensure_ascii=False, indent=2, default=str),
            file_name=f"q{key[0]}_review.json",
            mime="application/json",
        )
        markdown_column.download_button(
            "Download review report (.md)",
            review_markdown(pack),
            file_name=f"q{key[0]}_review.md",
            mime="text/markdown",
        )
        st.caption(
            "Both downloads describe the selection shown above and write nothing "
            "to the database."
        )


def _experiment_label(experiments: dict, value: str) -> str:
    return f"{experiments[value].get('label', value)} ({value[:8]})"


def _pick_experiment(container, label, options, key, experiments):
    """A selectbox that keeps its choice when the options around it change.

    Without a key Streamlit rebuilds the widget whenever its options change,
    which is what made the candidate reset to the first experiment as soon as
    anything else on the page moved.
    """
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    return container.selectbox(
        label,
        options,
        key=key,
        format_func=lambda value: _experiment_label(experiments, value),
    )


def _experiment_info(records, rows):
    """What one experiment holds: its questions, and which checks have scored them.

    Every experiment is listed here, including one nothing has been scored
    against, which is the case the comparison view cannot show.
    """
    experiments = {
        r["experiment_id"]: r.get("experiment", {})
        for r in records
        if r.get("experiment_id")
    }
    if not experiments:
        st.info(
            "No recorded experiments in this database. Legacy responses remain "
            "available in the question review."
        )
        return
    options = sorted(experiments, key=lambda e: _experiment_label(experiments, e))
    chosen = _pick_experiment(st, "Experiment", options, "info_experiment", experiments)
    group = [r for r in records if r.get("experiment_id") == chosen]
    ids = {r["response_id"] for r in group}
    stored = [row for row in rows if row["response_id"] in ids]
    st.caption(
        f"Gathered {min(r['timestamp'] for r in group)[:10]} to "
        f"{max(r['timestamp'] for r in group)[:10]}"
    )
    st.dataframe(
        [_display_counts(outcome_counts(group))],
        hide_index=True,
        width="stretch",
        column_config=_column_help(OUTCOME_COLUMNS),
    )
    table, pending = coverage(group, stored, METRICS)
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config=_column_help({**EXPERIMENT_INFO_COLUMNS, **METRIC_TOOLTIPS}),
    )
    st.caption(
        "Each check column counts the responses that have a stored result for "
        "that check, whatever the verdict was. A zero means the check has not "
        "been run against that question; n/a means it does not cover those "
        "runs, for example a deep research check on conversational runs."
    )
    if pending:
        st.markdown(
            f"**{len(pending)} check(s) cover a response here that has no "
            "stored result.** Score them with:"
        )
        st.code(
            f"python lex_eval/run_evals.py --experiment {chosen} "
            f"--metrics {' '.join(pending)}",
            language="bash",
        )
        st.caption(
            "Add --dry-run first to see what it would score. Results already "
            "stored are skipped. AI judge checks need OPENROUTER_API_KEY in "
            "lex_eval/.env."
        )
    else:
        st.caption(
            "Every check that covers these runs has a stored result for every "
            "response here."
        )
    with st.expander("Experiment conditions"):
        st.json(
            {
                k: v
                for k, v in experiments[chosen].get("config", {}).items()
                if k != "questions"
            }
        )


def _render_comparison_evidence(entries, sides, labels, records_by_id, changes):
    """The stored scores behind one check on one question, on both sides.

    Fed by the same selection the totals above are built from, so an
    individual verdict here always belongs to the counts it is part of. The
    rows are rendered by the question review's own metric panel, so a reason,
    its evidence and a missing result read the same in both views.
    """
    st.markdown("#### Evidence for one check")
    st.caption(
        "The stored scores behind the tables above, on the scoring version "
        "this comparison selected. Nothing here is rescored."
    )
    names = sorted(
        {entry["metric_name"] for entry in entries},
        # Name breaks the tie so a check the display order does not list keeps
        # the same place from one comparison to the next.
        key=lambda name: (_metric_sort_key({"metric_name": name}), name),
    )
    if not names:
        st.info("No matched check to inspect.")
        return
    check = st.selectbox("Check", names)
    for_check = [entry for entry in entries if entry["metric_name"] == check]
    with st.expander("Per question detail", expanded=True):
        st.caption(
            "Results for the selected check. Select a question below for its modes, responses and evidence."
        )
        st.table(
            [
                {
                    column: row.get(column, "Not compared")
                    for column in ("Question", "Baseline", "Candidate", "Change")
                }
                for row in changes
                if row["Check"] == check
            ]
        )
    key = st.selectbox(
        "Question",
        [entry["key"] for entry in for_check],
        format_func=lambda k: f"Q{k[0]} · {k[2]} · {k[3]}",
    )
    entry = next(item for item in for_check if item["key"] == key)
    st.caption(key[1])
    if entry["excluded"]:
        st.markdown(
            f":gray[**{entry['excluded']}**] &nbsp; :gray[These runs are in no "
            "total above.]"
        )
        return
    for column, label, side, total in zip(
        st.columns(2), labels, sides, entry["totals"], strict=True
    ):
        with column:
            st.markdown(f"**{label}**")
            identity = _identity([r for r in side if cohort_key(r) == key])
            st.markdown(_metric_row_label(total))
            _render_metric_body(
                total, identity, records_by_id, evidence_for_passes=True
            )


def _compare_experiments(records, rows):
    # An experiment with no stored metric result has nothing to compare, so it
    # is not offered here. Experiment info lists every experiment, scored or
    # not, and says what is missing.
    st.markdown(
        "Compare results from two recorded configurations on the questions "
        "both asked. Review changes in pass frequency and average score, then "
        "inspect the responses and stored reasons behind them."
    )
    scored = {row["response_id"] for row in rows}
    experiments = {
        r["experiment_id"]: r.get("experiment", {})
        for r in records
        if r.get("experiment_id") and r["response_id"] in scored
    }
    if len(experiments) < 2:
        st.info(
            "Comparison needs two recorded experiments with stored metric results. Experiments that have not been scored are listed under Experiment info. Legacy responses remain available in the question review; dates alone do not establish matching conditions."
        )
        st.button(
            "Review repeated responses instead",
            help="Opens the question review, where repeat attempts to the same "
            "question can be read side by side.",
            on_click=lambda: st.session_state.update(view="Review questions"),
        )
        return
    # The two experiments come first: everything below, including which
    # questions can be compared at all, follows from them.
    options = sorted(experiments, key=lambda e: _experiment_label(experiments, e))
    left, right = st.columns(2)
    baseline = _pick_experiment(
        left, "Baseline experiment", options, "compare_baseline", experiments
    )
    candidate = _pick_experiment(
        right,
        "Candidate experiment",
        [e for e in options if e != baseline],
        "compare_candidate",
        experiments,
    )
    sides = [
        [r for r in records if r.get("experiment_id") == exp]
        for exp in (baseline, candidate)
    ]
    shared = shared_cohorts(*sides)
    if not shared:
        st.info(
            "These two experiments have no question in common, asked with the "
            "same wording and the same modes, so there is nothing to compare."
        )
        return
    question_options = sorted({key[0] for key in shared})
    st.session_state["compare_questions"] = [
        q
        for q in st.session_state.get("compare_questions", [])
        if q in question_options
    ]
    chosen = st.multiselect(
        "Questions",
        question_options,
        key="compare_questions",
        format_func=lambda q: f"Q{q}",
        help="Only the questions both experiments asked, with the same wording "
        "and the same modes. Select none to use all of them.",
    )
    if chosen:
        sides = [[r for r in side if r["question_id"] in chosen] for side in sides]
    st.caption(
        "Matched question wording, snapshot and modes only. The latest shared scoring version is selected per metric; missing measurements remain visible. Two repeats describe observations, not statistical certainty."
    )
    scoped = apply_scope(rows)
    summary, changes, outcomes = compare(*sides, scoped)
    st.markdown("#### Coverage")
    st.dataframe(
        [summary],
        hide_index=True,
        width="stretch",
        column_config=_column_help(COMPARISON_SUMMARY_COLUMNS),
    )
    totals = sorted(
        metric_summary(*sides, scoped),
        key=lambda row: _metric_sort_key({"metric_name": row["Check"]}),
    )
    if totals:
        st.markdown("#### Results by check")
        st.dataframe(
            [change_counts(totals)],
            hide_index=True,
            width="stretch",
            column_config=_column_help(COMPARISON_CHANGE_COUNT_COLUMNS),
        )
        st.dataframe(
            totals,
            hide_index=True,
            width="stretch",
            column_config=_column_help(COMPARISON_METRIC_COLUMNS),
        )
        st.caption(
            "How often a check passed and its mean score can move in opposite "
            "directions, so both are shown. Neither says the answers are more "
            "legally correct."
        )
    st.markdown("#### Run outcomes")
    st.dataframe(
        [_display_counts(row) for row in outcomes],
        hide_index=True,
        width="stretch",
        column_config=_column_help(COMPARISON_OUTCOME_COLUMNS),
    )
    _render_comparison_evidence(
        matched_entries(*sides, scoped),
        sides,
        [_experiment_label(experiments, exp) for exp in (baseline, candidate)],
        {r["response_id"]: r for side in sides for r in side},
        changes,
    )
    with st.expander("Experiment conditions"):
        for exp in (baseline, candidate):
            st.write(_experiment_label(experiments, exp))
            st.json(
                {
                    k: v
                    for k, v in experiments[exp].get("config", {}).items()
                    if k != "questions"
                }
            )
    st.markdown("#### Full research log for one response")
    matched_ids = {r["response_id"] for side in sides for r in side}
    inspect = [r for r in records if r["response_id"] in matched_ids]
    selected = st.selectbox(
        "Inspect a response",
        inspect,
        format_func=lambda r: f"Response {r['response_id']}: Q{r['question_id']} ({r.get('experiment', {}).get('label', '')})",
    )
    _render_chat_interaction([selected])


def main() -> None:
    st.set_page_config(page_title="LexChat Eval", layout="wide")
    st.title("LexChat evaluation")
    st.caption(_reference_status_line())
    # The database path and the scoring selection are set once a session and
    # then repeat on every screen, so they are folded away and summarised in
    # the caption below the filters instead of leading the page.
    settings = st.expander("Settings: database and scoring selection")
    with settings:
        db_path = Path(st.text_input("Results database", str(RESPONSES_DB)))
    if not db_path.exists():
        st.info(
            "No results at this path. Expected a DuckDB file, or a directory "
            "of Parquet written by the deploy build."
        )
        return
    records, rows = load_dashboard(str(db_path), _results_mtime(db_path))
    if not records:
        st.info("No response attempts in this database.")
        return
    view = st.radio(
        "View",
        ["Review questions", "Compare experiments", "Experiment info"],
        horizontal=True,
        key="view",
    )
    if view == "Review questions":
        st.markdown(
            "Investigate individual answers and the evidence behind their "
            "scores. Select a question to review its failures, repeated "
            "attempts and research activity."
        )
    # The comparison view chooses its own experiments and then its own
    # questions from what those two share, so the filters below, which narrow
    # the records first, do not apply to it.
    if view == "Compare experiments":
        _compare_experiments(records, rows)
        return
    if view == "Experiment info":
        _experiment_info(records, rows)
        return
    active = {"Database": db_path.name}
    cols = st.columns(3)
    for col, field, label in zip(
        cols,
        ("llm_name", "chat_mode", "research_mode"),
        ("Model", "Chat mode", "Research mode"),
        strict=True,
    ):
        options = sorted({r.get(field) or "unknown" for r in records})
        selected = col.selectbox(label, ["All", *options])
        active[label] = selected
        if selected != "All":
            records = [r for r in records if (r.get(field) or "unknown") == selected]
    selected_questions = st.multiselect(
        "Questions",
        sorted({r["question_id"] for r in records}),
        format_func=lambda q: f"Q{q}",
    )
    if selected_questions:
        records = [r for r in records if r["question_id"] in selected_questions]
    ids = {r["response_id"] for r in records}
    rows = [r for r in rows if r["response_id"] in ids]
    experiments = {
        r.get("experiment_id")
        or "Legacy": r.get("experiment", {}).get("label", "Legacy, condition unknown")
        for r in records
    }
    exp = st.selectbox(
        "Experiment",
        ["All", *sorted(experiments)],
        format_func=lambda e: e if e == "All" else f"{experiments[e]} ({e[:8]})",
    )
    active["Experiment"] = exp
    if exp != "All":
        records = [r for r in records if (r.get("experiment_id") or "Legacy") == exp]
    ids = {r["response_id"] for r in records}
    rows = [r for r in rows if r["response_id"] in ids]
    runs = {
        r["scoring_run_id"]: r.get("scoring_label", r["scoring_run_id"])
        for r in rows
        if r.get("scoring_run_id")
    }
    with settings:
        scoring = st.selectbox(
            "Scoring selection",
            ["Latest stored", *sorted(runs)],
            format_func=lambda r: r if r == "Latest stored" else f"{runs[r]} ({r[:8]})",
        )
        st.caption(
            "One selected score per response and metric. Legacy scorer versions are unknown. An explicit scoring run shows its historical reference; the latest view excludes outdated reference verdicts."
        )
    scoring_label = (
        scoring if scoring == "Latest stored" else f"{runs[scoring]} ({scoring[:8]})"
    )
    active["Scoring"] = scoring_label
    if scoring != "Latest stored":
        rows = [r for r in rows if r.get("scoring_run_id") == scoring]
    else:
        rows = current_reference_rows(
            rows, _reference_answers(_reference_manifest_mtime())
        )
    rows = apply_scope(latest_results(rows))
    st.caption(" · ".join(f"{name}: {value}" for name, value in active.items()))
    if records:
        st.caption(
            f"Gathered {min(r['timestamp'] for r in records)[:10]} to {max(r['timestamp'] for r in records)[:10]}"
        )
    groups = question_groups(records)
    summaries = {
        key: _question_summary(key, group, rows) for key, group in groups.items()
    }
    left, right = st.columns(2)
    attention = left.checkbox(
        "Only questions needing attention",
        help="Hides questions where every run answered and every metric passed. "
        + NEEDS_ATTENTION_HELP,
    )
    configuration = right.checkbox(
        "Show run configuration columns",
        help="Model, chat mode, research mode, experiment and question type. "
        "Every row in one selection usually repeats them, and the selected "
        "question's header says them.",
    )
    keys = [
        key for key, row in summaries.items() if not attention or row["Needs attention"]
    ]
    keys.sort(key=lambda k: (not summaries[k]["Needs attention"], k))
    table = [
        {
            name: value
            for name, value in summaries[k].items()
            if configuration or name not in CONFIGURATION_COLUMNS
        }
        for k in keys
    ]
    # Clicking a row opens that question below, so the queue and the detail
    # view are the same control rather than two selections to keep in step.
    event = st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config=_column_help(QUESTION_COLUMNS, QUESTION_COLUMN_WIDTHS),
        on_select="rerun",
        selection_mode="single-row",
        key="question_table",
    )
    if not keys:
        st.info("No questions match this selection.")
        return
    chosen = getattr(event, "selection", {}).get("rows") if event is not None else None
    selected = st.selectbox(
        "Inspect question",
        keys,
        index=chosen[0] if chosen and chosen[0] < len(keys) else 0,
        format_func=lambda k: f"Q{k[0]} · {k[1]} · {k[2]} · {k[3]} · {k[5][:8]}",
    )
    _question_detail(selected, groups[selected], rows, scoring_label)


if __name__ == "__main__":
    main()
