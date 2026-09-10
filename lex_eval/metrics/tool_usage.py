"""
Custom metric to validate that the LLM used all expected legislation tools
*and* invoked them in the correct phase order.

Presence scoring, 1/3 for each of the three required tools:
    - delegate_research
    - Worker: search_legislation
    - Worker: search_legislation_sections

Order scoring (legislation_only mode only), the Worker tools must appear in
the phase order mandated by the Worker system prompt
(see ``LexChat/server_py/src/prompts.py``):

    1. Worker: search_legislation          (Phase 1, DISCOVER)
    2. Worker: search_legislation_sections (Phase 2, RETRIEVE PROVISIONS)
    3. Worker: get_legislation_text        (Phase 3, FALLBACK, optional)

A response must not only start the phases in this order, it must not loop
back to an earlier phase once a later one has begun. For example, calling
search_legislation again after search_legislation_sections has already
started is a violation, even though search_legislation's *first* call was
correctly before search_legislation_sections's first call. This catches
interleaved, multi-round re-querying that a first-occurrence-only check
would miss entirely.

The loop-back rule applies to single-shot ``research`` mode only.

For ``chat_mode == "conversational"`` it is not applied either. A single
conversational turn routinely asks about several instruments ("list the SSIs
made under this section"), and answering one means going back to search after
reading the first Act's sections. That is the same multi-Act interleaving the
deep-research exemption below was written for, and it accounted for 16 of the
22 order failures in the August 2026 question set, none of which was a
research fault.

For ``chat_mode == "deep_research"``, the loop-back rule is not applied: it
is checked per plan step instead of per run, and only for first-occurrence
order, not the no-revisit rule, since a single deep-research step routinely
covers more than one Act and legitimately interleaves discovery and
retrieval across them (see ``_check_tool_order`` for the measurement this is
based on). A step is still required to have called each earlier phase at
least once before a later one, e.g. a step that goes straight to
search_legislation_sections with no search_legislation call in that step is
a violation, since the top-level presence score only checks the whole run,
not each step.

Final score:
    - 1.0  all three required tools present AND correct order
    - 0.5  all three required tools present BUT wrong order
    - <1.0 one or more required tools missing (proportional; order is reported
           for transparency but does not cap the score further)
"""

from .base import BaseMetric
from ..testcase import LLMTestCase
from typing import List, Optional, Set

# The three tools that must all be present for a full score
REQUIRED_TOOLS: list[str] = [
    "delegate_research",
    "Worker: search_legislation",
    "Worker: search_legislation_sections",
]

PER_TOOL_SCORE = round(1 / len(REQUIRED_TOOLS), 10)

# Expected invocation order of Worker legislation tools, per the Worker system
# prompt phases. `get_legislation_text` is the optional fallback (Phase 3) and
# is NOT a required tool, but if it is called it must come after
# `search_legislation_sections`.
EXPECTED_TOOL_ORDER: list[str] = [
    "Worker: search_legislation",
    "Worker: search_legislation_sections",
    "Worker: get_legislation_text",
]

# Score assigned when all required tools are present but invoked out of order.
ORDER_VIOLATION_SCORE = 0.5


def _first_occurrence(tool_sequence: List[str], name: str) -> Optional[int]:
    """Return the index of the first occurrence of *name* in *tool_sequence*, or None."""
    for i, t in enumerate(tool_sequence):
        if t == name:
            return i
    return None


def _segment_by_delegation(tool_sequence: List[str]) -> List[List[str]]:
    """Split a flat ``tool_sequence`` into one Worker-tool list per delegation,
    using ``delegate_research`` entries as segment boundaries.

    ``audit_capture.py`` emits one ``delegate_research`` entry per delegation,
    which for a deep-research run means one per approved plan step, so this
    recovers step boundaries without needing any new capture field.
    """
    segments: List[List[str]] = []
    current: List[str] = []
    for t in tool_sequence:
        if t == "delegate_research":
            if current:
                segments.append(current)
            current = []
        else:
            current.append(t)
    if current:
        segments.append(current)
    return segments


def _check_first_occurrence_order(
    worker_seq: List[str],
    require_prerequisites: bool = False,
) -> tuple[bool, str, list[tuple[str, int]]]:
    """First-occurrence-only order check: each expected tool's first
    appearance must come in phase order. Does not penalise revisiting an
    earlier phase later in the sequence.

    ``present`` only lists phases that appear at all, so on its own this
    check is silent about a phase skipped entirely, e.g. a segment that goes
    straight to ``search_legislation_sections`` with no ``search_legislation``
    call has one present phase and no pair to compare, so it passes
    trivially. Pass ``require_prerequisites=True`` (used for a deep-research
    step, where nothing else checks that step's own presence) to also fail
    when a later phase is present but an earlier one it depends on is not.

    Returns ``(order_ok, detail, present)`` where *present* is the list of
    ``(short_name, index)`` pairs for tools that appeared at all, reusable by
    a stricter caller that also wants to check for revisits.
    """
    present: list[tuple[str, int]] = []
    present_names: set[str] = set()
    for name in EXPECTED_TOOL_ORDER:
        idx = _first_occurrence(worker_seq, name)
        if idx is not None:
            present.append((name.replace("Worker: ", ""), idx))
            present_names.add(name)

    if require_prerequisites and present:
        last_phase = max(EXPECTED_TOOL_ORDER.index(f"Worker: {n}") for n, _ in present)
        missing = [
            EXPECTED_TOOL_ORDER[i].replace("Worker: ", "")
            for i in range(last_phase)
            if EXPECTED_TOOL_ORDER[i] not in present_names
        ]
        if missing:
            return (
                False,
                f"{present[-1][0]} called without {', '.join(missing)} first",
                present,
            )

    order_ok = all(present[i][1] < present[i + 1][1] for i in range(len(present) - 1))
    if order_ok:
        return True, " → ".join(name for name, _ in present), present

    for i in range(len(present) - 1):
        if present[i][1] >= present[i + 1][1]:
            return False, f"{present[i + 1][0]} called before {present[i][0]}", present
    return False, "order violation", present


def _check_tool_order(
    tool_sequence: Optional[List[str]],
    research_mode: str,
    chat_mode: str = "research",
) -> tuple[bool, str]:
    """Validate that Worker legislation tools appear in the expected phase order.

    For ``chat_mode == "deep_research"``, checks first-occurrence order
    independently within each plan step (segmenting on ``delegate_research``
    boundaries) and stops there. Measured against six real deep-research
    responses, the stricter "never loop back to an earlier phase" rule below
    fails 13 of 25 real step-segments, because a single step routinely covers
    more than one Act, discovering and retrieving each in whatever order the
    Worker finds useful. First-occurrence-only order failed none of the same
    25 segments while still catching a step that skipped straight to
    retrieval before any discovery, so that's the check applied per step.

    For every other mode, also checks that the Worker never loops back to an
    earlier phase once a later one has begun, across the whole run.

    Only the ``Worker:``-prefixed entries are considered, ``delegate_research``
    is a Manager-level call and is excluded from the phase ordering.

    Returns ``(order_ok, detail)`` where *detail* is a short human-readable
    description of the observed order (or the first violation).
    """
    if research_mode != "legislation_only":
        return True, "skipped (non-legislation mode)"
    if not tool_sequence:
        return True, "n/a (no tool_sequence captured)"

    if chat_mode == "deep_research":
        segments = [
            [t for t in seg if t.startswith("Worker:")]
            for seg in _segment_by_delegation(tool_sequence)
        ]
        segments = [seg for seg in segments if seg]
        if not segments:
            return True, "n/a (no Worker tools in sequence)"
        for i, seg in enumerate(segments, 1):
            order_ok, detail, _ = _check_first_occurrence_order(
                seg, require_prerequisites=True
            )
            if not order_ok:
                return False, f"step {i}: {detail}"
        return True, f"{len(segments)} step(s), first-occurrence order OK"

    worker_seq = [t for t in tool_sequence if t.startswith("Worker:")]
    if not worker_seq:
        return True, "n/a (no Worker tools in sequence)"

    order_ok, detail, present = _check_first_occurrence_order(worker_seq)
    if not order_ok:
        return False, detail

    if chat_mode == "conversational":
        return True, " → ".join(name for name, _ in present)

    # First occurrences are in order. Now check the Worker didn't loop back to
    # an earlier phase after a later phase had already started, e.g. calling
    # search_legislation again after search_legislation_sections has begun.
    for i in range(len(present) - 1):
        earlier_name, _ = present[i]
        later_name, later_idx = present[i + 1]
        earlier_full = f"Worker: {earlier_name}"
        for j in range(later_idx + 1, len(worker_seq)):
            if worker_seq[j] == earlier_full:
                return False, (
                    f"{earlier_name} called again at step {j + 1} after "
                    f"{later_name} had already started (step {later_idx + 1})"
                )

    return True, " → ".join(name for name, _ in present)


class ToolUsageMetric(BaseMetric):
    """
    Scores tool usage by awarding 1/3 for each required tool present, and
    (for ``legislation_only`` mode) additionally validates that the Worker
    tools were invoked in the correct phase order and that the Worker never
    looped back to an earlier phase after a later one had begun.

    Score:
        - 1.0  all three required tools present AND correct order
        - 0.5  all three required tools present BUT wrong order
        - <1.0 one or more required tools missing (proportional; order is
               reported for transparency but does not cap the score further)

    Passes when ``score >= threshold`` (default threshold 1.0), so any order
    violation or missing tool is a fail.

    Args:
        threshold: Minimum score to pass (default 1.0).
        research_mode: ``legislation_only`` (default), ``case_law_only``, or
            ``legislation_and_case_law``. The order check only runs for
            ``legislation_only``.
        tool_sequence: Optional ordered list of tool names as captured by
            ``audit_capture`` (e.g. ``["delegate_research", "Worker:
            search_legislation", ...]``). When omitted, only presence is
            scored.
        chat_mode: ``research`` (default), ``conversational``, or
            ``deep_research``. For ``deep_research``, order is checked per
            plan step (segmented on ``delegate_research`` boundaries) using
            first-occurrence order only, not the no-revisit rule, see
            ``_check_tool_order``.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        research_mode: str = "legislation_only",
        tool_sequence: Optional[List[str]] = None,
        chat_mode: str = "research",
    ):
        self.threshold = threshold
        self.research_mode = research_mode
        self.tool_sequence = tool_sequence
        self.chat_mode = chat_mode
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        tools_used: Set[str] = set()
        # Tools whose completion event was never received from the stream (server-side
        # stream fault, the LLM did call them, but the result was never streamed back).
        incomplete_tools: Set[str] = set()

        if test_case.tools_called:
            for tool in test_case.tools_called:
                tools_used.add(tool.name)
                output = tool.output or ""
                if isinstance(output, str) and "no_completion_event" in output:
                    incomplete_tools.add(tool.name)

        if self.research_mode == "case_law_only":
            # A named judgment can be opened directly; an empty search is still
            # a valid attempt. Completion and claim support are separate checks.
            delegated = "delegate_research" in tools_used
            researched = bool(
                tools_used & {"Worker: search_case_law", "Worker: get_case_law_text"}
            )
            self.score = (int(delegated) + int(researched)) / 2
            self.success = self.score >= self.threshold
            self.reason = f"Case-law tools: delegated research {delegated}; searched or opened a judgment {researched}."
            return self.score

        present = [t for t in REQUIRED_TOOLS if t in tools_used]
        missing = [t for t in REQUIRED_TOOLS if t not in tools_used]

        all_required_present = len(present) == len(REQUIRED_TOOLS)

        # --- Presence score (1/3 per required tool) ---
        presence_score = len(present) / len(REQUIRED_TOOLS)

        # --- Order check (legislation_only only, when tool_sequence supplied) ---
        order_ok, order_detail = _check_tool_order(
            self.tool_sequence, self.research_mode, self.chat_mode
        )

        # --- Final score ---
        if all_required_present and not order_ok:
            # All tools present but wrong order → cap at ORDER_VIOLATION_SCORE.
            self.score = ORDER_VIOLATION_SCORE
        else:
            self.score = presence_score

        self.success = self.score >= self.threshold

        # --- Reason ---
        parts = [f"{t}: {'✓' if t in tools_used else '✗'}" for t in REQUIRED_TOOLS]
        self.reason = (
            f"Score {self.score:.3f} ({len(present)}/{len(REQUIRED_TOOLS)} tools used). "
            + " | ".join(parts)
        )
        if missing:
            self.reason += f" | Missing: {missing}"

        # Order section, always shown for legislation_only when a sequence
        # was supplied, even on missing-tool fails, for transparency.
        if self.research_mode == "legislation_only" and self.tool_sequence:
            order_icon = "✓" if order_ok else "✗"
            self.reason += f" | Order: {order_icon} {order_detail}"
            if all_required_present and not order_ok:
                self.reason += f" (expected: {' → '.join(n.replace('Worker: ', '') for n in EXPECTED_TOOL_ORDER)})"

        if incomplete_tools:
            self.reason += (
                f" | WARNING, stream incomplete (no result event received) for: "
                f"{sorted(incomplete_tools)}. LLM called the tool correctly; "
                f"server failed to return the completion event."
            )

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Tool Usage"
