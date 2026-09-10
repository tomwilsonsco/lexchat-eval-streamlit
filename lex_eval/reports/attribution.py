"""
Which step of a run is responsible for a poor answer?

A reviewer looking at a column of red metric scores cannot tell whether the
model reasoned badly or whether it was never shown the law. Those are different
bugs owned by different people. This module answers that for one response, from
data already captured, so the dashboard can say it in a line beside the scores.

Four verdicts, worst first:

``tech``
    The run did not finish cleanly: it errored, hit its tool call limit, or a
    tool call came back as an API error. Nothing below it means much.
``search``
    No tool call in the run turned up law the reference answer relies on.
``model``
    A tool call did turn that law up, and the answer does not cite it.
``no_law_lost``
    Every Act the reference answer relied on was turned up by a search and
    cited in the answer, so no step lost any law. **This is not a pass.** A
    response can cite the right Acts and the wrong sections of them and still
    land here; the question's metric rows are what judge that.
``not_attributable``
    Nothing Citation Agreement could measure: no reference answer, an answer
    too short to judge, or a reference that records no legislation. Not a
    verdict that the response was fine.

Every response gets one of the five, so the flag is never absent. Silence was
the one state a reader had to interpret, and it was read as approval.

**Why ``search`` names the search step and not the corpus.** "The law never
turned up" has three possible causes, and two are ruled out for this dataset
rather than by this code:

- *A gap in the corpus.* Ruled out on 1 Sep 2026: all 18 Acts then in this
  bucket were requested from the LEX API by identifier and all 18 came back.
  See TOM_TO_DO.md finding 41.
- *LexChat's jurisdiction filter dropping good results* (finding 41). Ruled out
  from the captured data itself: that filter can only keep results with an
  empty ``extent``, and 7,392 of the 8,345 search results captured in
  ``responses.db`` carry a non-empty one. ``utils/audit_capture.py`` sets no
  jurisdiction on the chat payload, and this confirms none was applied.

What remains is the search step: the queries LexChat chose, or the five result
cap. **Re-check both if the corpus changes or the eval starts setting a
jurisdiction filter**, because this verdict is only as good as they are.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..metrics.citation_agreement import (
    attributable,
    attribute_missing_acts,
    cited_provisions,
    reference_acts,
)
from ..reference.store import load_reference_answers
from ..utils.test_helpers import record_to_test_case

# Prefix LexChat writes into a tool result when the LEX API call itself failed.
_TOOL_ERROR_PREFIX = "Error executing tool"

TECH = "tech"
SEARCH = "search"
MODEL = "model"
NO_LAW_LOST = "no_law_lost"
NOT_ATTRIBUTABLE = "not_attributable"

# Worst first. A run that halted early is reported as halted even if it also
# missed an Act, because the missing Act is probably a consequence of it.
# ``no_law_lost`` outranks ``not_attributable`` so that a group holding one
# measurable run and one that could not be measured reports the measurement.
_ORDER = [TECH, SEARCH, MODEL, NO_LAW_LOST, NOT_ATTRIBUTABLE]

# Shown under the flag when the fault is not the model's, so a reader does not
# take the question's metric scores as a verdict on its legal reasoning.
_CAVEAT = {
    TECH: (
        "No Act was checked for this run: it stopped early, so there is nothing "
        "to attribute. The question's metric scores measure the partial answer "
        "it did produce."
    ),
    SEARCH: (
        "No search in this run found the Acts listed above, so the question's "
        "metric scores show what the answer did without them."
    ),
}


def _tech_detail(record: Dict[str, Any]) -> Optional[str]:
    """Why this run did not finish cleanly, or None if it did."""
    if record.get("is_error"):
        return record.get("error_message") or "the run errored"
    if record.get("max_turns_halted"):
        return "the research stopped at its tool call limit"
    failed = [
        t.get("name", "tool")
        for t in record.get("tools_called") or []
        if str(t.get("output") or "").lstrip().startswith(_TOOL_ERROR_PREFIX)
    ]
    if failed:
        return f"a tool call returned an API error ({failed[0]})"
    return None


def attribution_for_record(
    record: Dict[str, Any], references: Optional[Dict[int, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """The verdict for one response record. Always one of the five stages.

    *references* is the reference answers keyed by question_id. Pass one in to
    avoid re-reading them for every record.
    """
    detail = _tech_detail(record)
    if detail:
        return {"stage": TECH, "detail": detail, "ids": []}

    if references is None:
        references = load_reference_answers()
    reference = references.get(record.get("question_id"))
    if not reference:
        return {
            "stage": NOT_ATTRIBUTABLE,
            "detail": "no reference answer for this question",
            "ids": [],
        }

    # The same gate Citation Agreement scores behind. Without it the flag can
    # attribute a clarification request, or a question whose reference answer the
    # metric declined to measure, and contradict the row beneath it.
    if not attributable(reference, record.get("actual_output") or ""):
        return {
            "stage": NOT_ATTRIBUTABLE,
            "detail": "nothing Citation Agreement could measure for this response",
            "ids": [],
        }

    expected_acts = reference_acts(reference)
    test_case = record_to_test_case(record)
    never_retrieved, retrieved_not_cited = attribute_missing_acts(
        expected_acts,
        cited_provisions(test_case.actual_output or ""),
        test_case.tools_called or [],
    )
    if never_retrieved:
        return {
            "stage": SEARCH,
            "detail": "the search never turned up",
            "ids": never_retrieved,
        }
    if retrieved_not_cited:
        return {
            "stage": MODEL,
            "detail": "turned up by a search but not cited in the answer",
            "ids": retrieved_not_cited,
        }
    return {
        "stage": NO_LAW_LOST,
        "detail": (
            "every Act the reference answer relied on was found and cited; "
            "the question's metric scores judge what the answer did with them"
        ),
        "ids": [],
    }


def worst_attribution(
    records: List[Dict[str, Any]],
    references: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """The worst verdict across repeat runs of the same question.

    Worst rather than most common: one run losing the law is worth seeing even
    if another run found it, since it is the same question either way.
    """
    verdicts = [attribution_for_record(r, references) for r in records]
    if not verdicts:
        return None
    return min(verdicts, key=lambda v: _ORDER.index(v["stage"]))


def caveat(stage: str) -> str:
    """The sentence to show under the flag, or "" for stages that do not excuse
    the question's metric scores."""
    return _CAVEAT.get(stage, "")
