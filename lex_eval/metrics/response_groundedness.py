"""
Response groundedness metric.

Scores whether the final response to the user is strictly grounded in
the research agent's output, with no hallucinated facts.
"""

from __future__ import annotations

import difflib
import json
from typing import Literal

from .base import BaseMetric
from ..testcase import LLMTestCase
from pydantic import BaseModel

# Above this similarity, the final response is a near-verbatim relay of the
# research output: grounding is a provable fact, not a judgement call, so the
# judge is not invoked. glm-5.2 (which follows the Manager prompt's "do NOT
# condense, summarise, or restructure" instruction literally) measures
# 0.983-1.000 on stored runs; mistral-large-3 (which paraphrases) measures
# 0.095-0.763. 0.95 sits in the gap between the two clusters.
_NEAR_VERBATIM_THRESHOLD: float = 0.95

# Wrapper for the approved plan's scope note, added to the prompt so the judge
# treats it as a source alongside the research output. Without it, a true
# statement like "case law was excluded under the approved research plan" reads
# as unsupported, because the scope lives in the plan and never appears in the
# research output itself.
_SCOPE_BLOCK_TEMPLATE = """
Approved Research Scope (from the research plan, mode: {research_mode}):
{scope_note}

The response may describe this scope, for example by saying that case law was
excluded. Treat such a statement as grounded when it matches the scope above,
even though it does not appear in the research output.
"""


class _GroundednessJudgement(BaseModel):
    analysis: str
    verdict: Literal["pass", "fail"]
    reason: str


_PROMPT_TEMPLATE = """You are an expert legal evaluator. Your task is to decide whether a final response is strictly grounded in the provided research output.

Research Output:
{research_output}
{scope_block}
Final Response:
{actual_output}

Before deciding, explicitly identify:
- Any fact, legal assertion, or claim in the final response that does NOT appear in the research output.
- Any place where the response contradicts or misrepresents the research output.
- Any hedging, qualifications, or caveats present in the research output that are omitted in the final response in a way that changes meaning.

Then return exactly one verdict:
"fail" - you identified one or more unsupported claims or a meaningful misrepresentation.
"pass" - you identified none of those; only trivial wording differences, and all substantive claims are present in the research output.

A shorter response is not a failure on its own. Leaving material out is a failure only where the omission changes the meaning of what remains.

Provide your evaluation in strict JSON format exactly like this:
{{
    "analysis": "<A short paragraph explicitly identifying any hallucinated facts, contradictions, or omitted caveats you found above>",
    "verdict": "<pass or fail>",
    "reason": "<One sentence citing the specific hallucination or confirming full grounding>"
}}
"""


class ResponseGroundednessMetric(BaseMetric):
    """
    Evaluates whether the final response is grounded in the research
    agent's output, with no hallucinated or invented facts.

    Two steps. A response that is a near-verbatim copy of the research output
    passes without an LLM call, since grounding is then a provable fact. Anything
    reworded enough to matter goes to the judge, which returns pass or fail: fail
    if it finds an unsupported claim or a meaningful misrepresentation, pass if it
    finds only trivial wording differences. The score is 1.0 for a pass and 0.0
    for a fail.

    The verdict is asked for directly rather than as a 1-5 grade because the grade
    was unstable: a single grade step was enough to flip a verdict. See
    docs/metrics.md.

    research_output is not a standard LLMTestCase field so it is passed
    via the constructor, following the same pattern as ClaimSupportMetric.

    Args:
        research_output: The research agent's synthesised output for this question.
        model:           A judge model, see utils/judge.py.
        threshold:       Minimum score to pass. The score is binary, so this is
                         1.0 by default and there is no middle ground.
        scope_note:      The approved research plan's scope note, if the run had
                         a plan. Given to the judge so that a response
                         describing its own scope is not read as unsupported.
        research_mode:   The question's research mode, labelling the scope note.
    """

    def __init__(
        self,
        research_output: str,
        model,
        threshold: float = 1.0,
        scope_note: str | None = None,
        research_mode: str | None = None,
    ) -> None:
        self.research_output = research_output
        self.model = model
        self.threshold = threshold
        self.scope_note = scope_note
        self.research_mode = research_mode
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        actual_output = test_case.actual_output or ""
        ratio = difflib.SequenceMatcher(
            None, actual_output.strip(), self.research_output.strip()
        ).ratio()
        if ratio >= _NEAR_VERBATIM_THRESHOLD:
            self.score = 1.0
            self.reason = (
                f"Near-verbatim relay of research output (similarity={ratio:.2f}); "
                "judge not invoked."
            )
            self.success = True
            return self.score

        scope_block = (
            _SCOPE_BLOCK_TEMPLATE.format(
                research_mode=self.research_mode or "unspecified",
                scope_note=self.scope_note.strip(),
            )
            if self.scope_note and self.scope_note.strip()
            else ""
        )
        prompt = _PROMPT_TEMPLATE.format(
            research_output=self.research_output,
            scope_block=scope_block,
            actual_output=actual_output,
        )
        try:
            result = self.model.generate(prompt, schema=_GroundednessJudgement)
            if isinstance(result, _GroundednessJudgement):
                verdict = result.verdict
                self.reason = result.reason
            else:
                data = json.loads(str(result))
                verdict = str(data["verdict"]).strip().lower()
                self.reason = data["reason"]
            if verdict not in ("pass", "fail"):
                raise ValueError(f"judge returned verdict={verdict!r}")
            self.score = 1.0 if verdict == "pass" else 0.0
        except Exception as exc:
            # Scored 0 and flagged: the dashboard keeps rows whose reason starts
            # with "Judge error:" out of the mean (streamlit_report.py's
            # _NON_SCORED_PREFIXES), since no verdict exists for them.
            self.score = 0.0
            self.reason = f"Judge error: {exc}"

        self.success = self.score >= self.threshold
        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Response Groundedness"
