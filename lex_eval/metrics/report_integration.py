"""
Report integration metric.

Scores whether the final answer delivered to the user reflects every deep
research step's own substantive finding, rather than one being dropped when
the Manager condenses several step reports into a single response.
"""

from __future__ import annotations

import json
from typing import Literal

from .base import BaseMetric
from ..testcase import LLMTestCase
from pydantic import BaseModel

from .claim_support import _quote_found
from .structure import (
    _DELEGATE_TOOL_NAME,
    _cited_legislation_ids,
    _group_tools_by_delegation,
    _retrieved_legislation_ids,
    _retrieved_usable_content,
)


class _StepVerdict(BaseModel):
    label: Literal["represented", "dropped"]
    quote: str


_PROMPT_TEMPLATE = """You are checking whether a final answer to a UK legal question preserved one finding from the research that produced it. The research was carried out in separate steps of an approved plan; you are given one step's own report and the final answer the user actually received.

Do NOT use your own knowledge of the law. Do not judge whether the finding is right. Only check whether this one step's finding made it into the final answer, even if reworded, condensed, or merged with other findings.

Step's own report:
{step_report}

Final Answer:
{actual_output}

Decide:
- "represented" if the final answer reflects this step's finding, even briefly or in different words.
- "dropped" if the final answer does not reflect it at all.

For "represented", quote the words from the FINAL ANSWER that show it, copied exactly. For "dropped", leave the quote empty.

Provide your evaluation in strict JSON format exactly like this:
{{
    "label": "represented|dropped",
    "quote": "<exact words from the final answer, or empty>"
}}
"""


class ReportIntegrationMetric(BaseMetric):
    """
    Deep research only. Checks that every step's own substantive finding
    survives into the final answer, rather than being quietly dropped when
    the Manager condenses several step reports into one response.

    A step is in scope only if its own tool calls retrieved usable content
    and its own report cites one of the Acts that retrieval actually
    returned, not merely cites some URL (a report citing a sibling step's
    Act instead of its own has no finding of its own either). A step that
    retrieved nothing, and a step that retrieved but cited nothing of its
    own, both have no finding of their own for the final answer to have
    kept or dropped, so neither is scored here. Those two are
    GenuineGapMetric's and StepCompletionMetric's questions to answer, not
    this one.

    One judge call per in-scope step, asking only about that step's own
    finding, rather than one batched call listing every step alongside the
    full final answer, which was measured unstable. Asking one narrower
    question per call is the same fix ResponseGroundednessMetric already uses
    (a direct pass/fail per response rather than a multi-item grade), applied
    per step instead of per response. It reduced the noise without removing
    it: treat a single flagged step as a prompt to re-run rather than a
    finding, while a fully dropped run (score 0.0, several steps flagged) is
    not this kind of noise. See docs/metrics.md.

    The judge must quote the FINAL ANSWER to justify a "represented" label.
    A quote that isn't really there is downgraded to dropped, so the judge
    can't invent survival (same anti-hallucination shape as ClaimSupportMetric
    and ReferenceAnswerAgreementMetric).

    Score:
        0.0: no delegate_research call found; cannot be verified.
        1.0: no step both retrieved and reported a usable finding; nothing to
               check for integration.
        <1.0: fraction of in-scope steps whose finding is represented in the
               final answer; any fully dropped step drags the score down.

    Args:
        model:      A judge model, see utils/judge.py.
        threshold:  Minimum share of in-scope steps that must survive
                    (default 1.0: a fully dropped step's finding is a defect
                    nothing else in the harness catches).
    """

    def __init__(self, model, threshold: float = 1.0) -> None:
        self.model = model
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        groups = _group_tools_by_delegation(test_case)

        if not groups:
            self.score = 0.0
            self.success = False
            self.reason = (
                f"No '{_DELEGATE_TOOL_NAME}' tool call found; "
                "report integration cannot be verified."
            )
            return self.score

        in_scope = {
            i: g["report"]
            for i, g in enumerate(groups, 1)
            if _retrieved_usable_content(g["tools"])
            and (
                _retrieved_legislation_ids(g["tools"])
                & _cited_legislation_ids(g["report"])
            )
        }

        if not in_scope:
            self.score = 0.0
            self.success = False
            self.reason = (
                "Not measured: no step both retrieved and reported a usable finding; "
                "nothing to check for integration."
            )
            return self.score

        actual_output = test_case.actual_output or ""
        try:
            dropped = [
                i
                for i, report in in_scope.items()
                if not self._is_represented(report, actual_output)
            ]
        except Exception as exc:
            self.score = 0.0
            self.success = False
            self.reason = f"Judge error: {exc}"
            return self.score

        return self._score(dropped, len(in_scope), len(groups))

    def _is_represented(self, step_report: str, actual_output: str) -> bool:
        """One judge call: does actual_output reflect this one step's finding?"""
        prompt = _PROMPT_TEMPLATE.format(
            step_report=step_report, actual_output=actual_output
        )
        result = self.model.generate(prompt, schema=_StepVerdict)
        if isinstance(result, _StepVerdict):
            verdict = result
        else:
            data = json.loads(str(result))
            verdict = _StepVerdict(**data)
        return verdict.label == "represented" and _quote_found(
            verdict.quote, actual_output
        )

    def _score(self, dropped: list[int], n_in_scope: int, n_groups: int) -> float:
        n_represented = n_in_scope - len(dropped)
        self.score = n_represented / n_in_scope
        self.success = self.score >= self.threshold

        scope_note = (
            f" ({n_groups - n_in_scope} step(s) had nothing to check.)"
            if n_groups > n_in_scope
            else ""
        )

        if dropped:
            steps = ", ".join(str(s) for s in dropped)
            self.reason = (
                f"Step(s) {steps} (of {n_in_scope} step(s) with a reportable "
                "finding) are not reflected in the final answer." + scope_note
            )
        else:
            self.reason = (
                f"All {n_in_scope} step(s) with a reportable finding are "
                "reflected in the final answer." + scope_note
            )

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Report Integration"
