"""
Plan coverage metric.

Scores a deep-research plan's steps against the statements written alongside
the authored reference answer for the same question: does the plan the
lawyer approved actually set out to cover them, before any research happens.
"""

from __future__ import annotations

import json
from typing import List, Literal

from .base import BaseMetric
from ..testcase import LLMTestCase
from pydantic import BaseModel

# Attempts allowed for the judge to return one label per statement, no more and
# no fewer. See PlanCoverageMetric._label.
_MAX_LABEL_ATTEMPTS = 2


class _StepCoverage(BaseModel):
    index: int
    label: Literal["addressed", "not_addressed"]
    step: int


class _CoverageJudgement(BaseModel):
    points: List[_StepCoverage]


_PROMPT_TEMPLATE = """You are checking a proposed research plan for a UK legal question against a list of statements a correct answer has to make. The statements were written by the person who researched the question; the plan is what an AI system proposed to research, before doing any research.

Do NOT use your own knowledge of the law. Do not judge whether the statements are right, or whether the plan's steps are well written. Only judge whether, if every step were carried out, the plan would surface the information each statement needs.

Question:
{input}

Plan steps:
{steps}

Statements a correct answer has to make:
{statements}

Label each statement by its index:
- "addressed" if at least one step, if carried out, would surface the information this statement needs.
- "not_addressed" if no step would.

For "addressed", give the step number shown above that addresses it. For "not_addressed", give step 0.

Return exactly {n} labels, one per statement, in index order.

Provide your evaluation in strict JSON format exactly like this:
{{
    "points": [
        {{"index": 1, "label": "addressed|not_addressed", "step": <step number, or 0>}}
    ]
}}
"""


class PlanCoverageMetric(BaseMetric):
    """
    Evaluates whether a deep-research plan's steps set out to cover a
    question's reference statements, before any research has happened.

    Reuses the same fixed statement list as ReferenceAnswerAgreementMetric,
    rather than a separately authored "golden plan", so this costs no new
    lawyer review: the plan is judged against material already required for
    that metric. See docs/deep_research_plan.md for why.

    A step number the judge names for "addressed" is checked against the
    actual plan; a step number outside the plan's range is downgraded to
    not_addressed, so the judge can't invent coverage it can't point at, the
    same anti-hallucination shape as ReferenceAnswerAgreementMetric's quote
    check.

    This is a plan-quality check, not an execution check: it says nothing
    about whether the steps were actually carried out well, or whether the
    final answer used what was found. A plan whose steps are broad enough to
    cover a narrow question almost by definition (e.g. "retrieve the text of
    section X", against statements that are all facts inside that section)
    will score highly regardless of plan quality. This metric is most
    informative on questions whose statements span more ground than a single
    step could plausibly reach; it is not fixed for the narrow case, since it
    does not produce a wrong verdict there, only an uninformative one.

    Unlike ReferenceAnswerAgreementMetric, there is no "contradicted"
    outcome: a plan can fail to cover a point, but it does not assert
    anything the way a finished answer can, so is_successful() is simply
    score >= threshold.

    reason lists each statement's own text, one per line, under a "Not
    addressed" heading (shown first, so a reviewer sees the gap before
    everything the plan got right) and an "Addressed" heading, not just the
    count. Newline-separated rather than HTML: the dashboard renders it as
    line breaks, everything else (DB, pytest output) just sees plain text.

    Args:
        plan_steps: The approved plan's steps, in order, each a dict with at
                    least "title" and "detail" (LexChat's
                    ``research_plan.steps`` shape from
                    ``POST /api/research/plan``).
        statements: The reference statements for this question, in order.
        model:      A judge model, see utils/judge.py.
        threshold:  Minimum share of the statements the plan must address
                    (default 0.6).
    """

    def __init__(
        self,
        plan_steps: List[dict],
        statements: List[str],
        model,
        threshold: float = 0.6,
    ) -> None:
        self.plan_steps = list(plan_steps)
        self.statements = list(statements)
        self.model = model
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        numbered_steps = "\n".join(
            f"{i + 1}. {s.get('title', '')}: {s.get('detail', '')}"
            for i, s in enumerate(self.plan_steps)
        )
        numbered_statements = "\n".join(
            f"{i + 1}. {s}" for i, s in enumerate(self.statements)
        )
        prompt = _PROMPT_TEMPLATE.format(
            input=test_case.input or "",
            steps=numbered_steps,
            statements=numbered_statements,
            n=len(self.statements),
        )

        try:
            if not self.statements:
                raise ValueError("no reference statements supplied")
            if not self.plan_steps:
                raise ValueError("no plan steps supplied")
            points = self._label(prompt)
        except Exception as exc:
            self.score = 0.0
            self.success = False
            self.reason = f"Judge error: {exc}"
            return self.score

        return self._score_points(points)

    def _label(self, prompt: str) -> List[_StepCoverage]:
        """One label per statement, retried once if the judge returns the wrong set."""
        expected = list(range(1, len(self.statements) + 1))
        last: Exception | None = None
        for _ in range(_MAX_LABEL_ATTEMPTS):
            result = self.model.generate(prompt, schema=_CoverageJudgement)
            if isinstance(result, _CoverageJudgement):
                points = result.points
            else:
                data = json.loads(str(result))
                points = [_StepCoverage(**p) for p in data["points"]]
            if sorted(p.index for p in points) == expected:
                return points
            last = ValueError(
                f"judge returned {len(points)} label(s) for "
                f"{len(self.statements)} statements"
            )
        raise last  # type: ignore[misc]

    def _score_points(self, points: List[_StepCoverage]) -> float:
        n_steps = len(self.plan_steps)
        ordered = sorted(points, key=lambda p: p.index)

        addressed = [
            p for p in ordered if p.label == "addressed" and 1 <= p.step <= n_steps
        ]
        addressed_idx = {p.index for p in addressed}
        invalid_step = sum(
            1
            for p in ordered
            if p.label == "addressed" and not (1 <= p.step <= n_steps)
        )

        self.score = len(addressed) / len(ordered)
        self.success = self.score >= self.threshold

        self.reason = (
            f"Plan addresses {len(addressed)} of {len(ordered)} reference point(s)."
        )
        if invalid_step:
            self.reason += (
                f" {invalid_step} further 'addressed' label(s) counted as not "
                "addressed because the cited step number doesn't exist in the plan."
            )

        addressed_lines = [
            f"({p.index}) {self.statements[p.index - 1]}"
            for p in ordered
            if p.index in addressed_idx
        ]
        not_addressed_lines = [
            f"({p.index}) {self.statements[p.index - 1]}"
            for p in ordered
            if p.index not in addressed_idx
        ]
        # Not addressed first, so the gap is the first thing a reviewer sees
        # rather than scrolling past everything the plan got right.
        if not_addressed_lines:
            self.reason += "\nNot addressed:\n" + "\n".join(not_addressed_lines)
        if addressed_lines:
            self.reason += "\nAddressed:\n" + "\n".join(addressed_lines)

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Plan Coverage"
