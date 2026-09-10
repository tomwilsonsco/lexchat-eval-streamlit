"""
Reference answer agreement metric.

Scores a response against the statements written alongside the authored
reference ("gold") answer for the same question: how many of them does the
response also state, and does it contradict any.
"""

from __future__ import annotations

import json
from typing import List, Literal

from .base import BaseMetric
from ..testcase import LLMTestCase
from pydantic import BaseModel

from .claim_support import _quote_found

# Attempts allowed for the judge to return one label per statement, no more and
# no fewer. See ReferenceAnswerAgreementMetric._label.
_MAX_LABEL_ATTEMPTS = 2


class _Point(BaseModel):
    index: int
    label: Literal["stated", "contradicted", "missing"]
    quote: str


class _AgreementJudgement(BaseModel):
    points: List[_Point]


class _Contradiction(BaseModel):
    index: int
    contradicted: bool
    quote: str


class _ContradictionJudgement(BaseModel):
    findings: List[_Contradiction]


_PROMPT_TEMPLATE = """You are checking a response to a UK legal question against a list of statements a correct answer has to make. The statements were written by the person who researched the question.

Do NOT use your own knowledge of the law. Do not judge whether the statements are right. Only compare the statements against the response in front of you.

Question:
{input}

Statements:
{statements}

Response Under Test:
{actual_output}

Label each statement by its index:
- "stated" if the response makes the same point, even in different words.
- "contradicted" if the response asserts something incompatible with the statement.
- "missing" if the response neither makes the point nor contradicts it.

Ignore differences of wording, ordering, formatting and level of detail. A response that makes the point in passing has still made it.

For "stated" and "contradicted", quote the words from the RESPONSE UNDER TEST that justify the label, copied exactly. For "missing", leave the quote empty.

Return exactly {n} labels, one per statement, in index order.

Provide your evaluation in strict JSON format exactly like this:
{{
    "points": [
        {{"index": 1, "label": "stated|contradicted|missing", "quote": "<exact words from the response, or empty>"}}
    ]
}}
"""


_CONTRADICTION_PROMPT_TEMPLATE = """You are checking whether a response to a UK legal question contradicts any of a list of statements a correct answer has to make. The statements were written by the person who researched the question.

Do NOT use your own knowledge of the law. Do not judge whether the statements are right. Do not judge whether the response is complete. Only look for contradictions between the two documents in front of you.

Question:
{input}

Statements:
{statements}

Response Under Test:
{actual_output}

Your ONLY task is to find contradictions. Ignore whether the response covers the statements well; a statement the response never mentions is not a contradiction.

A contradiction is where the response asserts something that cannot both be true alongside the statement. The most important case to catch: the response states a point correctly in one place, then elsewhere asserts something wider or narrower that undoes it. Read the whole response, including sections far from where a point is first made, before deciding.

Pay particular attention to scope. If a statement says something applies to a closed or limited set, and the response says it applies to a wider or open ended set, that is a contradiction even if the response also recites the limited set correctly somewhere.

For each of the {n} statements, say whether the response contradicts it. When it does, quote the words from the RESPONSE UNDER TEST that are incompatible with the statement, copied exactly. Leave the quote empty when there is no contradiction.

Provide your evaluation in strict JSON format exactly like this:
{{
    "findings": [
        {{"index": 1, "contradicted": false, "quote": "<exact words from the response, or empty>"}}
    ]
}}
"""


class ReferenceAnswerAgreementMetric(BaseMetric):
    """
    Evaluates how many of a question's reference statements the response makes.

    The statements are written once, alongside the reference answer,
    and stored with it. The judge is given that fixed list and labels each entry
    stated, contradicted or missing. It never chooses the list itself, which is
    what makes the metric repeatable: when the judge picked the points on every
    run, 48% of calls disagreed with their own record's usual labelling and the
    denominator wandered between 6 and 9 for the same answer. Labelling a fixed
    list, that fell to 7%. See docs/metrics.md.

    This is the only metric that compares the response against material a person
    researched, so it is the only one that can catch a response that is faithful
    to its own retrieval but wrong about the law.

    A contradiction fails the metric outright, whatever the score, because a
    confidently wrong statement of law is worse than a missing one. A
    contradiction whose supporting quote is not actually in the response is
    ignored, so an invented quote cannot fail a record.

    Contradictions are looked for in a second judge call that does nothing else.
    Asked in the same breath as "does the response make this point", the
    contradiction question loses: once the judge finds a passage stating the
    point it labels the statement stated and stops reading. On the answer that
    prompted this design, which recited a closed list of nine professions
    correctly and then 65 lines later called the reservation open ended, the
    single combined call caught the contradiction 0 times in 5 and a call asking
    only about contradictions caught it 4 times in 5, with no false positives on
    two answers that get the same point right.

    Args:
        statements: The reference statements for this question, in order. With
                    the usual 5, the score can only be 0.0, 0.2, 0.4, 0.6, 0.8
                    or 1.0, and the default threshold means at least 3 of the 5.
        model:      A judge model, see utils/judge.py.
        threshold:  Minimum share of the statements the response must state
                    (default 0.6).
    """

    def __init__(self, statements: List[str], model, threshold: float = 0.6) -> None:
        self.statements = list(statements)
        self.model = model
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(self.statements))
        prompt = _PROMPT_TEMPLATE.format(
            input=test_case.input or "",
            statements=numbered,
            actual_output=test_case.actual_output or "",
            n=len(self.statements),
        )

        try:
            if not self.statements:
                raise ValueError("no reference statements supplied")
            points = self._label(prompt)
            contradicted = self._contradictions(test_case)
        except Exception as exc:
            self.score = 0.0
            self.success = False
            self.reason = f"Judge error: {exc}"
            return self.score

        return self._score_points(points, contradicted, test_case.actual_output or "")

    def _contradictions(self, test_case: LLMTestCase) -> set[int]:
        """Statement indexes the response contradicts, by a judge call of its own.

        Only findings whose quote is genuinely in the response are returned, so
        an invented quote cannot fail a record.
        """
        numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(self.statements))
        prompt = _CONTRADICTION_PROMPT_TEMPLATE.format(
            input=test_case.input or "",
            statements=numbered,
            actual_output=test_case.actual_output or "",
            n=len(self.statements),
        )
        result = self.model.generate(prompt, schema=_ContradictionJudgement)
        if isinstance(result, _ContradictionJudgement):
            findings = result.findings
        else:
            data = json.loads(str(result))
            findings = [_Contradiction(**f) for f in data["findings"]]

        self.contradiction_findings = [f.model_dump() for f in findings]
        valid = range(1, len(self.statements) + 1)
        return {
            f.index
            for f in findings
            if f.contradicted
            and f.index in valid
            and _quote_found(f.quote, test_case.actual_output or "")
        }

    def _label(self, prompt: str) -> List[_Point]:
        """One label per statement, retried once if the judge returns the wrong set.

        A short or padded label list would silently move the denominator, which
        is the failure this metric exists to remove, so it is rejected rather
        than scored. Observed on roughly 1 call in 22, and it has not survived a
        retry, so one is enough.
        """
        expected = list(range(1, len(self.statements) + 1))
        last: Exception | None = None
        for _ in range(_MAX_LABEL_ATTEMPTS):
            result = self.model.generate(prompt, schema=_AgreementJudgement)
            if isinstance(result, _AgreementJudgement):
                points = result.points
            else:
                data = json.loads(str(result))
                points = [_Point(**p) for p in data["points"]]
            if sorted(p.index for p in points) == expected:
                return points
            last = ValueError(
                f"judge returned {len(points)} label(s) for "
                f"{len(self.statements)} statements"
            )
        raise last  # type: ignore[misc]

    def _score_points(
        self, points: List[_Point], contradicted: set[int], actual_output: str
    ) -> float:
        ordered = sorted(points, key=lambda p: p.index)

        stated = [p for p in ordered if p.label == "stated"]
        # Either judge call can spot a contradiction; both verify their quote
        # against the response first.
        found = set(contradicted) | {
            p.index
            for p in ordered
            if p.label == "contradicted" and _quote_found(p.quote, actual_output)
        }

        self.details = {
            "statements": self.statements,
            "points": [p.model_dump() for p in ordered],
            "contradicted_indexes": sorted(found),
            "contradiction_findings": getattr(self, "contradiction_findings", []),
        }
        self.score = len(stated) / len(ordered)
        self.success = self.score >= self.threshold and not found

        if found:
            first = sorted(found)[0]
            self.reason = (
                "Contradicts the reference answer: "
                f"{self.statements[first - 1]} "
                f"(states {len(stated)} of {len(ordered)} reference points)."
            )
            if len(found) > 1:
                self.reason += f" {len(found) - 1} further contradiction(s) found."
        else:
            self.reason = f"States {len(stated)} of {len(ordered)} reference points."

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Reference Answer Agreement"
