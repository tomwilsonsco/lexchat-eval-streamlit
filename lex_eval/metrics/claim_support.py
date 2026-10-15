"""
Claim support metric.

Scores how many of the legal claims in the research agent's output can be
traced to the legal text the run actually retrieved from the API.
"""

from __future__ import annotations

import json
import re
from typing import List, Literal

from .base import BaseMetric
from ..testcase import LLMTestCase
from pydantic import BaseModel

# Budgeted for a judge with a roughly 1M-token context window (see
# OPENROUTER_JUDGE_MODEL in .env for whichever model is currently configured).
# The largest stored retrieval context is ~411k chars, so on current data
# nothing is ever omitted.
_MAX_CONTEXT_CHARS: int = (1_048_576 - 30_000) * 4  # ≈ 4 074 304 chars

_MAX_CLAIMS = 8


class _Claim(BaseModel):
    claim: str
    label: Literal["supported", "unsupported", "absence"]
    quote: str


class _ClaimSupportJudgement(BaseModel):
    claims: List[_Claim]


_PROMPT_TEMPLATE = """You are checking a research agent's report about UK law against the legal text the agent retrieved.

Your task is to decide, claim by claim, whether the retrieved text supports what the report says.

Do NOT use your own knowledge of the law. Do not judge whether the report is right about the law. Only compare the two documents in front of you.

Retrieved Legal Text:
{retrieval_context}
{truncation_note}
Research Report:
{research_report}

First identify the main legal claims the report makes about what the law says, at most {max_claims} of them. Prefer the claims a reader would rely on: what a provision says, what it requires or permits, who it applies to, and the report's overall conclusion. Ignore differences of wording, ordering and formatting.

A statement that something was not retrieved, or that the database does not contain it, is not a claim about the law. Do not list it.

Then label each claim:
- "supported" if the retrieved text says it, even in different words.
- "unsupported" if the retrieved text does not say it.
- "absence" if the claim is that the law does NOT do something, for example that an Act imposes no consultation requirement, or that a later Act did not amend a provision.

For "supported", quote the words from the RETRIEVED LEGAL TEXT that justify the label, copied exactly. For "unsupported" and "absence", leave the quote empty.

Provide your evaluation in strict JSON format exactly like this:
{{
    "claims": [
        {{"claim": "<the claim, in one sentence>", "label": "supported|unsupported|absence", "quote": "<exact words from the retrieved text, or empty>"}}
    ]
}}
"""


# How much of the judge's quote has to be found in the text. Judges copy
# faithfully but trim mid-sentence and reflow list punctuation, so requiring
# the whole quote rejected 39 genuine quotes out of 42 when measured. Matching
# the opening of the quote separated genuine from invented evidence cleanly.
_QUOTE_MATCH_CHARS = 40


def _normalise(text: str) -> str:
    """Lower-case, and reduce punctuation and whitespace to single spaces."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (text or "").lower())).strip()


def _quote_found(quote: str, context: str) -> bool:
    """
    Whether the judge's quote really comes from *context*.

    The first 40 characters of the quote must appear in the text, ignoring
    case, spacing and punctuation. A shorter quote must appear in full.
    """
    normalised = _normalise(quote)
    return bool(normalised) and normalised[:_QUOTE_MATCH_CHARS] in _normalise(context)


class ClaimSupportMetric(BaseMetric):
    """
    Evaluates how many of the research agent's legal claims can be traced to
    the legal text the agent actually saw.

    This is the metric that catches the agent inventing legal content, as
    opposed to citing an Act it never retrieved (Citation Grounding, which is
    deterministic and Act level).

    The judge must quote the passage that supports each claim it passes. A
    quote that is not actually in that text is rejected and the claim counted
    unsupported, so a judge that invents its own evidence cannot pass a record.
    Judges trim and reflow the passages they quote, so the check matches the
    opening of the quote rather than the whole of it (see ``_quote_found``).

    Claims that the law does NOT do something (no consultation requirement, no
    amendment to a section) are counted but not scored: nothing can be quoted
    to prove an absence, so scoring them would mark down a report for saying
    something true. They are named in the reason so a reviewer can still spot a
    wrong one. If every claim is one of these there is nothing to trace and the
    score is 1.0.

    The text to judge against comes from the LLMTestCase's retrieval_context
    (joined to a single string). Callers must set that to what the agent saw:
    LexChat summarises large tool results before the agent reads them, so use
    ``utils.test_helpers.agent_visible_context`` rather than the raw captured
    retrieval_context. research_output is passed via the constructor as it is
    not a standard LLMTestCase field.

    Read this metric at the aggregate, not per record: the mean holds steady
    across sweeps while individual records move, because the judge re-chooses
    which claims the report makes on every run. No stored list can prevent that
    here, since the claims come from the report, which is new text on every
    gather run. See docs/metrics.md.

    Args:
        research_output: The research agent's synthesised output.
        model:           A judge model, see utils/judge.py.
        threshold:       Minimum share of claims that must be supported
                         (default 0.8).
    """

    def __init__(self, research_output: str, model, threshold: float = 0.8) -> None:
        self.research_output = research_output
        self.model = model
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        # Join retrieval context items, skipping (not truncating the whole
        # list at) any individual item too large to fit the remaining budget.
        context_items = test_case.retrieval_context or []
        kept: list[str] = []
        omitted = 0
        total = 0
        for item in context_items:
            if total + len(item) > _MAX_CONTEXT_CHARS:
                omitted += 1
                continue
            kept.append(item)
            total += len(item)
        retrieval_context_str = "\n\n".join(kept)

        truncation_note = (
            f"\n({omitted} further retrieval item(s) omitted for length; "
            "do not treat their absence as evidence that something is "
            "unsupported.)\n"
            if omitted
            else ""
        )

        prompt = _PROMPT_TEMPLATE.format(
            retrieval_context=retrieval_context_str,
            truncation_note=truncation_note,
            research_report=self.research_output,
            max_claims=_MAX_CLAIMS,
        )

        try:
            result = self.model.generate(prompt, schema=_ClaimSupportJudgement)
            if isinstance(result, _ClaimSupportJudgement):
                claims = result.claims
            else:
                data = json.loads(str(result))
                claims = [_Claim(**c) for c in data["claims"]]
            if not claims:
                raise ValueError("judge returned no claims")
        except Exception as exc:
            self.score = 0.0
            self.success = False
            self.reason = f"Judge error: {exc}"
            return self.score

        return self._score_claims(claims, retrieval_context_str)

    def _score_claims(self, claims: List[_Claim], retrieval_context: str) -> float:
        self.details = {
            "claims": [
                {
                    **c.model_dump(),
                    "quote_found": _quote_found(c.quote, retrieval_context),
                }
                for c in claims
            ]
        }
        absence = [c for c in claims if c.label == "absence"]
        scorable = [c for c in claims if c.label != "absence"]

        supported = [
            c
            for c in scorable
            if c.label == "supported" and _quote_found(c.quote, retrieval_context)
        ]
        unquoted = sum(
            1
            for c in scorable
            if c.label == "supported" and not _quote_found(c.quote, retrieval_context)
        )
        unsupported = [c for c in scorable if c.label == "unsupported"]

        if not scorable:
            self.score = 0.0
            self.success = False
            self.reason = (
                f"Not measured: All {len(absence)} claim(s) are claims of absence, which "
                f"cannot be traced to a passage: {absence[0].claim}"
            )
            return self.score

        self.score = len(supported) / len(scorable)
        self.success = self.score >= self.threshold

        self.reason = (
            f"Traced {len(supported)} of {len(scorable)} claims to the text "
            "the agent saw"
        )
        if unsupported:
            self.reason += f"; unsupported: {unsupported[0].claim}"
        self.reason += "."
        if unquoted:
            self.reason += (
                f" {unquoted} further claim(s) counted unsupported because the "
                "quoted passage is not in the retrieved text."
            )
        if absence:
            self.reason += (
                f" {len(absence)} claim(s) of absence not scored, since nothing "
                f"can be quoted to prove one: {absence[0].claim}"
            )

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Claim Support"
