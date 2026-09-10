"""
Custom metric to measure consistency across multiple responses to the same
question, either from the same LLM (repeatability) or across different LLMs
(cross-model agreement).

Uses TF vectorisation (no IDF) with cosine similarity. Skipping IDF ensures
that shared legal terminology is not down-weighted when comparing a small
number of responses, giving more meaningful scores.

The similarity score alone decides pass or fail. Any legislation.gov.uk
section cited in one answer but not the other is listed in the reason as a
diagnostic, because cosine similarity can't see a single flipped section
number buried among hundreds of otherwise-identical tokens.
"""

from .base import BaseMetric
from ..testcase import LLMTestCase
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Set
import numpy as np
import re

_SECTION_CITATION_RE = re.compile(
    r"(?:https?://)?(?:www\.)?legislation\.gov\.uk/[^\s)]+?/section/\d+[A-Za-z]*",
    re.IGNORECASE,
)


def _preprocess(text: str) -> str:
    """Strip markdown formatting and normalise whitespace."""
    # Drop markdown link targets (keeping the visible link text): every
    # citation shares the same legislation.gov.uk URL prefix, which would
    # otherwise inflate similarity with content-free boilerplate.
    text = re.sub(r"\]\([^)]*\)", " ", text)
    text = re.sub(r"[*_#>`\[\]()]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _extract_citations(text: str) -> Set[str]:
    """Return the set of legislation.gov.uk section citation URLs in *text*."""
    return {
        m.group(0).lower().rstrip("/")
        for m in _SECTION_CITATION_RE.finditer(text or "")
    }


def _vectorize(texts: List[str]):
    """
    Fit a TF vectorizer on *texts*, falling back to no stop-word removal
    if all tokens in a document are stop words (which would otherwise raise
    a ValueError).  Returns None if vectorization is not possible.
    """
    for stop_words in ("english", None):
        vectorizer = TfidfVectorizer(
            stop_words=stop_words,
            ngram_range=(1, 2),
            use_idf=False,
            sublinear_tf=True,
        )
        try:
            return vectorizer.fit_transform(texts)
        except ValueError:
            continue
    return None


class ConsistencyMetric(BaseMetric):
    """
    Measures how consistent a response is compared to one or more
    reference responses to the same question.

    Responses are vectorised with TF (no IDF) so that shared legal
    terminology retains its full weight rather than being penalised for
    appearing across the small comparison corpus. Cosine similarity then
    captures directional agreement independent of response length.

    Pass or fail is the similarity score against the threshold, nothing
    else. Sections cited in one answer but not the other are listed in the
    reason so a flipped section number stays visible, but they do not
    decide the result: an agent searching a live corpus twice will touch
    different secondary provisions each run, which is expected rather than
    a defect.

    Args:
        reference_outputs: Other answers to compare against.
        threshold:         Minimum mean similarity to pass (default 0.4).
    """

    def __init__(
        self,
        reference_outputs: List[str],
        threshold: float = 0.4,
    ):
        self.threshold = threshold
        self.reference_outputs = reference_outputs
        self.score = 0.0
        self.reason = ""
        self.success = False

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        actual = _preprocess(test_case.actual_output or "")
        refs = [_preprocess(r) for r in self.reference_outputs]

        if not actual:
            self.score = 0.0
            self.success = False
            self.reason = "Actual output is empty."
            return self.score

        if not refs:
            self.score = 0.0
            self.success = False
            self.reason = "No reference outputs provided."
            return self.score

        # Vectorise with TF only (use_idf=False) so shared domain terms keep
        # their weight. Bigrams capture legal phrases like "good faith".
        all_texts = [actual] + refs
        tfidf_matrix = _vectorize(all_texts)

        if tfidf_matrix is None:
            self.score = 0.0
            self.success = False
            self.reason = "Could not vectorize responses (possibly empty inputs)."
            return self.score

        actual_vec = tfidf_matrix[0]
        ref_vecs = tfidf_matrix[1:]

        sims = cosine_similarity(actual_vec, ref_vecs)[0]
        self.score = float(np.mean(sims))

        citation_mismatch = self._find_citation_mismatch(test_case.actual_output or "")

        self.success = self.score >= self.threshold
        self.reason = (
            f"Mean cosine similarity: {self.score:.3f} "
            f"(across {len(all_texts)} responses, "
            f"threshold: {self.threshold})"
        )

        # A run that produced nothing scores 0.000 against this one by
        # construction, not by measurement, and the score alone reads as two
        # answers that disagree. Say which it is. The score still stands: a
        # model that answers once and not the next time is inconsistent, and
        # that is the result, not an artefact to be filtered away.
        empty_refs = sum(1 for r in refs if not r)
        if empty_refs:
            self.reason += (
                f". {empty_refs} of {len(refs)} comparison run(s) produced no "
                "answer to compare against, so the similarity to those is 0.000 "
                "because they are empty, not because the answers differ"
            )
        if citation_mismatch is not None:
            self.reason += (
                f". For information, section citations differ from a reference "
                f"(this does not affect the result): {citation_mismatch}"
            )
        return self.score

    def _find_citation_mismatch(self, actual_raw: str) -> Optional[str]:
        """
        Return a description of the first reference whose cited
        legislation.gov.uk sections differ from the response's, or None if
        the response has no citations to check or all references agree.

        Reported in the reason only. The check is skipped when the response
        has no section citations at all (the Worker prompt allows bold-text
        citation as a fallback).
        """
        actual_citations = _extract_citations(actual_raw)
        if not actual_citations:
            return None

        for ref_raw in self.reference_outputs:
            ref_citations = _extract_citations(ref_raw)
            if ref_citations != actual_citations:
                missing = ref_citations - actual_citations
                extra = actual_citations - ref_citations
                return (
                    f"missing {sorted(missing)}, extra {sorted(extra)}"
                    if missing or extra
                    else "citation sets differ"
                )
        return None

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        return "Consistency (Cosine)"
