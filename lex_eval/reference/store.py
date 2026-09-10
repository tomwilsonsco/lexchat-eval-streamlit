"""Reading and writing the reference-answer set.

Two artefacts per question:

* an entry in `reference_answers.json`, the machine-readable manifest metrics read.
  This is the record; whether a question has been answered is decided from it.
* `q{id}.md`, a generated view of that record for a lawyer to review: the answer,
  the key statements, the citations to mark up, a decision, and the research
  trail as an appendix. Editing it changes nothing that is scored.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(__file__).parent.parent / "data"
QUESTIONS_PATH = DATA_DIR / "questions.json"
ANSWERS_DIR = DATA_DIR / "reference_answers"
MANIFEST_NAME = "reference_answers.json"


def load_questions(path: Path = QUESTIONS_PATH) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest(answers_dir: Path = ANSWERS_DIR) -> List[Dict[str, Any]]:
    """Every reference answer written so far, or [] if there are none yet."""
    path = answers_dir / MANIFEST_NAME
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def is_built(record: Optional[Dict[str, Any]]) -> bool:
    """Whether a manifest record is a usable reference answer.

    This, not the presence of `q{id}.md`, is what says a question is done. The
    Markdown is a generated view, so deleting it should cost a re-render and
    not a rebuild.
    """
    return bool(
        record
        and (record.get("final_answer") or "").strip()
        and (record.get("statements") or [])
    )


def load_reference_answers(
    answers_dir: Path = ANSWERS_DIR, *, verified_only: bool = False
) -> Dict[int, Dict[str, Any]]:
    """Reference answers keyed by question_id, for metrics to score against.

    Drafts are included by default. Excluding them would leave almost every
    question unscored while the signed set grows, so evaluation uses both and
    labels a draft's scores as agreement with its author rather than legal
    correctness. Pass `verified_only=True` for a signed-off-only view.

    "Verified" means `effective_verified()`: a complete sign-off given against
    the version of the reference that is here now, not merely a `verified: true`
    somebody typed in.
    """
    return {
        r["question_id"]: r
        for r in load_manifest(answers_dir)
        if not verified_only or effective_verified(r)
    }


REVIEW_NAME = "review.json"

# What the lawyer decides. "Approve" is the only verdict that can turn
# `verified` on; anything else is work to do before this can be ground truth.
APPROVE = "Approve"
CHANGES_REQUIRED = "Changes required"


def new_review_block() -> Dict[str, Any]:
    """The lawyer decision, empty. Nothing here is populated by generation."""
    return {
        "verified": False,
        "verified_by": None,
        "verified_at": None,
        "verdict": None,
        "citations_reviewed": False,
        "required_citations": [],
        "corrections": "",
        "notes": "",
        "signed_reference_sha256": None,
    }


def normalise_review(review: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """A review block in the current shape, whatever shape it was stored in.

    Keys the current shape does not have are dropped, which is how the old
    answer-only `answer_sha256` and hand-set `stale` fields fall away: staleness
    is now derived from the fingerprint, not stored and trusted.
    """
    block = new_review_block()
    for key in block:
        value = (review or {}).get(key)
        if value is not None:
            block[key] = value
    return block


# A provision id: a legislation type, a year, a number, then any provision path
# below it, e.g. ukpga/2018/12 or asp/2009/12/section/35a.
# Modern ids date the Act by calendar year (`ukpga/2018/12`). Acts before 1963
# are dated by regnal year instead, which takes two segments
# (`ukpga/eliz2/5-6/31`), so both shapes have to be accepted or no provision
# of the Occupiers' Liability Act 1957 can ever be marked Required.
_PROVISION_ID_RE = re.compile(
    r"^[a-z]{2,10}/(\d{4}|[a-z]{2,10}\d*/[0-9\-]+)/[A-Za-z0-9.\-]+(/[A-Za-z0-9.\-]+)*$"
)


def citation_id(citation: str) -> Optional[str]:
    """One required citation as a canonical provision id, or None if it is not one.

    Accepts a legislation.gov.uk URL or a bare provision id. Anything else, a
    link to another site, a case reference, a note typed into the field, has no
    id and cannot be scored against.
    """
    text = (citation or "").strip()
    if not text:
        return None
    if "://" in text or text.lower().startswith("www."):
        from ..metrics.structure import provision_id_from_url

        if "legislation.gov.uk" not in text.lower():
            return None
        identifier = provision_id_from_url(text)
    else:
        identifier = text.strip("/").lower()
    return identifier if _PROVISION_ID_RE.match(identifier or "") else None


def normalise_citations(citations: Optional[List[str]]) -> List[str]:
    """Required citations as canonical provision ids, sorted and deduplicated.

    An entry that is not a citation is kept as its own text rather than dropped,
    so that the fingerprint still changes when somebody corrects it and
    `review_problems()` can name it.
    """
    return sorted(
        {
            citation_id(c) or c.strip().lower()
            for c in citations or []
            if c and c.strip()
        }
    )


def reference_fingerprint(record: Dict[str, Any]) -> str:
    """A hash of everything a lawyer's approval of this reference rests on.

    Covers the question, the answer, the statements the judge is shown, the
    citations approved as required, and the retrieval evidence the answer was
    written from, legislation and judgments alike. Anything else, timestamps,
    notes, the order tools are displayed in, can change without invalidating
    the approval.
    """
    review = normalise_review(record.get("review"))
    material = {
        "question_id": record.get("question_id"),
        "question": (record.get("question") or "").strip(),
        "research_mode": record.get("research_mode"),
        "final_answer": (record.get("final_answer") or "").strip(),
        "statements": [s.strip() for s in record.get("statements") or []],
        "citations_reviewed": bool(review["citations_reviewed"]),
        "required_citations": normalise_citations(review["required_citations"]),
        "sources_retrieved": sorted(
            f"{s.get('legislation_id', '')} {s.get('uri', '')}"
            for s in record.get("sources_retrieved") or []
        ),
        "cases_retrieved": sorted(
            f"{c.get('ncn', '')} {c.get('url', '')}"
            for c in record.get("cases_retrieved") or []
        ),
        # One hash for all the retrieved text, because the record keeps the text
        # as a flat list and not per source.
        "retrieved_text": _sha256("\n".join(record.get("retrieval_context") or [])),
    }
    return _sha256(json.dumps(material, sort_keys=True, ensure_ascii=False))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def citation_problems(record: Dict[str, Any]) -> List[str]:
    """Why each approved citation cannot be used as the expected answer, if any.

    A required citation has to be a legislation.gov.uk provision, has to appear
    in the answer, and has to be one the research actually read. A citation
    failing any of those would make the scored expectation something the
    reference cannot support.

    Every judgment the answer cites has to have been read too. Nothing scores a
    case citation, so a case the research never opened would otherwise pass
    sign-off unchallenged.
    """
    from ..metrics.citation_agreement import _is_covered, cited_provisions

    review = normalise_review(record.get("review"))
    cited = cited_provisions(record.get("final_answer") or "")
    retrieved = {
        citation_id(s.get("uri") or "") or ""
        for s in record.get("sources_retrieved") or []
    }

    problems = []
    for raw in review["required_citations"]:
        identifier = citation_id(raw)
        if not identifier:
            problems.append(f"not a legislation.gov.uk provision: {raw!r}")
        elif not _is_covered(identifier, cited):
            problems.append(f"required but not cited in the answer: {identifier}")
        elif not _is_covered(identifier, retrieved):
            problems.append(f"required but never retrieved: {identifier}")

    read = {
        (c.get("url") or "").rstrip("/") for c in record.get("cases_retrieved") or []
    }
    for url in cited_judgments(record.get("final_answer") or ""):
        if url not in read:
            problems.append(f"cited in the answer but never read: {url}")
    return problems


def review_problems(record: Dict[str, Any]) -> List[str]:
    """Everything stopping this record's approval from counting, in plain words.

    Empty for a record nobody has approved, and empty for a good approval. A
    non-empty list means somebody has set `verified: true` on something that
    cannot stand as a lawyer's approval of what is in the record now.
    """
    review = normalise_review(record.get("review"))
    if not review["verified"]:
        return []

    problems = []
    if review["verdict"] != APPROVE:
        problems.append(
            f"the decision is {review['verdict'] or 'missing'}, not {APPROVE!r}"
        )
    if not review["verified_by"]:
        problems.append("no reviewer is recorded")
    if not review["verified_at"]:
        problems.append("no review date is recorded")
    if not review["citations_reviewed"]:
        problems.append("the citations have not been marked up")
    problems += citation_problems(record)
    return problems


def review_state(record: Dict[str, Any]) -> str:
    """Where this reference stands, in one phrase a reviewer can act on."""
    review = normalise_review(record.get("review"))
    if review["verdict"] == CHANGES_REQUIRED:
        return CHANGES_REQUIRED
    if not review["verified"]:
        return "Draft"
    # Staleness first: when the approval was given against a different version,
    # what is wrong with it as an approval of this one is beside the point.
    if review["signed_reference_sha256"] != reference_fingerprint(record):
        return "Stale"
    if review_problems(record):
        return "Sign-off unusable"
    return "Verified"


def effective_verified(record: Dict[str, Any]) -> bool:
    """Whether this reference carries a lawyer approval that still holds.

    A sign-off that does not say `Approve`, is missing its reviewer, date or
    citation decision, names a citation the answer does not make or the
    research never read, or was given against a version of the answer,
    statements, citations or evidence that has since changed, is not an
    approval of what is here now.
    """
    return review_state(record) == "Verified"


def reference_version(record: Dict[str, Any]) -> tuple[str, str]:
    """Which reference a metric scored against: its fingerprint and standing.

    Stored on every reference-metric result, so a score calculated against an
    answer that has since been corrected, or against a draft that has since
    been signed off, can be told apart from a current one.
    """
    fingerprint = record.get("reference_sha256") or reference_fingerprint(record)
    return fingerprint, "verified" if effective_verified(record) else "draft"


def current_reference_versions(
    answers_dir: Path = ANSWERS_DIR,
) -> Dict[int, tuple[str, str]]:
    """The current version of every reference answer, keyed by question_id."""
    return {
        qid: reference_version(record)
        for qid, record in load_reference_answers(answers_dir).items()
    }


def apply_review(record: Dict[str, Any], review: Optional[Dict[str, Any]]) -> None:
    """Attach a review block to a record and stamp the reference fingerprint.

    A new approval, one with `verified: true` and no signed fingerprint yet, is
    stamped with the version in front of it. Clearing `signed_reference_sha256`
    back to null is therefore how a maintainer records that the lawyer has
    confirmed a changed version.
    """
    record["review"] = normalise_review(review)
    fingerprint = reference_fingerprint(record)
    if record["review"]["verified"] and not record["review"]["signed_reference_sha256"]:
        record["review"]["signed_reference_sha256"] = fingerprint
    record["reference_sha256"] = fingerprint


def write(records: List[Dict[str, Any]], answers_dir: Path = ANSWERS_DIR) -> Path:
    """Merge records into the manifest and write their Markdown files."""
    answers_dir.mkdir(parents=True, exist_ok=True)
    merged = {r["question_id"]: r for r in load_manifest(answers_dir)}
    merged.update({r["question_id"]: r for r in records})

    manifest_path = answers_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps([merged[k] for k in sorted(merged)], indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    for record in records:
        (answers_dir / f"q{record['question_id']}.md").write_text(
            render_markdown(record), encoding="utf-8"
        )
    return manifest_path


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------

# The rendered answer sits between these two markers so a test can pull the
# exact text back out and compare it with the manifest. Nothing else in the
# document may be inserted between them.
ANSWER_BEGIN = "<!-- BEGIN REFERENCE ANSWER -->"
ANSWER_END = "<!-- END REFERENCE ANSWER -->"

_MODE_NOTE = {
    "legislation_only": (
        "Legislation only. Case law is out of scope, so an answer that turns on "
        "a decided case is not expected to cite one here."
    ),
    "case_law_only": "Case law only. Legislation is out of scope for this answer.",
    "legislation_and_case_law": "Legislation and case law are both in scope.",
}


def answer_section(markdown: str) -> str:
    """The reference answer as it appears in a rendered `q{id}.md`.

    Raises if the markers are missing, which means the file was written by an
    older renderer or edited by hand. Either way it is no longer a view of the
    record and should be regenerated.
    """
    start = markdown.find(ANSWER_BEGIN)
    end = markdown.find(ANSWER_END)
    if start < 0 or end < start:
        raise ValueError("no marked reference answer in this Markdown")
    return markdown[start + len(ANSWER_BEGIN) : end].strip()


# What each state means to the person opening the file.
_STATE_NOTE = {
    "Draft": "Nobody has reviewed this yet.",
    CHANGES_REQUIRED: "Reviewed and not approved. The corrections are in section 5.",
    "Verified": "Approved by the reviewer named in section 5.",
    "Stale": (
        "This was approved, but the answer, statements, required citations or "
        "retrieved material have changed since. The approval no longer covers "
        "what is in this file and the reviewer needs to see it again."
    ),
    "Sign-off unusable": (
        "This is marked approved, but the approval cannot stand as recorded. "
        "What is wrong is listed under the status above, and it is not counted "
        "as approved until that is put right."
    ),
}


def _fmt_plan(plan: Optional[Dict[str, Any]]) -> str:
    if not plan or not plan.get("steps"):
        return "_No plan recorded._"
    return "\n".join(
        f"{step['id']}. **{step['title']}**\n   {step['detail']}\n"
        for step in plan["steps"]
    ).rstrip()


def _fmt_problems(problems: List[str]) -> str:
    """What is stopping an approval from counting, if anything is."""
    if not problems:
        return ""
    listed = "\n".join(f"> - {p}" for p in problems)
    return f"\n> **The approval on file cannot be used:**\n{listed}\n"


def _fmt_statements(statements: Optional[List[str]], approved: bool) -> str:
    """The statements, each with a place to accept or amend it until approved."""
    if not statements:
        return (
            "_None recorded. Write them in `.authored/q{id}/statements.json`, then "
            "run `python -m lex_eval.reference.build --statements-only`._"
        )
    if approved:
        return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
    return "\n\n".join(
        f"{i + 1}. {s}\n   - Accept / Amend (write the replacement here):"
        for i, s in enumerate(statements)
    )


def cited_judgments(text: str) -> List[str]:
    """Find Case Law judgment links in *text*, deduplicated and sorted.

    Sentence punctuation is stripped along with the trailing slash: the URL
    pattern stops at a bracket but not at a full stop, so a judgment cited at
    the end of a sentence would otherwise never match one that was read.
    """
    from ..metrics.structure import _CASE_LAW_DOMAIN, _URL_RE

    return sorted(
        {
            url.rstrip("/.,;:")
            for url in _URL_RE.findall(text or "")
            if _CASE_LAW_DOMAIN in url.lower()
        }
    )


def _fmt_judgments(record: Dict[str, Any]) -> str:
    """Every judgment link in the answer, for the lawyer to check.

    Not marked up as Required or Background: `Citation Agreement` reads
    legislation.gov.uk provisions only, so nothing scores a judgment citation
    today. This table is here so a case the answer leans on can still be
    checked against what the research actually read.
    """
    cited = cited_judgments(record.get("final_answer") or "")
    if not cited:
        return "_The answer cites no judgments._"

    read = {
        (c.get("url") or "").rstrip("/"): c for c in record.get("cases_retrieved") or []
    }
    lines = [
        "| Judgment | Neutral citation | Read during research |",
        "| --- | --- | --- |",
    ]
    for url in cited:
        case = read.get(url) or {}
        title = (case.get("title") or "").replace("|", "\\|")
        lines.append(
            f"| [{title or url}]({url}) | {case.get('ncn', '')} "
            f"| {'yes' if case else 'no'} |"
        )
    return "\n".join(lines)


def _fmt_citations(record: Dict[str, Any]) -> str:
    """Every legislation link in the answer, for the lawyer to mark up.

    The links are read with the same parser the Citation Agreement metric uses,
    so this table is exactly the list that metric expects a response to cite
    while the reference is a draft. Once the lawyer marks them up, the ones
    marked `Required` become that expectation on their own.
    """
    from ..metrics.citation_agreement import _is_covered, cited_provisions
    from ..metrics.structure import provision_id_from_url

    cited = sorted(cited_provisions(record.get("final_answer") or ""))
    if not cited:
        if record.get("research_mode") == "case_law_only":
            return (
                "_Case law only, so no legislation is expected. Citation Agreement "
                "does not measure this question._"
            )
        return (
            "_The answer contains no legislation.gov.uk links. Citation Agreement "
            "cannot measure this question until the provisions it relies on are "
            "written into the answer as links._"
        )

    retrieved = {
        provision_id_from_url(s.get("uri") or ""): s
        for s in (record.get("sources_retrieved") or [])
    }
    review = normalise_review(record.get("review"))
    required = set(normalise_citations(review["required_citations"]))
    lines = [
        "| Provision | Title | Read during research | Required / Background / Remove |",
        "| --- | --- | --- | --- |",
    ]
    for pid in cited:
        source = retrieved.get(pid) or {}
        # An instrument-level citation is read if any provision of it was read,
        # the same rule the score and the citation checks use.
        read = bool(source) or _is_covered(pid, set(retrieved))
        if source:
            title = (source.get("title") or "").replace("|", "\\|")
        else:
            title = "the instrument as a whole" if read else "_not retrieved_"
        if pid in required:
            mark = "**Required**"
        elif review["citations_reviewed"]:
            mark = "Background"
        else:
            mark = ""
        lines.append(
            f"| [{pid}](https://www.legislation.gov.uk/{pid}) | {title} "
            f"| {'yes' if read else 'no'} | {mark} |"
        )
    return "\n".join(lines)


def _fmt_retrieved(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "_Nothing retrieved._"
    lines = ["| Provision | Legislation | Extent | URI |", "| --- | --- | --- | --- |"]
    for s in sources:
        title = (s.get("title") or "").replace("|", "\\|")
        lines.append(
            f"| {title} | `{s.get('legislation_id', '')}` "
            f"| {', '.join(s.get('extent') or [])} | {s.get('uri', '')} |"
        )
    return "\n".join(lines)


def _fmt_discovered(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "_Every Act and SI found in search was also retrieved._"
    return "\n".join(
        f"- `{s.get('legislation_id', '')}`, {s.get('title', '')}" for s in sources
    )


def _fmt_cases_retrieved(cases: List[Dict[str, Any]]) -> str:
    if not cases:
        return "_No judgment text was read._"
    lines = [
        "| Judgment | Neutral citation | Court | Date | URL |",
        "| --- | --- | --- | --- | --- |",
    ]
    for c in cases:
        title = (c.get("title") or "").replace("|", "\\|")
        lines.append(
            f"| {title} | {c.get('ncn', '')} | {c.get('court', '')} "
            f"| {c.get('date', '')} | {c.get('url', '')} |"
        )
    return "\n".join(lines)


def _fmt_cases_discovered(cases: List[Dict[str, Any]]) -> str:
    if not cases:
        return "_Every judgment found in search was also read._"
    return "\n".join(
        f"- {c.get('ncn', '')}, {c.get('title', '')} ({c.get('url', '')})"
        for c in cases
    )


# Which halves of the citation schedule and the research appendix a mode gets.
# A section about a source the question excluded is noise the reviewer has to
# read past, and its "nothing retrieved" placeholder reads as a gap in the
# research rather than as a source that was never in scope.
_SHOWS_LEGISLATION = {"legislation_only", "legislation_and_case_law"}
_SHOWS_CASE_LAW = {"case_law_only", "legislation_and_case_law"}


# Where the material in front of the reviewer came from. One complete sentence
# per mode rather than a name and a service spliced together, so the grammar and
# the line wrapping stay with the words they belong to.
_PROVENANCE = {
    "legislation_only": (
        "The legislation below was retrieved from the live LEX\n"
        "service using the same search tools AILA uses."
    ),
    "case_law_only": (
        "The judgments below were retrieved from the National\n"
        "Archives Find Case Law service using the same search tools AILA uses."
    ),
    "legislation_and_case_law": (
        "The legislation and judgments below were retrieved\n"
        "from the live LEX service and the National Archives Find Case Law\n"
        "service, using the same search tools AILA uses."
    ),
}

# What the case law tools cannot reach, shown to any reviewer whose question was
# researched with them. Without it a reference answer can inherit AILA's own
# retrieval blind spot, be signed off, and then be used to score AILA.
_COVERAGE_NOTE = (
    "\n\n**What these tools cannot see:** Find Case Law does not index the Court\n"
    "of Session or the sheriff courts, and its coverage of older judgments is\n"
    "patchy. An answer saying a point could not be found may only mean it could\n"
    "not be found here, so please check anything the answer states is not the law."
)


# What section 4 asks of the reviewer, which is not the same job in every mode.
# Judgments are shown to be checked, not marked up: no metric scores a case
# citation, so asking for Required/Background/Remove against them would be
# asking for a decision nothing acts on.
_CITATION_ASKS = {
    "legislation_only": (
        "**Section 4, the citations.** Which of these must a correct answer cite? "
        "Mark\n   each one Required, Background or Remove."
    ),
    "case_law_only": (
        "**Section 4, the judgments.** Is each one really the case the answer says "
        "it is,\n   and is a case the answer should have cited missing? There is "
        "nothing to mark\n   up here."
    ),
    "legislation_and_case_law": (
        "**Section 4, the citations.** Which legislation must a correct answer "
        "cite? Mark\n   each one Required, Background or Remove. The judgments "
        "below need no markup,\n   just tell us in section 5 if one is wrong or "
        "missing."
    ),
}


def _appendix_counts(record: Dict[str, Any]) -> str:
    """The rows of the appendix summary that this question's mode has numbers for."""
    mode = record.get("research_mode", "legislation_only")
    rows = []
    if mode in _SHOWS_LEGISLATION:
        rows.append(
            f"| Full-Act fallback used | {'yes' if record.get('fallback_used') else 'no'} |"
        )
        rows.append(
            f"| Provisions retrieved | {len(record.get('sources_retrieved') or [])} |"
        )
    if mode in _SHOWS_CASE_LAW:
        rows.append(f"| Judgments read | {len(record.get('cases_retrieved') or [])} |")
    return "".join(f"{row}\n" for row in rows)


def _citation_section(record: Dict[str, Any]) -> str:
    """Section 4, the parts of it this question's research mode calls for."""
    mode = record.get("research_mode", "legislation_only")
    parts = []
    if mode in _SHOWS_LEGISLATION:
        parts.append(
            """Every legislation link in the answer. Mark each one `Required` if a correct
answer has to cite it, `Background` if it is context, or `Remove` if it does not
belong in the answer at all.

"""
            + _fmt_citations(record)
        )
    if mode in _SHOWS_CASE_LAW:
        parts.append(
            """Judgments the answer cites. Nothing scores these yet, so there is nothing to
mark up: check them, and say in section 5 if one is wrong or missing.

"""
            + _fmt_judgments(record)
        )
    return "\n" + "\n\n".join(parts) + "\n"


def _research_appendix(record: Dict[str, Any]) -> str:
    """Appendix subsections 6.2 onward, numbered in the order the mode shows them.

    6.1 is the research plan, so these start at 6.2. Which of them appear
    depends on the mode, hence the numbering here rather than in the template.
    """
    mode = record.get("research_mode", "legislation_only")
    sections: List[tuple] = []
    if mode in _SHOWS_LEGISLATION:
        sections.append(
            (
                "Provisions retrieved",
                """Every provision the answer was permitted to rely on. A citation in section 2
that does not appear below is unsupported by this run's retrieval.""",
                _fmt_retrieved(record.get("sources_retrieved") or []),
            )
        )
        sections.append(
            (
                "Legislation found but never read",
                """These appeared in search results, so they exist and were located, but their text
was never retrieved. Citing one is a weaker claim than citing a provision above.""",
                _fmt_discovered(record.get("sources_discovered") or []),
            )
        )
    if mode in _SHOWS_CASE_LAW:
        sections.append(
            (
                "Judgments read",
                "Every judgment whose full text the answer was permitted to rely on.",
                _fmt_cases_retrieved(record.get("cases_retrieved") or []),
            )
        )
        sections.append(
            (
                "Judgments found but never read",
                """These came back from a case law search but their text was never retrieved.
Citing one is a weaker claim than citing a judgment above.""",
                _fmt_cases_discovered(record.get("cases_discovered") or []),
            )
        )
    return "\n\n".join(
        f"### 6.{i + 2} {title}\n\n{blurb}\n\n{body}"
        for i, (title, blurb, body) in enumerate(sections)
    )


def render_markdown(record: Dict[str, Any]) -> str:
    """Render one reference answer for lawyer review.

    The answer, the key statements and the citations come first, because those
    are what the reviewer decides on. How the answer was researched is an
    appendix: it is there to be checked if a citation looks wrong, not to be
    signed off.
    """
    r = record
    review = normalise_review(r.get("review"))
    state = review_state(r)
    mode = r.get("research_mode", "legislation_only")

    return f"""# Q{r['question_id']} reference answer for review

**Status: {state}.** {_STATE_NOTE.get(state, '')}{_fmt_problems(review_problems(r))}

**How this was produced:** {_PROVENANCE.get(mode, _PROVENANCE['legislation_only'])}
The answer was drafted from that retrieved text by {r.get('author', 'unknown')}.
Nobody legally qualified has checked it.{_COVERAGE_NOTE if mode in _SHOWS_CASE_LAW else ''}

**Why this is required:** We test AILA by asking it the same questions over and
over and comparing what it says against a fixed expected answer, which is the
only way to tell whether a change to the system has made its legal answers
better or worse. That expected answer is currently drafted by an AI model, and
even the best of them state the law confidently when they have it wrong, so
scoring one AI against another proves nothing on its own. Your sign-off is what
turns this from a draft into a legal benchmark we can rely on.

**What we need from you:** three checks, then your decision written into
section 5.

1. **Section 2, the answer.** Is it right, and is anything missing or
   misleading?
2. **Section 3, the key statements.** Are these the points a correct answer has
   to make? Mark each one Accept or Amend.
3. {_CITATION_ASKS.get(mode, _CITATION_ASKS['legislation_only'])}
4. **Section 5, your decision. Please complete every line of it**, including
   Approve or Changes required, your name and the date. Nothing counts as
   reviewed until section 5 is filled in, whether or not you have written
   comments elsewhere.

Section 6 is the research trail behind the answer. For information only.

## 1. Question

> {r['question']}

{_MODE_NOTE.get(mode, f'Research mode `{mode}`.')}

{('**Scope of this answer:** ' + r['plan']['scope_note']) if (r.get('plan') or {}).get('scope_note') else ''}

## 2. Reference answer

{ANSWER_BEGIN}

{r.get('final_answer', '') or '_none_'}

{ANSWER_END}

## 3. Key statements used by evaluation

The points a correct answer has to make, most important first. These are the
fixed list the `Reference Answer Agreement` metric scores a response against:
the judge is shown these and the response, never the answer above, and labels
each one stated, contradicted or missing. Editing them changes what that metric
measures, so they need your approval as much as the answer does.

{_fmt_statements(r.get('statements'), state == 'Verified')}

## 4. Citation schedule
{_citation_section(r)}
## 5. Decision

This is the section we need you to complete. Please fill in every line.

- **Approve / Changes required:** {review['verdict'] or '_not yet reviewed_'}
- **Reviewer:** {review['verified_by'] or '_not yet reviewed_'}
- **Date:** {review['verified_at'] or '_not yet reviewed_'}
- **Citations in section 4 marked up:** {'yes' if review['citations_reviewed'] else 'not yet'}
- **What is wrong, missing or misleading:** {review['corrections'] or '_nothing recorded_'}
- **Notes:** {review['notes'] or '_none_'}

A decision recorded here is copied into `.authored/q{r['question_id']}/review.json`
by a maintainer, who then re-renders this file. Editing this file changes
nothing that is scored.

## 6. Appendix, how this answer was researched

You are not asked to approve any of this. It is here so a citation that looks
wrong can be traced back to what was read.

| | |
| --- | --- |
| Research mode | `{mode}` |
| Written | {r['generated_at']} |
| Tool calls | {' → '.join(r.get('tool_sequence') or []) or '_none_'} |
{_appendix_counts(r)}| Reference version | `{r.get('reference_sha256') or reference_fingerprint(r)}` |
| Version approved | `{review['signed_reference_sha256'] or 'none'}` |

### 6.1 Research plan

{_fmt_plan(r.get('plan'))}

{_research_appendix(r)}
"""
