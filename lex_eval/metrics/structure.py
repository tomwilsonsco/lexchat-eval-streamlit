"""
Metrics that validate Worker Agent output quality.

MandatoryStructureMetric:  checks the 4-part Markdown heading structure.
CitationPassthroughMetric: checks that Worker references reach the final response.
CitationGroundingMetric:   checks that Worker citations were actually retrieved.
CitationReadMetric:        checks that cited Acts had their text read, not just
                            their title seen in a search result.
CitationDomainMetric:      checks that Worker citation URLs are on legislation.gov.uk.
GenuineGapMetric:          checks that an empty retrieval is disclosed, not papered over.
StepCompletionMetric:      checks that a step's own retrieval reached that step's own
                            report (deep research only).

All seven metrics inspect every ``delegate_research`` tool-call output, which is where
the Worker Agent's response is surfaced, one per delegation, so a deep-research run
with several approved plan steps produces several outputs, and none is skipped.

They differ in how they then score, and the distinction matters when reading a
deep-research result:

* Scoped per step, so one step cannot be excused by a sibling: MandatoryStructure,
  GenuineGap, StepCompletion.
* Scored over the run as a whole, pooling every report against every tool call:
  CitationGrounding, CitationRead, CitationDomain. CitationRead additionally reports
  a per-step observation in its reason, without scoring it (see ``_sibling_read_note``).
"""

import json
import re
from urllib.parse import urlparse

from .base import BaseMetric
from ..testcase import LLMTestCase

_DELEGATE_TOOL_NAME = "delegate_research"

# Matches http(s) URLs; stops at whitespace or markdown link-closing chars.
_URL_RE = re.compile(r"https?://[^\s\)\]>,\"']+")

# Bare keywords to search for (case-insensitive, colon optional).  Tolerates
# variation in bold markers, numbering, and trailing colons, e.g.:
#   "**Summary Answer (BLUF):**"        ✓
#   "### **1. Summary Answer (BLUF):**" ✓
#   "### 1. **Summary Answer (BLUF):**" ✓
#   "### **3. Jurisdiction & Status**"  ✓  (no colon)
#
# Each entry may be either a single string or a list of acceptable
# alternatives. The summary heading accepts both "Summary Answer (BLUF)"
# and "Summary Answer", the (BLUF) qualifier is a stylistic hint in the
# Worker system prompt (see LexChat/server_py/src/config.py), not a
# semantic requirement, so either form passes. Likewise "Jurisdiction &
# Status"/"Jurisdiction & Currency" accept the spelled-out "and", models
# routinely paraphrase the prompt's literal "&" this way.
REQUIRED_HEADINGS = {
    "legislation_only": [
        ["Summary Answer (BLUF)", "Summary Answer"],
        "Detailed Analysis",
        ["Jurisdiction & Status", "Jurisdiction and Status"],
        "References",
    ],
    "case_law_only": [
        ["Summary Answer (BLUF)", "Summary Answer"],
        "Key Cases",
        "Analysis",
        ["Jurisdiction & Currency", "Jurisdiction and Currency"],
        "References",
    ],
    "legislation_and_case_law": [
        ["Summary Answer (BLUF)", "Summary Answer"],
        "Statutory Framework",
        "Key Cases",
        ["Jurisdiction & Status", "Jurisdiction and Status"],
        "References",
    ],
}

# A heading match must sit at the start of its line, after only "decoration"
# characters (whitespace, #, *, digits, '.', '-', ':'), this is what lets a
# bare substring check for something like "References" tell a real heading
# apart from the word appearing mid-sentence in ordinary legal prose (e.g.
# "references to the 1978 Act..."), without requiring a literal Markdown
# '#' that real Worker output doesn't always use.
_HEADING_LINE_PREFIX = r"[\s#*\d.\-:]*"


def _group_tools_by_delegation(test_case: LLMTestCase) -> list[dict]:
    """Split ``test_case.tools_called`` into one group per delegation.

    ``audit_capture.py`` appends a ``delegate_research`` entry immediately
    followed by that delegation's own ``Worker: ...`` tool entries, one
    delegation at a time, mirroring the server's own audit-trace structure.
    Splitting the flat list at each ``delegate_research`` entry recovers
    exactly those per-step boundaries: a single-shot run has one group, a
    deep-research run has one per approved plan step, in order.

    Each group is ``{"report": <that delegation's output>, "tools": [...]}``.
    """
    if not test_case.tools_called:
        return []
    groups: list[dict] = []
    for tool in test_case.tools_called:
        if tool.name == _DELEGATE_TOOL_NAME:
            raw = tool.output
            groups.append(
                {"report": raw if isinstance(raw, str) else str(raw), "tools": []}
            )
        elif groups:
            groups[-1]["tools"].append(tool)
    return groups


def _get_delegate_outputs(test_case: LLMTestCase) -> list[str]:
    """Return every ``delegate_research`` tool-call output, in order.

    A single-shot run has exactly one. A deep-research run has one per
    approved plan step, and every step's report must be checked, not just
    the first, so a bad step can't hide behind a good one.
    """
    return [g["report"] for g in _group_tools_by_delegation(test_case)]


# The two tools that return legal text, as opposed to search hits that
# return only titles and links.
_TEXT_TOOLS = (
    "Worker: search_legislation_sections",
    "Worker: get_legislation_text",
)


def _usable_output(output) -> bool:
    """True if a tool call's *output* is real content rather than a failure.

    A LEX tool failure comes back as non-empty prose starting "Error
    executing tool: " (``LexChat/server_py/src/agent/tools/executor.py``),
    not an empty string, so that's excluded explicitly rather than counted
    as usable content.
    """
    if not output:
        return False
    text = output if isinstance(output, str) else str(output)
    return not text.startswith("Error executing tool:")


def _retrieved_usable_content(tools: list) -> bool:
    """True if any ``search_legislation_sections``/``get_legislation_text``
    call among *tools* returned non-empty output, i.e. retrieval wasn't empty.
    """
    return any(
        tool.name in _TEXT_TOOLS and _usable_output(tool.output) for tool in tools
    )


class MandatoryStructureMetric(BaseMetric):
    """
    Ensures the Worker Agent strictly adhered to the mandatory Markdown structure
    mandated by its system prompt for the given research mode.

    Looks for the headings inside the ``delegate_research`` tool-call output
    rather than the top-level actual_output, because the Worker's response is
    surfaced as the return value of that tool.

    Matching is case-insensitive and ignores surrounding bold markers /
    numbering so minor formatting variations don't cause false failures.

    Score:
        1.0: all mandatory headings present (pass)
        0.0: one or more headings missing, or no delegate_research call found
    """

    def __init__(
        self, threshold: float = 1.0, research_mode: str = "legislation_only"
    ) -> None:
        self.threshold = threshold
        self.research_mode = research_mode
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        dr_outputs = _get_delegate_outputs(test_case)

        if not dr_outputs:
            self.score = 0.0
            self.success = False
            self.reason = (
                f"No '{_DELEGATE_TOOL_NAME}' tool call found; "
                "structure cannot be verified."
            )
            return self.score

        headings = REQUIRED_HEADINGS.get(
            self.research_mode, REQUIRED_HEADINGS["legislation_only"]
        )

        def _heading_present(heading, lowered: str) -> bool:
            variants = heading if isinstance(heading, list) else [heading]
            return any(
                re.search(
                    rf"(?m)^{_HEADING_LINE_PREFIX}{re.escape(v.lower())}", lowered
                )
                for v in variants
            )

        step_failures = []
        for i, dr_output in enumerate(dr_outputs, 1):
            lowered = dr_output.lower()
            missing = [h for h in headings if not _heading_present(h, lowered)]
            if missing:
                display = [h[0] if isinstance(h, list) else h for h in missing]
                label = f"step {i}: " if len(dr_outputs) > 1 else ""
                step_failures.append(f"{label}{', '.join(display)}")

        if step_failures:
            self.score = 0.0
            self.success = False
            self.reason = f"Missing mandatory headings: {'; '.join(step_failures)}"
        else:
            self.score = 1.0
            self.success = True
            self.reason = (
                "All mandatory Markdown headings present in Worker output."
                if len(dr_outputs) == 1
                else f"All mandatory Markdown headings present in all "
                f"{len(dr_outputs)} Worker report(s)."
            )

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Research Output Structure"


class CitationPassthroughMetric(BaseMetric):
    """
    Checks that every reference link from the Worker output is present in
    the final response delivered to the user.

    Score:
        0.0: Failure A: no URLs found in Worker output at all.
        0.5: Failure B: one or more Worker links are missing from the
                          final response (citation links were dropped).
        1.0: Pass: every Worker URL is present in the final response.

    Threshold defaults to 1.0, so both failure modes are recorded as fails.
    """

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        dr_outputs = _get_delegate_outputs(test_case)

        if not dr_outputs:
            self.score = 0.0
            self.success = False
            self.reason = (
                f"No '{_DELEGATE_TOOL_NAME}' tool call found; "
                "citations cannot be verified."
            )
            return self.score

        worker_links: set[str] = set()
        for dr_output in dr_outputs:
            worker_links.update(_URL_RE.findall(dr_output))

        if not worker_links:
            self.score = 0.0
            self.success = False
            self.reason = "Failure A: no reference links found in Worker output."
            return self.score

        actual = test_case.actual_output or ""
        passed_through = {link for link in worker_links if link in actual}
        missing = worker_links - passed_through

        if missing:
            self.score = 0.5
            self.success = False
            self.reason = (
                f"Failure B: {len(missing)} of {len(worker_links)} Worker "
                "link(s) missing from final response."
            )
        else:
            self.score = 1.0
            self.success = True
            self.reason = (
                f"Pass: all {len(worker_links)} Worker link(s) present in "
                "final response."
            )

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Reference Links"


def _url_path(url: str) -> str:
    """
    The legislation.gov.uk path a URL points at, lowercased, without a leading
    ``id/`` segment or a trailing slash, e.g. ``ukpga/1978/29/section/10c``.

    Mirrors how the LexChat server builds legislation_id from a search result
    URI (``LexChat/server_py/src/agent/tools/lex.py::_slim_search_results``:
    URL path, minus a leading ``id/`` segment).
    """
    path = urlparse(url).path.strip("/").lower().rstrip(".,;:")
    if path.startswith("id/"):
        path = path[3:]
    return path


def provision_id_from_url(url: str) -> str:
    """
    Derive the provision-level id (e.g. ``ukpga/2018/12/section/6``) from a
    legislation.gov.uk URL.

    Unlike ``_legislation_id_from_url`` this keeps the section or schedule, so
    a citation to section 3 of an Act is not treated as a citation to
    section 6 of the same Act.
    """
    return _url_path(url)


# Path segments that begin a provision, a version, or a format suffix. An Act
# id is everything before the first of these, so the id survives whatever
# follows it: "/section/10C", "/schedule/1", "/made", "/section/12/england+wales".
#
# Counting segments instead does not work. An Act dated by calendar year has
# three (ukpga/1978/29) but one dated by regnal year has four
# (ukpga/Edw7/4/31 is the Shop Hours Act 1904), and the corpus holds both.
_PROVISION_SUFFIXES = frozenset(
    {
        "section",
        "regulation",
        "schedule",
        "article",
        "part",
        "chapter",
        "paragraph",
        "rule",
        "crossheading",
        "division",
        "appendix",
        "annex",
        "note",
        "signature",
        "introduction",
        "body",
        "contents",
        "made",
        "enacted",
        "adopted",
        "created",
        "revised",
        "prospective",
    }
)

# A single path segment of a real legislation id: letters, digits, and the
# ".-_" that regnal and local Act ids use (1-2geo5, 26geo5_1edw8, ch.lxxxiv).
# Anything else, a percent-encoding, a bracket, an ampersand, a space, means
# the model wrote something that is not an id at all.
_ID_SEGMENT_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _legislation_id_from_url(url: str) -> str:
    """
    Derive the Act-level legislation_id (e.g. ``ukpga/1978/29``) from a
    legislation.gov.uk URL, whether it points at the Act itself or a specific
    section/schedule within it (e.g. ``ukpga/1978/29/section/10C``).

    Everything from the first provision or version segment onward is dropped,
    which is the granularity search_legislation_sections and
    get_legislation_text are called at.
    """
    segments = _url_path(url).split("/")
    for i, segment in enumerate(segments):
        if segment in _PROVISION_SUFFIXES:
            segments = segments[:i]
            break
    return "/".join(segments)


def _is_plausible_legislation_id(identifier: str) -> bool:
    """Whether *identifier* could be a legislation id at all.

    Deliberately permissive: this only rejects text that no id could be, so
    that a citation to a real Act is never discarded. A model sometimes writes
    a link body that is prose rather than a reference, e.g.
    ``.../id/[UNCLEAR: no document reference provided]``. Treating one of
    those as a cited Act makes it a fabricated citation, which reads as an
    accusation about an Act rather than what it is, a broken link.

    Three segments is the floor because no legislation id has fewer: every id
    the LEX API returned across the stored responses has at least
    type/year/number, and every citation below that floor was junk, including
    a bare ``ukpga/1988`` written alongside correct links to
    ``ukpga/1988/41/section/65`` in the same report.
    """
    segments = identifier.split("/")
    return len(segments) >= 3 and all(_ID_SEGMENT_RE.match(s) for s in segments)


_LEGISLATION_DOMAIN = "legislation.gov.uk"
_CASE_LAW_DOMAIN = "caselaw.nationalarchives.gov.uk"

# The domains each research mode's Worker system prompt tells it to cite
# (LexChat/server_py/src/prompts.py, CITATION PROTOCOL in each Worker prompt).
# One entry per mode, and each is exactly what that prompt allows:
#   legislation_only         legislation.gov.uk and nothing else.
#   case_law_only            case law only. That prompt requires findings to be
#                            grounded "EXCLUSIVELY in case law retrieved via the
#                            search_case_law tool", with every legal proposition
#                            citing a specific case, and never mentions
#                            legislation.gov.uk. Allowing it here would let a run
#                            score 1.0 for citing a source its brief excluded.
#   legislation_and_case_law both, since that prompt's protocol names both.
_PERMITTED_DOMAINS = {
    "legislation_only": (_LEGISLATION_DOMAIN,),
    "case_law_only": (_CASE_LAW_DOMAIN,),
    "legislation_and_case_law": (_LEGISLATION_DOMAIN, _CASE_LAW_DOMAIN),
}


def _on_domain(url: str, domain: str) -> bool:
    """True if *url*'s host is *domain* or a subdomain of it."""
    netloc = urlparse(url).netloc.lower()
    return netloc == domain or netloc.endswith(f".{domain}")


def _is_legislation_url(url: str) -> bool:
    """True if *url* points at legislation.gov.uk.

    This is what separates an Act citation from a judgment citation, so it
    gates every place a URL is turned into a legislation_id. A case law link
    such as caselaw.nationalarchives.gov.uk/uksc/2025/13 would otherwise
    become Act id "uksc/2025", which no legislation tool call can ever have
    retrieved.
    """
    return _on_domain(url, _LEGISLATION_DOMAIN)


def _permitted_domains(research_mode: str) -> tuple:
    """The citation domains this research mode's Worker prompt permits."""
    return _PERMITTED_DOMAINS.get(research_mode, (_LEGISLATION_DOMAIN,))


def _leading_json(raw: str) -> dict:
    """
    Parse the JSON object at the start of *raw*, ignoring anything after it.

    ``search_legislation`` returns its JSON results followed by a plain-text
    "[NEXT STEP: ...]" hint for the Worker, so a plain ``json.loads`` of the
    whole string raises "Extra data" and yields nothing.
    """
    return json.JSONDecoder().raw_decode(raw.lstrip())[0]


def _retrieved_legislation_ids(tools: list) -> set:
    """
    Return the set of legislation_ids that *tools* actually retrieved:
    results returned by ``search_legislation``, plus the legislation_id
    argument passed to ``search_legislation_sections`` / ``get_legislation_text``.

    Ids are lowercased, because that is how they arrive from a citation URL.
    The API sends them cased (``ukpga/Edw7/4/31``) and roughly one id in
    fourteen has a capital in it, so comparing raw would report every Act
    dated by regnal year as never retrieved.

    Pass a single group's ``tools`` list to scope this to one delegation, or
    ``test_case.tools_called`` for the whole run.
    """
    ids: set = set()
    if not tools:
        return ids

    for tool in tools:
        if tool.name == "Worker: search_legislation":
            raw = tool.output
            try:
                data = _leading_json(raw) if isinstance(raw, str) else raw
                for r in (data or {}).get("results", []):
                    lid = r.get("legislation_id")
                    if lid:
                        ids.add(lid.lower())
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue
        elif tool.name in (
            "Worker: search_legislation_sections",
            "Worker: get_legislation_text",
        ):
            params = tool.input_parameters or {}
            lid = params.get("legislation_id")
            if lid:
                ids.add(lid.lower())

    return ids


def _cited_legislation_ids(report: str) -> set:
    """Act-level ids cited by legislation.gov.uk URL in *report*.

    Case law links are skipped: there is no legislation retrieval to check
    them against. So is a link body that is not an id at all, which
    :func:`_malformed_citations` counts instead.
    """
    return {
        lid
        for lid in (
            _legislation_id_from_url(u)
            for u in _URL_RE.findall(report or "")
            if _is_legislation_url(u)
        )
        if lid and _is_plausible_legislation_id(lid)
    }


def _malformed_citations(report: str) -> set:
    """legislation.gov.uk link bodies in *report* that are not ids.

    Reported, never scored. These are broken links rather than claims about a
    source, so they are not the failure Citation Grounding exists to catch.
    """
    return {
        lid
        for lid in (
            _legislation_id_from_url(u)
            for u in _URL_RE.findall(report or "")
            if _is_legislation_url(u)
        )
        if lid and not _is_plausible_legislation_id(lid)
    }


def _sibling_read_note(groups: list) -> str:
    """Steps citing an Act a *sibling* step read but they did not, i.e. a step
    asserting ahead of its own evidence.

    Reported in the reason, never scored, since a step's References section
    lists Acts it did not read. See docs/metrics.md.
    """
    if len(groups) < 2:
        return ""
    read_per = [_read_legislation_ids(g["tools"]) for g in groups]
    all_read: set = set().union(*read_per)
    notes = []
    for i, g in enumerate(groups, 1):
        sibling = (_cited_legislation_ids(g["report"]) - read_per[i - 1]) & all_read
        if sibling:
            notes.append(f"step {i} cites {sorted(sibling)}")
    if not notes:
        return ""
    return (
        " Diagnostic, not scored: "
        + "; ".join(notes)
        + ", which another step read but that step did not."
    )


def _read_legislation_ids(tools: list) -> set:
    """
    Return the set of legislation_ids whose *text* was actually read: the
    legislation_id argument of every ``search_legislation_sections`` /
    ``get_legislation_text`` call among *tools* that returned usable output.

    Narrower than ``_retrieved_legislation_ids`` on purpose. That function
    also counts an Act that merely turned up in a ``search_legislation``
    results list, which returns titles and links but no legal text.

    Lowercased for the same reason as ``_retrieved_legislation_ids``.
    """
    ids: set = set()
    for tool in tools or []:
        if tool.name in _TEXT_TOOLS and _usable_output(tool.output):
            lid = (tool.input_parameters or {}).get("legislation_id")
            if lid:
                ids.add(lid.lower())
    return ids


class CitationGroundingMetric(BaseMetric):
    """
    Checks that every Act cited in the Worker's report was actually retrieved
    by this run's own tool calls, rather than invented from pattern-matching.

    Catches fabrication (a citation to something never retrieved), not
    wrongness (a citation to a real, retrieved Act that doesn't actually
    answer the question, which is a substantive-correctness question this
    rule-based check can't make).

    Score:
        0.0: no delegate_research call found; citations cannot be verified.
        1.0: no legislation.gov.uk citation URLs in Worker output (nothing
               to falsely ground).
        0.0: one or more cited Acts were never retrieved by search_legislation,
               search_legislation_sections, or get_legislation_text in this run.
        1.0: every cited Act was retrieved by this run.

    No partial credit: unlike Reference Links, where "some links survived" is
    a meaningfully different failure from "none did", one fabricated citation
    is a full failure regardless of how many others were genuine.
    """

    def __init__(self, threshold: float = 1.0) -> None:
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        dr_outputs = _get_delegate_outputs(test_case)

        if not dr_outputs:
            self.score = 0.0
            self.success = False
            self.reason = (
                f"No '{_DELEGATE_TOOL_NAME}' tool call found; "
                "citation grounding cannot be verified."
            )
            return self.score

        cited_ids: set[str] = set()
        malformed: set[str] = set()
        for dr_output in dr_outputs:
            cited_ids |= _cited_legislation_ids(dr_output)
            malformed |= _malformed_citations(dr_output)

        broken_note = (
            f" Diagnostic, not scored: {len(malformed)} legislation.gov.uk "
            "link(s) whose address is not a legislation id, e.g. "
            f"{sorted(malformed)[0]!r}."
            if malformed
            else ""
        )

        if not cited_ids:
            self.score = 1.0
            self.success = True
            self.reason = (
                "No legislation.gov.uk citations found in Worker output; "
                "nothing to ground." + broken_note
            )
            return self.score

        retrieved_ids = _retrieved_legislation_ids(test_case.tools_called)
        fabricated = cited_ids - retrieved_ids

        if fabricated:
            self.score = 0.0
            self.success = False
            self.reason = (
                f"Fabricated citation(s): {sorted(fabricated)} cited in Worker "
                "output but never retrieved by search_legislation, "
                "search_legislation_sections, or get_legislation_text in this run."
                + broken_note
            )
        else:
            self.score = 1.0
            self.success = True
            self.reason = (
                f"All {len(cited_ids)} cited Act(s) were retrieved by this "
                "run's tool calls." + broken_note
            )

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Citation Grounding"


class CitationReadMetric(BaseMetric):
    """
    Checks that the Worker actually read every Act it cites: an Act whose text
    was pulled counts as read, one that only appeared as a title in a
    search_legislation results list does not.

    Catches a report making claims about a real, correctly linked source it
    never opened, e.g. saying an Act "was commenced by" an SI whose text
    nobody retrieved. Citation Grounding passes those, because it accepts a
    search hit as grounding.

    Only legislation.gov.uk citations are counted; case law links have no
    legislation retrieval to check them against.

    Score:
        0.0: no delegate_research call found; cannot be verified.
        1.0: no legislation.gov.uk citations in Worker output; nothing to check.
        else: the fraction of cited Acts whose text was read. Threshold is 1.0,
              so a single unread cited Act fails.
    """

    def __init__(self, threshold: float = 1.0) -> None:
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
                "citation reading cannot be verified."
            )
            return self.score

        cited_ids: set = set()
        for g in groups:
            cited_ids |= _cited_legislation_ids(g["report"])

        if not cited_ids:
            self.score = 1.0
            self.success = True
            self.reason = (
                f"No {_LEGISLATION_DOMAIN} citations found in Worker "
                "output; nothing to check."
            )
            return self.score

        read_ids = _read_legislation_ids(test_case.tools_called)
        unread = cited_ids - read_ids

        self.score = (len(cited_ids) - len(unread)) / len(cited_ids)
        self.success = self.score >= self.threshold

        if unread:
            self.reason = (
                f"Cited without reading: {sorted(unread)}. "
                f"{len(cited_ids) - len(unread)} of {len(cited_ids)} cited Act(s) "
                "had their text retrieved; the rest were cited on the strength of "
                "a search result title, or were never looked up at all."
            )
        else:
            self.reason = (
                f"All {len(cited_ids)} cited Act(s) had their text retrieved by "
                "this run."
            )

        self.reason += _sibling_read_note(groups)

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Citation Read"


class CitationDomainMetric(BaseMetric):
    """
    Checks that every citation URL in the Worker's report points to a domain
    its system prompt told it to cite.

    Which domains those are follows the research mode: legislation.gov.uk for
    legislation_only, and caselaw.nationalarchives.gov.uk as well for the two
    case law modes, whose prompts mandate that format for judgments.

    Score:
        0.0: no delegate_research call found; domains cannot be verified.
        1.0: no citation URLs in Worker output (nothing to check).
        0.0: one or more citation URLs point to some other domain.
        1.0: every citation URL is on a permitted domain.

    No partial credit, same reasoning as Citation Grounding: one invented
    domain is a full failure regardless of how many other citations are fine.

    Args:
        threshold:     Minimum score to pass (default 1.0).
        research_mode: ``legislation_only`` (default), ``case_law_only``, or
            ``legislation_and_case_law``. Decides the permitted domain set.
    """

    def __init__(
        self, threshold: float = 1.0, research_mode: str = "legislation_only"
    ) -> None:
        self.threshold = threshold
        self.research_mode = research_mode
        self.score = 0.0
        self.success = False
        self.reason = ""

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        dr_outputs = _get_delegate_outputs(test_case)

        if not dr_outputs:
            self.score = 0.0
            self.success = False
            self.reason = (
                f"No '{_DELEGATE_TOOL_NAME}' tool call found; "
                "citation domains cannot be verified."
            )
            return self.score

        cited_urls: set[str] = set()
        for dr_output in dr_outputs:
            cited_urls.update(_URL_RE.findall(dr_output))

        if not cited_urls:
            self.score = 1.0
            self.success = True
            self.reason = "No citation URLs found in Worker output; nothing to check."
            return self.score

        permitted = _permitted_domains(self.research_mode)
        off_domain = {
            u for u in cited_urls if not any(_on_domain(u, d) for d in permitted)
        }
        permitted_str = " or ".join(permitted)

        if off_domain:
            self.score = 0.0
            self.success = False
            self.reason = (
                f"Off-domain citation(s): {sorted(off_domain)} do not point to "
                f"{permitted_str}, the domain(s) the Worker's system prompt "
                f"permits in {self.research_mode} mode."
            )
        else:
            self.score = 1.0
            self.success = True
            self.reason = (
                f"All {len(cited_urls)} citation URL(s) are on {permitted_str}."
            )

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Citation Domain"


# The Worker system prompt's mandated sentence for an empty result
# (LexChat/server_py/src/prompts.py, WORKER_SYSTEM_PROMPT, legislation-only block):
# "If the API data does not answer the specific question, state: ... DO NOT attempt
# to fill gaps with internal training data."
_GENUINE_GAP_PHRASE = (
    "The available database does not contain information on this specific issue."
)

# Wordings that disclose the gap without using the research prompt's mandated
# sentence. Two groups, and both are needed:
#   - paraphrases of the mandated sentence, worth 0.5 in research mode because
#     the wording there is prescribed and was not used;
#   - the phrasing the conversational Worker prompt asks for in its own words,
#     "If the retrieved text does not answer the question, say so plainly and
#     suggest the user switch to Research mode". A conversational report that
#     follows that instruction literally matches none of the paraphrases above,
#     so without these entries it scored 0.0 for doing exactly what it was told.
# "not answer" also covers "do not answer" and "cannot answer"; "n't answer"
# covers the contracted forms.
_GENUINE_GAP_KEYWORDS = (
    "does not contain information",
    "no relevant",
    "could not find",
    "no information",
    "unable to find",
    "not answer",
    "n't answer",
    "switch to research mode",
)


class GenuineGapMetric(BaseMetric):
    """
    When a step's own tool calls failed to retrieve any usable legislation
    section text, checks that step's own report says so plainly instead of
    presenting a confident but unsupported answer.

    Only applies to ``legislation_only`` mode: the mandated disclosure sentence and
    the tools this check inspects (search_legislation_sections, get_legislation_text)
    are specific to legislation retrieval.

    In conversational mode the Worker prompt asks for a plain statement of the
    gap and mandates no exact sentence, so a paraphrase scores full marks there
    rather than the 0.5 partial credit it gets in research mode.

    Scoped per step (a single-shot run has exactly one): a step whose own
    retrieval succeeded is judged on its own report, not excused because a
    sibling step elsewhere in the run happened to retrieve something.

    Score:
        0.0: no delegate_research call found; cannot be verified.
        Not measured: research_mode is not legislation_only.
        1.0: every step either retrieved usable content itself (nothing to
               disclose) or, having retrieved nothing, disclosed that plainly.
        0.5: the worst such step disclosed the gap only as a paraphrase of
               the mandated sentence (research mode only).
        0.0: the worst such step didn't disclose the gap at all.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        research_mode: str = "legislation_only",
        chat_mode: str = "research",
    ) -> None:
        self.threshold = threshold
        self.research_mode = research_mode
        self.chat_mode = chat_mode
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
                "genuine gap disclosure cannot be verified."
            )
            return self.score

        if self.research_mode != "legislation_only":
            self.score = 0.0
            self.success = False
            self.reason = (
                f"Not measured: research_mode is '{self.research_mode}', "
                "not legislation_only."
            )
            return self.score

        # Each step is judged on its own retrieval: a step whose own tool
        # calls found usable content has nothing to disclose, regardless of
        # whether a sibling step's retrieval was empty.
        step_scores = []
        n_retrieved = 0
        # The conversational Worker prompt says "say so plainly" and gives no
        # sentence to copy, so a paraphrase is the required behaviour there,
        # not a partial one.
        paraphrase_score = 1.0 if self.chat_mode == "conversational" else 0.5
        for g in groups:
            if _retrieved_usable_content(g["tools"]):
                step_scores.append(1.0)
                n_retrieved += 1
                continue
            lowered = (g["report"] or "").lower()
            if _GENUINE_GAP_PHRASE.lower() in lowered:
                step_scores.append(1.0)
            elif any(kw in lowered for kw in _GENUINE_GAP_KEYWORDS):
                step_scores.append(paraphrase_score)
            else:
                step_scores.append(0.0)

        self.score = min(step_scores)
        self.success = self.score >= self.threshold

        if all(s == 1.0 for s in step_scores):
            n_disclosed = len(groups) - n_retrieved
            if len(groups) == 1:
                self.reason = (
                    "Retrieval returned usable section/full-text content; "
                    "nothing to disclose."
                    if n_retrieved
                    else "Retrieval was empty and the Worker disclosed this "
                    "as required."
                )
            elif not n_disclosed:
                self.reason = (
                    f"All {len(groups)} steps retrieved usable content; "
                    "nothing to disclose."
                )
            else:
                self.reason = (
                    f"All {len(groups)} steps are clear: {n_retrieved} retrieved "
                    f"usable content, {n_disclosed} had empty retrieval and "
                    "disclosed it."
                )
        else:
            worst = step_scores.index(min(step_scores)) + 1
            where = (
                f"Step {worst} of {len(groups)}" if len(groups) > 1 else "The report"
            )
            if self.score == 0.5:
                self.reason = (
                    f"{where} had empty retrieval and disclosed the gap, but "
                    "only as a paraphrase, not the mandated wording."
                )
            else:
                self.reason = (
                    f"{where} had empty retrieval and does not disclose this; "
                    "answered without an honest gap statement."
                )

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Genuine Gap"


class StepCompletionMetric(BaseMetric):
    """
    Deep research only. Checks that every step's own retrieval reached that
    step's own report, catching a step that made tool calls returning legal
    text and then reported nothing (e.g. because it hit a tool-call budget
    limit mid-step), while its sibling steps report normally and every other
    metric in this file scores the run perfectly.

    "Cited" means cites one of the Acts this step's own tool calls actually
    retrieved, not merely cites some URL, a report citing an unrelated Act
    (e.g. one a sibling step retrieved) passes no more here than a report
    citing nothing at all.

    Score:
        0.0: no delegate_research call found; cannot be verified.
        1.0: no step both retrieved usable content and cited none of it.
        <1.0: fraction of steps that pass; any lost step drags the score down.

    Only steps that retrieved usable legislation are scored. Empty retrieval
    is GenuineGapMetric's question; no eligible steps means not measured.
    """

    def __init__(self, threshold: float = 1.0) -> None:
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
                "step completion cannot be verified."
            )
            return self.score

        eligible = {
            i: g
            for i, g in enumerate(groups, 1)
            if _retrieved_usable_content(g["tools"])
        }
        if not eligible:
            self.score = 0.0
            self.success = False
            self.reason = (
                "Not measured: no captured step retrieved usable legislation text."
            )
            return self.score
        failed = [
            i
            for i, g in eligible.items()
            if not (
                _retrieved_legislation_ids(g["tools"])
                & _cited_legislation_ids(g["report"])
            )
        ]
        self.score = (len(eligible) - len(failed)) / len(eligible)
        self.success = self.score >= self.threshold

        if failed:
            steps = ", ".join(str(i) for i in failed)
            self.reason = (
                f"Step(s) {steps} retrieved legislation text but "
                "their own report cites none of it."
            )
        else:
            self.reason = (
                f"All {len(eligible)} eligible step(s) carried their retrieved legislation "
                "text into their own report."
                if len(eligible) > 1
                else "The report cites the legal text it retrieved."
            )

        return self.score

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self) -> str:  # type: ignore[override]
        return "Step Completion"
