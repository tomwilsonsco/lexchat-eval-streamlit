"""Matched experiment comparisons using compatible stored scoring results."""

from collections import defaultdict

from lex_eval.reports.data import (
    NOT_APPLICABLE,
    NO_STORED_RESULT,
    aggregate_metrics,
    latest_results,
    measured,
    outcome_counts,
    unmeasured_state,
)


def cohort_key(record):
    return (
        record["question_id"],
        record["question"],
        record.get("chat_mode"),
        record.get("research_mode"),
        record.get("question_hash"),
    )


def signature(row):
    return (
        row.get("metric_version"),
        row.get("reference_sha256"),
        row.get("reference_mode"),
        row.get("threshold"),
    )


def _cohorts(records):
    grouped = defaultdict(list)
    for rec in records:
        grouped[cohort_key(rec)].append(rec)
    return grouped


def shared_cohorts(baseline, candidate):
    """The question, wording and mode combinations present in both experiments."""
    return _cohorts(baseline).keys() & _cohorts(candidate).keys()


def matched_entries(baseline, candidate, rows):
    """The selected verdicts for every matched cohort and check.

    The evidence panel calls this so it shows the exact stored rows the
    totals were built from, rather than reselecting scores of its own.
    """
    groups = [_cohorts(records) for records in (baseline, candidate)]
    return _matched_entries(groups, groups[0].keys() & groups[1].keys(), rows)


# Why a shared question is left out when one side has no score for it, such
# as Consistency on a question one experiment asked only once. Totals over
# different questions would make the missing score look like a change.
NOT_STORED_ON_A_SIDE = "Not compared: no stored result on one side"


def _matched_entries(groups, matched, rows):
    """One entry per matched cohort and metric, with the selected verdicts.

    An entry either carries a reason it could not be compared, or every
    side's response ids and their aggregated verdicts. A scoring version is
    used only when every side has it. The per question table, the per check
    summary and the overview are all built from these, so they select the
    same scoring version and exclude the same rows.
    """
    entries = []
    for key in sorted(matched):
        ids = [{r["response_id"] for r in side[key]} for side in groups]
        metric_keys = {
            r["test_name"] for r in rows if r["response_id"] in set().union(*ids)
        }
        for metric in sorted(metric_keys):
            sides = [
                [
                    r
                    for r in rows
                    if r["response_id"] in side and r["test_name"] == metric
                ]
                for side in ids
            ]
            entry = {
                "key": key,
                "test_name": metric,
                "metric_name": next(
                    (
                        r["metric_name"]
                        for side in sides
                        for r in side
                        if r.get("metric_name")
                    ),
                    metric,
                ),
                "ids": ids,
                "stored": [bool(side) for side in sides],
            }
            if not all(sides):
                entries.append({**entry, "excluded": NOT_STORED_ON_A_SIDE})
                continue
            versions = [
                {signature(r) for r in side if r.get("metric_version")}
                for side in sides
            ]
            common = set.intersection(*versions)
            if not common:
                entries.append(
                    {
                        **entry,
                        "excluded": "Not comparable: scoring versions differ or are unknown",
                    }
                )
                continue
            # Select the most recently used version that exists on both sides.
            version = max(
                common,
                key=lambda v: max(
                    (r.get("run_at") or "", r.get("id") or 0)
                    for side in sides
                    for r in side
                    if signature(r) == v
                ),
            )
            selected = [
                latest_results([r for r in side if signature(r) == version])
                for side in sides
            ]
            judges = {
                r.get("judge_llm")
                for side in selected
                for r in side
                if measured(r) and r.get("judge_llm")
            }
            if len(judges) > 1:
                entries.append(
                    {**entry, "excluded": "Not comparable: different actual judges"}
                )
                continue
            entries.append(
                {
                    **entry,
                    "excluded": None,
                    "selected": selected,
                    "totals": [aggregate_metrics(side)[0] for side in selected],
                }
            )
    return entries


def _unmeasured(total, ids):
    """Runs a check should have measured and did not.

    A run the check does not apply to, such as a conversational run for a deep
    research check, is not counted: nothing went wrong there.
    """
    not_applicable = sum(
        unmeasured_state(r) == NOT_APPLICABLE for r in total["not_scored_rows"]
    )
    return len(ids) - total["measured_count"] - not_applicable


def _direction(rates, states=None):
    if rates[1] > rates[0]:
        return "More passes"
    if rates[1] < rates[0]:
        return "Fewer passes"
    if states and states[0] == states[1]:
        return states[0]
    return "Same pass frequency"


def compare(baseline, candidate, rows):
    """Return matched verdicts and explicit exclusions; legacy versions cannot match."""
    groups = [_cohorts(records) for records in (baseline, candidate)]
    left, right = groups
    matched = left.keys() & right.keys()
    output = []
    for item in _matched_entries(groups, matched, rows):
        key = item["key"]
        entry = {
            "Question": f"Q{key[0]}",
            "Chat mode": key[2],
            "Research mode": key[3],
            "Check": item["metric_name"],
        }
        if item["excluded"]:
            output.append({**entry, "Change": item["excluded"]})
            continue
        ids, totals = item["ids"], item["totals"]
        responses = {}
        for label, total, side_ids, side in zip(
            ("Baseline", "Candidate"), totals, ids, item["selected"], strict=True
        ):
            entry[label] = (
                f"{total['pass_count']}/{total['measured_count']} measured passes; {_unmeasured(total, side_ids)} unmeasured or missing"
            )
            responses[f"{label} responses"] = ", ".join(
                str(r["response_id"]) for r in side
            )
        unmeasured = [
            _unmeasured(t, side_ids) for t, side_ids in zip(totals, ids, strict=True)
        ]
        if not any(t["measured_count"] for t in totals):
            # Only runs the check excludes by design are Not applicable. A run
            # that failed to score, such as a judge error, is Not measured.
            change = "Not measured" if any(unmeasured) else "Not applicable"
        elif any(unmeasured) or not all(t["measured_count"] for t in totals):
            change = "Incomplete measurements"
        else:
            change = _direction(
                [t["pass_count"] / t["measured_count"] for t in totals],
                [t["state"] for t in totals],
            )
        output.append({**entry, "Change": change, **responses})
    summary = {
        "Matched questions and modes": len(matched),
        "Baseline only": len(left.keys() - right.keys()),
        "Candidate only": len(right.keys() - left.keys()),
    }
    outcomes = [
        {
            "Experiment": label,
            **outcome_counts([r for key in matched for r in side[key]]),
            **run_totals([r for key in matched for r in side[key]]),
        }
        for label, side in zip(("Baseline", "Candidate"), groups, strict=True)
    ]
    return summary, output, outcomes


# The rule a reviewer reads under the per check summary. A gap this many runs
# wide or narrower can come from asking the same questions again.
NORMAL_VARIATION_RUNS = 2
NORMAL_VARIATION_RULE = (
    f"A change of {NORMAL_VARIATION_RUNS} runs or fewer, out of about 24, is "
    "within normal run-to-run variation."
)

# What the per check summary's change column is called. It compares how often
# a check passed, which can move the other way from the mean score, so the
# header says which of the two it is.
PASS_FREQUENCY_CHANGE = "Change in pass frequency"
PASS_RATE_DELTA = "Pass rate change (points)"

# Every change a check can end up in, in the order they are counted. Fixed so
# a column keeps its place from one comparison to the next.
CHANGE_STATES = (
    "Better",
    "Worse",
    "Within normal variation",
    "Not measured",
    "Not comparable",
)


def _change(sides):
    """Better, Worse or Within normal variation, by the gap in runs.

    The gap is the difference in pass rate times the larger side's measured
    runs, so two sides that measured different numbers of runs are compared
    on the same footing.
    """
    rates = [side["passes"] / side["measured"] for side in sides]
    runs = abs(rates[1] - rates[0]) * max(side["measured"] for side in sides)
    if round(runs) <= NORMAL_VARIATION_RUNS:
        return "Within normal variation"
    return "Better" if rates[1] > rates[0] else "Worse"


def change_counts(summary_rows):
    """How many checks moved which way, from the rows of `metric_summary`."""
    counts = dict.fromkeys(CHANGE_STATES, 0)
    for row in summary_rows:
        state = row[PASS_FREQUENCY_CHANGE]
        counts[state] = counts.get(state, 0) + 1
    return {"Checks": len(summary_rows), **counts}


def metric_summary(baseline, candidate, rows):
    """One row per check, totalled over the questions both experiments answered.

    Questions whose scoring versions or judges differ between the two sides are
    counted in "Not compared" and are left out of the totals, so an unmatched
    scoring version cannot look like a change. Consistency gives one verdict
    per question, so it is counted per question. Better and Worse rows come
    first, then the rest, each largest change first.
    """
    groups = [_cohorts(records) for records in (baseline, candidate)]
    matched = groups[0].keys() & groups[1].keys()
    totals = {}
    for item in _matched_entries(groups, matched, rows):
        row = totals.setdefault(
            item["test_name"],
            {
                "Check": item["metric_name"],
                "sides": [
                    dict(passes=0, measured=0, score=0.0, missing=0) for _ in (0, 1)
                ],
                "compared": 0,
                "excluded": 0,
                "mismatched": 0,
            },
        )
        if item["excluded"]:
            row["excluded"] += 1
            row["mismatched"] += item["excluded"] != NOT_STORED_ON_A_SIDE
            continue
        row["compared"] += 1
        for side, total, ids in zip(
            row["sides"], item["totals"], item["ids"], strict=True
        ):
            if item["test_name"] == "consistency":
                if total["measured_count"]:
                    side["passes"] += total["pass_count"] == total["measured_count"]
                    side["measured"] += 1
                    side["score"] += total["score"]
                elif _unmeasured(total, ids):
                    side["missing"] += 1
                continue
            side["passes"] += total["pass_count"]
            side["measured"] += total["measured_count"]
            side["score"] += total["score"] * total["measured_count"]
            side["missing"] += _unmeasured(total, ids)
    output = []
    for row in totals.values():
        sides = row["sides"]
        delta = None
        if not row["compared"]:
            # Comparable only fails when versions or judges differ. A check
            # one side never stored is unmeasured there, not incomparable.
            change = "Not comparable" if row["mismatched"] else "Not measured"
        elif not all(side["measured"] for side in sides):
            change = "Not measured"
        else:
            change = _change(sides)
            delta = round(
                100 * (sides[1]["passes"] / sides[1]["measured"])
                - 100 * (sides[0]["passes"] / sides[0]["measured"])
            )
        entry = {"Check": row["Check"], PASS_FREQUENCY_CHANGE: change}
        for label, side in zip(("Baseline", "Candidate"), sides, strict=True):
            entry[f"{label} passes"] = side["passes"]
            entry[f"{label} measured"] = side["measured"]
            entry[f"{label} mean"] = (
                round(side["score"] / side["measured"], 2) if side["measured"] else None
            )
        entry |= {
            PASS_RATE_DELTA: delta,
            "Unmeasured runs": sides[0]["missing"] + sides[1]["missing"],
            "Questions compared": row["compared"],
            "Not compared": row["excluded"],
        }
        output.append(entry)
    return sorted(
        output,
        key=lambda r: (
            CHANGE_STATES.index(r[PASS_FREQUENCY_CHANGE]) > 1,
            -abs(r[PASS_RATE_DELTA] or 0),
        ),
    )


def run_totals(records):
    """Cost, time and summarisation model over a set of runs."""
    costs = [r["total_cost_usd"] for r in records if r.get("total_cost_usd")]
    times = [r["total_ms"] / 1000 for r in records if r.get("total_ms")]
    return {
        "Total cost ($)": round(sum(costs), 2) if costs else None,
        "Mean cost ($)": round(sum(costs) / len(costs), 3) if costs else None,
        "Mean time (s)": round(sum(times) / len(times)) if times else None,
        "Slowest (s)": round(max(times)) if times else None,
        "Summarisation model": ", ".join(
            sorted({r.get("summarisation_llm") or "n/a" for r in records})
        ),
    }


def overview_rows(sides, rows):
    """One row per experiment, totalled over the questions every side asked.

    Returns how many question and mode combinations were matched, and for each
    side its run outcomes, cost, time and the passes of every check, counted
    only on questions where every side has a stored result. A check reads
    "Not comparable" when no shared question had a scoring version common to
    every side, and "Not compared" when no shared question had a result on
    every side. Consistency gives one verdict per question, so it is counted
    per question rather than per response.
    """
    groups = [_cohorts(records) for records in sides]
    matched = set.intersection(*(set(g) for g in groups)) if groups else set()
    output = []
    for group in groups:
        records = [r for key in matched for r in group[key]]
        counts = outcome_counts(records)
        output.append(
            {
                "Responses": counts["Attempts"],
                "Answered": counts["Answer"],
                "Clarification": counts["Clarification"],
                "Error": counts["Error"],
                "No response text": counts["No answer"],
                "Limit reached": counts["Turn-cap flags"],
                **run_totals(records),
                "checks": {},
            }
        )
    tallies = {}
    for item in _matched_entries(groups, matched, rows):
        tally = tallies.setdefault(
            item["metric_name"],
            {
                "sides": [[0, 0, False] for _ in groups],
                "compared": 0,
                "excluded": 0,
                "mismatched": 0,
            },
        )
        for side, stored in zip(tally["sides"], item["stored"], strict=True):
            side[2] = side[2] or stored
        if item["excluded"]:
            tally["excluded"] += 1
            tally["mismatched"] += item["excluded"] != NOT_STORED_ON_A_SIDE
            continue
        tally["compared"] += 1
        for side, total in zip(tally["sides"], item["totals"], strict=True):
            if item["test_name"] == "consistency":
                if total["measured_count"]:
                    side[0] += total["pass_count"] == total["measured_count"]
                    side[1] += 1
            else:
                side[0] += total["pass_count"]
                side[1] += total["measured_count"]
    for name, tally in tallies.items():
        for row, (passes, measured_count, stored) in zip(
            output, tally["sides"], strict=True
        ):
            if not stored:
                text = NO_STORED_RESULT
            elif not tally["compared"]:
                text = "Not comparable" if tally["mismatched"] else "Not compared"
            elif not measured_count:
                text = "Not measured"
            else:
                text = f"{passes}/{measured_count}"
                if tally["excluded"]:
                    text += f" ({tally['excluded']} not compared)"
            row["checks"][name] = text
    return len(matched), output


def up_to_date(experiment_ids, rows):
    """The experiments with a score on the newest version of every check.

    A check writes a row even for a response it cannot measure, so an
    experiment missing a check entirely has not been scored with it. The
    exception is Consistency, which writes nothing for a question asked once.

    The overview selects these by default, because one experiment scored with
    an older version makes that check Not comparable for all of them.
    """
    newest = {}
    for row in rows:
        if not row.get("metric_version"):
            continue
        order = (row.get("run_at") or "", row.get("id") or 0)
        if row["test_name"] not in newest or order > newest[row["test_name"]][0]:
            newest[row["test_name"]] = (order, row["metric_version"])
    versions = defaultdict(lambda: defaultdict(set))
    for row in rows:
        versions[row.get("experiment_id")][row["test_name"]].add(
            row.get("metric_version")
        )
    return [
        exp
        for exp in experiment_ids
        if all(
            version in versions[exp][test]
            for test, (_, version) in newest.items()
            if test != "consistency" or versions[exp][test]
        )
    ]


def _md_row(cells):
    return "| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |"


def _md_table(header, rows):
    return "\n".join(
        [_md_row(header), _md_row(["---"] * len(header)), *map(_md_row, rows)]
    )


def overview_markdown(mode, matched, entries, checks, date):
    """The overview as the two tables a findings document opens with.

    *entries* holds one dict per experiment: its column name, label, model,
    run totals and the "checks" text from `overview_rows`. *checks* is check
    name to row label, in display order. A check no experiment measured is
    left out, as a findings table would leave it out.
    """
    blank = {"Not measured", "No stored result"}
    run_header = [
        "Experiment",
        "Model",
        "Summarisation model",
        "Responses",
        "Answered",
        "Clarification",
        "Error",
        "Total cost ($)",
        "Mean cost ($)",
        "Mean time (s)",
        "Slowest (s)",
    ]
    runs = [
        [e.get(h) if e.get(h) is not None else "n/a" for h in run_header]
        for e in entries
    ]
    rows = []
    for name, label in checks.items():
        cells = [e["checks"].get(name, "No stored result") for e in entries]
        if not all(cell in blank for cell in cells):
            rows.append([label, *cells])
    return (
        "\n\n".join(
            [
                f"# Overview, {mode} ({date})",
                f"Date: {date}. {matched} question and mode combination(s) asked by "
                "every experiment below; questions only some of them asked are left "
                "out. Each check is passes out of measured runs, Consistency is "
                "counted per question, and rows no experiment measured are omitted.",
                "## Runs and cost",
                _md_table(run_header, runs),
                "## Pass counts by check",
                _md_table(["Check", *(e["Column"] for e in entries)], rows),
            ]
        )
        + "\n"
    )
