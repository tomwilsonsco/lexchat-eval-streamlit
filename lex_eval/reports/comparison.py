"""Matched experiment comparisons using compatible stored scoring results."""

from collections import defaultdict

from lex_eval.reports.data import (
    aggregate_metrics,
    latest_results,
    measured,
    outcome_counts,
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


def _matched_entries(groups, matched, rows):
    """One entry per matched cohort and metric, with the selected verdicts.

    An entry either carries a reason it could not be compared, or the two
    sides' response ids and their aggregated verdicts. Both the per question
    table and the per check summary are built from these, so they select the
    same scoring version and exclude the same rows.
    """
    entries = []
    for key in sorted(matched):
        ids = [{r["response_id"] for r in side[key]} for side in groups]
        metric_keys = {
            r["test_name"] for r in rows if r["response_id"] in ids[0] | ids[1]
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
            }
            versions = [
                {signature(r) for r in side if r.get("metric_version")}
                for side in sides
            ]
            common = versions[0] & versions[1]
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
                f"{total['pass_count']}/{total['measured_count']} measured passes; {len(side_ids) - total['measured_count']} unmeasured or missing"
            )
            responses[f"{label} responses"] = ", ".join(
                str(r["response_id"]) for r in side
            )
        if any(
            t["measured_count"] != len(side_ids)
            for t, side_ids in zip(totals, ids, strict=True)
        ):
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
        }
        for label, side in zip(("Baseline", "Candidate"), groups, strict=True)
    ]
    return summary, output, outcomes


def _side_text(passes, measured_count, score_total, unmeasured):
    mean = f"mean {score_total / measured_count:.2f}" if measured_count else "no mean"
    text = f"{passes}/{measured_count} measured passes · {mean}"
    return f"{text} · {unmeasured} unmeasured" if unmeasured else text


# Every direction a check can end up in, in the order they are counted. Fixed
# so a column keeps its place from one comparison to the next.
# What the per check summary's direction column is called. It compares how
# often a check passed, which can move the other way from the mean score, so
# the header says which of the two it is.
PASS_FREQUENCY_CHANGE = "Change in pass frequency"

CHANGE_STATES = (
    "More passes",
    "Fewer passes",
    "Same pass frequency",
    "Not measured",
    "Not comparable",
)


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
    scoring version cannot look like a change.
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
            },
        )
        if item["excluded"]:
            row["excluded"] += 1
            continue
        row["compared"] += 1
        for side, total, ids in zip(
            row["sides"], item["totals"], item["ids"], strict=True
        ):
            side["passes"] += total["pass_count"]
            side["measured"] += total["measured_count"]
            side["score"] += total["score"] * total["measured_count"]
            side["missing"] += len(ids) - total["measured_count"]
    output = []
    for row in totals.values():
        sides = row["sides"]
        if not row["compared"]:
            change = "Not comparable"
        elif not all(side["measured"] for side in sides):
            change = "Not measured"
        else:
            change = _direction([side["passes"] / side["measured"] for side in sides])
        output.append(
            {
                "Check": row["Check"],
                **{
                    label: _side_text(
                        side["passes"], side["measured"], side["score"], side["missing"]
                    )
                    for label, side in zip(
                        ("Baseline", "Candidate"), sides, strict=True
                    )
                },
                PASS_FREQUENCY_CHANGE: change,
                "Questions compared": row["compared"],
                "Not compared": row["excluded"],
            }
        )
    return output
