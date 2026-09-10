"""Facts from audit traces, without additional scores or judge calls."""

import json


def _search_outcome(tool: dict, name: str) -> tuple[int | None, str | None]:
    """How many items a search returned, and any error it reported.

    Reads the stored result text when it is there. A deploy copy keeps the
    count in ``result_count`` and folds the error into ``error`` instead of
    carrying the text, so this reads that pair when the text has gone. The
    count is None whenever it cannot be known either way.
    """
    error = tool.get("error")
    if "result_count" in tool:
        return tool["result_count"], error
    raw = tool.get("raw_result")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        parsed = None
    items = (
        parsed
        if name == "search_legislation_sections"
        else parsed.get("results") if isinstance(parsed, dict) else None
    )
    if not error and isinstance(parsed, dict):
        error = parsed.get("error")
    return (len(items) if isinstance(items, list) else None), error


def searches(record: dict) -> list[dict]:
    audit = record.get("audit_json") or {}
    if isinstance(audit, str):
        audit = json.loads(audit)
    rows = []
    for index, step in enumerate(audit.get("delegations", []), 1):
        for tool in step.get("tools", []):
            name = tool.get("name", "")
            if not name.startswith("search_"):
                continue
            count, error = _search_outcome(tool, name)
            if error:
                state, count = "Error", None
            elif tool.get("truncated") or tool.get("budget_blocked"):
                state, count = "Unknown / incomplete", None
            elif count is not None:
                state = "Empty" if not count else "Results returned"
            else:
                state = "Unknown / incomplete"
            rows.append(
                {
                    "Response": record["response_id"],
                    "Step": step.get("step", index),
                    "Tool": name,
                    "Arguments": json.dumps(tool.get("args") or {}, ensure_ascii=False),
                    "Outcome": state,
                    "Returned": count,
                    "Cache reused": bool(
                        tool.get("memo_hit") or tool.get("local_cache_hit")
                    ),
                    "Error": str(error or ""),
                }
            )
    for index, row in enumerate(rows):
        row["Later nonempty search in this step"] = row["Outcome"] == "Empty" and any(
            later["Step"] == row["Step"]
            and later["Tool"] == row["Tool"]
            and (later["Returned"] or 0) > 0
            for later in rows[index + 1 :]
        )
    return rows


def search_summary(rows: list[dict]) -> list[dict]:
    """Search calls counted per response, tool and outcome.

    Grouping on the outcome means a group's calls either all carry a count or
    all carry none, so "Items returned" is never a partial sum.
    """
    totals: dict[tuple, dict] = {}
    for row in rows:
        key = (row["Response"], row["Tool"], row["Outcome"])
        entry = totals.setdefault(
            key,
            {
                "Response": key[0],
                "Tool": key[1],
                "Outcome": key[2],
                "Searches": 0,
                "Items returned": None,
            },
        )
        entry["Searches"] += 1
        if row["Returned"] is not None:
            entry["Items returned"] = (entry["Items returned"] or 0) + row["Returned"]
    return [totals[key] for key in sorted(totals)]


def plan_steps(record: dict) -> list[dict]:
    audit = record.get("audit_json") or {}
    if isinstance(audit, str):
        audit = json.loads(audit)
    return [
        {
            "Step": d.get("step", i),
            "Title": d.get("title", ""),
            "Tools": len(d.get("tools", [])),
            "Report": d.get("report") or "",
            "Error": d.get("error"),
            "Reformatted": bool(d.get("reformatted")),
        }
        for i, d in enumerate(audit.get("delegations", []), 1)
    ]
