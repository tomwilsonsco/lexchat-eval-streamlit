"""
The data containers metrics score against.

``LLMTestCase`` is one captured LexChat response in the shape the metrics
expect: the question, the final answer, the legal text retrieved, and the
tools called. ``utils/test_helpers.py::record_to_test_case`` builds one from
a DuckDB ``responses`` row.

Only the fields the metrics actually read are defined here. Every read site
guards with ``or ""`` / ``or []``, so unset fields default to ``None``.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class ToolCall(BaseModel):
    """One tool invocation captured from the response stream.

    ``output`` is deliberately untyped: LexChat's tools return strings for
    some calls and JSON objects for others, and the metrics branch on which
    they got. Coercing it to a string here would break that.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = ""
    input_parameters: Optional[Dict[str, Any]] = None
    output: Any = None


class LLMTestCase(BaseModel):
    """One captured response, ready to be scored."""

    model_config = ConfigDict(extra="ignore")

    input: str = ""
    actual_output: Optional[str] = None
    retrieval_context: Optional[List[str]] = None
    tools_called: Optional[List[ToolCall]] = None
