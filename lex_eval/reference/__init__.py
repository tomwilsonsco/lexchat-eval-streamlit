"""Reference ("gold") answers for the evaluation question set.

Answers are researched against the live LEX API using LexChat's own legislation
tools, so they rest on exactly the material LexChat would have retrieved, and are
written up from that retrieved text by their recorded author. Build them with:

    python -m lex_eval.reference.build

Metrics read them with `load_reference_answers()`, which returns only answers a
lawyer has signed off unless asked otherwise.
"""

from .store import load_reference_answers, load_manifest

__all__ = ["load_reference_answers", "load_manifest"]
