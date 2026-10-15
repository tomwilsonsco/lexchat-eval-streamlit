"""
The base class every metric in this package subclasses.

A metric is scored by calling ``measure(test_case)``, which sets ``score``,
``reason`` and ``success`` on the metric instance. The test function then
reads those back and passes them to ``utils/collector.py::attach_metric``,
which is what puts a row in the metric's ``eval_<metric>`` table.

Subclasses set their own state in ``__init__`` and do not call
``super().__init__()``; the attributes below are class-level defaults so
that a metric which never assigns one still reads back a sensible value.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..testcase import LLMTestCase


class BaseMetric(ABC):
    # The score to beat for `success` to be True. Set by every subclass.
    threshold: float = 0.5

    # Written by `measure`.
    score: float = 0.0
    reason: str = ""
    success: bool = False

    # Set only when a metric could not be scored at all, e.g. the judge call
    # failed. A metric that scored normally leaves this None.
    error: Optional[str] = None

    # The AI judge, for metrics that use one. Any object with a
    # `.generate(prompt, schema=...)` method, see utils/judge.py. Left None
    # by the offline metrics.
    model: Optional[Any] = None

    @abstractmethod
    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        """Score *test_case*, set score/reason/success, and return the score.

        Never validates the type of *test_case*: tests pass lightweight
        stand-ins carrying only the fields a given metric reads.
        """
        raise NotImplementedError

    def is_successful(self) -> bool:
        """Whether the last `measure` call passed. Overridden by every metric."""
        return bool(self.success)

    @property
    def __name__(self) -> str:
        """The metric's display name, used as `metric_name` in the database."""
        return type(self).__name__
