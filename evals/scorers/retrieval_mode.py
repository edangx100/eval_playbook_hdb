"""
Retrieval mode scorer for the eval harness.

Checks whether the final trace step used the correct retrieval mode.

Two cases:
  Constrained (test case has expected_retrieval_mode: hybrid):
    The agent MUST have used hybrid retrieval. Score = 1.0 if the final
    TraceStep.retrieval_mode is "hybrid", 0.0 otherwise. This applies to
    street_hint cases, where vector search is required for meaningful results.

  Unconstrained (no expected_retrieval_mode in test case):
    Either "structured" or "hybrid" is acceptable — the orchestrator picks the
    mode based on count and street hints, and both are correct in different
    situations. Score is always 1.0.

Score is always 0.0 if there are no trace steps, since the mode cannot be
determined from an empty trace.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalModeScore:
    score: float              # 1.0 = pass, 0.0 = fail
    expected_mode: str | None # required mode from test case, or None if unconstrained
    actual_mode: str | None   # retrieval_mode from the final trace step, or None
    reason: str               # human-readable explanation of the outcome


def _to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def _final_retrieval_mode(actual_dict: dict[str, Any]) -> str | None:
    """Extract retrieval_mode from the last step in the trace, or None if no trace."""
    trace = actual_dict.get("trace") or []
    if not trace:
        return None
    last_step = trace[-1]
    if hasattr(last_step, "model_dump"):
        last_step = last_step.model_dump()
    elif not isinstance(last_step, dict):
        last_step = dict(last_step)
    return last_step.get("retrieval_mode")


def score(
    actual: Any,
    test_case: dict[str, Any],
) -> RetrievalModeScore:
    """
    Public entry point — score retrieval mode correctness.

    Reads expected_retrieval_mode from the test case (set on street_hint cases
    in the benchmark). If present, the actual final-step mode must match exactly.
    If absent, any valid mode is accepted.
    """
    actual_dict = _to_dict(actual)
    actual_mode = _final_retrieval_mode(actual_dict)
    expected_mode: str | None = test_case.get("expected_retrieval_mode")

    if actual_mode is None:
        return RetrievalModeScore(
            score=0.0,
            expected_mode=expected_mode,
            actual_mode=None,
            reason="no trace steps — retrieval mode cannot be determined",
        )

    if expected_mode is None:
        # Unconstrained: structured and hybrid are both legitimate outcomes.
        return RetrievalModeScore(
            score=1.0,
            expected_mode=None,
            actual_mode=actual_mode,
            reason=f"no expected_retrieval_mode constraint; actual mode '{actual_mode}' is acceptable",
        )

    # Constrained (e.g. street_hint cases require hybrid).
    if actual_mode == expected_mode:
        return RetrievalModeScore(
            score=1.0,
            expected_mode=expected_mode,
            actual_mode=actual_mode,
            reason=f"correct — actual mode '{actual_mode}' matches expected '{expected_mode}'",
        )

    return RetrievalModeScore(
        score=0.0,
        expected_mode=expected_mode,
        actual_mode=actual_mode,
        reason=f"wrong mode — expected '{expected_mode}' but got '{actual_mode}'",
    )
