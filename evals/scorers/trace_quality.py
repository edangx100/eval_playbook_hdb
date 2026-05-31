"""
Trace quality scorer for the eval harness.

Checks that the planner loop produced a structurally complete trace.

Three sub-scores:

  step_completeness       — fraction of TraceStep entries that have a non-null
                            action and a non-null count.  1.0 vacuously when the
                            trace is empty (nothing to be incomplete about — the
                            empty-trace case is caught by the composite guard).

  adjustment_completeness — fraction of relax/tighten steps that carry both a
                            non-null adjustment AND a non-null adjustment_note.
                            1.0 vacuously when there are no relax/tighten steps.

  final_step_valid        — 1.0 if the final step has action in {accept, clarify}
                            AND a non-null, non-empty retrieval_mode; 0.0 otherwise.

Composite score:
  - 0.0 immediately for an empty trace (no steps = nothing to evaluate).
  - Otherwise the arithmetic mean of the three sub-scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TraceQualityScore:
    score: float                    # composite score (0.0–1.0)
    step_completeness: float        # fraction of steps with non-null action + count
    adjustment_completeness: float  # fraction of relax/tighten steps with adjustment + note
    final_step_valid: float         # 1.0 if final step is accept/clarify with retrieval_mode
    n_steps: int
    n_adjustment_steps: int         # count of relax/tighten steps
    reason: str


def _to_dict(obj: Any) -> dict[str, Any]:
    # actual arrives as a live SearchResponse (Pydantic) in production and as a plain
    # dict in unit tests — model_dump() is required for the Pydantic path because
    # dict(pydantic_obj) returns internal metadata rather than field values.
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def _normalize_step(step: Any) -> dict[str, Any]:
    # Each element in the trace list may be a TraceStep Pydantic object (live run)
    # or a plain dict (fixture / serialised output) — normalise to dict so the rest
    # of the scorer can use .get() uniformly without branching on type.
    if hasattr(step, "model_dump"):
        return step.model_dump()
    if isinstance(step, dict):
        return step
    return dict(step)


def _get_trace(actual_dict: dict[str, Any]) -> list[dict[str, Any]]:
    # `or []` handles both a missing "trace" key and an explicit None value without
    # raising a TypeError when iterating.
    return [_normalize_step(s) for s in (actual_dict.get("trace") or [])]


def _step_completeness(trace: list[dict[str, Any]]) -> float:
    """Fraction of steps with non-null action and non-null count."""
    if not trace:
        # Return 1.0 rather than 0.0 so the empty-trace penalty lives entirely in the
        # composite guard (score=0.0 for empty trace), not split across sub-scores.
        return 1.0
    # Both fields must be present: action identifies what the planner decided,
    # count is what drove that decision. A step missing either is uninterpretable.
    complete = sum(
        1 for s in trace
        if s.get("action") is not None and s.get("count") is not None
    )
    return complete / len(trace)


def _adjustment_completeness(trace: list[dict[str, Any]]) -> tuple[float, int]:
    """
    Fraction of relax/tighten steps with non-null adjustment AND adjustment_note.
    Returns (score, n_adjustment_steps).
    """
    steps = [s for s in trace if s.get("action") in {"relax", "tighten"}]
    if not steps:
        # No adjustment steps means the planner accepted immediately — valid, not a flaw.
        return 1.0, 0
    # adjustment names the specific filter dimension changed (e.g. widen_time_window);
    # adjustment_note is the human-readable explanation surfaced in Braintrust traces.
    # Both are required — a step with only one is still incomplete for diagnostic purposes.
    complete = sum(
        1 for s in steps
        if s.get("adjustment") is not None and s.get("adjustment_note") is not None
    )
    return complete / len(steps), len(steps)


def _final_step_valid(trace: list[dict[str, Any]]) -> tuple[float, str]:
    """1.0 if the final step has action in {accept, clarify} and a retrieval_mode."""
    if not trace:
        return 0.0, "empty trace"
    final = trace[-1]
    action = final.get("action")
    retrieval_mode = final.get("retrieval_mode")

    # relax/tighten/initial as the last action means the loop exited before reaching
    # a terminal decision — the orchestrator should never produce this state.
    if action not in {"accept", "clarify"}:
        return 0.0, f"final step action is '{action}' (expected accept or clarify)"
    # retrieval_mode records which path (structured SQL vs hybrid vector+SQL) was used;
    # a null value means the orchestrator never committed to a mode before stopping.
    if not retrieval_mode:
        return 0.0, f"final step action='{action}' but retrieval_mode is missing/null"
    return 1.0, f"final step action='{action}', retrieval_mode='{retrieval_mode}'"


def score(
    actual: Any,
    # test_case is not read here — trace quality is self-contained in the trace itself.
    # The parameter exists because the eval runner calls every scorer with the same
    # (actual, test_case) signature; omitting it would break the uniform call site.
    test_case: dict[str, Any],  # noqa: ARG001
) -> TraceQualityScore:
    """
    Public entry point — checks structural completeness of the agent trace.

    actual must contain a "trace" key with a list of TraceStep-compatible objects
    (Pydantic models or plain dicts).  test_case is accepted but not read — it is
    present to satisfy the uniform scorer signature used by the eval runner.

    Returns score=0.0 for an empty trace.  For a non-empty trace, returns the
    arithmetic mean of step_completeness, adjustment_completeness, and
    final_step_valid.
    """
    actual_dict = _to_dict(actual)
    trace = _get_trace(actual_dict)

    if not trace:
        return TraceQualityScore(
            score=0.0,
            step_completeness=0.0,
            adjustment_completeness=0.0,
            final_step_valid=0.0,
            n_steps=0,
            n_adjustment_steps=0,
            reason="empty trace — no steps to evaluate",
        )

    step_comp = _step_completeness(trace)
    adj_comp, n_adj = _adjustment_completeness(trace)
    final_valid, final_reason = _final_step_valid(trace)

    composite = (step_comp + adj_comp + final_valid) / 3.0

    # Only surface failing sub-scores in the reason to keep it readable; the final
    # step outcome is always included because it's the most diagnostic field.
    reason_parts: list[str] = []
    if step_comp < 1.0:
        reason_parts.append(f"step_completeness={step_comp:.2f} (some steps missing action/count)")
    if adj_comp < 1.0:
        reason_parts.append(
            f"adjustment_completeness={adj_comp:.2f} "
            f"(relax/tighten step missing adjustment or note)"
        )
    reason_parts.append(final_reason)

    return TraceQualityScore(
        score=composite,
        step_completeness=step_comp,
        adjustment_completeness=adj_comp,
        final_step_valid=final_valid,
        n_steps=len(trace),
        n_adjustment_steps=n_adj,
        reason="; ".join(reason_parts),
    )
