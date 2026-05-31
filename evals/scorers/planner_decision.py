"""
Planner decision scorer for the eval harness.

Three sub-scores returned in PlannerDecisionScore:

  final_action_score    — 1.0 if the final TraceStep.action matches the expected
                          outcome for the test case category; 0.0 otherwise.
                          Unconstrained categories (edge) always score 1.0 here.

  adjustment_compliance — fraction of relax/tighten trace steps that carry a
                          non-null, non-"none" adjustment value. 1.0 vacuously
                          when there are no relax/tighten steps.

  fallback_correct      — whether the deterministic fallback path (identified by
                          "(fallback)" in step_name) fires only when expected:
                            1.0  fallback_stress case and fallback triggered
                            0.5  fallback_stress case but fallback not triggered
                            1.0  non-fallback_stress case and no fallback fired
                            0.0  non-fallback_stress case and fallback fired

Composite score: 0.0 when final_action_score == 0.0 (hard fail on wrong final
action); otherwise the arithmetic mean of all three sub-scores.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Expected final TraceStep.action per test-case category.
# Categories absent from this dict are unconstrained (any final action is acceptable).
_EXPECTED_FINAL_ACTION: dict[str, str] = {
    "easy": "accept",
    "sparse": "accept",       # relax then accept
    "broad": "accept",        # tighten then accept
    "street_hint": "accept",
    "ambiguous": "clarify",
    "multi_turn": "accept",
    "fallback_stress": "accept",
    # "edge" omitted — contradictory constraints may produce clarify or accept
}

_FALLBACK_MARKER = "(fallback)"


@dataclass
class PlannerDecisionScore:
    score: float                  # composite score (0.0–1.0)
    final_action_score: float     # 1.0 = final action matches expected
    adjustment_compliance: float  # fraction of relax/tighten steps with valid adjustment
    fallback_correct: float       # 1.0 correct; 0.0 unexpected; 0.5 stress not triggered
    expected_final_action: str | None
    actual_final_action: str | None
    fallback_fired: bool
    reason: str


def _to_dict(obj: Any) -> dict[str, Any]:
    # actual may be a live SearchResponse (Pydantic) or a pre-serialised dict from fixtures
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def _normalize_step(step: Any) -> dict[str, Any]:
    # TraceStep objects arrive as Pydantic models when called live; as plain dicts from YAML fixtures
    if hasattr(step, "model_dump"):
        return step.model_dump()
    if isinstance(step, dict):
        return step
    return dict(step)


def _get_trace(actual_dict: dict[str, Any]) -> list[dict[str, Any]]:
    return [_normalize_step(s) for s in (actual_dict.get("trace") or [])]


def _final_action(trace: list[dict[str, Any]]) -> str | None:
    return trace[-1].get("action") if trace else None


def _has_fallback(trace: list[dict[str, Any]]) -> bool:
    # The orchestrator appends "(fallback)" to step_name when it skips the planner and
    # applies a hard-coded default — cheaper to detect via naming convention than a new field
    return any(_FALLBACK_MARKER in (step.get("step_name") or "") for step in trace)


def _adjustment_compliance(trace: list[dict[str, Any]]) -> float:
    """Fraction of relax/tighten steps with a non-null, non-'none' adjustment."""
    steps = [s for s in trace if s.get("action") in {"relax", "tighten"}]
    if not steps:
        # No relax/tighten steps means the planner never needed to adjust; vacuously perfect
        return 1.0
    compliant = sum(
        1 for s in steps
        if s.get("adjustment") and s.get("adjustment") != "none"
    )
    return compliant / len(steps)


def score(
    actual: Any,
    test_case: dict[str, Any],
) -> PlannerDecisionScore:
    """
    Public entry point — scores how well the planner loop behaved for the given test case.

    Computes three sub-scores from the trace embedded in actual (a SearchResponse or dict):
      - final_action_score    → 1.0 if the last TraceStep.action matches the category's
                                expected outcome; 1.0 also for unconstrained categories.
      - adjustment_compliance → fraction of relax/tighten steps that carry a non-null,
                                non-"none" adjustment field.
      - fallback_correct      → whether the deterministic fallback fired only when expected
                                (see module docstring for the 0.0 / 0.5 / 1.0 breakdown).

    test_case must contain a "category" key that maps to _EXPECTED_FINAL_ACTION.
    Categories absent from that dict (e.g. "edge") are unconstrained and always
    score 1.0 for final_action_score.

    Returns score=0.0 when final_action_score == 0.0 (hard fail); otherwise the
    arithmetic mean of all three sub-scores.
    """
    actual_dict = _to_dict(actual)
    trace = _get_trace(actual_dict)
    category: str = test_case.get("category", "")

    if not trace:
        return PlannerDecisionScore(
            score=0.0,
            final_action_score=0.0,
            adjustment_compliance=0.0,
            fallback_correct=0.0,
            expected_final_action=_EXPECTED_FINAL_ACTION.get(category),
            actual_final_action=None,
            fallback_fired=False,
            reason="empty trace — planner decisions cannot be evaluated",
        )

    # Sub-score 1: final action
    expected_final = _EXPECTED_FINAL_ACTION.get(category)
    actual_final = _final_action(trace)

    if expected_final is None:
        final_action_score = 1.0
        action_reason = f"category '{category}' is unconstrained; actual='{actual_final}'"
    elif actual_final == expected_final:
        final_action_score = 1.0
        action_reason = f"correct — final action '{actual_final}' matches expected '{expected_final}'"
    else:
        final_action_score = 0.0
        action_reason = (
            f"wrong final action '{actual_final}' "
            f"(expected '{expected_final}' for category '{category}')"
        )

    # Sub-score 2: adjustment compliance
    adj_compliance = _adjustment_compliance(trace)

    # Sub-score 3: fallback correctness
    fallback_fired = _has_fallback(trace)
    if category == "fallback_stress":
        if fallback_fired:
            fallback_correct = 1.0
            fallback_reason = "fallback triggered as expected for fallback_stress case"
        else:
            # 0.5 rather than 0.0: the test case may simply be too easy, not a conformance bug
            fallback_correct = 0.5
            fallback_reason = "fallback_stress case — fallback did not trigger (test may be too easy)"
    else:
        if fallback_fired:
            fallback_correct = 0.0
            fallback_reason = (
                f"unexpected fallback in category '{category}' — LLM conformance issue"
            )
        else:
            fallback_correct = 1.0
            fallback_reason = f"no unexpected fallback for category '{category}'"

    # A wrong final action is disqualifying — the planner reached the wrong conclusion
    # regardless of how well it adjusted along the way
    if final_action_score == 0.0:
        composite = 0.0
    else:
        composite = (final_action_score + adj_compliance + fallback_correct) / 3.0

    reason_parts = [action_reason]
    if adj_compliance < 1.0:  # only surface when something went wrong, to keep the reason readable
        reason_parts.append(f"adjustment compliance={adj_compliance:.2f}")
    reason_parts.append(fallback_reason)

    return PlannerDecisionScore(
        score=composite,
        final_action_score=final_action_score,
        adjustment_compliance=adj_compliance,
        fallback_correct=fallback_correct,
        expected_final_action=expected_final,
        actual_final_action=actual_final,
        fallback_fired=fallback_fired,
        reason="; ".join(reason_parts),
    )
