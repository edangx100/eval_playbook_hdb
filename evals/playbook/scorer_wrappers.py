"""Braintrust scorer wrapper functions for the HDB eval harness.

Each wrapper adapts a scorer module's score() function to the Braintrust
callable interface: (input, output) → Score | list[Score] | None.

Braintrust uses the function __name__ as the score column name in the UI,
so the wrappers are named after the check they implement.

Returning None tells Braintrust to skip the scorer for this test case —
used when the check is absent from the case's 'checks' list. Skipped scorers
show up as None entries in EvalResult.scores; the diagnosis pass ignores
those so a skipped scorer does not get counted as a failure.
"""
from __future__ import annotations

from typing import Any

from braintrust.score import Score
from openai import OpenAI

import evals.scorers.llm_judge as _llm_judge_mod
import evals.scorers.planner_decision as _planner_decision_mod
import evals.scorers.reranking_quality as _reranking_quality_mod
import evals.scorers.retrieval_mode as _retrieval_mode_mod
import evals.scorers.retrieval_quality as _retrieval_quality_mod
import evals.scorers.target_extraction as _target_extraction_mod
import evals.scorers.trace_quality as _trace_quality_mod

# Set by run.py after braintrust.login() via set_proxy_client(). Read by the
# llm_judge wrapper at call time — Python module globals are resolved at call
# time, not at import time, so assigning before Eval() is sufficient.
_proxy_client: OpenAI | None = None


def set_proxy_client(client: OpenAI) -> None:
    """Inject the Braintrust-wrapped OpenRouter client before Eval() is called."""
    global _proxy_client
    _proxy_client = client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _in_checks(check_name: str, input_dict: dict[str, Any]) -> bool:
    """Return True if check_name is listed in the relevant 'checks' field.

    For multi-turn cases checks live on each turn, not at the top level.
    Scorers evaluate the final turn, so we look at turns[-1]["checks"].
    """
    if "turns" in input_dict:
        last_turn = input_dict["turns"][-1]
        return check_name in (last_turn.get("checks") or [])
    return check_name in (input_dict.get("checks") or [])


def _effective_turn(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Return the turn dict that scorers should evaluate against.

    For multi-turn cases this is the last turn (which has expected_target_delta,
    expected_count_range, etc.). For single-turn cases it's input_dict itself.
    """
    if "turns" in input_dict:
        return input_dict["turns"][-1]
    return input_dict


def _prior_expected_target(input_dict: dict[str, Any]) -> dict[str, Any] | None:
    """Return the expected_target from the turn before the final one, or None.

    Used by retrieval_quality delta scoring to reconstruct the effective constraint
    set (preserved fields come from the prior turn's expected_target).
    """
    if "turns" not in input_dict or len(input_dict["turns"]) < 2:
        return None
    return input_dict["turns"][-2].get("expected_target") or {}


# ---------------------------------------------------------------------------
# Scorer wrappers
# ---------------------------------------------------------------------------

def target_extraction(input_dict: dict[str, Any], output: dict[str, Any]) -> list[Score] | None:
    """Score field-level accuracy of the extracted Target against expected_target.

    For multi-turn cases, scores against expected_target_delta on the last turn.
    prior_actual (the previous turn's extracted Target) is required for delta
    scoring so the scorer can verify that preserved fields were not changed.
    """
    if not _in_checks("target_extraction", input_dict):
        return None
    result = _target_extraction_mod.score(
        output.get("target", {}),
        _effective_turn(input_dict),
        prior_actual=output.get("prior_target"),
    )
    return [
        Score(name="target_extraction", score=result.f1),
        Score(name="target_extraction_precision", score=result.precision),
        Score(name="target_extraction_recall", score=result.recall),
    ]


def retrieval_quality(input_dict: dict[str, Any], output: dict[str, Any]) -> Score | None:
    """Score the fraction of results satisfying the hard constraints in the test case.

    For multi-turn delta cases, passes prior_expected_target so the scorer can
    reconstruct the full effective constraint set (preserved fields + updated fields).
    """
    if not _in_checks("retrieval_quality", input_dict):
        return None
    result = _retrieval_quality_mod.score(
        output,
        _effective_turn(input_dict),
        prior_expected_target=_prior_expected_target(input_dict),
    )
    return Score(name="retrieval_quality", score=result.score)


def retrieval_mode(input_dict: dict[str, Any], output: dict[str, Any]) -> Score | None:
    """Score whether the orchestrator chose the correct retrieval mode."""
    if not _in_checks("retrieval_mode", input_dict):
        return None
    result = _retrieval_mode_mod.score(output, input_dict)
    return Score(
        name="retrieval_mode",
        score=result.score,
        metadata={"expected": result.expected_mode, "actual": result.actual_mode, "reason": result.reason},
    )


def planner_decision(input_dict: dict[str, Any], output: dict[str, Any]) -> list[Score] | None:
    """Score planner loop correctness: final action, adjustment compliance, fallback behaviour."""
    if not _in_checks("planner_decision", input_dict):
        return None
    result = _planner_decision_mod.score(output, input_dict)
    return [
        Score(name="planner_decision", score=result.score),
        Score(name="planner_final_action", score=result.final_action_score),
        Score(name="planner_adjustment_compliance", score=result.adjustment_compliance),
        Score(name="planner_fallback_correct", score=result.fallback_correct),
    ]


def reranking_quality(input_dict: dict[str, Any], output: dict[str, Any]) -> Score | None:
    """Score whether reranking improved area proximity vs the raw candidate pool."""
    if not _in_checks("reranking_quality", input_dict):
        return None
    result = _reranking_quality_mod.score(output, input_dict)
    return Score(
        name="reranking_quality",
        score=result.score,
        metadata={"top_n_maad": result.top_n_maad, "pool_maad": result.pool_maad, "reason": result.reason},
    )


def trace_quality(input_dict: dict[str, Any], output: dict[str, Any]) -> list[Score] | None:
    """Score structural completeness of the planner trace."""
    if not _in_checks("trace_quality", input_dict):
        return None
    result = _trace_quality_mod.score(output, input_dict)
    return [
        Score(name="trace_quality", score=result.score),
        Score(name="trace_step_completeness", score=result.step_completeness),
        Score(name="trace_adjustment_completeness", score=result.adjustment_completeness),
        Score(name="trace_final_step_valid", score=result.final_step_valid),
    ]


def llm_judge(input_dict: dict[str, Any], output: dict[str, Any]) -> Score | None:
    """LLM-as-judge: score whether the returned comparables are defensible for the query."""
    if not _in_checks("llm_judge", input_dict):
        return None
    result = _llm_judge_mod.score(output, input_dict, client=_proxy_client)
    return Score(name="llm_judge", score=result.score, metadata=result.metadata or {})


ALL_SCORERS = [
    target_extraction,
    retrieval_quality,
    retrieval_mode,
    planner_decision,
    reranking_quality,
    trace_quality,
    llm_judge,
]
