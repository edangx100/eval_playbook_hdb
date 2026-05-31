"""
Retrieval quality scorer for the eval harness.

Measures the fraction of returned results that satisfy the hard constraints
implied by the test case's expected_target: town, flat_type, date range
(derived from months_back), and floor area (target ± tolerance sqm).

Only constraints present and non-None in expected_target are enforced —
missing keys are skipped, so the scorer is safe to run against partial cases.

Scoring:
  score = n_pass / n_results
  1.0 when no constraints are active or results list is empty (trivially satisfied).

Multi-turn support: pass prior_expected_target (the expected_target from the
previous turn) so preserved field values can be merged with the delta's updated
fields to reconstruct the full effective constraint set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class RetrievalQualityScore:
    score: float                              # fraction of results passing all active constraints
    n_results: int                            # total results evaluated
    n_pass: int                               # results that passed every active constraint
    constraint_pass_rates: dict[str, float] = field(default_factory=dict)  # per-constraint rate
    details: list[dict[str, str]] = field(default_factory=list)  # per-result breakdown


def _cutoff_year_month(months_back: int) -> str:
    """Return the earliest YYYY-MM that still falls within months_back of today."""
    today = date.today()
    # Convert to 0-based absolute month index, subtract, then convert back.
    total_months = today.year * 12 + (today.month - 1) - months_back
    y, m0 = divmod(total_months, 12)
    return f"{y:04d}-{m0 + 1:02d}"


def _normalise_results(results: Any) -> list[dict[str, Any]]:
    out = []
    for r in results:
        if hasattr(r, "model_dump"):
            out.append(r.model_dump())
        elif isinstance(r, dict):
            out.append(r)
        else:
            out.append(dict(r))
    return out


def _to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def score_retrieval_quality(
    actual: Any,
    expected_target: dict[str, Any],
) -> RetrievalQualityScore:
    """
    Core scorer: checks each result in actual.results against the hard
    constraints implied by expected_target.

    Returns score = n_pass / n_results, where a result "passes" only if it
    satisfies every active constraint. A constraint is "active" when its
    corresponding key is present and non-None in expected_target.
    """
    actual_dict = _to_dict(actual)
    results = _normalise_results(actual_dict.get("results", []))

    if not results:
        # Nothing to evaluate — automatically satisfied.
        return RetrievalQualityScore(score=1.0, n_results=0, n_pass=0)

    # --- Build active constraints ---
    # Each constraint is stored as a callable(result_dict) -> bool, keyed by a
    # human-readable name. We only add a constraint when the corresponding field
    # is present and non-None in expected_target, so partial test cases (e.g.
    # those that only specify town) skip the unconstrained fields entirely.
    constraints: dict[str, Any] = {}

    town = expected_target.get("town")
    if town is not None:
        t = str(town).strip().upper()
        # Default argument _t captures t at lambda-definition time, avoiding
        # the classic Python loop-closure bug where all lambdas share the same
        # variable reference. Same pattern is used for all constraints below.
        constraints["town"] = lambda r, _t=t: (
            str(r.get("town") or "").strip().upper() == _t
        )

    flat_type = expected_target.get("flat_type")
    if flat_type is not None:
        ft = str(flat_type).strip().upper()
        constraints["flat_type"] = lambda r, _ft=ft: (
            str(r.get("flat_type") or "").strip().upper() == _ft
        )

    months_back = expected_target.get("months_back")
    if months_back is not None:
        # Convert months_back to an absolute "YYYY-MM" cutoff. A result passes
        # if its month string is lexicographically >= the cutoff — this works
        # because YYYY-MM strings sort chronologically.
        cutoff = _cutoff_year_month(int(months_back))
        constraints["date_range"] = lambda r, _c=cutoff: (
            bool(r.get("month")) and str(r["month"]) >= _c
        )

    floor_area_target = expected_target.get("floor_area_target")
    if floor_area_target is not None:
        # Default tolerance mirrors Target.floor_area_tolerance default in models.py.
        # No benchmark test case overrides this, so it's always 5.0 in practice.
        tolerance = float(expected_target.get("floor_area_tolerance") or 5.0)
        lo = float(floor_area_target) - tolerance
        hi = float(floor_area_target) + tolerance
        constraints["floor_area"] = lambda r, _lo=lo, _hi=hi: (
            r.get("floor_area_sqm") is not None
            and _lo <= float(r["floor_area_sqm"]) <= _hi
        )

    if not constraints:
        # No active constraints — every result trivially passes.
        return RetrievalQualityScore(
            score=1.0,
            n_results=len(results),
            n_pass=len(results),
        )

    # --- Evaluate each result ---
    # Track per-constraint pass counts separately so we can report which specific
    # constraint was the most common failure mode (useful for diagnosis).
    constraint_pass_counts: dict[str, int] = {k: 0 for k in constraints}
    details: list[dict[str, str]] = []
    n_pass = 0

    for r in results:
        row_detail: dict[str, str] = {}
        all_pass = True
        for name, check in constraints.items():
            passed = check(r)
            row_detail[name] = "pass" if passed else "fail"
            if passed:
                constraint_pass_counts[name] += 1
            else:
                # Don't break early — record every constraint's outcome so
                # the details can show multiple simultaneous failures per row.
                all_pass = False
        row_detail["overall"] = "pass" if all_pass else "fail"
        if all_pass:
            n_pass += 1
        details.append(row_detail)

    n = len(results)
    return RetrievalQualityScore(
        score=n_pass / n,
        n_results=n,
        n_pass=n_pass,
        constraint_pass_rates={k: v / n for k, v in constraint_pass_counts.items()},
        details=details,
    )


def score(
    actual: Any,
    test_case: dict[str, Any],
    prior_expected_target: dict[str, Any] | None = None,
) -> RetrievalQualityScore:
    """
    Public entry point — derives the effective expected_target from the test case
    and delegates to score_retrieval_quality.

    Single-turn: uses test_case["expected_target"] directly.

    Multi-turn delta (test_case has expected_target_delta): reconstructs the
    effective constraint set by starting from prior_expected_target (values for
    preserved fields) and overlaying the delta's updated fields. Raises ValueError
    if prior_expected_target is not provided for a delta turn.

    If no expected_target is present and the turn is not a delta, all constraints
    are skipped and score = 1.0 (trivially satisfied).
    """
    if "expected_target_delta" in test_case:
        if prior_expected_target is None:
            raise ValueError(
                "prior_expected_target is required for multi-turn delta scoring"
            )
        delta = test_case["expected_target_delta"]
        # Seed effective constraints with values for preserved fields from the prior turn.
        effective: dict[str, Any] = {
            field_name: prior_expected_target[field_name]
            for field_name in delta.get("preserved", [])
            if field_name in prior_expected_target
        }
        # Overlay the fields that were explicitly updated this turn.
        effective.update(delta.get("updated", {}))
        return score_retrieval_quality(actual, effective)

    return score_retrieval_quality(actual, test_case.get("expected_target") or {})
