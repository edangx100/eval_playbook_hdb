"""
Reranking quality scorer for the eval harness.

Measures whether the top-N reranked results are closer to the target floor area
than the raw candidate pool they were drawn from. The key question: did the
deterministic scoring algorithm (agent/scoring.py) surface the most area-relevant
candidates to the top?

Metric: mean absolute area deviation (MAAD) from floor_area_target.
  score = 1.0  top-N MAAD < pool MAAD  (reranking improved area proximity)
  score = 0.0  top-N MAAD >= pool MAAD (reranking did not improve)

Inputs expected in actual:
  results    — the top-N reranked rows (list of dicts with floor_area_sqm).
               Corresponds to SearchResponse.results.
  candidates — the full pre-reranked pool (list of dicts with floor_area_sqm).
               Not stored in SearchResponse; must be added by the eval runner.
               If absent, the scorer cannot compare and returns a vacuous 1.0.

Skips gracefully (returns 1.0) when:
  - floor_area_target is not set in expected_target (no area dimension to measure)
  - candidates is missing from actual (pool data unavailable)
  - insufficient rows have a non-null floor_area_sqm to compute MAAD
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RerankingQualityScore:
    score: float              # 1.0 if improved, 0.0 if not; 1.0 vacuously when data is missing
    top_n_maad: float | None  # mean absolute area deviation of top-N results (None when skipped)
    pool_maad: float | None   # mean absolute area deviation of full candidate pool (None when skipped)
    n_results: int            # number of top-N rows evaluated
    n_candidates: int         # size of the candidate pool (0 when absent)
    reason: str


def _to_dict(obj: Any) -> dict[str, Any]:
    # actual may be a live SearchResponse (Pydantic) or a plain dict from fixtures
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def _normalise_rows(rows: Any) -> list[dict[str, Any]]:
    # Handles lists of Pydantic models (ResultRow), plain dicts, or dict-likes
    out = []
    for r in (rows or []):
        if hasattr(r, "model_dump"):
            out.append(r.model_dump())
        elif isinstance(r, dict):
            out.append(r)
        else:
            out.append(dict(r))
    return out


def _maad(rows: list[dict[str, Any]], target: float) -> float | None:
    """Mean absolute area deviation from target across rows with non-null floor_area_sqm."""
    deviations = [
        abs(float(r["floor_area_sqm"]) - target)
        for r in rows
        if r.get("floor_area_sqm") is not None
    ]
    if not deviations:
        return None
    return sum(deviations) / len(deviations)


def score(
    actual: Any,
    test_case: dict[str, Any],
) -> RerankingQualityScore:
    """
    Public entry point — compares top-N mean absolute area deviation (MAAD) against pool MAAD for the given test case.

    Reads floor_area_target from test_case["expected_target"]. If absent, returns a
    vacuous 1.0 immediately — the scorer has no area dimension to measure against.

    actual must contain:
      - results    — list of reranked rows (floor_area_sqm per row)
      - candidates — list of pool rows (floor_area_sqm per row); if missing, the scorer
                     returns a vacuous 1.0 with a note rather than failing, because the
                     eval runner may not always surface the pre-reranking pool.

    score = 1.0 when top-N MAAD < pool MAAD (strict less-than), 0.0 otherwise.
    Equal deviation is not enough — reranking must measurably improve area proximity.
    """
    actual_dict = _to_dict(actual)
    expected_target = test_case.get("expected_target") or {}
    floor_area_target = expected_target.get("floor_area_target")

    if floor_area_target is None:
        return RerankingQualityScore(
            score=1.0,
            top_n_maad=None,
            pool_maad=None,
            n_results=0,
            n_candidates=0,
            reason="floor_area_target not specified — reranking quality skipped",
        )

    target = float(floor_area_target)
    results = _normalise_rows(actual_dict.get("results"))
    candidates = _normalise_rows(actual_dict.get("candidates"))

    if not candidates:
        # Pool data is not carried in SearchResponse; the eval runner must inject it.
        # Without the pool we cannot measure relative improvement, so score vacuously.
        return RerankingQualityScore(
            score=1.0,
            top_n_maad=None,
            pool_maad=None,
            n_results=len(results),
            n_candidates=0,
            reason="candidates pool not available — cannot compare reranking quality",
        )

    top_n_maad = _maad(results, target)
    pool_maad = _maad(candidates, target)

    if top_n_maad is None or pool_maad is None:
        # Rows exist but none have a non-null floor_area_sqm — treat as vacuous pass.
        return RerankingQualityScore(
            score=1.0,
            top_n_maad=top_n_maad,
            pool_maad=pool_maad,
            n_results=len(results),
            n_candidates=len(candidates),
            reason="insufficient floor_area_sqm data to compute deviation",
        )

    improved = top_n_maad < pool_maad  # strict: equal deviation does not count as improvement
    n = len(results)
    return RerankingQualityScore(
        score=1.0 if improved else 0.0,
        top_n_maad=top_n_maad,
        pool_maad=pool_maad,
        n_results=n,
        n_candidates=len(candidates),
        reason=(
            f"top-{n} MAAD {top_n_maad:.2f} sqm < pool MAAD {pool_maad:.2f} sqm — reranking improved"
            if improved
            else f"top-{n} MAAD {top_n_maad:.2f} sqm >= pool MAAD {pool_maad:.2f} sqm — no improvement"
        ),
    )
