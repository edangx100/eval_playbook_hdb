"""
Deterministic scoring for HDB comparable flats.

Score each candidate based on distance from target preferences.
Lower scores indicate better matches.

Scoring formula:
  score = 0.45*area_score + 0.25*lease_score + 0.15*storey_score + 0.10*recency_score
        + 0.05*model_score

Where:
  - area_score: deviation from target floor area (normalized by tolerance)
  - lease_score: shortfall in remaining lease (normalized)
  - storey_score: deviation from preferred storey band
  - recency_score: age of transaction (normalized by time window)
  - model_score: mismatch penalty for preferred flat model hint
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal, Mapping

from hdb_search_agents.agent.models import Target


# Scoring weights (must sum to 1.0 for interpretability).
WEIGHT_AREA = 0.45
WEIGHT_LEASE = 0.25
WEIGHT_STOREY = 0.15
WEIGHT_RECENCY = 0.10
WEIGHT_MODEL = 0.05


def area_score(
    floor_area_sqm: float,
    target_area: float | None,
    tolerance: float,
) -> float:
    """
    Score based on deviation from target floor area.

    Returns:
        0.0 if within tolerance or no target specified.
        Positive value proportional to distance beyond tolerance.
    """
    # No penalty if no target or invalid tolerance
    if target_area is None or tolerance <= 0:
        return 0.0

    deviation = abs(floor_area_sqm - target_area)

    # Within tolerance = perfect score (0.0)
    if deviation <= tolerance:
        return 0.0

    # Beyond tolerance: penalty grows linearly with excess deviation
    # e.g., 10 sqm beyond 5 sqm tolerance = score 1.0
    return (deviation - tolerance) / tolerance


def lease_score(
    remaining_lease_months: int,
    min_lease_months: int | None,
) -> float:
    """
    Score based on shortfall from minimum desired lease.

    Returns:
        0.0 if lease meets or exceeds minimum (or no minimum specified).
        Positive value for shortfall in years.
    """
    # No penalty if no minimum specified
    if min_lease_months is None:
        return 0.0

    # Shortfall in years (e.g., 50 years when minimum is 60 = score 10.0)
    shortfall_months = max(0, min_lease_months - remaining_lease_months)
    return shortfall_months / 12.0


def storey_score(
    storey_mid: int,
    storey_preference: Literal["low", "mid", "high"] | None,
) -> float:
    """
    Score based on deviation from preferred storey band.

    Storey bands:
      - low: floors 1-4 (target mid = 2.5)
      - mid: floors 5-10 (target mid = 7.5)
      - high: floors 11+ (target mid = 15)

    Returns:
        0.0 if no preference specified.
        Distance in floors from target band midpoint (normalized by 10).
    """
    # No penalty if no preference
    if storey_preference is None:
        return 0.0

    # Target midpoints for each preference band
    band_midpoints = {
        "low": 2.5,   # floors 1-4
        "mid": 7.5,   # floors 5-10
        "high": 15.0, # floors 11+
    }

    target_mid = band_midpoints[storey_preference]
    deviation = abs(storey_mid - target_mid)

    # Normalize: e.g., floor 2 vs high (15) = deviation 13 → score 1.3
    return deviation / 10.0


def recency_score(
    month_date: date,
    months_back: int,
    reference_date: date | None = None,
) -> float:
    """
    Score based on transaction age relative to lookback window.

    Args:
        month_date: Transaction date
        months_back: Lookback window in months
        reference_date: Reference date for age calculation (defaults to today)

    Returns:
        0.0 for most recent transactions.
        1.0 for transactions at the edge of the lookback window.
        >1.0 for transactions beyond the window (should be filtered out).
    """
    if reference_date is None:
        reference_date = date.today()

    if months_back <= 0:
        return 0.0

    # Convert age to months (30.44 = average days per month)
    age_days = (reference_date - month_date).days
    age_months = age_days / 30.44

    # Normalize: e.g., 6 months old in 12-month window = score 0.5
    return age_months / months_back


def model_score(
    flat_model: str | None,
    flat_model_hint: str | None,
) -> float:
    """
    Score based on matching a preferred flat model hint.

    Returns:
        0.0 if no hint provided or model matches.
        1.0 if hint is provided and model does not match.
    """
    # No preference set means no model penalty.
    if not flat_model_hint:
        return 0.0
    # Missing model data can't satisfy the hint, so treat as mismatch.
    if not flat_model:
        return 1.0
    model_value = flat_model.strip().lower()
    hint_value = flat_model_hint.strip().lower()
    return 0.0 if hint_value in model_value else 1.0


def calculate_score(
    row: Mapping[str, Any],
    target: Target,
    reference_date: date | None = None,
) -> float:
    """
    Calculate weighted composite score for a candidate row.

    Args:
        row: Database row with fields (floor_area_sqm, remaining_lease_months,
             storey_mid, month_date)
        target: User's target preferences
        reference_date: Reference date for recency calculation (defaults to today)

    Returns:
        Composite score (lower is better).
        0.0 indicates perfect match on all dimensions.
    """
    # Extract numeric fields from row
    floor_area_sqm = float(row.get("floor_area_sqm", 0))
    remaining_lease_months = int(row.get("remaining_lease_months", 0))
    storey_mid = int(row.get("storey_mid", 0))

    # Normalize month_date to date object (handles date/datetime/string)
    month_date_value = row.get("month_date")
    if isinstance(month_date_value, datetime):
        month_date = month_date_value.date()
    elif isinstance(month_date_value, date):
        month_date = month_date_value
    elif isinstance(month_date_value, str):
        month_date = datetime.strptime(month_date_value, "%Y-%m-%d").date()
    else:
        month_date = date.today()  # fallback (rare)

    # Calculate individual component scores
    area = area_score(
        floor_area_sqm,
        target.floor_area_target,
        target.floor_area_tolerance,
    )

    lease = lease_score(
        remaining_lease_months,
        target.min_remaining_lease_years * 12 if target.min_remaining_lease_years else None,
    )

    storey = storey_score(
        storey_mid,
        target.storey_preference,
    )

    recency = recency_score(
        month_date,
        target.months_back,
        reference_date,
    )

    # Penalize candidates that do not match a preferred flat model hint.
    model = model_score(
        row.get("flat_model"),
        target.flat_model_hint,
    )

    # Combine component scores into a single relevance score (lower is better).
    # Weights emphasize area/lease and keep model as a light preference.
    score = (
        WEIGHT_AREA * area +
        WEIGHT_LEASE * lease +
        WEIGHT_STOREY * storey +
        WEIGHT_RECENCY * recency +
        WEIGHT_MODEL * model
    )

    return score


def rank_candidates(
    rows: list[Mapping[str, Any]],
    target: Target,
    reference_date: date | None = None,
) -> list[tuple[float, Mapping[str, Any]]]:
    """
    Score and rank candidate rows by similarity to target.

    Args:
        rows: List of database rows to score
        target: User's target preferences
        reference_date: Reference date for recency calculation (defaults to today)

    Returns:
        List of (score, row) tuples sorted by score (ascending = best first).
    """
    # Score each row
    scored = [
        (calculate_score(row, target, reference_date), row)
        for row in rows
    ]

    # Sort ascending: lower score = better match
    scored.sort(key=lambda x: x[0])

    return scored


def rerank_top_k(
    rows: list[Mapping[str, Any]],
    target: Target,
    k: int = 30,
    reference_date: date | None = None,
) -> list[Mapping[str, Any]]:
    """
    Rerank candidates and return top k results.

    Args:
        rows: List of database rows to rerank
        target: User's target preferences
        k: Number of top results to return
        reference_date: Reference date for recency calculation (defaults to today)

    Returns:
        Top k rows sorted by score (best matches first).
    """
    # Rank all rows then extract top k (discard scores)
    ranked = rank_candidates(rows, target, reference_date)
    return [row for _, row in ranked[:k]]
