"""
Target Agent accuracy scorer for the eval harness.

Computes field-level precision / recall / F1 between an extracted Target and the
expected Target from a benchmark test case. This measures how reliably the Target
Agent converts natural language queries into structured intent.

Two modes (dispatched via score()):
  Single-turn: each field in expected_target is checked against the actual extraction.
               Fields are TP (correct), FP+FN (wrong value), or FN (not extracted).
  Multi-turn:  expected_target_delta splits fields into "updated" (must match new values)
               and "preserved" (must be unchanged from the prior turn's extraction).

Field comparison is type-aware:
  - String fields (town, flat_type, etc.)  → case-insensitive equality
  - Numeric fields (floor_area, months_back, etc.) → ±10% relative tolerance
  - Bool fields → boolean equality
"""
from __future__ import annotations   # if Python <3.10

from dataclasses import dataclass, field
from typing import Any

# These sets must stay in sync with the field types declared in agent/models.py:Target.
# They determine how _fields_match compares values for each field name.
_STRING_FIELDS = {"town", "flat_type", "street_hint", "flat_model_hint", "storey_preference"}
_NUMERIC_FIELDS = {
    "floor_area_target",
    "floor_area_tolerance",
    "min_remaining_lease_years",
    "months_back",
    "price_budget_max",
}
_BOOL_FIELDS = {"enforce_street_hint", "enforce_price_budget"}


@dataclass
class TargetExtractionScore:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int
    details: dict[str, str] = field(default_factory=dict)  # field -> "tp"|"fp+fn"|"fn"|"skip"

    @classmethod
    def from_counts(
        cls, tp: int, fp: int, fn: int, details: dict[str, str]
    ) -> "TargetExtractionScore":
        # Default to 1.0 when the denominator is zero: an empty expected set is
        # trivially satisfied, so precision and recall are both perfect.
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return cls(
            precision=precision,
            recall=recall,
            f1=f1,
            tp=tp,
            fp=fp,
            fn=fn,
            details=details,
        )


def _to_dict(target: Any) -> dict[str, Any]:
    # Normalise to plain dict so callers work with both Pydantic models and raw dicts.
    if hasattr(target, "model_dump"):
        return target.model_dump()
    return dict(target)


def _fields_match(field_name: str, actual_val: Any, expected_val: Any) -> bool:
    # Callers already check for None before calling, but guard defensively.
    if actual_val is None:
        return False
    if field_name in _STRING_FIELDS:
        return str(actual_val).strip().upper() == str(expected_val).strip().upper()
    if field_name in _NUMERIC_FIELDS:
        try:
            a, e = float(actual_val), float(expected_val)
        except (TypeError, ValueError):
            return False
        # Guard against zero expected value before computing relative tolerance.
        if e == 0:
            return a == 0
        return abs(a - e) / abs(e) <= 0.10  # ±10% tolerance per spec
    if field_name in _BOOL_FIELDS:
        return bool(actual_val) == bool(expected_val)
    return actual_val == expected_val


def score_target_extraction(
    actual: Any,
    expected_target: dict[str, Any],
) -> TargetExtractionScore:
    """Score single-turn extraction: compare actual Target against expected_target."""
    actual_dict = _to_dict(actual)
    details: dict[str, str] = {}
    tp = fp = fn = 0

    for field_name, expected_val in expected_target.items():
        # None in expected_target means "don't care" — field is intentionally untested.
        if expected_val is None:
            details[field_name] = "skip"
            continue
        actual_val = actual_dict.get(field_name)
        if actual_val is None:
            # Field not extracted at all — only a recall miss, not a precision hit.
            details[field_name] = "fn"
            fn += 1
        elif _fields_match(field_name, actual_val, expected_val):
            details[field_name] = "tp"
            tp += 1
        else:
            # Extracted but wrong value: hurts precision (a bad prediction was made)
            # AND recall (the relevant field was not correctly matched).
            # This IR double-counting is what yields P = R = 0.75 when 3 of 4
            # expected fields are present-but-wrong, giving F1 = 0.75.
            details[field_name] = "fp+fn"
            fp += 1
            fn += 1

    return TargetExtractionScore.from_counts(tp=tp, fp=fp, fn=fn, details=details)


def score_target_delta(
    actual: Any,
    expected_delta: dict[str, Any],
    prior_actual: Any,
) -> TargetExtractionScore:
    """
    Score multi-turn extraction against expected_target_delta.

    updated fields: actual[field] must match expected_delta["updated"][field].
    preserved fields: actual[field] must equal prior_actual[field] (unchanged).
    """
    actual_dict = _to_dict(actual)
    prior_dict = _to_dict(prior_actual)
    details: dict[str, str] = {}
    tp = fp = fn = 0

    for field_name, expected_val in expected_delta.get("updated", {}).items():
        if expected_val is None:
            details[field_name] = "skip"
            continue
        actual_val = actual_dict.get(field_name)
        if actual_val is None:
            details[field_name] = "fn"
            fn += 1
        elif _fields_match(field_name, actual_val, expected_val):
            details[field_name] = "tp"
            tp += 1
        else:
            details[field_name] = "fp+fn"
            fp += 1
            fn += 1

    for field_name in expected_delta.get("preserved", []):
        prior_val = prior_dict.get(field_name)
        actual_val = actual_dict.get(field_name)
        key = f"{field_name}(preserved)"
        # Both None means neither turn populated the field — preservation holds trivially.
        if prior_val is None and actual_val is None:
            details[key] = "tp"
            tp += 1
        elif prior_val is not None and _fields_match(field_name, actual_val, prior_val):
            details[key] = "tp"
            tp += 1
        else:
            # Covers: prior had a value but actual changed it (dropped or mutated),
            # OR prior was None but actual introduced a value (agent hallucinated a field).
            # Both violate preservation, so score as FN.
            details[key] = "fn"
            fn += 1

    return TargetExtractionScore.from_counts(tp=tp, fp=fp, fn=fn, details=details)


def score(
    actual: Any,
    test_case: dict[str, Any],
    prior_actual: Any = None,
) -> TargetExtractionScore:
    """
    Public entry point — routes to the correct scorer based on test case type.

    A test case contains either:
      - expected_target       → single-turn: checks that actual matches the listed fields.
      - expected_target_delta → multi-turn: checks that the agent correctly updated fields
                                the new message changed AND preserved fields already set.

    prior_actual is the Target extracted on the previous conversation turn. It is only
    needed for delta scoring (to verify preservation) and is therefore optional — but
    omitting it when expected_target_delta is present raises ValueError immediately,
    since preservation cannot be checked without it.

    If neither key is present, an empty dict is passed to score_target_extraction,
    which returns F1 = 1.0 (nothing to check → trivially satisfied).
    """
    if "expected_target_delta" in test_case:
        if prior_actual is None:
            raise ValueError("prior_actual is required for multi-turn delta scoring")
        return score_target_delta(actual, test_case["expected_target_delta"], prior_actual)
    # Empty dict fallback: no fields to check → all metrics default to 1.0 (trivially satisfied).
    return score_target_extraction(actual, test_case.get("expected_target", {}))
