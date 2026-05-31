"""Planner Agent: Decides iterative refinement actions and manages adjustments."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Mapping

from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModelSettings

from hdb_search_agents.agent.models import PlannerDecision, Stats, Target
from hdb_search_agents.agent.prompts import PLANNER_PROMPT
from hdb_search_agents.agent.target import get_openrouter_model, _has_text

# Guardrails for candidate pool size
MIN_CANDIDATE_COUNT = 30
MAX_CANDIDATE_COUNT = 200
MAX_RELAX_STEPS = 4
MAX_TIGHTEN_STEPS = 4

# Incremental relax steps, in the order the agent should try them.
RELAX_MONTHS_SEQUENCE = (12, 18, 24)
RELAX_SQM_TOLERANCE_SEQUENCE = (5.0, 8.0, 12.0)
TIGHTEN_MONTHS_SEQUENCE = (24, 12, 6)
TIGHTEN_SQM_TOLERANCE_SEQUENCE = (12.0, 8.0, 5.0, 3.0)
TIGHTEN_LEASE_YEARS_SEQUENCE = (60, 70, 80, 90)
RELAX_QUESTION_LABELS = {
    "widen_time_window": "time window",
    "widen_sqm_tolerance": "floor area range",
    "drop_storey_preference": "storey preference",
}
TIGHTEN_QUESTION_LABELS = {
    "narrow_time_window": "time window",
    "narrow_sqm_tolerance": "floor area range",
    "raise_min_lease_years": "minimum lease",
    "require_street_hint": "street name",
    "cap_price_budget": "price budget",
}


@lru_cache(maxsize=1)
def get_planner_agent() -> Agent[PlannerDecision]:
    """Planner agent decides the next loop action from observation context."""
    model_settings = OpenRouterModelSettings(
        openrouter_usage={"include": True}, temperature=0
    )
    agent: Agent[PlannerDecision] = Agent(
        get_openrouter_model(),
        model_settings=model_settings,
        system_prompt=PLANNER_PROMPT,
        output_type=PlannerDecision,
        retries=2,
    )
    return agent


def _planner_payload(
    *,
    query: str,
    target: Target,
    count: int,
    stats: Stats,
    filters: Mapping[str, Any],
    history: list[dict[str, Any]],
    relax_steps: int,
    tighten_steps: int,
) -> str:
    """Pass structured context so the planner can reason about thresholds and history."""
    payload = {
        "query": query,
        "target": target.model_dump(),
        "filters": dict(filters),
        "conflicts": _filter_conflicts(filters),
        "count": count,
        "stats": stats.model_dump(),
        "history": history,
        "min_count": MIN_CANDIDATE_COUNT,
        "max_count": MAX_CANDIDATE_COUNT,
        "relax_steps": relax_steps,
        "max_relax_steps": MAX_RELAX_STEPS,
        "tighten_steps": tighten_steps,
        "max_tighten_steps": MAX_TIGHTEN_STEPS,
        "available_relax_adjustments": _available_relax_adjustments(target),
        "available_tighten_adjustments": _available_tighten_adjustments(target),
    }
    return json.dumps(payload, sort_keys=True)


def _filter_conflicts(filters: Mapping[str, Any]) -> list[str]:
    """Catch contradictory min/max ranges before querying."""
    conflicts: list[str] = []
    # Validate numeric range filters that could be inverted by user input or parsing.
    sqm_min = filters.get("sqm_min")
    sqm_max = filters.get("sqm_max")
    if sqm_min is not None and sqm_max is not None:
        sqm_min_f = float(sqm_min) if not isinstance(sqm_min, bool) else None
        sqm_max_f = float(sqm_max) if not isinstance(sqm_max, bool) else None
        if sqm_min_f is not None and sqm_max_f is not None and sqm_min_f > sqm_max_f:
            conflicts.append(f"floor area min {sqm_min_f:g} > max {sqm_max_f:g}")
    # Storey bounds are also range-based and should be monotonic.
    storey_min = filters.get("storey_min")
    storey_max = filters.get("storey_max")
    if storey_min is not None and storey_max is not None:
        storey_min_i = int(storey_min) if not isinstance(storey_min, bool) else None
        storey_max_i = int(storey_max) if not isinstance(storey_max, bool) else None
        if storey_min_i is not None and storey_max_i is not None and storey_min_i > storey_max_i:
            conflicts.append(f"storey min {storey_min_i} > max {storey_max_i}")
    # Lease windows can be tightened in both directions; ensure they are consistent.
    lease_min = filters.get("remaining_lease_months_min")
    lease_max = filters.get("remaining_lease_months_max")
    if lease_min is not None and lease_max is not None:
        lease_min_i = int(lease_min) if not isinstance(lease_min, bool) else None
        lease_max_i = int(lease_max) if not isinstance(lease_max, bool) else None
        if lease_min_i is not None and lease_max_i is not None and lease_min_i > lease_max_i:
            conflicts.append(
                f"remaining lease min {lease_min_i} months > max {lease_max_i} months"
            )
    return conflicts


def _next_relax_value(current: float, steps: tuple[float, ...]) -> float | None:
    """Pick the next larger relax value in the configured sequence."""
    for value in steps:
        if current < value:
            return value
    return None


def _next_tighten_value(current: float, steps: tuple[float, ...]) -> float | None:
    """Pick the next smaller tighten value in the configured sequence."""
    for value in steps:
        if current > value:
            return value
    return None


def _available_relax_adjustments(target: Target) -> list[str]:
    """Enumerate relax adjustments that can still expand the candidate pool."""
    available: list[str] = []
    months_back = target.months_back or 12
    # Only include time-window expansion if there is a wider step left.
    next_months = _next_relax_value(float(months_back), RELAX_MONTHS_SEQUENCE)
    if next_months is not None and int(next_months) != months_back:
        available.append("widen_time_window")
    if target.floor_area_target is not None:
        tolerance = target.floor_area_tolerance or 0.0
        # Only include tolerance expansion if it increases the current band.
        next_tolerance = _next_relax_value(tolerance, RELAX_SQM_TOLERANCE_SEQUENCE)
        if next_tolerance is not None and next_tolerance != tolerance:
            available.append("widen_sqm_tolerance")
    if target.storey_preference is not None:
        # Storey preference is optional; dropping it can broaden recall.
        available.append("drop_storey_preference")
    return available


def _available_tighten_adjustments(target: Target) -> list[str]:
    """Enumerate tighten adjustments that can still reduce the candidate pool."""
    available: list[str] = []
    months_back = target.months_back or 12
    # Only include time-window tightening if a narrower step exists.
    next_months = _next_tighten_value(float(months_back), TIGHTEN_MONTHS_SEQUENCE)
    if next_months is not None and int(next_months) != months_back:
        available.append("narrow_time_window")
    if target.floor_area_target is not None:
        tolerance = target.floor_area_tolerance or 0.0
        # Only include tolerance tightening if it reduces the current band.
        next_tolerance = _next_tighten_value(
            tolerance, TIGHTEN_SQM_TOLERANCE_SEQUENCE
        )
        if next_tolerance is not None and next_tolerance != tolerance:
            available.append("narrow_sqm_tolerance")
    current_lease = target.min_remaining_lease_years or 0
    # Lease tightening can be applied if there's a higher threshold step.
    next_lease = _next_relax_value(float(current_lease), TIGHTEN_LEASE_YEARS_SEQUENCE)
    if next_lease is not None and int(next_lease) != current_lease:
        available.append("raise_min_lease_years")
    # Tighten options that only make sense when the user supplied extra hints.
    if _has_text(target.street_hint) and not target.enforce_street_hint:
        available.append("require_street_hint")
    if target.price_budget_max is not None and not target.enforce_price_budget:
        available.append("cap_price_budget")
    return available


def _apply_relax_adjustment(
    target: Target, adjustment: str
) -> tuple[Target, str, str] | None:
    """Apply the planner-selected relaxation and return the update + description."""
    if adjustment == "widen_time_window":
        months_back = target.months_back or 12
        next_months = _next_relax_value(float(months_back), RELAX_MONTHS_SEQUENCE)
        if next_months is None or int(next_months) == months_back:
            return None
        updated = target.model_copy()
        updated.months_back = int(next_months)
        return (
            updated,
            "widen_time_window",
            f"widened time window to last {int(next_months)} months",
        )

    if adjustment == "widen_sqm_tolerance":
        if target.floor_area_target is None:
            return None
        tolerance = target.floor_area_tolerance or 0.0
        next_tolerance = _next_relax_value(tolerance, RELAX_SQM_TOLERANCE_SEQUENCE)
        if next_tolerance is None or next_tolerance == tolerance:
            return None
        updated = target.model_copy()
        updated.floor_area_tolerance = float(next_tolerance)
        return (
            updated,
            "widen_sqm_tolerance",
            f"widened floor area tolerance to ±{next_tolerance:g} sqm",
        )

    if adjustment == "drop_storey_preference":
        if target.storey_preference is None:
            return None
        updated = target.model_copy()
        updated.storey_preference = None
        return updated, "drop_storey_preference", "removed storey preference"

    return None


def _apply_tighten_adjustment(
    target: Target, adjustment: str
) -> tuple[Target, str, str] | None:
    """Apply the planner-selected tightening and return the update + description."""
    if adjustment == "narrow_time_window":
        # Move to the next smaller time window to reduce the candidate pool.
        months_back = target.months_back or 12
        next_months = _next_tighten_value(float(months_back), TIGHTEN_MONTHS_SEQUENCE)
        if next_months is None or int(next_months) == months_back:
            return None
        updated = target.model_copy()
        updated.months_back = int(next_months)
        return (
            updated,
            "narrow_time_window",
            f"narrowed time window to last {int(next_months)} months",
        )

    if adjustment == "narrow_sqm_tolerance":
        # Only tighten sqm tolerance if the user specified a target area.
        if target.floor_area_target is None:
            return None
        tolerance = target.floor_area_tolerance or 0.0
        next_tolerance = _next_tighten_value(
            tolerance, TIGHTEN_SQM_TOLERANCE_SEQUENCE
        )
        if next_tolerance is None or next_tolerance == tolerance:
            return None
        updated = target.model_copy()
        updated.floor_area_tolerance = float(next_tolerance)
        return (
            updated,
            "narrow_sqm_tolerance",
            f"narrowed floor area tolerance to ±{next_tolerance:g} sqm",
        )

    if adjustment == "raise_min_lease_years":
        # Raise the lease floor to the next configured threshold.
        current_lease = target.min_remaining_lease_years or 0
        next_lease = _next_relax_value(
            float(current_lease), TIGHTEN_LEASE_YEARS_SEQUENCE
        )
        if next_lease is None or int(next_lease) == current_lease:
            return None
        updated = target.model_copy()
        updated.min_remaining_lease_years = int(next_lease)
        if current_lease:
            return (
                updated,
                "raise_min_lease_years",
                f"raised minimum remaining lease to {int(next_lease)} years",
            )
        return (
            updated,
            "raise_min_lease_years",
            f"set minimum remaining lease to {int(next_lease)} years",
        )

    if adjustment == "require_street_hint":
        if not _has_text(target.street_hint) or target.enforce_street_hint:
            return None
        updated = target.model_copy()
        # Flip the flag so the street hint becomes a hard SQL filter.
        updated.enforce_street_hint = True
        return updated, "require_street_hint", "required street hint filter"

    if adjustment == "cap_price_budget":
        if target.price_budget_max is None or target.enforce_price_budget:
            return None
        updated = target.model_copy()
        # Lock the price budget as a hard cap for tightening.
        updated.enforce_price_budget = True
        return (
            updated,
            "cap_price_budget",
            f"capped price at {int(target.price_budget_max):,} SGD",
        )

    return None


def _join_options(options: list[str]) -> str:
    """Join short option labels into a readable phrase."""
    if not options:
        return ""
    # Use simple grammar for 1/2/3+ items to keep the question natural.
    if len(options) == 1:
        return options[0]
    if len(options) == 2:
        return f"{options[0]} or {options[1]}"
    return ", ".join(options[:-1]) + f", or {options[-1]}"


def _format_adjustment_options(
    adjustments: list[str], labels: Mapping[str, str]
) -> str:
    """Map adjustment IDs to user-facing option labels."""
    options: list[str] = []
    for adjustment in adjustments:
        label = labels.get(adjustment)
        # Deduplicate labels so we don't repeat the same suggestion.
        if label and label not in options:
            options.append(label)
    return _join_options(options)


def _clarify_conflict_note(conflicts: list[str]) -> str:
    """Ask for correction when constraints directly contradict each other."""
    detail = "; ".join(conflicts)
    # Keep the note direct so the user can revise the conflicting values.
    return (
        "I see conflicting constraints "
        f"({detail}). Which values should I use instead?"
    )


def _clarify_results_note(
    *,
    count: int,
    available_relax: list[str],
    available_tighten: list[str],
    reason: str | None,
) -> str:
    """Provide a short clarifying question tailored to the result size."""
    if count <= 0:
        # No matches: ask which dimension to broaden, preferably from planner options.
        options = _format_adjustment_options(available_relax, RELAX_QUESTION_LABELS)
        if options:
            note = (
                "I couldn't find any comps with those constraints. "
                f"Should I broaden the search on {options}?"
            )
        else:
            note = (
                "I couldn't find any comps with those constraints. "
                "Which part should I relax (time window, floor area, storey, lease, price)?"
            )
    elif count < MIN_CANDIDATE_COUNT:
        # Sparse matches: suggest broadening on available relax axes.
        options = _format_adjustment_options(available_relax, RELAX_QUESTION_LABELS)
        if options:
            note = (
                f"I only found {count} comps. "
                f"Should I broaden the search on {options}?"
            )
        else:
            note = (
                f"I only found {count} comps. "
                "Which part should I relax (time window, floor area, storey, lease, price)?"
            )
    elif count > MAX_CANDIDATE_COUNT:
        # Oversized matches: ask for narrowing on available tighten axes.
        options = _format_adjustment_options(available_tighten, TIGHTEN_QUESTION_LABELS)
        if options:
            note = (
                f"I found too many comps ({count}). "
                f"Can you narrow by {options}?"
            )
        else:
            note = (
                "I found too many comps. Which constraints should I add "
                "(floor area, price, lease, storey, street)?"
            )
    else:
        # Fallback if we cannot match a bucket (should be rare).
        note = "I need more detail to refine the search."
    if reason:
        note = f"{note} ({reason})"
    return note


def _clarifying_note(target: Target, *, missing_town: bool, missing_flat_type: bool) -> str:
    """Ask for required fields before running any DB queries."""
    if missing_town and missing_flat_type:
        return (
            "I need the town and flat type before searching. "
            "Which town and flat type should I use?"
        )
    if missing_town:
        flat_type = target.flat_type.strip()
        return (
            "I need the town before searching. "
            f"Which town should I use for {flat_type} flats?"
        )
    town = target.town.strip()
    return (
        "I need the flat type before searching. "
        f"Which flat type should I use in {town}?"
    )


def _build_note(
    *,
    target: Target,
    relax_notes: list[str],
    tighten_notes: list[str],
    tighten_capped: bool,
    tighten_exhausted: bool,
    count: int,
) -> str | None:
    """Consolidate missing fields + relax decisions into a single optional message."""
    note_parts: list[str] = []
    missing: list[str] = []
    missing_town = not _has_text(target.town)
    # Allow street-hint queries to proceed without a town in hybrid mode.
    if missing_town and _has_text(target.street_hint):
        missing_town = False
    if missing_town:
        # Town is required for hard filtering.
        missing.append("town")
    if not _has_text(target.flat_type):
        # Flat type is required for hard filtering.
        missing.append("flat_type")
    if missing:
        note_parts.append("Missing required fields: " + ", ".join(missing) + ".")
    if relax_notes:
        # Surface any relax steps so the user can see what changed.
        note_parts.append("Relaxed search: " + "; ".join(relax_notes) + ".")
    if tighten_notes:
        # Surface any tighten steps so the user can see what changed.
        note_parts.append("Tightened search: " + "; ".join(tighten_notes) + ".")
    if count < MIN_CANDIDATE_COUNT:
        # Encourage broader inputs when the pool stays too small.
        note_parts.append("Few comps found; consider broadening the search.")
    if count > MAX_CANDIDATE_COUNT:
        # Tailor the message based on whether any tightening was actually applied.
        if tighten_notes:
            # Tightening happened but the pool is still oversized.
            if tighten_capped or tighten_exhausted:
                note_parts.append(
                    "Still too many comps after tightening; add more constraints "
                    "(town, floor area, price, lease)."
                )
            else:
                note_parts.append(
                    "Too many comps; consider tightening the search further."
                )
        else:
            # No tighten steps were applied, so don't imply that they were.
            note_parts.append(
                "Too many comps; add more constraints (floor area, price, lease) "
                "or allow tightening."
            )
    if not note_parts:
        return None
    return " ".join(note_parts)
