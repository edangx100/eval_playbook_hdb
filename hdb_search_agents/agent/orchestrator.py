"""Loop Orchestrator: Coordinates Target and Planner agents for iterative search."""

from __future__ import annotations

import asyncio
from decimal import Decimal
import math
import re
from typing import Any, Mapping

from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from hdb_search_agents.agent.models import SearchResponse, Stats, Target, TraceStep
from hdb_search_agents.agent.planner import (
    MAX_RELAX_STEPS,
    MAX_TIGHTEN_STEPS,
    MIN_CANDIDATE_COUNT,
    MAX_CANDIDATE_COUNT,
    get_planner_agent,
    _planner_payload,
    _filter_conflicts,
    _available_relax_adjustments,
    _available_tighten_adjustments,
    _apply_relax_adjustment,
    _apply_tighten_adjustment,
    _clarifying_note,
    _clarify_conflict_note,
    _clarify_results_note,
    _build_note,
)
from hdb_search_agents.agent.target import get_target_agent, _target_to_filters, _has_text
from hdb_search_agents.agent.tools import (
    count_candidates as _count_candidates,
    fetch_candidates as _fetch_candidates,
    hybrid_fetch_candidates as _hybrid_fetch_candidates,
    price_stats as _price_stats,
)
from hdb_search_agents.agent.scoring import rerank_top_k
from hdb_search_agents.db.queries import bm25_boost_reason as _bm25_boost_reason

# Guardrails for result pagination.
DEFAULT_RESULT_LIMIT = 25
MAX_RESULT_LIMIT = 50

# Reranking configuration
DEFAULT_RERANK_TOP_K = 30  # Return top 30 best matches after scoring
MAX_FETCH_FOR_RERANK = 500  # Fetch up to 500 rows for reranking


def _clamp_limit(limit: int | None) -> int:
    """Ensure tool calls never request extreme row counts."""
    if limit is None:
        return DEFAULT_RESULT_LIMIT
    return max(1, min(int(limit), MAX_RESULT_LIMIT))


def _to_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _to_float(value: Any) -> float | None:
    """SQLAlchemy row values can be Decimal; normalize to native floats."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    """Normalize numeric text/Decimal into ints for the response schema."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _trim_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Reduce DB rows to the response fields expected by SearchResponse."""
    trimmed: list[dict[str, Any]] = []
    for row in rows:
        trimmed.append(
            {
                "month": _to_str(row.get("month")),
                "town": _to_str(row.get("town")),
                "flat_type": _to_str(row.get("flat_type")),
                "block": _to_str(row.get("block")),
                "street_name": _to_str(row.get("street_name")),
                "storey_range": _to_str(row.get("storey_range")),
                "floor_area_sqm": _to_float(row.get("floor_area_sqm")),
                "remaining_lease": _to_str(row.get("remaining_lease")),
                "resale_price": _to_int(row.get("resale_price")),
            }
        )
    return trimmed


def _fallback_parse_target(query: str, target: Target) -> Target:
    """Fill obvious fields when the LLM returns empty output."""
    updates: dict[str, Any] = {}
    query_text = query or ""

    # Fallback to simple regex parsing only when the LLM left fields empty.
    if not _has_text(target.flat_type):
        match = re.search(r"\b([1-5])\s*[- ]?\s*room\b", query_text, re.IGNORECASE)
        if match:
            updates["flat_type"] = f"{match.group(1)} ROOM"

    # Extract a short location phrase after "near/around" as a street hint.
    if not _has_text(target.street_hint):
        match = re.search(
            r"\b(?:near|around|nearby|close to|close)\s+([^,]+)",
            query_text,
            re.IGNORECASE,
        )
        if match:
            hint = match.group(1)
            hint = re.split(
                r"[;]|\b(last|past|within|for|with|sqm|sq\s*m)\b",
                hint,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            hint = hint.strip(" .")
            # Discard if the remaining text is purely numeric — it's a measurement
            # ("around 95 sqm"), not a location name.
            if hint and not re.match(r"^~?\d+(\.\d+)?$", hint):
                tokens = hint.split()
                if len(tokens) > 3:
                    hint = " ".join(tokens[:3])
                updates["street_hint"] = hint

    # Map descriptive wording to known HDB flat models.
    if not _has_text(target.flat_model_hint):
        match = re.search(
            r"\bpremium\s+apartment(?:-?ish)?\b",
            query_text,
            re.IGNORECASE,
        )
        if match:
            updates["flat_model_hint"] = "Premium Apartment"

    # Pick up simple "95 sqm" style targets for area filtering.
    if target.floor_area_target is None:
        match = re.search(r"(~?\d+(?:\.\d+)?)\s*sq\s*m", query_text, re.IGNORECASE)
        if match:
            try:
                updates["floor_area_target"] = float(match.group(1).lstrip("~"))
            except ValueError:
                pass

    # Always attempt to extract an explicit time window from the query text.
    # The Target model defaults months_back=12, so the LLM's conformance failure
    # (returning only {"town": "..."}) silently leaves months_back=12 even when
    # the query says "last 6 months". The regex is specific enough to be safe.
    match = re.search(r"\b(last|past)\s+(\d+)\s+months?\b", query_text, re.IGNORECASE)
    if match:
        updates["months_back"] = int(match.group(2))

    # Treat "long lease" as a minimum lease preference.
    if target.min_remaining_lease_years is None:
        match = re.search(r"\blong\s+(remaining\s+)?lease\b", query_text, re.IGNORECASE)
        if match:
            updates["min_remaining_lease_years"] = 80

    # Rescue storey_preference when the LLM under-extracts it.
    # The LLM frequently returns only {"town": "..."} and leaves storey_preference=None.
    if target.storey_preference is None:
        if re.search(r"\bhigh[- ]?floor\b|\bhigh\s+storey\b|\bupper\s+floor\b", query_text, re.IGNORECASE):
            updates["storey_preference"] = "high"
        elif re.search(r"\blow[- ]?floor\b|\blow\s+storey\b|\blower\s+floor\b", query_text, re.IGNORECASE):
            updates["storey_preference"] = "low"
        elif re.search(r"\bmid[- ]?floor\b|\bmiddle\s+floor\b|\bmid\s+storey\b|\bmedium\s+floor\b", query_text, re.IGNORECASE):
            updates["storey_preference"] = "mid"

    if not updates:
        return target
    return target.model_copy(update=updates)


# Fields carried forward from the previous conversation turn when the current
# turn leaves them empty. On a follow-up like "actually I want mid floor", the
# Target Agent sometimes drops earlier details (most often flat_type); this
# deterministically inherits them so a refinement never silently discards the
# flat type / town the user already gave.
_MULTITURN_INHERIT_FIELDS = (
    "town",
    "flat_type",
    "street_hint",
    "flat_model_hint",
    "floor_area_target",
    "storey_preference",
    "min_remaining_lease_years",
    "price_budget_max",
)


def _merge_with_prior_target(target: Target, prior: Any) -> Target:
    """Inherit prior-turn Target fields that the current turn left empty.

    `prior` may be a Target or a plain dict (e.g. ``Target.model_dump()``). The
    current turn always wins; the prior turn only fills gaps.
    """
    if isinstance(prior, Target):
        prior_data = prior.model_dump()
    elif isinstance(prior, dict):
        prior_data = prior
    else:
        return target
    updates: dict[str, Any] = {}
    for field in _MULTITURN_INHERIT_FIELDS:
        current = getattr(target, field, None)
        prior_value = prior_data.get(field)
        if (current is None or current == "") and prior_value not in (None, ""):
            updates[field] = prior_value
    if not updates:
        return target
    return target.model_copy(update=updates)


def _stats_from_rows(rows: list[dict[str, Any]]) -> Stats:
    """Compute summary stats directly from fetched rows (used for hybrid results)."""
    # Pull numeric prices and ignore missing/invalid values.
    prices: list[float] = []
    for row in rows:
        value = row.get("resale_price")
        if value is None:
            continue
        try:
            prices.append(float(value))
        except (TypeError, ValueError):
            continue

    # Return empty stats when no prices are available.
    if not prices:
        return Stats(median=0, p25=0, p75=0, min=0, max=0, count=0)

    prices.sort()

    # Linear interpolation percentile for small samples.
    def percentile(percent: float) -> float:
        if not prices:
            return 0.0
        rank = (len(prices) - 1) * percent
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return float(prices[int(rank)])
        return float(prices[lower] * (upper - rank) + prices[upper] * (rank - lower))

    return Stats(
        median=percentile(0.5),
        p25=percentile(0.25),
        p75=percentile(0.75),
        min=float(prices[0]),
        max=float(prices[-1]),
        count=len(prices),
    )


async def run_iterative_search(
    query: str,
    *,
    message_history: list | None = None,
    prior_target: "Target | dict | None" = None,
    verbose: bool = True
) -> SearchResponse:
    """Parse intent, then iterate on filters until the candidate pool is in range.

    Multi-Turn Conversation Support:
    This function supports multi-turn conversations using PydanticAI's native message history.

    TWO-AGENT ARCHITECTURE:
    - Target Agent: Extracts structured intent from natural language queries
      - NEEDS message_history for multi-turn context (e.g., "Make it high floor" after "Find 4-room in Bedok")
      - Messages are preserved across turns via SearchResponse.messages
    - Planner Agent: Makes stateless decisions based on structured JSON payloads
      - DOES NOT need message_history (each decision is independent within a single search)
      - Not involved in multi-turn conversations

    Args:
        query: Natural language search query
        message_history: Optional list of Message objects from previous conversation turns.
                        Only passed to Target Agent for understanding follow-up queries.
        prior_target: Optional Target (or dict) extracted on the previous turn. Used to
                        deterministically inherit fields the follow-up turn left empty,
                        so a refinement never silently drops the flat type / town already given.
        verbose: Whether to print debug information

    Returns:
        SearchResponse with results, stats, trace, and Target Agent message history
        (Planner Agent messages are not preserved as they are not conversational)
    """
    target_agent = get_target_agent()
    planner_agent = get_planner_agent()
    if verbose:
        print("Parsing target...", flush=True)

    # Target Agent: Extract structured intent with multi-turn context
    # Pass message_history so the agent can understand follow-up queries like:
    # Turn 1: "Find 4-room in Bedok"
    # Turn 2: "Make it high floor" (agent knows "it" refers to the 4-room in Bedok)
    try:
        target_result = await target_agent.run(query, message_history=message_history)
        target = target_result.output
        # Capture ALL messages (including input message_history + new messages from this run)
        # This accumulates conversation history across turns for the Target Agent
        agent_messages = target_result.all_messages()
    except ModelHTTPError as exc:
        # The extraction model hit a hard provider error — e.g. a context-length 400
        # (a verbose model can blow past the window when retries accumulate) or a
        # transient HTTP/rate-limit error. Degrade gracefully: start from an empty
        # target and let the regex fallback + missing-field clarify handle the query,
        # rather than crashing the whole request.
        if verbose:
            print(
                f"Target agent HTTP error (status {exc.status_code}); "
                "falling back to regex extraction",
                flush=True,
            )
        target = Target()
        agent_messages = list(message_history) if message_history else []
    except Exception as exc:
        # Target extraction failed - this usually means LLM API issues
        raise RuntimeError(f"Failed to parse search query: {type(exc).__name__}: {exc}") from exc

    target = _fallback_parse_target(query, target)
    # Multi-turn: inherit fields the follow-up turn left empty (e.g. the model
    # dropped flat_type when the user only refined floor/area) from the prior turn.
    if prior_target is not None:
        target = _merge_with_prior_target(target, prior_target)
    # Require town/flat type before any DB calls to avoid noisy broad queries.
    missing_town = not _has_text(target.town)
    missing_flat_type = not _has_text(target.flat_type)
    # Allow street-hint queries (e.g., "near Compassvale") to proceed without town,
    # since hybrid retrieval can handle them without a strict town filter.
    allow_missing_town = missing_town and _has_text(target.street_hint)
    
    if (missing_town and not allow_missing_town) or missing_flat_type:
        # Short-circuit with a clarifying question instead of running the loop.
        note = _clarifying_note(
            target, missing_town=missing_town, missing_flat_type=missing_flat_type
        )
        # Create a single-step trace showing we need clarification before any DB queries.
        trace = [
            TraceStep(
                step_number=1,
                step_name="Missing required fields",
                filters=_target_to_filters(target),
                count=0,
                action="clarify",
                adjustment_note="Town or flat type is missing",
                retrieval_mode="structured",
            )
        ]
        return SearchResponse(
            target=target,
            filters=_target_to_filters(target),
            count=0,
            # Return an empty stats object to keep the response schema stable.
            stats=Stats(median=0, p25=0, p75=0, min=0, max=0, count=0),
            results=[],
            note=note,
            trace=trace,
            messages=agent_messages,
        )
    original_target = target
    # Work on a mutable copy so we can record relax steps without losing the original intent.
    current_target = target.model_copy(deep=True)
    # Initialize loop state and counters for tracking how filters evolve.
    filters = _target_to_filters(current_target)
    # Guard against inverted range filters before querying the DB.
    conflicts = _filter_conflicts(filters)
    if conflicts:
        # Short-circuit with a fix-up question when constraints contradict.
        note = _clarify_conflict_note(conflicts)
        # Create a single-step trace documenting the detected conflicts.
        trace = [
            TraceStep(
                step_number=1,
                step_name="Filter conflicts detected",
                filters=dict(filters),
                count=0,
                action="clarify",
                adjustment_note="; ".join(conflicts),
                retrieval_mode="structured",
            )
        ]
        return SearchResponse(
            target=original_target,
            filters=filters,
            count=0,
            stats=Stats(median=0, p25=0, p75=0, min=0, max=0, count=0),
            results=[],
            note=note,
            trace=trace,
            messages=agent_messages,
        )
    relax_steps = 0
    relax_notes: list[str] = []
    tighten_notes: list[str] = []
    history: list[dict[str, Any]] = []
    tighten_steps = 0
    tighten_capped = False
    tighten_exhausted = False
    # Initialize trace list to record all loop iterations and decisions.
    trace: list[TraceStep] = []

    # Observe the initial candidate pool before any planner adjustments.
    try:
        count = _count_candidates(filters)
        current_stats = _price_stats(filters)
    except RuntimeError as exc:
        # Database connection or query failed
        raise RuntimeError(f"Database error during initial search: {exc}") from exc
    if verbose:
        print(f"Initial count={count}", flush=True)

    # Record the initial observation as the first trace step.
    # This captures the starting filters and count before any adjustments.
    trace.append(
        TraceStep(
            step_number=1,
            step_name="Initial count",
            filters=dict(filters),
            count=count,
            action="initial",
            retrieval_mode="structured",
        )
    )

    planner_failed = False
    # Iteratively ask the planner whether to accept, relax, tighten, or clarify.
    while True:
        planner_input = _planner_payload(
            query=query,
            target=current_target,
            count=count,
            stats=current_stats,
            filters=filters,
            history=history,
            relax_steps=relax_steps,
            tighten_steps=tighten_steps,
        )
        # Planner Agent: Make stateless decision based on current search state
        # NO message_history is passed - each planner decision is independent
        # The planner only sees the structured payload (target, filters, count, stats, history)
        # and doesn't participate in multi-turn conversations
        try:
            decision = (await planner_agent.run(planner_input)).output
        except (UnexpectedModelBehavior, ModelHTTPError):
            # UnexpectedModelBehavior: the model couldn't produce a valid PlannerDecision
            # (e.g. a weak model returns adjustment=null and exhausts retries).
            # ModelHTTPError: a hard provider error such as a context-length 400 (a verbose
            # reasoning model can balloon past the window as pydantic-ai retries accumulate)
            # or a transient rate-limit. Both are treated the same way: keep the loop alive
            # via the deterministic fallback below instead of crashing the whole request.
            available_tighten = _available_tighten_adjustments(current_target)
            available_relax = _available_relax_adjustments(current_target)
            # When the planner LLM fails to produce a valid PlannerDecision (e.g., a weak
            # model outputs adjustment=null for a tighten action and exhausts retries), we
            # apply the first available adjustment deterministically rather than immediately
            # surfacing a clarify response to the user.  This keeps the loop alive through
            # transient model conformance failures without requiring user interaction.
            if count > MAX_CANDIDATE_COUNT and tighten_steps < MAX_TIGHTEN_STEPS and available_tighten:
                # Pool is oversized and we have room to tighten — pick the first valid move.
                fallback_adjustment = available_tighten[0]
                tightened = _apply_tighten_adjustment(current_target, fallback_adjustment)
                if tightened is not None:
                    if verbose:
                        print(f"Planner error; applying fallback tighten: {fallback_adjustment}", flush=True)
                    current_target, adjustment_id, tighten_note = tightened
                    tighten_notes.append(tighten_note)
                    tighten_steps += 1
                    filters = _target_to_filters(current_target)
                    try:
                        count = _count_candidates(filters)
                        current_stats = _price_stats(filters)
                    except RuntimeError as exc:
                        raise RuntimeError(f"Database error during fallback tighten: {exc}") from exc
                    history.append({"step": len(history) + 1, "action": "tighten", "adjustment": adjustment_id, "note": tighten_note, "count": count})
                    trace.append(
                        TraceStep(
                            step_number=len(trace) + 1,
                            # "(fallback)" in the name makes planner errors visible in the trace UI.
                            step_name=f"Tighten (fallback): {tighten_note}",
                            filters=dict(filters),
                            count=count,
                            action="tighten",
                            adjustment=adjustment_id,
                            adjustment_note=tighten_note,
                            retrieval_mode="structured",
                        )
                    )
                    continue
            elif count < MIN_CANDIDATE_COUNT and relax_steps < MAX_RELAX_STEPS and available_relax:
                # Pool is too sparse and we have room to relax — pick the first valid move.
                fallback_adjustment = available_relax[0]
                relaxed = _apply_relax_adjustment(current_target, fallback_adjustment)
                if relaxed is not None:
                    if verbose:
                        print(f"Planner error; applying fallback relax: {fallback_adjustment}", flush=True)
                    current_target, adjustment_id, relax_note = relaxed
                    relax_steps += 1
                    relax_notes.append(relax_note)
                    filters = _target_to_filters(current_target)
                    try:
                        count = _count_candidates(filters)
                        current_stats = _price_stats(filters)
                    except RuntimeError as exc:
                        raise RuntimeError(f"Database error during fallback relax: {exc}") from exc
                    history.append({"step": len(history) + 1, "action": "relax", "adjustment": adjustment_id, "note": relax_note, "count": count})
                    trace.append(
                        TraceStep(
                            step_number=len(trace) + 1,
                            step_name=f"Relax (fallback): {relax_note}",
                            filters=dict(filters),
                            count=count,
                            action="relax",
                            adjustment=adjustment_id,
                            adjustment_note=relax_note,
                            retrieval_mode="structured",
                        )
                    )
                    continue
            error_reason = "planner output error; please rephrase or retry"
            note = _clarify_results_note(
                count=count,
                available_relax=available_relax,
                available_tighten=available_tighten,
                reason=error_reason,
            )
            # Record the failure as a trace step for visibility in the UI.
            trace.append(
                TraceStep(
                    step_number=len(trace) + 1,
                    step_name="Planner output error",
                    filters=dict(filters),
                    count=count,
                    action="clarify",
                    adjustment_note=error_reason,
                    retrieval_mode="structured",
                )
            )
            if _has_text(current_target.street_hint):
                if verbose:
                    print("Planner output error; proceeding to hybrid retrieval", flush=True)
                planner_failed = True
                break
            # Return early with a clarification prompt rather than raising.
            return SearchResponse(
                target=original_target,
                filters=filters,
                count=count,
                stats=current_stats,
                results=[],
                note=note,
                trace=trace,
                messages=agent_messages,
            )
        action = decision.action
        if verbose:
            parts = [f"Planner action={action}"]
            if decision.adjustment and decision.adjustment != "none":
                parts.append(f"adjustment={decision.adjustment}")
            if decision.reason:
                parts.append(f"reason={decision.reason}")
            print(" ".join(parts), flush=True)

        # Exit the loop when the planner is satisfied with the candidate set.
        if action == "accept":
            break
        # If the planner needs more info, check if hybrid retrieval can help before giving up.
        if action == "clarify":
            # If we have a street hint and low count, try hybrid retrieval instead of clarifying
            if count < MIN_CANDIDATE_COUNT and _has_text(current_target.street_hint):
                if verbose:
                    print(f"Count={count}, have street_hint - will try hybrid retrieval instead of clarifying", flush=True)
                # Break out of loop and proceed to hybrid retrieval
                break

            # Ask a targeted follow-up based on result size and adjustable levers.
            note = _clarify_results_note(
                count=count,
                available_relax=_available_relax_adjustments(current_target),
                available_tighten=_available_tighten_adjustments(current_target),
                reason=decision.reason,
            )
            # Record the clarify decision in the trace before returning.
            # This happens when the planner determines it needs user input to proceed.
            trace.append(
                TraceStep(
                    step_number=len(trace) + 1,
                    step_name="Clarify needed",
                    filters=dict(filters),
                    count=count,
                    action="clarify",
                    adjustment_note=decision.reason,
                    retrieval_mode="structured",
                )
            )
            return SearchResponse(
                target=original_target,
                filters=filters,
                count=count,
                stats=current_stats,
                results=[],
                note=note,
                trace=trace,
                messages=agent_messages,
            )
        # Tighten path: adjust constraints to reduce an oversized candidate pool.
        if action == "tighten":
            # Stop tightening after a fixed number of steps to avoid long loops.
            if tighten_steps >= MAX_TIGHTEN_STEPS:
                tighten_capped = True
                break
            # Compute which tighten moves are valid for the current target state.
            available = _available_tighten_adjustments(current_target)
            adjustment = decision.adjustment or "none"
            # Bail if the planner didn't pick an allowed adjustment for this target.
            if adjustment == "none" or adjustment not in available:
                tighten_exhausted = True
                break
            tightened = _apply_tighten_adjustment(current_target, adjustment)
            if tightened is None:
                # No further tightening rules apply; exit the loop.
                tighten_exhausted = True
                break
            current_target, adjustment_id, tighten_note = tightened
            # Track the tighten step and rebuild filters for the next count.
            tighten_notes.append(tighten_note)
            tighten_steps += 1
            filters = _target_to_filters(current_target)
            try:
                count = _count_candidates(filters)
                current_stats = _price_stats(filters)
            except RuntimeError as exc:
                # Database error during tighten step
                raise RuntimeError(f"Database error during tighten adjustment: {exc}") from exc
            # Record the applied adjustment for trace/debugging.
            history.append(
                {
                    "step": len(history) + 1,
                    "action": "tighten",
                    "adjustment": adjustment_id,
                    "note": tighten_note,
                    "count": count,
                }
            )
            # Record the tighten adjustment in the trace with updated filters and count.
            # This captures the effect of narrowing constraints to reduce candidate pool.
            trace.append(
                TraceStep(
                    step_number=len(trace) + 1,
                    step_name=f"Tighten: {tighten_note}",
                    filters=dict(filters),
                    count=count,
                    action="tighten",
                    adjustment=adjustment_id,
                    adjustment_note=tighten_note,
                    retrieval_mode="structured",
                )
            )
            if verbose:
                # Emit a concise trace line to show what changed and the new count.
                print(
                    f"Tighten step {len(tighten_notes)}: {tighten_note} (count={count})",
                    flush=True,
                )
            continue
        # Relax path: loosen constraints when results are too sparse.
        if action == "relax":
            if relax_steps >= MAX_RELAX_STEPS:
                break
            available = _available_relax_adjustments(current_target)
            adjustment = decision.adjustment or "none"
            # Avoid applying invalid relax steps; return current results instead.
            if adjustment == "none" or adjustment not in available:
                break
            relaxed = _apply_relax_adjustment(current_target, adjustment)
            if relaxed is None:
                # No more relax options left; return what we have.
                break
            current_target, adjustment_id, relax_note = relaxed
            relax_steps += 1
            relax_notes.append(relax_note)
            filters = _target_to_filters(current_target)
            try:
                count = _count_candidates(filters)
                current_stats = _price_stats(filters)
            except RuntimeError as exc:
                # Database error during relax step
                raise RuntimeError(f"Database error during relax adjustment: {exc}") from exc
            # Record the applied adjustment for trace/debugging.
            history.append(
                {
                    "step": len(history) + 1,
                    "action": "relax",
                    "adjustment": adjustment_id,
                    "note": relax_note,
                    "count": count,
                }
            )
            # Record the relax adjustment in the trace with updated filters and count.
            # This captures the effect of broadening constraints to increase candidate pool.
            trace.append(
                TraceStep(
                    step_number=len(trace) + 1,
                    step_name=f"Relax: {relax_note}",
                    filters=dict(filters),
                    count=count,
                    action="relax",
                    adjustment=adjustment_id,
                    adjustment_note=relax_note,
                    retrieval_mode="structured",
                )
            )
            if verbose:
                print(f"Relax step {relax_steps}: {relax_note} (count={count})", flush=True)
            continue

    # Final retrieval: decide whether to use hybrid or structured mode
    # Trigger hybrid mode when:
    # 1. street_hint is present (address-like / landmark-like query), OR
    # 2. count < MIN_CANDIDATE_COUNT (need better recall)
    use_hybrid = False
    query_text_used_for_embedding = None
    bm25_query_text = None
    retrieval_mode = "structured"
    bm25_used = None
    bm25_reason = None

    if _has_text(current_target.street_hint):
        use_hybrid = True
        query_text_used_for_embedding = current_target.street_hint
        # Keep BM25 focused on the strongest lexical signal (street hint) to avoid
        # over-constraining the BM25 query with unrelated prompt wording.
        bm25_query_text = current_target.street_hint
        if verbose:
            print(f"Triggering hybrid mode: street_hint present ('{current_target.street_hint}')", flush=True)
    elif count < MIN_CANDIDATE_COUNT and count > 0:
        use_hybrid = True
        # Build query text from target fields for semantic search
        query_text_used_for_embedding = query
        bm25_query_text = query
        if verbose:
            print(f"Triggering hybrid mode: count ({count}) < {MIN_CANDIDATE_COUNT}", flush=True)
    else:
        # Check for lexical BM25 triggers (keywords) even when counts are in-range.
        bm25_reason = _bm25_boost_reason(filters, query)
        if bm25_reason is not None:
            # Enable hybrid retrieval so BM25 can surface lexical matches.
            use_hybrid = True
            query_text_used_for_embedding = query
            bm25_query_text = query
            if verbose:
                print(f"Triggering hybrid mode: {bm25_reason}", flush=True)

    # Fetch candidate pool for reranking
    fetch_limit = min(count, MAX_FETCH_FOR_RERANK) if count > 0 else DEFAULT_RESULT_LIMIT

    try:
        if use_hybrid and query_text_used_for_embedding:
            retrieval_mode = "hybrid"
            # Remove street_name hard filter when using hybrid mode - the street hint is already
            # encoded in the query embedding and will be matched via vector similarity
            hybrid_filters = dict(filters)
            if "street_name" in hybrid_filters:
                del hybrid_filters["street_name"]
                if verbose:
                    print(f"Removed street_name hard filter for hybrid mode", flush=True)
            # Street hints are strong lexical signals: force BM25 and document the reason.
            if _has_text(current_target.street_hint):
                bm25_used = True
                bm25_reason = "street_hint present"
                use_bm25 = True
            elif bm25_reason is not None:
                bm25_used = True
                use_bm25 = True
            else:
                # Otherwise let the BM25 trigger logic decide based on query wording.
                bm25_reason = _bm25_boost_reason(hybrid_filters, bm25_query_text or query)
                bm25_used = bm25_reason is not None
                use_bm25 = None
            # Embed the hint (if any) while scoring BM25 against the full user query.
            raw_rows = _hybrid_fetch_candidates(
                hybrid_filters,
                bm25_query_text or query,
                limit=fetch_limit,
                use_bm25=use_bm25,
                embed_query_text=query_text_used_for_embedding,
                bm25_query_text=bm25_query_text or query,
            )
            if verbose:
                print(f"Fetched {len(raw_rows)} candidates via hybrid retrieval", flush=True)
        else:
            raw_rows = _fetch_candidates(filters, limit=fetch_limit)
            if verbose:
                print(f"Fetched {len(raw_rows)} candidates via structured retrieval", flush=True)
    except RuntimeError as exc:
        # Retrieval failed - re-raise with context
        if use_hybrid:
            raise RuntimeError(f"Hybrid retrieval failed: {exc}") from exc
        else:
            raise RuntimeError(f"Structured retrieval failed: {exc}") from exc

    # Deterministic rerank: score each candidate and return top k best matches
    # This ensures results are ordered by similarity to target, not just by retrieval order
    reranked_rows = rerank_top_k(
        raw_rows,
        target=current_target,  # Use adjusted target (with relaxations applied)
        k=DEFAULT_RERANK_TOP_K,
    )

    # Surface planner failures in verbose mode so the run isn't silent.
    if planner_failed and verbose:
        print("Proceeding with hybrid retrieval after planner error", flush=True)

    # Always log the rerank size in verbose mode for debugging.
    if verbose:
        print(f"Reranked to top {len(reranked_rows)} results", flush=True)

    # When a street hint exists in hybrid mode, prefer rows that actually contain it.
    if use_hybrid and _has_text(current_target.street_hint) and reranked_rows:
        hint = current_target.street_hint.upper()
        hint_matches = [
            row
            for row in reranked_rows
            if hint in str(row.get("street_name", "")).upper()
        ]
        if hint_matches:
            reranked_rows = hint_matches

    # Normalize rows for the response and recompute stats when hybrid changed the set.
    trimmed_rows = _trim_rows(reranked_rows)
    if use_hybrid:
        count = len(trimmed_rows)
        stats = _stats_from_rows(trimmed_rows)
    else:
        stats = current_stats
    # Summarize any missing fields and filter adjustments for the user-facing response.
    note = _build_note(
        target=original_target,
        relax_notes=relax_notes,
        tighten_notes=tighten_notes,
        tighten_capped=tighten_capped,
        tighten_exhausted=tighten_exhausted,
        count=count,
    )
    # Add final trace step showing acceptance of the candidate pool and retrieval.
    # This is the terminal step indicating the loop exited successfully.
    trace.append(
        TraceStep(
            step_number=len(trace) + 1,
            step_name="Accept and fetch results",
            filters=dict(filters),
            count=count,
            action="accept",
            retrieval_mode=retrieval_mode,
            query_text_used_for_embedding=query_text_used_for_embedding,
            topk=fetch_limit if use_hybrid else None,
            bm25_used=bm25_used,
            bm25_boost_reason=bm25_reason,
        )
    )
    return SearchResponse(
        target=original_target,
        filters=filters,
        count=count,
        stats=stats,
        results=trimmed_rows,
        note=note,
        trace=trace,  # Include the complete trace in the response
        messages=agent_messages,  # Include message history for multi-turn context
    )


def run_basic_search(query: str, *, verbose: bool = True) -> SearchResponse:
    """Sync wrapper for CLI/tests."""
    return asyncio.run(run_iterative_search(query, verbose=verbose))


if __name__ == "__main__":
    import sys

    from hdb_search_agents.agent.tracing import initialize_braintrust_tracing
    from settings import Settings

    # Initialize BrainTrust tracing if configured
    settings = Settings()
    initialize_braintrust_tracing(
        api_key=settings.braintrust_api_key,
        parent=settings.braintrust_parent,
    )

    prompt = " ".join(sys.argv[1:]).strip() or "Find 4-room in ANG MO KIO"
    output = run_basic_search(prompt)
    print(output.model_dump())
