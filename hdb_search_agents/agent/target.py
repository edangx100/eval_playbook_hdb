"""Target Agent: Extract structured intent from natural language queries."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic_ai import Agent, ToolOutput
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openrouter import OpenRouterProvider

from hdb_search_agents.agent.models import Target
from hdb_search_agents.agent.prompts import TARGET_PROMPT
from settings import Settings


def _validate_api_key(api_key: str) -> None:
    """Guard against missing or placeholder keys to fail fast in dev."""
    if not api_key or api_key.strip().upper() in {"REPLACE_ME", "YOUR_OPENROUTER_API_KEY"}:
        raise ValueError("OPENROUTER_API_KEY is not set; update .env to enable OpenRouter.")


def build_openrouter_model(settings: Settings) -> OpenRouterModel:
    """Construct a model bound to an explicit provider (useful for tests)."""
    _validate_api_key(settings.openrouter_api_key)
    provider = OpenRouterProvider(api_key=settings.openrouter_api_key)
    return OpenRouterModel(settings.openrouter_model_name, provider=provider)


@lru_cache(maxsize=1)
def get_openrouter_model() -> OpenRouterModel:
    """Cache the model to reuse the underlying HTTP client across calls."""
    settings = Settings()
    return build_openrouter_model(settings)


def _normalize_text(value: str) -> str:
    """Normalize free-form text to match DB values (uppercase + collapsed whitespace)."""
    return " ".join(value.strip().upper().split())


def _normalize_flat_type(value: str) -> str:
    """Handle variants like "4-room" by stripping punctuation before normalization."""
    return _normalize_text(value.replace("-", " "))


def _has_text(value: str | None) -> bool:
    """Treat empty/whitespace-only strings and LLM null literals as missing input."""
    if not isinstance(value, str):
        return False
    stripped = value.strip()
    return stripped != "" and stripped.lower() not in {"null", "none"}


def _target_to_filters(target: Target) -> dict[str, Any]:
    """Convert structured intent into the db/queries.py filter contract."""
    filters: dict[str, Any] = {}

    # Normalize categorical fields to match stored values.
    if _has_text(target.town):
        filters["town"] = _normalize_text(target.town)

    if _has_text(target.flat_type):
        filters["flat_type"] = _normalize_flat_type(target.flat_type)

    # Only enforce street hints when the planner tightens with that constraint.
    if _has_text(target.street_hint) and target.enforce_street_hint:
        filters["street_hint"] = _normalize_text(target.street_hint)

    # Default months_back when the model omits it or returns 0/None.
    months_back = target.months_back or 12
    filters["months_back"] = max(1, int(months_back))

    if target.floor_area_target is not None:
        # Build an inclusive sqm band around the target area.
        tolerance = max(target.floor_area_tolerance or 0.0, 0.0)
        sqm_min = max(target.floor_area_target - tolerance, 0.0)
        sqm_max = target.floor_area_target + tolerance
        filters["sqm_min"] = sqm_min
        filters["sqm_max"] = sqm_max

    if target.min_remaining_lease_years is not None:
        # Stored as months in the DB, so convert years -> months.
        filters["remaining_lease_months_min"] = int(
            target.min_remaining_lease_years * 12
        )

    # Price caps stay soft until the planner explicitly tightens on price.
    if target.enforce_price_budget and target.price_budget_max is not None:
        filters["resale_price_max"] = int(target.price_budget_max)

    # Translate storey preference into a coarse numeric band.
    if target.storey_preference == "low":
        # Simple storey bands for the MVP.
        filters["storey_max"] = 4
    elif target.storey_preference == "mid":
        filters["storey_min"] = 5
        filters["storey_max"] = 10
    elif target.storey_preference == "high":
        filters["storey_min"] = 11

    return filters


@lru_cache(maxsize=1)
def get_target_agent() -> Agent[Target]:
    """Separate agent that only extracts the Target schema (no DB calls)."""
    model_settings = OpenRouterModelSettings(
        openrouter_usage={"include": True}, temperature=0
    )
    agent: Agent[Target] = Agent(
        get_openrouter_model(),
        model_settings=model_settings,
        system_prompt=TARGET_PROMPT,
        output_type=ToolOutput(
            Target,
            description=(
                "Call this tool exactly ONCE with ALL fields populated in a single JSON object. "
                "Do NOT call it multiple times with one field each."
            ),
        ),
    )
    return agent
