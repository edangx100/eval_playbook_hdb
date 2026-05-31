from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_ai.messages import ModelMessage


# Shared Pydantic models for agent input/output.
# User intent distilled into structured filters/preferences.
class Target(BaseModel):
    town: str | None = Field(default=None, description="HDB town name")
    flat_type: str | None = Field(default=None, description="Flat type, e.g. 4 ROOM")
    street_hint: str | None = Field(default=None, description="Street name hint")
    # Flat model is the HDB design category stored in data (e.g., "Premium Apartment", "Model A").
    flat_model_hint: str | None = Field(
        default=None, description="Flat model hint, e.g. Premium Apartment"
    )
    floor_area_target: float | None = Field(
        default=None, description="Desired floor area in sqm"
    )
    floor_area_tolerance: float = Field(
        default=5.0, ge=0, description="Allowed +/- sqm tolerance"
    )
    storey_preference: Literal["low", "mid", "high"] | None = Field(
        default=None, description="Preferred storey band"
    )
    min_remaining_lease_years: int | None = Field(
        default=None, ge=0, description="Minimum remaining lease in years"
    )
    months_back: int = Field(
        default=12, ge=1, description="Lookback window in months"
    )
    price_budget_max: int | None = Field(
        default=None, ge=0, description="Maximum resale price"
    )
    # Planner toggles these to turn optional hints into hard filters.
    enforce_street_hint: bool = Field(
        default=False,
        description="Whether to require the street hint as a hard filter",
    )
    enforce_price_budget: bool = Field(
        default=False,
        description="Whether to apply price_budget_max as a hard filter",
    )

    @field_validator("town", "flat_type", mode="before")
    @classmethod
    def _normalise_categorical(cls, v: Any) -> Any:
        # LLMs sometimes output the literal string "null" or "none" for absent fields.
        # Coerce those to Python None; otherwise uppercase and collapse whitespace so
        # values like "Ang Mo Kio" match the DB's stored form "ANG MO KIO".
        if not isinstance(v, str):
            return v
        stripped = v.strip()
        if stripped.lower() in {"null", "none", ""}:
            return None
        return " ".join(stripped.upper().split())

    @field_validator("street_hint", "flat_model_hint", mode="before")
    @classmethod
    def _coerce_null_strings(cls, v: Any) -> Any:
        # Same null-literal coercion for optional hint fields; no uppercasing needed
        # since street names and model names are not stored in a normalised form.
        if isinstance(v, str) and v.strip().lower() in {"null", "none", ""}:
            return None
        return v


# One step in the agent's query/relax/tighten plan.
class QueryStep(BaseModel):
    name: str = Field(..., description="Human-readable step name")
    hard_filters: dict[str, Any] = Field(
        default_factory=dict, description="SQL-style hard filters"
    )
    soft_preferences: dict[str, Any] = Field(
        default_factory=dict, description="Soft preferences for scoring/relaxation"
    )
    action: Literal["search", "relax", "tighten", "rerank", "summarize"] = Field(
        ..., description="Agent action type"
    )


# Planner decision for the iterative loop (relax/tighten/accept).
PlannerAdjustment = Literal[
    "widen_time_window",
    "widen_sqm_tolerance",
    "drop_storey_preference",
    "narrow_time_window",
    "narrow_sqm_tolerance",
    "raise_min_lease_years",
    "require_street_hint",
    "cap_price_budget",
    "none",
]


class PlannerDecision(BaseModel):
    action: Literal["relax", "tighten", "accept", "clarify"] = Field(
        ..., description="Planner action for the next loop step"
    )
    adjustment: PlannerAdjustment | None = Field(
        default=None,
        description=(
            "REQUIRED (non-null) when action is 'relax' or 'tighten' — "
            "pick one value from available_relax_adjustments or available_tighten_adjustments "
            "in the payload. Must be null only when action is 'accept' or 'clarify'."
        ),
    )
    reason: str | None = Field(
        default=None, description="Short rationale for the decision"
    )

    @model_validator(mode="after")
    def _ensure_adjustment_for_action(self) -> "PlannerDecision":
        # Force relax/tighten decisions to include an explicit adjustment.
        if self.action in {"relax", "tighten"}:
            if self.adjustment is None or self.adjustment == "none":
                raise ValueError("adjustment is required for relax/tighten actions")
        return self


# Summary stats computed from a candidate set.
class Stats(BaseModel):
    median: float = Field(..., description="Median resale price")
    p25: float = Field(..., description="25th percentile resale price")
    p75: float = Field(..., description="75th percentile resale price")
    min: float = Field(..., description="Minimum resale price")
    max: float = Field(..., description="Maximum resale price")
    count: int = Field(..., ge=0, description="Number of rows in sample")


# Core fields used to display a comparable sale.
class ResultRow(BaseModel):
    month: str | None = Field(default=None, description="Sale month (YYYY-MM)")
    town: str | None = Field(default=None, description="HDB town")
    flat_type: str | None = Field(default=None, description="Flat type, e.g. 4 ROOM")
    block: str | None = Field(default=None, description="Block number")
    street_name: str | None = Field(default=None, description="Street name")
    storey_range: str | None = Field(default=None, description="Storey range, e.g. 10 TO 12")
    floor_area_sqm: float | None = Field(default=None, description="Floor area in sqm")
    remaining_lease: str | None = Field(default=None, description="Remaining lease text")
    resale_price: int | None = Field(default=None, description="Resale price in SGD")


# Trace step for agent loop observability.
class TraceStep(BaseModel):
    step_number: int = Field(..., ge=1, description="Sequential step number")
    step_name: str = Field(..., description="Human-readable step name")
    filters: dict[str, Any] = Field(default_factory=dict, description="Filters applied at this step")
    count: int = Field(..., ge=0, description="Count observed at this step")
    action: Literal["initial", "relax", "tighten", "accept", "clarify"] | None = Field(
        default=None, description="Planner action taken at this step"
    )
    adjustment: str | None = Field(
        default=None, description="Adjustment applied (for relax/tighten actions)"
    )
    adjustment_note: str | None = Field(
        default=None, description="Human-readable note about the adjustment"
    )
    retrieval_mode: Literal["structured", "hybrid"] = Field(
        default="structured", description="Retrieval mode used (structured SQL or hybrid vector+SQL)"
    )
    query_text_used_for_embedding: str | None = Field(
        default=None, description="Query text used for embedding generation (hybrid mode only)"
    )
    topk: int | None = Field(
        default=None, description="Top-k limit for hybrid retrieval (hybrid mode only)"
    )
    bm25_used: bool | None = Field(
        default=None, description="Whether BM25 ranking was used (hybrid mode only)"
    )
    bm25_boost_reason: str | None = Field(
        default=None, description="Reason BM25 was enabled (hybrid mode only)"
    )


# Final payload returned by the agent to the caller/UI.
class SearchResponse(BaseModel):
    target: Target = Field(..., description="Parsed user intent")
    filters: dict[str, Any] = Field(default_factory=dict, description="Filters used for search")
    count: int = Field(..., ge=0, description="Total matching rows")
    stats: Stats = Field(..., description="Summary price stats for the match set")
    results: list[ResultRow] = Field(default_factory=list, description="Comparable sales")
    note: str | None = Field(default=None, description="Optional note or warning")
    trace: list[TraceStep] = Field(default_factory=list, description="Agent loop trace steps")
    messages: list[ModelMessage] = Field(
        default_factory=list,
        description="Agent message history for multi-turn context"
    )
