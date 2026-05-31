"""
LLM-as-judge scorer for the eval harness.

Uses autoevals.LLMClassifier with a custom rubric to judge whether the returned
HDB resale comparables are defensible for the given query. Three-point scale:

  GOOD    1.0  Results clearly match the query constraints (right town, flat type,
               area band, date range).
  PARTIAL 0.5  Results mostly match with minor mismatches (one constraint slightly
               off, or count borderline).
  POOR    0.0  Results are largely irrelevant or contradict core constraints.

LLM calls go to OpenRouter using settings.openrouter_api_key and the model set by
JUDGE_OPENROUTER_MODEL_NAME (default: z-ai/glm-5). When the braintrust package is
installed, autoevals automatically wraps the OpenAI client with braintrust.wrap_openai
so scorer calls appear in experiment traces without changing the routing.
use_cot=True surfaces the judge's step-by-step rationale in the Braintrust UI.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# When run directly as a script (python evals/scorers/llm_judge.py), Python does not
# add the project root to sys.path automatically, so `settings` would be unfindable.
# This insert is a no-op when the module is imported normally with PYTHONPATH set.
if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from autoevals.llm import LLMClassifier
from openai import OpenAI

from settings import settings
from evals.prompts import LLM_JUDGE_PROMPT as _PROMPT_TEMPLATE


# openai.OpenAI is an HTTP client library, not a model choice.
# Setting base_url to OpenRouter routes all requests there instead of OpenAI's servers;
# the `model` parameter below controls which model runs (z-ai/glm-5 by default).
# autoevals.LLMClassifier requires an openai-SDK-compatible client (.chat.completions.create),
# so openai.OpenAI is the required transport — no OpenAI model is ever called.
# When the braintrust package is installed, autoevals wraps this client automatically
# so scorer calls appear in Braintrust traces without any code changes here.
_openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.openrouter_api_key,
)

# Module-level classifier using the direct OpenRouter client. Used by score()
# when no client is injected (standalone smoke test, unit tests) and exported
# so tests can inspect classifier configuration (choice_scores, model, etc.)
# and patch .eval() without making live API calls.
llm_judge = LLMClassifier(
    name="llm_judge",
    prompt_template=_PROMPT_TEMPLATE,
    choice_scores={"GOOD": 1.0, "PARTIAL": 0.5, "POOR": 0.0},
    model=settings.judge_openrouter_model_name,
    use_cot=True,
    client=_openrouter_client,
)


# --- input normalisation helpers ---

def _to_dict(obj: Any) -> dict[str, Any]:
    # actual arrives as a live SearchResponse (Pydantic) in production; as a plain
    # dict in fixtures. model_dump() is required for Pydantic — dict(pydantic_obj)
    # returns internal metadata rather than field values.
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


def _normalise_rows(rows: Any) -> list[dict[str, Any]]:
    # Each row may be a ResultRow Pydantic object or a plain dict.
    out = []
    for r in (rows or []):
        if hasattr(r, "model_dump"):
            out.append(r.model_dump())
        elif isinstance(r, dict):
            out.append(r)
        else:
            out.append(dict(r))
    return out


# --- prompt formatting helpers ---

def _format_target(target: dict[str, Any]) -> str:
    """Summarise the extracted target fields for inclusion in the judge prompt."""
    lines = []
    if target.get("town"):
        lines.append(f"  town: {target['town']}")
    if target.get("flat_type"):
        lines.append(f"  flat_type: {target['flat_type']}")
    if target.get("floor_area_target") is not None:
        tol = target.get("floor_area_tolerance", 5.0)
        lines.append(f"  floor_area: {target['floor_area_target']} ± {tol} sqm")
    if target.get("months_back") is not None:
        lines.append(f"  months_back: {target['months_back']}")
    if target.get("storey_preference"):
        lines.append(f"  storey_preference: {target['storey_preference']}")
    if target.get("street_hint"):
        lines.append(f"  street_hint: {target['street_hint']}")
    if target.get("min_remaining_lease_years") is not None:
        lines.append(f"  min_remaining_lease_years: {target['min_remaining_lease_years']}")
    return "\n".join(lines) if lines else "(no structured target extracted)"


def _format_results(results: list[dict[str, Any]], max_rows: int = 10) -> str:
    """Format the top results as a compact list for the judge prompt.

    Capped at max_rows so the prompt stays concise — the judge needs enough
    rows to assess constraint compliance but not all 30.
    """
    if not results:
        return "(no results)"
    rows = []
    for r in results[:max_rows]:
        parts = []
        if r.get("month"):
            parts.append(str(r["month"]))
        if r.get("town"):
            parts.append(str(r["town"]))
        if r.get("flat_type"):
            parts.append(str(r["flat_type"]))
        if r.get("floor_area_sqm") is not None:
            parts.append(f"{r['floor_area_sqm']} sqm")
        if r.get("storey_range"):
            parts.append(str(r["storey_range"]))
        if r.get("resale_price") is not None:
            parts.append(f"${r['resale_price']:,}")
        rows.append("  " + ", ".join(parts))
    if len(results) > max_rows:
        rows.append(f"  ... ({len(results) - max_rows} more rows not shown)")
    return "\n".join(rows)


def _extract_query(test_case: dict[str, Any]) -> str:
    """Return the query string from a single-turn or multi-turn test case.

    For multi-turn cases the last turn's query is used — it represents the most
    recent user intent, which is what the final results should satisfy.
    """
    if "query" in test_case:
        return test_case["query"]
    turns = test_case.get("turns") or []
    if turns:
        return turns[-1].get("query", "")
    return ""


# --- public scorer entry point ---

def score(
    actual: Any,
    test_case: dict[str, Any],
    # Optional injected client. When run.py passes the Braintrust proxy client,
    # calls are routed through the proxy so token usage and latency appear in
    # the experiment trace. Falls back to _openrouter_client for standalone use
    # (e.g. the smoke test at the bottom of this file).
    client: OpenAI | None = None,
) -> Any:
    """Judge whether the returned comparables are defensible for the query.

    Formats actual (SearchResponse or dict) and test_case into the judge prompt,
    then calls llm_judge.eval(). Returns an autoevals Score object:
      .score    — 1.0 (GOOD), 0.5 (PARTIAL), or 0.0 (POOR)
      .metadata — {"choice": "GOOD"|"PARTIAL"|"POOR", "rationale": "<CoT text>"}

    The rationale field is populated because use_cot=True is set on llm_judge,
    making it visible in the Braintrust experiment trace for each test case.
    """
    actual_dict = _to_dict(actual)
    # Judge against what the agent actually produced, not what the test case expected.
    target_dict = _to_dict(actual_dict.get("target") or {})
    results = _normalise_rows(actual_dict.get("results"))
    count = actual_dict.get("count", len(results))
    query = _extract_query(test_case)

    # When a client is injected (e.g. the Braintrust proxy from run.py), build a
    # new classifier with it so calls are routed through the proxy and token usage
    # is captured. LLMClassifier is cheap to construct — the cost is the API call.
    # Without an injected client, reuse the module-level instance (direct OpenRouter).
    if client is not None:
        judge = LLMClassifier(
            name="llm_judge",
            prompt_template=_PROMPT_TEMPLATE,
            choice_scores={"GOOD": 1.0, "PARTIAL": 0.5, "POOR": 0.0},
            model=settings.judge_openrouter_model_name,
            use_cot=True,
            client=client,
        )
    else:
        judge = llm_judge

    return judge.eval(
        # output and expected are unused by our template but are required positional
        # arguments by the LLMClassifier.eval() signature.
        output="",
        expected="",
        query=query,
        target_summary=_format_target(target_dict),
        count=count,
        results_summary=_format_results(results),
    )


if __name__ == "__main__":
    # Live smoke test — makes a real API call to OpenRouter to verify end-to-end wiring:
    # correct model, correct client config, and that the judge returns a valid label.
    # Empty results for a common query (Bishan 4-room) should always score POOR,
    # so the expected output is deterministic enough to be a useful sanity check.
    result = score(
        {"target": {"town": "BISHAN", "flat_type": "4 ROOM"}, "results": [], "count": 0},
        {"query": "4-room in Bishan"},
    )
    print("score:", result.score)
    print("metadata:", result.metadata)
    # Guard: any label outside the three defined scores means the classifier misbehaved.
    assert result.score in (0.0, 0.5, 1.0), f"unexpected score: {result.score}"
    print("Live API test OK")
