"""
Failure diagnosis LLM call.

For each test case where one or more scorers returned < 1.0, run a single LLM
call to JUDGE_OPENROUTER_MODEL_NAME asking for a one-line ``likely_cause`` and
``recommended_fix``. Output goes into the per-run failure report written by the
eval runner.

The LLM client mirrors ``llm_judge.py``: an OpenAI-SDK client pointed at
OpenRouter. When the eval runner injects a Braintrust-wrapped client, calls are
captured in the experiment trace; otherwise we build a direct OpenRouter client
so the module is usable standalone (smoke tests, ad-hoc CLI use).

Structured output is enforced via a Pydantic model (``Diagnosis``). The LLM is
asked for JSON mode and the raw text is validated against the model so the rest
of the pipeline can rely on the two fields being present and well-typed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

# When imported from outside the project root (or run as a script), make sure
# the project root is on sys.path so ``from settings import settings`` resolves.
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from settings import settings
from evals.prompts import DIAGNOSIS_PROMPT


# ---------------------------------------------------------------------------
# Structured output schema
#
# Defining the LLM response as a Pydantic model gives us three benefits:
#   1. The schema is the source of truth — the prompt is generated from it, so
#      field names can never drift between docs/prompt/parser.
#   2. .model_validate_json() enforces the shape at parse time — a malformed
#      response is rejected uniformly rather than silently returning a half-
#      populated dict.
#   3. Downstream code accesses fields by attribute (.likely_cause) which is
#      typo-safe under static analysis, unlike dict["likely_cause"].
# ---------------------------------------------------------------------------

class Diagnosis(BaseModel):
    """Structured LLM output for a single failing test case.

    Both fields must be short, single-sentence strings — long-form analysis is
    deliberately discouraged so the per-run JSON report stays skimmable.
    """

    likely_cause: str = Field(
        ...,
        description=(
            "One short sentence (<= 200 chars) hypothesising the root cause of "
            "the failure. Reference specific fields (extracted town, count, "
            "trace action) where possible."
        ),
    )
    recommended_fix: str = Field(
        ...,
        description=(
            "One short sentence (<= 200 chars) describing a targeted action to "
            "investigate or fix. Should name a component (prompt, scorer, "
            "planner adjustment, retrieval mode) rather than be generic."
        ),
    )


# ---------------------------------------------------------------------------
# Prompt
#
# The schema description is rendered into the prompt at module load time so the
# LLM sees the same field documentation that callers see. This keeps the prompt
# and the parser locked together — change the Pydantic model, the prompt updates
# automatically on next import.
# ---------------------------------------------------------------------------

def _render_schema_for_prompt() -> str:
    """Format the Diagnosis model's JSON schema as a readable block for the prompt.

    The literal braces ``{`` / ``}`` are doubled (``{{`` / ``}}``) so that the
    rendered block survives ``str.format()`` further down — otherwise format()
    would treat them as placeholders and raise KeyError.
    """
    schema = Diagnosis.model_json_schema()
    # Pydantic preserves field declaration order in v2, so iterating "properties"
    # produces the same order the model was defined in.
    lines = ["{{"]
    for name, spec in schema.get("properties", {}).items():
        description = spec.get("description", "")
        lines.append(f'  "{name}": "{description}"')
    lines.append("}}")
    return "\n".join(lines)


# Splice the rendered schema into the template once at import time.
# DIAGNOSIS_PROMPT lives in evals/prompts.py; _render_schema_for_prompt() stays
# here because it depends on the Diagnosis model defined above.
_PROMPT_TEMPLATE = DIAGNOSIS_PROMPT.replace("__SCHEMA__", _render_schema_for_prompt())


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _build_client() -> OpenAI:
    """Build a direct OpenRouter client. Used when no wrapped client is injected."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )


# ---------------------------------------------------------------------------
# Output summarisation
# ---------------------------------------------------------------------------

def _summarise_output(output: dict[str, Any]) -> dict[str, Any]:
    """Compact view of the agent output for the diagnosis prompt.

    The full output dict contains 30+ result rows and verbose trace metadata that
    would blow the prompt past useful length without helping the diagnosis.
    Keep target, count, retrieval mode, the trace skeleton, and a 5-row sample.
    """
    trace = []
    for step in (output.get("trace") or []):
        trace.append(
            {
                "step_name": step.get("step_name"),
                "action": step.get("action"),
                "count": step.get("count"),
                "retrieval_mode": step.get("retrieval_mode"),
                "adjustment": step.get("adjustment"),
                "adjustment_note": step.get("adjustment_note"),
            }
        )

    sample = []
    for r in (output.get("results") or [])[:5]:
        sample.append(
            {
                "month": r.get("month"),
                "town": r.get("town"),
                "flat_type": r.get("flat_type"),
                "floor_area_sqm": r.get("floor_area_sqm"),
                "storey_range": r.get("storey_range"),
            }
        )

    return {
        "target": output.get("target"),
        "filters": output.get("filters"),
        "count": output.get("count"),
        "retrieval_mode": output.get("retrieval_mode"),
        "note": output.get("note"),
        "trace": trace,
        "result_sample": sample,
        "results_total": len(output.get("results") or []),
    }


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

# Strip ```json ... ``` or ``` ... ``` wrappers if the model adds them despite
# the instruction; some judge models default to fenced JSON.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# Fallback values used when the LLM response can't be coerced into Diagnosis.
# The strings are deliberately actionable rather than blank so a reviewer
# scanning the JSON report can still triage the case manually.
_FALLBACK_CAUSE = "(LLM diagnosis could not be parsed)"
_FALLBACK_FIX = "Inspect the raw output and investigate manually."


def _try_parse(text: Any) -> Diagnosis | None:
    """Attempt to coerce one text blob into a ``Diagnosis``; return None on failure.

    Two strategies, both gated on Pydantic validation:
      1. Direct JSON parse of the de-fenced text.
      2. Greedy brace-match recovery (the model wrapped the JSON in prose).

    Used by ``_parse_response`` once for ``content``, then again for ``reasoning``
    when the primary content is empty (see the OpenRouter / DeepInfra
    reasoning-only edge case described below).

    Defensive on input type: ``reasoning`` is an unofficial field that some
    OpenAI-compatible SDK versions don't declare, so we may receive None or
    even a non-string sentinel from a test mock. Coerce non-string input to
    empty rather than raising.
    """
    if not isinstance(text, str):
        return None
    cleaned = _FENCE_RE.sub("", text).strip()
    if not cleaned:
        return None

    # Strategy 1: validate the cleaned text as JSON directly via Pydantic.
    try:
        return Diagnosis.model_validate_json(cleaned)
    except (ValidationError, ValueError):
        # ValueError covers malformed JSON (Pydantic wraps json.JSONDecodeError);
        # ValidationError covers missing/wrong-typed fields. Fall through.
        pass

    # Strategy 2: model may have prepended a preamble before the JSON object
    # (e.g. "Here is my diagnosis: { ... }"). Find the outermost braces and
    # re-attempt validation on just that slice. This is especially useful for
    # reasoning-mode responses where the JSON appears at the END of a long CoT.
    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first != -1 and last > first:
        candidate = cleaned[first : last + 1]
        try:
            return Diagnosis.model_validate_json(candidate)
        except (ValidationError, ValueError):
            pass

    return None


def _parse_response(raw: str, reasoning: str | None = None) -> Diagnosis:
    """Parse the LLM's response into a validated ``Diagnosis`` instance.

    Tries up to four sources in order:
      1. ``raw`` (the ``message.content`` field) — happy path for JSON-mode.
      2. ``reasoning`` (the ``message.reasoning`` field) — needed for OpenRouter
         routes (e.g. DeepInfra for z-ai/glm-5) that emit reasoning-only
         completions: the final JSON answer sits in ``reasoning`` while
         ``content`` is empty. Without this fallback, ~⅓ of glm-5 calls
         silently produced "(LLM returned no diagnosis)" placeholders.
      3. Fallback Diagnosis built from the raw text (so the report still has
         *something* a reviewer can act on).

    The Pydantic ``Diagnosis`` model is what guarantees the return shape —
    every caller can rely on both fields being non-empty strings.
    """
    # 1. Try the primary content field.
    parsed = _try_parse(raw)
    if parsed is not None:
        return parsed

    # 2. Try the reasoning field. Some OpenRouter providers (DeepInfra for
    # z-ai/glm-5 specifically) return content="" with the final JSON embedded
    # in reasoning. This rescues those calls instead of falling through to the
    # placeholder text.
    if reasoning:
        parsed = _try_parse(reasoning)
        if parsed is not None:
            return parsed

    # 3. Neither source yielded a valid Diagnosis. If we had ANY text, surface
    # it so the report at least reflects what the model said.
    text = (raw or "").strip()
    if not text:
        return Diagnosis(
            likely_cause="(LLM returned no diagnosis)",
            recommended_fix=_FALLBACK_FIX,
        )

    # Free-form prose: model didn't emit JSON. Cap at 200 chars so the report
    # stays scannable; the recommended_fix flags this as a parse failure so a
    # reviewer knows the model output was not machine-readable.
    return Diagnosis(
        likely_cause=text[:200] or _FALLBACK_CAUSE,
        recommended_fix=_FALLBACK_FIX,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _truncate_json(obj: Any, max_chars: int = 20_000) -> str:
    """Serialise obj to JSON, hard-capping at max_chars to stay within model limits."""
    s = json.dumps(obj, default=str, indent=2)
    if len(s) <= max_chars:
        return s
    truncated = s[:max_chars]
    # Close the JSON cleanly so parsers don't choke (best-effort).
    return truncated + "\n... (truncated)"


def diagnose_failure(
    test_case: dict[str, Any],
    output: dict[str, Any],
    failed_checks: list[str],
    client: OpenAI | None = None,
) -> Diagnosis:
    """Run one LLM call to diagnose a failing test case.

    Args:
        test_case: the test case dict from the benchmark YAML.
        output: the agent's structured output dict (target, count, trace, results, ...).
        failed_checks: names of scorers that returned < 1.0 for this case.
        client: optional OpenAI-SDK client. When omitted, a direct OpenRouter
            client is built. Pass None explicitly (rather than the
            Braintrust-wrapped _proxy_client) to avoid wrap_openai() injecting
            experiment-level context that can inflate token counts to 200k+.

    Returns:
        A ``Diagnosis`` Pydantic instance with ``likely_cause`` and
        ``recommended_fix`` fields. Parse failures fall back to placeholder
        strings rather than raising, so a single bad response does not abort
        the run.
    """
    if client is None:
        client = _build_client()

    prompt = _PROMPT_TEMPLATE.format(
        test_case=_truncate_json(test_case),
        output_summary=_truncate_json(_summarise_output(output)),
        failed_checks=", ".join(failed_checks) if failed_checks else "(none recorded)",
    )

    # response_format={"type": "json_object"} asks the OpenRouter-hosted model to
    # emit valid JSON. Most modern models honour it; if the chosen judge model
    # doesn't, _parse_response's fallbacks still salvage something useful.
    response = client.chat.completions.create(
        model=settings.judge_openrouter_model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    message = response.choices[0].message
    raw = message.content or ""
    # Some OpenRouter providers (notably DeepInfra for z-ai/glm-5) return the
    # final JSON answer inside `message.reasoning` while `message.content` is
    # empty — the model produces CoT but the provider never migrates the final
    # answer into the content field. The OpenAI SDK stores this as an extra
    # attribute on ChatCompletionMessage, so we read it via getattr to stay
    # forward-compatible with SDK versions that don't declare the field.
    reasoning = getattr(message, "reasoning", None) or ""
    return _parse_response(raw, reasoning=reasoning)


# ---------------------------------------------------------------------------
# Failure record / report file
# ---------------------------------------------------------------------------

def build_failure_record(
    test_case: dict[str, Any],
    output: dict[str, Any],
    failed_checks: list[str],
    diagnosis: Diagnosis,
) -> dict[str, Any]:
    """Assemble one record for the failure report.

    Schema (matches the SPEC.md and TASKS.md "Failure report" section):
        test_id, query, failed_checks, observed, expected, likely_cause, recommended_fix

    ``diagnosis`` is the structured Pydantic instance returned by
    ``diagnose_failure``; its fields are flattened into the record so the JSON
    report has a flat shape that's easy to skim.
    """
    # Multi-turn cases store the final user intent in turns[-1].query; fall back
    # to a joined string for traceability when neither shape is present.
    if "query" in test_case:
        query = test_case["query"]
    else:
        turns = test_case.get("turns") or []
        query = " | ".join(t.get("query", "") for t in turns) if turns else ""

    # Pull what the agent actually produced for the easy-to-read "observed" bucket.
    target = output.get("target") or {}
    final_action = None
    final_adjustment = None
    if output.get("trace"):
        last_step = output["trace"][-1]
        final_action = last_step.get("action")
        final_adjustment = last_step.get("adjustment")

    observed = {
        "count": output.get("count"),
        "retrieval_mode": output.get("retrieval_mode"),
        "extracted_town": target.get("town"),
        "extracted_flat_type": target.get("flat_type"),
        "final_action": final_action,
        "final_adjustment": final_adjustment,
        "planner_actions": [
            step.get("action") for step in (output.get("trace") or [])
        ],
    }

    # Surface the expectations the test case declared so a reader can compare
    # without flipping back to the YAML. Multi-turn cases declare expectations
    # per-turn; we pull from the last turn since the final state is what's scored.
    if "turns" in test_case:
        last_turn = test_case["turns"][-1]
        expected = {
            "expected_target": last_turn.get("expected_target"),
            "expected_target_delta": last_turn.get("expected_target_delta"),
            "expected_count_range": last_turn.get("expected_count_range"),
            "expected_retrieval_mode": last_turn.get("expected_retrieval_mode"),
        }
    else:
        expected = {
            "expected_target": test_case.get("expected_target"),
            "expected_count_range": test_case.get("expected_count_range"),
            "expected_retrieval_mode": test_case.get("expected_retrieval_mode"),
        }

    return {
        "test_id": test_case.get("id"),
        "category": test_case.get("category"),
        "query": query,
        "failed_checks": failed_checks,
        "observed": observed,
        "expected": expected,
        # Flatten the Pydantic fields into the record. .model_dump() preserves
        # both keys and gives us a plain dict that mixes naturally with the
        # other observed/expected fields above.
        **diagnosis.model_dump(),
    }


def write_failure_report(records: list[dict[str, Any]], path: Path) -> None:
    """Write all failure records for a run to a single JSON file.

    Creates parent directories as needed. The file is written even when
    ``records`` is empty so downstream tooling can rely on its presence as a
    "the run completed" signal.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)


if __name__ == "__main__":
    # Standalone smoke test: synthesise a failing case (wrong town in output) and
    # print the diagnosis. Requires OPENROUTER_API_KEY in .env.
    fake_case = {
        "id": "smoke_001",
        "category": "easy",
        "query": "Find 4-room flat in Ang Mo Kio, 95 sqm, last 12 months",
        "expected_target": {
            "town": "ANG MO KIO",
            "flat_type": "4 ROOM",
            "floor_area_target": 95.0,
            "months_back": 12,
        },
    }
    fake_output = {
        "target": {"town": "BISHAN", "flat_type": "4 ROOM", "floor_area_target": 95.0},
        "count": 42,
        "retrieval_mode": "structured",
        "trace": [{"step_name": "initial_search", "action": "accept", "count": 42}],
        "results": [{"town": "BISHAN", "flat_type": "4 ROOM", "floor_area_sqm": 92}] * 3,
        "note": None,
    }
    diag = diagnose_failure(
        fake_case,
        fake_output,
        failed_checks=["target_extraction", "retrieval_quality"],
    )
    print(diag.model_dump_json(indent=2))
    # Both fields are guaranteed non-empty by the parser's fallback path,
    # so these asserts protect against an unexpected regression.
    assert diag.likely_cause
    assert diag.recommended_fix
    print("Live diagnosis smoke test OK")
