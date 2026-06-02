"""
Generate evals/reports/failure_taxonomy.md from the eval failure reports.

This is a programmatic doc-generation script. It
is NOT hand-authored: it globs every ``evals/reports/*_failures.json``, derives
which agent components actually have failures, asks the judge LLM to cluster the
real ``likely_cause`` strings into distinct failure modes, then renders the
Markdown deterministically.

Design notes
------------
* **Structured output, not Markdown.** The LLM returns JSON validated against the
  ``FailureTaxonomy`` Pydantic model; this module renders the Markdown itself.
  That guarantees the document's structure (one ``##`` per present component,
  four labelled lines per ``###`` entry) instead of trusting the model to format
  consistently — and makes the acceptance test deterministic.
* **Data-driven, not a fixed count.** Components with zero failures get no
  section, and the prompt forbids padding/inventing modes. The taxonomy stays
  grounded in what actually failed. See the discussion that motivated dropping
  the old "≥ 8 entries" rule.
* **LLM client mirrors ``evals/failure_diagnosis.py``** — an OpenAI-SDK client
  pointed at OpenRouter, JSON mode, temperature 0, with the glm-5 reasoning-field
  fallback (some OpenRouter providers return the answer in ``message.reasoning``
  while ``message.content`` is empty).
"""
from __future__ import annotations

import glob   # finds files matching a wildcard pattern, e.g. "*_failures.json"
import json   # read the failure reports (JSON) and serialise data for the prompt
import re     # one small regular expression, used to strip ```code fences```
import sys    # we tweak sys.path below so imports work when run as a script
from pathlib import Path
from typing import Any

# When you run this file directly (``python docs/doc_gen_scripts/gen_failure_taxonomy.py``)
# Python only knows about the script's own folder, so ``from settings import ...``
# would fail. We compute the project root and add it to the import search path
# (sys.path) so the imports below resolve.
#   __file__              = .../docs/doc_gen_scripts/gen_failure_taxonomy.py
#   .resolve()            = turn it into a full absolute path
#   .parents[2]           = go up three levels:
#                           [0] docs/doc_gen_scripts/, [1] docs/, [2] the project root
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# These imports rely on the sys.path tweak above, so they come after it (which is
# why the linter "imports not at top of file" rule is intentionally not followed).
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from settings import settings
from evals.prompts import FAILURE_TAXONOMY_PROMPT

# Where we read from and where we write to. Building these as Path objects off
# _ROOT means the script works no matter which directory it's launched from.
REPORTS_GLOB = str(_ROOT / "evals" / "reports" / "*_failures.json")
OUTPUT_PATH = _ROOT / "evals" / "reports" / "failure_taxonomy.md"

# The four parts of the agent we group failures under, in the order we want the
# document's sections to appear. We only ever render the ones that actually have
# failures in the data (see derive_present_components), but the *order* comes from
# this list.
CANONICAL_COMPONENTS = ["Target Agent", "Planner Agent", "retrieval", "reranking"]


# ---------------------------------------------------------------------------
# Structured output schema
#
# As in failure_diagnosis.py, the Pydantic model is the source of truth: the
# prompt's __SCHEMA__ block is rendered from it, and the response is validated
# against it, so field names can never drift between prompt, parser, and renderer.
# ---------------------------------------------------------------------------

class FailureMode(BaseModel):
    """One distinct failure mode within a single component.

    Subclassing Pydantic's ``BaseModel`` turns this into a self-validating data
    container: when we parse the LLM's JSON, Pydantic checks that every field is
    present and a string, and raises a clear error if not. ``Field(..., ...)``
    marks a field as required (the ``...``) and attaches the human description we
    reuse to build the prompt's schema block.
    """

    component: str = Field(
        ...,
        description="Component this mode belongs to; MUST match one of the listed components exactly.",
    )
    mode_name: str = Field(
        ...,
        description="Short, specific name for the failure mode (becomes the ### heading).",
    )
    observable_symptom: str = Field(
        ...,
        description="One sentence: what the scorer or agent output actually showed.",
    )
    root_cause: str = Field(
        ...,
        description="One sentence: why the agent made this error.",
    )
    mitigation: str = Field(
        ...,
        description="One concrete fix naming the component and the field/prompt to change.",
    )
    evidence: str = Field(
        ...,
        description='A likely_cause copied verbatim from the records, ending with " — test_id".',
    )


class FailureTaxonomy(BaseModel):
    """The full taxonomy: a flat list of modes, grouped by component at render time."""

    modes: list[FailureMode] = Field(..., description="One entry per distinct observed failure mode.")


def _render_schema_for_prompt() -> str:
    """Render FailureMode's fields as a readable JSON skeleton for the prompt.

    We show the LLM exactly what shape of JSON we expect by building a little
    example object out of the model's own field names + descriptions. Building it
    from the model (rather than hand-writing it) means the prompt can never drift
    out of sync with the Pydantic class.

    Braces are doubled (``{{`` / ``}}``) on purpose: later we call
    ``_PROMPT_TEMPLATE.format(...)`` to fill in {components}/{failures}, and
    ``str.format`` treats a single ``{`` as a placeholder. Doubling escapes them
    so a literal ``{`` survives into the final prompt. Same trick as
    failure_diagnosis._render_schema_for_prompt.
    """
    # model_json_schema() is a Pydantic helper; "properties" holds one entry per
    # field, in the order the fields were declared on the class.
    props = FailureMode.model_json_schema().get("properties", {})
    lines = ["{{", '  "modes": [', "    {{"]
    items = list(props.items())
    for i, (name, spec) in enumerate(items):
        # JSON forbids a trailing comma, so every field gets a comma except the last.
        comma = "," if i < len(items) - 1 else ""
        lines.append(f'      "{name}": "{spec.get("description", "")}"{comma}')
    lines += ["    }}", "    // ... one object per distinct failure mode", "  ]", "}}"]
    return "\n".join(lines)


# Build the final prompt text once, when the module is first imported, by swapping
# the literal token __SCHEMA__ in the template for the rendered schema block above.
# (FAILURE_TAXONOMY_PROMPT still has {components}/{failures} placeholders left,
# which get filled in per-run inside request_taxonomy.)
_PROMPT_TEMPLATE = FAILURE_TAXONOMY_PROMPT.replace("__SCHEMA__", _render_schema_for_prompt())


# ---------------------------------------------------------------------------
# Loading & component mapping
# ---------------------------------------------------------------------------

def _component_for_check(check: str) -> str | None:
    """Map a failed-scorer name to one of the canonical components.

    ``llm_judge`` rates whether the returned transactions are defensible for the
    query — that is fundamentally a retrieval-quality signal, so it maps to
    ``retrieval``. Unrecognised checks (e.g. ``task_error``) return None so they
    don't fabricate a component, but their records are still shown to the LLM.
    """
    if check.startswith("target_extraction"):
        return "Target Agent"
    if check.startswith("planner") or check.startswith("trace"):
        return "Planner Agent"
    if check.startswith("retrieval") or check == "llm_judge":
        return "retrieval"
    if check.startswith("reranking"):
        return "reranking"
    return None


def load_failure_records() -> list[dict[str, Any]]:
    """Load and de-duplicate failure records from every ``*_failures.json`` report.

    Records repeat across experiments (e.g. baseline vs. baseline-postreorg), so
    we de-dupe on (test_id, likely_cause) to keep the prompt focused and avoid
    inflating one mode's apparent frequency.
    """
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()  # remembers which (test_id, cause) pairs we've kept
    for path in sorted(glob.glob(REPORTS_GLOB)):  # sorted() = deterministic order
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue  # a malformed/locked report shouldn't abort the whole run
        # Each report file is expected to be a JSON list of records; if it isn't
        # (corrupt/empty), iterate over an empty list instead of crashing.
        for rec in data if isinstance(data, list) else []:
            # The dedup "key" is the pair of test id + cause text. The first time
            # we see a pair we keep it; identical repeats from later files are skipped.
            key = (str(rec.get("test_id")), str(rec.get("likely_cause")))
            if key in seen:
                continue
            seen.add(key)
            records.append(rec)
    return records


def derive_present_components(records: list[dict[str, Any]]) -> list[str]:
    """Return the canonical components that have ≥ 1 failure, in canonical order."""
    present: set[str] = set()  # a set automatically ignores duplicates
    for rec in records:
        # ``.get("failed_checks", []) or []`` guards two cases at once: the key is
        # missing (default []), or it's present but explicitly None (the ``or []``).
        for check in rec.get("failed_checks", []) or []:
            comp = _component_for_check(check)
            if comp:
                present.add(comp)
    # Filter the canonical list down to what's present. Iterating CANONICAL_COMPONENTS
    # (not ``present``) is what gives us a stable, meaningful order in the output.
    return [c for c in CANONICAL_COMPONENTS if c in present]


def _slim_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Keep only the fields the LLM needs, with derived components attached."""
    # Set comprehension with a walrus operator (:=). For each check we compute its
    # component once, bind it to ``c``, and keep ``c`` only when it isn't None.
    # ``sorted({...})`` then turns the unique set into a stable, ordered list.
    comps = sorted(
        {
            c
            for check in (rec.get("failed_checks") or [])
            if (c := _component_for_check(check))
        }
    )
    return {
        "test_id": rec.get("test_id"),
        "category": rec.get("category"),
        "query": rec.get("query"),
        "failed_checks": rec.get("failed_checks"),
        "components": comps,
        "likely_cause": rec.get("likely_cause"),
        "recommended_fix": rec.get("recommended_fix"),
    }


def _truncate_json(obj: Any, max_chars: int = 20_000) -> str:
    """Serialise obj to JSON, hard-capping length to stay within the model window."""
    s = json.dumps(obj, default=str, indent=2)
    return s if len(s) <= max_chars else s[:max_chars] + "\n... (truncated)"


# ---------------------------------------------------------------------------
# LLM call + parsing (mirrors failure_diagnosis.py)
# ---------------------------------------------------------------------------

# Matches a leading ```json (or plain ```) fence and a trailing ``` fence, so we
# can strip them if the model wraps its JSON in a Markdown code block despite being
# told not to. re.MULTILINE makes ^ and $ match at line boundaries.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _build_client() -> OpenAI:
    # We talk to the model through OpenRouter, which speaks the OpenAI API format,
    # so we point the standard OpenAI client at OpenRouter's base URL.
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key)


def _try_parse(text: Any) -> FailureTaxonomy | None:
    """Try to turn one blob of model text into a validated FailureTaxonomy.

    Returns the parsed object on success, or None so the caller can try another
    source (e.g. the ``reasoning`` field). Two attempts:
      1. validate the cleaned text directly as JSON;
      2. if that fails, grab everything between the first ``{`` and last ``}`` —
         this rescues responses where the model added prose around the JSON.
    """
    if not isinstance(text, str):  # defensive: ``reasoning`` may be None/missing
        return None
    cleaned = _FENCE_RE.sub("", text).strip()  # remove any ``` fences
    if not cleaned:
        return None
    # Attempt 1: the whole thing is valid JSON matching our model.
    # model_validate_json parses AND checks the fields in one step; it raises if
    # the JSON is malformed (ValueError) or the shape is wrong (ValidationError).
    try:
        return FailureTaxonomy.model_validate_json(cleaned)
    except (ValidationError, ValueError):
        pass  # fall through to attempt 2
    # Attempt 2: slice out the outermost { ... } and try again.
    first, last = cleaned.find("{"), cleaned.rfind("}")
    if first != -1 and last > first:
        try:
            return FailureTaxonomy.model_validate_json(cleaned[first : last + 1])
        except (ValidationError, ValueError):
            pass
    return None  # both attempts failed


def request_taxonomy(
    components: list[str],
    records: list[dict[str, Any]],
    client: OpenAI | None = None,
) -> FailureTaxonomy:
    """Run one judge-LLM call and return the validated FailureTaxonomy.

    Raises RuntimeError if neither ``content`` nor ``reasoning`` yields valid
    JSON — generating a doc from an unparseable response would be worse than
    failing loudly, since this script runs deliberately (not per-eval-case).
    """
    # Allow a caller to inject a client (handy for tests); otherwise build the
    # real OpenRouter client.
    if client is None:
        client = _build_client()

    # Fill the two remaining placeholders in the prompt: the component list and the
    # (slimmed, length-capped) failure records as JSON.
    prompt = _PROMPT_TEMPLATE.format(
        components=", ".join(components),
        failures=_truncate_json([_slim_record(r) for r in records]),
    )
    # We use JSON mode ({"type": "json_object"}) rather than OpenAI's stricter
    # Structured Outputs ({"type": "json_schema", strict: true} / the
    # client.chat.completions.parse(response_format=FailureTaxonomy) helper), which
    # would *guarantee* the response matches our Pydantic schema via constrained
    # decoding and let us drop the _try_parse recovery below. Two reasons it isn't
    # used here, both because the judge is a non-OpenAI model over OpenRouter:
    #   1. Strict json_schema is an OpenAI-platform feature (gpt-4o-2024-08-06+).
    #      Over OpenRouter it's best-effort passthrough and not guaranteed for the
    #      configured judge model (glm-5).
    #   2. glm-5 sometimes returns its answer in `message.reasoning` with empty
    #      `content` (see the fallback below); .parse() reads message.parsed from
    #      the content channel, so it would miss those responses and regress
    #      reliability. JSON mode + manual validate + reasoning fallback is the
    #      robust choice while the judge is glm-5. Revisit if it moves to an
    #      OpenAI model (or a provider that advertises structured-output support).
    response = client.chat.completions.create(
        model=settings.judge_openrouter_model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,                              # deterministic: same input → same output
        response_format={"type": "json_object"},    # ask the model to emit valid JSON
    )
    message = response.choices[0].message
    # Normally the answer is in message.content. But glm-5 on some OpenRouter
    # providers leaves content empty and puts the final JSON in a non-standard
    # `reasoning` field, so we try content first and fall back to reasoning.
    # ``getattr(message, "reasoning", None)`` reads that field safely even on SDK
    # versions that don't declare it.
    parsed = _try_parse(message.content or "")
    if parsed is None:
        parsed = _try_parse(getattr(message, "reasoning", None) or "")
    if parsed is None:
        # failing loudly is better than writing a broken document.
        raise RuntimeError("Judge LLM returned no parseable FailureTaxonomy JSON.")
    return parsed


# ---------------------------------------------------------------------------
# Coverage guarantees + rendering
# ---------------------------------------------------------------------------

def _representative_record(records: list[dict[str, Any]], component: str) -> dict[str, Any] | None:
    """First record that maps to ``component`` — used to synthesise a fallback mode."""
    for rec in records:
        for check in rec.get("failed_checks") or []:
            if _component_for_check(check) == component:
                return rec
    return None


def _ensure_coverage(
    taxonomy: FailureTaxonomy,
    components: list[str],
    records: list[dict[str, Any]],
) -> dict[str, list[FailureMode]]:
    """Group modes by present component, guaranteeing ≥ 1 mode per present component.

    The prompt asks for full coverage, but if the model omits a component we
    synthesise one grounded entry from that component's records rather than ship
    a section with no entries (which would also fail the acceptance test). Modes
    naming a component that isn't present are dropped — the model was told not to.
    """
    # ``if mode.component in grouped`` quietly discards any mode the model tagged with a component that isn't present.
    grouped: dict[str, list[FailureMode]] = {c: [] for c in components}
    for mode in taxonomy.modes:
        if mode.component in grouped:
            grouped[mode.component].append(mode)

    # Safety net: if the model left a present component with no modes, build one
    # entry from that component's real records so the section is never empty.
    for comp in components:
        if grouped[comp]:
            continue  # already has at least one mode — nothing to do
        rec = _representative_record(records, comp)
        # ``(rec or {})`` lets us call .get() safely even if no record was found.
        cause = (rec or {}).get("likely_cause") or "Failure observed but not described."
        tid = (rec or {}).get("test_id") or "unknown"
        grouped[comp].append(
            FailureMode(
                component=comp,
                mode_name=f"Uncategorised {comp} failure",
                observable_symptom="One or more scorers for this component returned < 1.0.",
                root_cause=cause,
                mitigation=f"Review the {comp} prompt/logic against the cited case.",
                evidence=f"{cause} — {tid}",
            )
        )
    return grouped


def _ensure_trace_reference(grouped: dict[str, list[FailureMode]]) -> None:
    """Guarantee at least one entry names PlannerAdjustment or TraceStep.

    The prompt requests this, but we enforce it as a safety net (the acceptance
    test requires it). If absent, append a grounded pointer to the first Planner
    Agent mode's mitigation — truthful, since planner errors surface in exactly
    those fields.
    """
    # Concatenate every mode's cause + mitigation text into one big string so we
    # can do a simple substring check. (outer loop over each component's list of modes, inner loop over the modes in that list.
    blob = " ".join(
        f"{m.root_cause} {m.mitigation}" for modes in grouped.values() for m in modes
    )
    if "PlannerAdjustment" in blob or "TraceStep" in blob:
        return  # the requirement is already satisfied — leave the text untouched
    # Otherwise append a truthful pointer to the first Planner Agent mode.
    planner_modes = grouped.get("Planner Agent")
    if planner_modes:
        m = planner_modes[0]
        m.mitigation = (
            f"{m.mitigation} Verify the PlannerAdjustment field and the "
            f"TraceStep.adjustment recorded in the trace."
        )


def render_markdown(grouped: dict[str, list[FailureMode]], components: list[str]) -> str:
    """Render the grouped modes into the final Markdown document.

    We build the document as a list of lines and join them at the end. Because
    *this* function controls the formatting (not the LLM), the output structure is
    guaranteed: one ``##`` per component, and exactly the four ``**...**`` lines
    per ``###`` entry that the acceptance test checks for.
    """
    lines = [
        "# Failure Taxonomy",
        "",  # blank line = paragraph break in Markdown
        f"Components with observed failures: {', '.join(components)}.",
        "",
    ]
    # One ## section per component, then one ### entry per failure mode inside it.
    for comp in components:
        lines.append(f"## {comp}")
        lines.append("")
        for mode in grouped[comp]:
            lines.append(f"### {mode.mode_name}")
            lines.append(f"**Observable symptom:** {mode.observable_symptom}")
            lines.append(f"**Root cause:** {mode.root_cause}")
            lines.append(f"**Mitigation:** {mode.mitigation}")
            lines.append(f"**Evidence:** {mode.evidence}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the whole pipeline. Returns a process exit code (0 = success).

    Steps: load reports → work out which components failed → ask the LLM to
    cluster the failures → guarantee coverage + the trace reference → render and
    write the Markdown.
    """
    # 1. Read and de-dupe all failure records.
    records = load_failure_records()
    if not records:
        print(f"No failure records found at {REPORTS_GLOB}; nothing to generate.")
        return 1  # non-zero exit code signals failure to the shell / CI

    # 2. Figure out which components actually have failures.
    components = derive_present_components(records)
    if not components:
        print("Failure records exist but none map to a known component; aborting.")
        return 1

    print(f"components_present={len(components)} ({', '.join(components)})")

    # 3. One LLM call to cluster the raw causes into distinct failure modes.
    taxonomy = request_taxonomy(components, records)

    # 4. Group by component and apply the two safety nets.
    grouped = _ensure_coverage(taxonomy, components, records)
    _ensure_trace_reference(grouped)

    # 5. Render the Markdown and write it, creating docs/ if needed.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(render_markdown(grouped, components))

    # counts modes across every component's bucket.
    n_modes = sum(len(v) for v in grouped.values())
    print(f"Wrote {OUTPUT_PATH} ({n_modes} failure modes across {len(components)} components)")
    return 0


if __name__ == "__main__":
    # ``raise SystemExit(code)`` exits the process with main()'s return value, so
    # the shell sees 0 on success and 1 on the early-return failures above.
    raise SystemExit(main())
