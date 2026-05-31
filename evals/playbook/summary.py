"""Summary report generation for HDB eval runs.

Computes pass-rate stats deterministically from the Braintrust EvalResultWithSummary,
then makes one LLM call to cluster failures into themes and generate the narrative
sections. Output is a Markdown file written to evals/reports/<experiment>_summary.md.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field as PydanticField, ValidationError

from settings import settings
from evals.prompts import SUMMARY_PROMPT

_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Structured output schema
#
# _FailureTheme / _SummaryNarrative are the structured output contract for the
# one LLM call that writes the narrative sections. Field descriptions double as
# prompt documentation; model_validate_json() enforces the shape at parse time.
# ---------------------------------------------------------------------------

class _FailureTheme(BaseModel):
    title: str = PydanticField(description="Short descriptive title (5–8 words)")
    case_count: int = PydanticField(description="Number of distinct cases in this theme")
    description: str = PydanticField(description="One sentence describing the failure pattern")
    quotes: list[str] = PydanticField(description='1–2 verbatim likely_cause strings with source, format: "…" — test_id')
    next_step: str = PydanticField(description="One concrete action to address this theme")


class _SummaryNarrative(BaseModel):
    themes: list[_FailureTheme] = PydanticField(
        description="3–5 failure themes ordered by case_count descending"
    )


def _render_summary_schema() -> str:
    """Serialise _SummaryNarrative's JSON schema for embedding in the prompt.

    Sending the schema rather than prose keeps the prompt and parser in sync:
    if _FailureTheme gains a new field, the schema updates automatically on next import.
    """
    return json.dumps(_SummaryNarrative.model_json_schema(), indent=2)


# Strips a leading ```json (or plain ```) fence and a trailing ``` fence if the
# model wraps its JSON in a Markdown block despite being told not to.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

# How many times to call the judge for the narrative before giving up. glm-5
# returns an unparseable response on a minority of calls; retrying independent
# draws recovers almost all of them (see SUMMARY_FIX.md).
_SUMMARY_LLM_ATTEMPTS = 3


def _try_parse(text: Any) -> _SummaryNarrative | None:
    """Validate one blob of model text into a _SummaryNarrative, or return None.

    Returning None lets the caller try another source (e.g. the ``reasoning``
    field). Mirrors gen_failure_taxonomy._try_parse: first validate the cleaned
    text as JSON, then fall back to slicing the outermost ``{ ... }`` to rescue
    responses that wrapped the JSON in prose.
    """
    if not isinstance(text, str):  # defensive: ``reasoning`` may be None/missing
        return None
    cleaned = _FENCE_RE.sub("", text).strip()
    if not cleaned:
        return None
    try:
        return _SummaryNarrative.model_validate_json(cleaned)
    except (ValidationError, ValueError):
        pass
    first, last = cleaned.find("{"), cleaned.rfind("}")
    if first != -1 and last > first:
        try:
            return _SummaryNarrative.model_validate_json(cleaned[first : last + 1])
        except (ValidationError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def write_summary(
    eval_result: Any,
    experiment_name: str,
    failures_path: Path,
) -> Path:
    """Generate {experiment_name}_summary.md from eval results and the failure report.

    Stats are computed deterministically from eval_result.results. The narrative
    (Top Failure Modes + Recommended Next Steps) comes from one LLM call whose
    output is validated against _SummaryNarrative.

    Returns the path to the written summary file.
    """
    failure_records: list[dict[str, Any]] = json.loads(failures_path.read_text())
    failed_ids = {r["test_id"] for r in failure_records}

    total = len(eval_result.results)
    pass_count = total - len(failed_ids)
    pass_pct = 100.0 * pass_count / total if total else 0.0

    # --- deterministic stats from EvalResult rows ---

    cat_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "total": 0})
    for row in eval_result.results:
        cat = row.input.get("category", "unknown")
        tid = row.input.get("id", "")
        cat_stats[cat]["total"] += 1
        if tid not in failed_ids:
            cat_stats[cat]["passed"] += 1

    # row.scores is Dict[str, Optional[float]]; None means the scorer was
    # skipped for that case — excluded from the mean.
    scorer_totals: dict[str, list[float]] = defaultdict(list)
    for row in eval_result.results:
        scores = getattr(row, "scores", {}) or {}
        for name, score in scores.items():
            if score is not None:
                scorer_totals[name].append(score)
    scorer_means = {
        name: 100.0 * sum(vals) / len(vals)
        for name, vals in scorer_totals.items()
    }
    sorted_scorers = sorted(scorer_means.items(), key=lambda x: (-x[1], x[0]))

    # --- build static Markdown sections ---

    cat_order = ["easy", "sparse", "broad", "street_hint", "ambiguous", "edge", "multi_turn", "fallback_stress"]
    cat_rows = [
        f"| {cat:<16} | {cat_stats[cat]['passed']:>6} | {cat_stats[cat]['total']:>5} |"
        f" {100.0 * cat_stats[cat]['passed'] / cat_stats[cat]['total']:>9.1f}% |"
        for cat in cat_order if cat in cat_stats
    ]
    scorer_rows = [f"| `{name}` | {mean:>7.2f}% |" for name, mean in sorted_scorers]

    lines: list[str] = [
        f"# Eval Summary — {experiment_name}",
        "",
        f"**Date:** {date.today().isoformat()}",
        f"**Dataset:** `evals/datasets/hdb_compare_benchmark.yaml` ({total} cases)",
        f"**Failure report:** `evals/reports/{experiment_name}_failures.json`",
        "",
        "---",
        "",
        "## Overall Pass Rate",
        "",
        f"**{pass_count} / {total} cases fully passed — {pass_pct:.1f}%**",
        "",
        "A case 'fully passes' when every scorer in its `checks` list returns 1.0.",
        "",
        "### By category",
        "",
        "| Category         | Passed | Total | Pass rate |",
        "|------------------|-------:|------:|----------:|",
        *cat_rows,
        "",
        "---",
        "",
        "## Per-Scorer Aggregate Scores",
        "",
        "| Scorer | Score |",
        "|--------|------:|",
        *scorer_rows,
        "",
        "---",
        "",
    ]

    if not failure_records:
        lines += [
            "## Top Failure Modes", "", "_No failures._", "",
            "---", "",
            "## Recommended Next Steps", "", "_No failures — nothing to action._",
        ]
    else:
        slim = [
            {
                "test_id": r.get("test_id"),
                "category": r.get("category"),
                "failed_checks": r.get("failed_checks"),
                "likely_cause": r.get("likely_cause"),
                "recommended_fix": r.get("recommended_fix"),
            }
            for r in failure_records
        ]
        # Direct OpenRouter client (not Braintrust-wrapped) — wrap_openai() can
        # inflate token counts when called outside an active experiment context.
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        )
        # Build the prompt once, outside the retry loop, since it never changes
        # between attempts (the failure records and schema are fixed).
        prompt = SUMMARY_PROMPT.format(
            failures=json.dumps(slim, indent=2),
            schema=_render_summary_schema(),
        )
        # Retry loop (Layer 1 of SUMMARY_FIX.md). glm-5 over OpenRouter is
        # non-deterministic even at temperature 0: it leaves `content` empty and
        # returns the JSON in a `reasoning` field that is sometimes truncated or
        # buried in a long chain-of-thought, so ~40% of single calls are
        # unparseable. Because each call is an *independent* draw, simply asking
        # again usually yields a clean response — three attempts cuts the
        # per-run miss rate from ~40% to ~0.4**3 ≈ 6%.
        narrative = None  # holds the parsed _SummaryNarrative once a call succeeds
        for _ in range(_SUMMARY_LLM_ATTEMPTS):
            response = client.chat.completions.create(
                model=settings.judge_openrouter_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            message = response.choices[0].message
            # Normal case: the answer is in `content`. glm-5 frequently returns
            # empty `content` and puts the JSON in the non-standard `reasoning`
            # field instead, so fall back to that when `content` doesn't parse.
            narrative = _try_parse(message.content or "")
            if narrative is None:
                narrative = _try_parse(getattr(message, "reasoning", None) or "")
            # Got a valid narrative — stop retrying.
            if narrative is not None:
                break
        # Every attempt failed to parse. Raising here (rather than re-trying
        # forever) keeps the run bounded; Layer 3 of SUMMARY_FIX.md would instead
        # degrade gracefully and still write the deterministic stats.
        if narrative is None:
            raise RuntimeError(
                f"Judge LLM returned no parseable summary narrative after "
                f"{_SUMMARY_LLM_ATTEMPTS} attempts (empty content and reasoning)."
            )

        lines.append("## Top Failure Modes")
        for i, theme in enumerate(narrative.themes, 1):
            lines += [
                "",
                f"### {i}. {theme.title} ({theme.case_count} cases)",
                "",
                theme.description,
            ]
            for q in theme.quotes:
                lines += ["", f"> *\"{q}\"*"]
        lines += ["", "---", "", "## Recommended Next Steps", ""]
        for i, theme in enumerate(narrative.themes, 1):
            lines.append(f"{i}. **{theme.title} ({theme.case_count} cases):** {theme.next_step}")

    out_path = _ROOT / "evals" / "reports" / f"{experiment_name}_summary.md"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
