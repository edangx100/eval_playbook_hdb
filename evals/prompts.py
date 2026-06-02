"""Prompt templates used by the HDB eval harness LLM calls and doc-generation scripts."""
from __future__ import annotations

# Placeholders: {test_case}, {output_summary}, {failed_checks}.
# __SCHEMA__ is replaced at import time in failure_diagnosis.py with the rendered
# Diagnosis model schema, keeping the prompt and parser locked together.
DIAGNOSIS_PROMPT = """\
You are diagnosing a failure in an HDB resale comparable-flat search agent.
The agent ran a test case and one or more automated scorers returned < 1.0.

Test case:
{test_case}

Agent output (summary):
{output_summary}

Scorers that returned < 1.0:
{failed_checks}

Respond with a single JSON object that matches this schema (string values only):
__SCHEMA__

Output ONLY the JSON object. No prose, no markdown fences.\
"""

# Chevron template ({{var}}) for autoevals.LLMClassifier — variables injected by score().
LLM_JUDGE_PROMPT = """
You are evaluating whether HDB (Housing Development Board) resale flat comparable \
transactions returned by a search agent are defensible for the given user query.

User query: {{query}}

Extracted target:
{{target_summary}}

Results ({{count}} transactions returned):
{{results_summary}}

Rate the quality of these results against the query constraints:
GOOD    — Results clearly match all constraints (right town, flat type, area band, date range).
PARTIAL — Results mostly match with minor mismatches (one constraint slightly off, or count borderline).
POOR    — Results are largely irrelevant or directly contradict core constraints.
""".strip()

# Placeholders: {failures} — slim failure records JSON, {schema} — _SummaryNarrative schema.
SUMMARY_PROMPT = """\
You are summarising the results of an evaluation run for an HDB resale flat search agent.

Below are all failure records from this run. Each record has a `likely_cause` and \
`recommended_fix` from a per-case LLM diagnosis.

Failure records:
{failures}

Respond with a single JSON object that conforms to this schema:
{schema}

Rules:
- Group failures into 3–5 themes by their likely_cause patterns
- Order themes by case_count descending
- quotes: copy likely_cause verbatim from the records; append " — test_id"
- next_step: one concrete action naming the component to fix

Output ONLY the JSON object. No prose, no markdown fences.\
"""

# ---------------------------------------------------------------------------
# Doc-generation prompts (used by evals/docs/doc_gen_scripts/ and
# evals/playbook/gen_failure_taxonomy.py)
# ---------------------------------------------------------------------------

# Placeholders: {components}, {failures}
# __SCHEMA__ is replaced at import time in gen_failure_taxonomy.py with the
# rendered FailureTaxonomy model schema.
FAILURE_TAXONOMY_PROMPT = """\
You are producing a failure taxonomy for an HDB resale flat comparable-search agent.

The agent uses a two-agent PydanticAI architecture:
- Target Agent — extracts structured Target (town, flat_type, floor_area, storey,
  months_back) from natural language.
- Planner Agent — decides accept / relax / tighten / clarify per loop iteration,
  producing a PlannerAdjustment (field + direction). Each iteration is recorded as
  a TraceStep (filters, count, action, adjustment, retrieval_mode).

Components that have failures in this data (use these EXACT names for the
`component` field, and NO other component):
{components}

Failure records from all evaluation experiments:
{failures}

Identify the DISTINCT failure modes in these records. Respond with a single JSON
object that matches this schema:
__SCHEMA__

Rules:
- Output one entry per distinct failure mode you actually observe. Do NOT pad,
  split, or invent modes to reach any target count.
- Every entry's `component` MUST be one of the components listed above, and you
  should cover each listed component with at least one mode.
- Merge near-duplicate likely_cause strings into a single mode rather than
  emitting one entry per occurrence.
- `evidence` must copy a `likely_cause` verbatim from the records and end with
  " — test_id" (the id of the case it came from).
- At least one entry must name a specific PlannerAdjustment field or TraceStep
  field in its root_cause or mitigation.

Output ONLY the JSON object. No prose, no markdown fences.\
"""
