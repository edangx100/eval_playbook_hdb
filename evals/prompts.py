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

# Placeholders: {summaries}
EVAL_METHODOLOGY_PROMPT = """\
You are writing the evaluation methodology document for an HDB resale flat \
comparable-search agent.

Experiment summaries (before/after score evidence):
{summaries}

Write a Markdown document titled "# Evaluation Methodology". Include exactly these
four ## sections in this order:

## Constraint Match Rate
Explain why constraint match rate (hard-filter precision/recall on town, flat_type,
floor_area, storey, date range) was chosen as the primary retrieval quality signal
rather than ground-truth ranking. Cite actual scorer scores from the summaries above
as evidence. Use the term "constraint match" at least once.

## LLMClassifier as Judge
Defend the choice of autoevals LLMClassifier over a bespoke rule-based judge:
calibration, extensibility, and cost. Reference `llm_judge` scorer scores from the
summaries. Use the term "LLMClassifier" at least once.

## Deterministic Fallback Trade-offs
Discuss the trade-offs of the deterministic fallback retrieval path (structured SQL
without vector search): when it fires, why it is kept, and its failure modes. Use
the term "fallback" at least once. Cite `planner_fallback_correct` or
`retrieval_mode` scorer evidence where available.

## Multi-turn Harness Replay
Explain how the eval harness replays multi-turn conversations: turn sequencing,
message_history accumulation, and why single-turn replays are insufficient. Use
the term "multi-turn" at least once. Cite `multi_turn` category pass rates as
evidence.

Each section must be ≥ 3 sentences and grounded in the provided summaries.
Output only the Markdown document. No preamble, no fences.\
"""

# Placeholders: {summaries}, {failures}
PLAYBOOK_PROMPT = """\
You are writing the AI Performance Engineering Playbook for the HDB resale flat \
comparable-search agent project.

Experiment summaries (before/after scores, improvement decisions):
{summaries}

Failure records (likely_cause, recommended_fix):
{failures}

Write a Markdown document titled "# AI Performance Engineering Playbook". Include
exactly these nine ## sections in this order:

## 1. Quality Bars
Define the pass/fail thresholds used in this project. Reference overall pass rates
and per-category rates from the summaries. Explain the rationale for ≥ 30 results
as the agent's retrieval target.

## 2. Benchmark Design
Describe the benchmark dataset structure: categories (easy, sparse, broad,
street_hint, ambiguous, edge, multi_turn, fallback_stress), how cases were authored,
and why this split covers the agent's failure surface.

## 3. Target Extraction Evaluation
Explain how target extraction is evaluated (precision, recall, field-level checks).
Reference `target_extraction`, `target_extraction_precision`, and
`target_extraction_recall` scorer scores. Describe the most common extraction
failure modes from the failure records.

## 4. Retrieval Quality Evaluation
Explain the retrieval quality scorers (`retrieval_quality`, `retrieval_mode`,
`planner_fallback_correct`). Describe how constraint match rate is measured and
what score ranges indicate healthy vs. degraded retrieval.

## 5. Trace Inspection
Explain how TraceStep records are used to diagnose planner behaviour:
`trace_quality`, `trace_step_completeness`, `trace_adjustment_completeness`,
`planner_decision`, `planner_final_action`. Cite actual scores from the summaries.

## 6. Common Failure Modes
Summarise the top recurring failure patterns from the failure records, grouped by
component. Quote at least two `likely_cause` strings verbatim, each ending with
" — test_id".

## 7. Regression Testing Workflow
Describe the workflow for running evals before and after a change: dataset freeze,
experiment naming convention, re-running the improvement loop, comparing summary
reports. Reference the eval runner CLI.

## 8. Human Review Protocol
Describe when and how a human should review eval output: thresholds that trigger
manual inspection, which scorer scores to prioritise, how to read a failure record,
and when to override an LLM judge verdict.

## 9. Prompt and Planner Improvement Process
Describe the iterative improvement loop: diagnose → fix prompt or planner → re-eval
→ compare before/after. Reference specific scorer improvements seen between
experiments in the summaries.

Each section must contain ≥ 3 sentences grounded in the provided data.
Output only the Markdown document. No preamble, no fences.\
"""
