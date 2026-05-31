# Eval Summary — baseline_before

**Date:** 2026-05-31
**Dataset:** `evals/datasets/hdb_compare_benchmark.yaml` (14 cases)
**Failure report:** `evals/reports/baseline_before_failures.json`

---

## Overall Pass Rate

**1 / 14 cases fully passed — 7.1%**

A case 'fully passes' when every scorer in its `checks` list returns 1.0.

### By category

| Category         | Passed | Total | Pass rate |
|------------------|-------:|------:|----------:|
| easy             |      1 |     8 |      12.5% |
| multi_turn       |      0 |     6 |       0.0% |

---

## Per-Scorer Aggregate Scores

| Scorer | Score |
|--------|------:|
| `planner_adjustment_compliance` |  100.00% |
| `planner_fallback_correct` |  100.00% |
| `reranking_quality` |  100.00% |
| `target_extraction_precision` |  100.00% |
| `trace_adjustment_completeness` |  100.00% |
| `trace_final_step_valid` |  100.00% |
| `trace_quality` |  100.00% |
| `trace_step_completeness` |  100.00% |
| `retrieval_quality` |   91.90% |
| `target_extraction` |   89.27% |
| `planner_decision` |   87.50% |
| `planner_final_action` |   87.50% |
| `target_extraction_recall` |   82.98% |
| `llm_judge` |   62.50% |

---

## Top Failure Modes

### 1. Multi-turn context preservation failures (6 cases)

The agent fails to carry forward extracted fields from previous turns, resulting in null values and retrieval failures.

> *"Agent failed to preserve `flat_type` from Turn 1, setting it to null in Turn 2 instead of inheriting '4 ROOM', causing the retrieval to fail with count 0. — multiturn_001"*

> *"Agent failed to preserve flat_type='3 ROOM' from turn 1 when processing turn 2, resulting in null flat_type and 0 results despite user explicitly mentioning '3-room flat'. — multiturn_006"*

### 2. Planner over-relaxation of explicit constraints (4 cases)

The planner relaxes explicit user constraints like time windows and storey preferences to meet result count targets.

> *"Agent relaxed months_back from 9 to 12 in filters, violating user's explicit 'last 9 months' constraint; target correctly extracted 9 but planner overrode it. — easy_001"*

> *"The planner dropped the user's explicit 'high floor' storey preference during relaxation (step 4) and never restored it, returning results with low/mid storeys like '07 TO 09'. — easy_004"*

### 3. Result count reporting inconsistencies (3 cases)

The agent reports mismatched counts between the trace and final results_total, failing consistency checks.

> *"The agent's trace shows an 'accept' action with count 115, but results_total is 30, suggesting a mismatch in reported vs actual results that the LLM judge flagged as inconsistent or incomplete. — easy_003"*

> *"The agent output has a discrepancy between the trace count (111) and `results_total` (30), failing the LLM judge's consistency check. — easy_006"*

### 4. Extraction failures for flat_type values (2 cases)

The extraction component fails to parse specific flat_type values like 'Executive' and '4 ROOM' from user queries.

> *"The extraction component failed to map 'Executive' to the `flat_type` field, resulting in a null value and a clarification action instead of a search. — easy_007"*

> *"Target extraction failed to parse 'Executive' as flat_type and '18 months' as months_back from turn 1, and multi-turn context preservation did not carry these values forward to turn 2. — multiturn_005"*

---

## Recommended Next Steps

1. **Multi-turn context preservation failures (6 cases):** Fix the multi-turn context merger logic to correctly inherit and accumulate fields from previous turns.
2. **Planner over-relaxation of explicit constraints (4 cases):** Update planner relaxation policy to never widen or drop explicit user constraints like months_back and storey_preference.
3. **Result count reporting inconsistencies (3 cases):** Correct the result aggregation logic to ensure results_total matches the final count reported in the trace.
4. **Extraction failures for flat_type values (2 cases):** Update the extraction prompt and schema to recognize 'Executive' and other flat_type variations as valid values.
