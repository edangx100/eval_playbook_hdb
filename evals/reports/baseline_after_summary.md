# Eval Summary — baseline_after

**Date:** 2026-05-31
**Dataset:** `evals/datasets/hdb_compare_benchmark.yaml` (14 cases)
**Failure report:** `evals/reports/baseline_after_failures.json`

---

## Overall Pass Rate

**6 / 14 cases fully passed — 42.9%**

A case 'fully passes' when every scorer in its `checks` list returns 1.0.

### By category

| Category         | Passed | Total | Pass rate |
|------------------|-------:|------:|----------:|
| easy             |      2 |     8 |      25.0% |
| multi_turn       |      4 |     6 |      66.7% |

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
| `target_extraction` |   97.55% |
| `target_extraction_recall` |   95.83% |
| `retrieval_quality` |   87.86% |
| `planner_decision` |   87.50% |
| `planner_final_action` |   87.50% |
| `llm_judge` |   62.50% |

---

## Top Failure Modes

### 1. Storey Preference Constraint Violations During Relaxation (3 cases)

The planner drops or ignores user-specified floor/storey preferences during constraint relaxation, returning results that violate the original query intent.

> *""Agent dropped `storey_preference` during relaxation, accepting low-floor results (e.g., '07 TO 09') that violate the user's 'high floor' requirement." — easy_004"*

> *""The planner dropped the user-specified 'low floor' constraint during relaxation, returning high-floor units (e.g., '16 TO 18') that violate the query intent." — easy_008"*

### 2. Time Window Constraint Violations During Relaxation (3 cases)

The planner automatically widens user-specified time windows during relaxation, violating explicit temporal constraints in the query.

> *""The planner relaxed the 'months_back' filter from 9 to 12 to boost count from 15 to 31, violating the user's specific time constraint." — easy_001"*

> *""Agent violated the explicit 'last 12 months' query constraint by relaxing the time window to 18 months during the retrieval adjustment phase." — easy_002"*

### 3. Target Extraction Failures for Flat Types (1 cases)

The extraction component fails to map user-specified flat types to the correct schema fields, causing downstream search failures.

> *""Extraction failed to map 'Executive flat' to `flat_type`, resulting in a null value that triggered a 'clarify' action instead of a search." — easy_007"*

### 4. Multi-turn State Preservation Failures (1 cases)

The agent fails to preserve context from previous turns, losing critical query parameters across the conversation.

> *""Agent failed to preserve `flat_type` from Turn 1 and incorrectly reset `months_back` to 12, violating state preservation rules." — multiturn_005"*

---

## Recommended Next Steps

1. **Storey Preference Constraint Violations During Relaxation (3 cases):** Update the planner relaxation logic to treat storey_preference as a hard constraint that cannot be dropped without user confirmation.
2. **Time Window Constraint Violations During Relaxation (3 cases):** Update the planner logic to treat time-window constraints as hard limits that cannot be automatically widened.
3. **Target Extraction Failures for Flat Types (1 cases):** Update the target extraction prompt to correctly identify 'Executive' and other flat types as valid `flat_type` values.
4. **Multi-turn State Preservation Failures (1 cases):** Update the agent's state management logic to correctly carry forward preserved fields across conversation turns.
