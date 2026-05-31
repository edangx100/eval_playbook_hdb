# Failure Taxonomy

Components with observed failures: Target Agent, Planner Agent, retrieval.

## Target Agent

### Multi-turn context preservation failure
**Observable symptom:** Target fields from previous turns are set to null in subsequent turns instead of being preserved.
**Root cause:** Multi-turn state merger fails to carry forward previously extracted fields when processing new constraints.
**Mitigation:** Update Target Agent's multi-turn context merger logic to preserve all existing fields when merging with new turn constraints.
**Evidence:** Agent failed to preserve `flat_type` from Turn 1, setting it to null in Turn 2 instead of inheriting '4 ROOM', causing the retrieval to fail with count 0. — multiturn_001

### Flat type extraction failure
**Observable symptom:** Target extraction returns null for flat_type when user specifies 'Executive' or similar non-standard flat type terms.
**Root cause:** Extraction prompt or schema does not recognize 'Executive' as a valid flat_type value.
**Mitigation:** Update Target Agent extraction prompt to include 'Executive' as a recognized flat_type value.
**Evidence:** The extraction component failed to map 'Executive' to the `flat_type` field, resulting in a null value and a clarification action instead of a search. — easy_007

## Planner Agent

### Clarification action on incomplete target
**Observable symptom:** Planner produces clarification action instead of search when required target field is null.
**Root cause:** Target Agent extraction failure leaves flat_type null, triggering Planner Agent's clarification logic instead of search execution.
**Mitigation:** Update Target Agent extraction prompt to recognize 'Executive' as valid flat_type value, preventing null fields that trigger clarification.
**Evidence:** The extraction component failed to map 'Executive' to the `flat_type` field, resulting in a null value and a clarification action instead of a search. — easy_007

## retrieval

### Explicit time constraint relaxation
**Observable symptom:** Retrieval results include records outside the user's specified months_back window.
**Root cause:** Planner Agent's relaxation logic widens the months_back filter in TraceStep to increase result count, overriding explicit user constraints.
**Mitigation:** Update Planner Agent relaxation policy to never widen months_back beyond user-specified value in the filters field.
**Evidence:** Agent relaxed months_back from 9 to 12 in filters, violating user's explicit 'last 9 months' constraint; target correctly extracted 9 but planner overrode it. — easy_001

### Storey preference constraint relaxation
**Observable symptom:** Retrieval results include flats with storey ranges outside the user's specified storey_preference.
**Root cause:** Planner Agent drops storey_preference from filters during relaxation to meet minimum result count targets.
**Mitigation:** Update Planner Agent relaxation hierarchy to preserve storey_preference in filters over meeting minimum count thresholds.
**Evidence:** The planner dropped the user's explicit 'high floor' storey preference during relaxation (step 4) and never restored it, returning results with low/mid storeys like '07 TO 09'. — easy_004

### Trace count discrepancy
**Observable symptom:** The count field in TraceStep does not match the final results_total in agent output.
**Root cause:** Result aggregation logic fails to synchronize the trace count with the actual retrieved result count.
**Mitigation:** Fix retrieval result aggregation to ensure results_total matches the count field in the final TraceStep.
**Evidence:** The agent output has a discrepancy between the trace count (111) and `results_total` (30), failing the LLM judge's consistency check. — easy_006
