"""
Eval runner for the HDB comparable-flats agent.

Entry point for all evaluation runs. Initialises the Braintrust project context
(hdb-compare-agents-eval, separate from the production project), routes autoevals
LLM calls through the Braintrust proxy, and runs Eval() with all scorers.

After Eval() completes, the runner walks its returned ``EvalResultWithSummary``
to find per-case failures (any score < 1.0 or a task crash) and calls the
failure-diagnosis LLM once per failing case. This matches the pattern Braintrust
support recommended: do not collect scores via module-level state — let the SDK
hand us the per-case bundle (input, output, scores, error) when Eval finishes.

Usage:
    python evals/playbook/run.py --experiment-name baseline
    python evals/playbook/run.py --experiment-name smoke --subset easy_001,easy_002
    python evals/playbook/run.py --experiment-name smoke --subset easy
    python evals/playbook/run.py --experiment-name t1 --dataset evals/datasets/hdb_compare_benchmark.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path when run directly (python evals/playbook/run.py).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import braintrust
from braintrust import Eval
from openai import OpenAI

from hdb_search_agents.agent.orchestrator import run_iterative_search
from settings import settings

from evals.playbook.failure_diagnosis import build_failure_record, diagnose_failure, write_failure_report
from evals.playbook.scorer_wrappers import ALL_SCORERS, set_proxy_client
from evals.playbook.summary import write_summary

_EVAL_PROJECT = settings.eval_project
_DEFAULT_DATASET = _ROOT / settings.eval_dataset_path

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Braintrust setup
# ---------------------------------------------------------------------------

def _build_proxy_client() -> OpenAI:
    """Return an OpenAI-compatible OpenRouter client wrapped for Braintrust tracing.

    wrap_openai() intercepts calls client-side and sends telemetry to Braintrust;
    actual HTTP requests still go directly to OpenRouter, so any model OpenRouter
    supports works without Braintrust needing to know about it.
    """
    client = OpenAI(base_url=_OPENROUTER_BASE_URL, api_key=settings.openrouter_api_key)
    return braintrust.wrap_openai(client)


# ---------------------------------------------------------------------------
# task() — calls the agent, returns a serialisable output dict
#
# SINGLE-TURN: input has a top-level 'query' key.
# MULTI-TURN: input has a top-level 'turns' list instead of 'query'.
#   Replays turns in order, threading response.messages as message_history
#   into each subsequent call. The final turn's output is what scorers receive.
#   prior_target carries the second-to-last turn's extracted Target so delta
#   scorers can check preservation (target_extraction, retrieval_quality).
# ---------------------------------------------------------------------------

async def task(input_dict: dict[str, Any]) -> dict[str, Any]:
    """Run the agent on a test case. Single-turn cases have a 'query' key; multi-turn cases have a 'turns' list."""
    if "turns" in input_dict:
        turns = input_dict["turns"]
        messages = None
        prior_target: dict[str, Any] | None = None
        response = None
        for turn in turns:
            if response is not None:
                prior_target = response.target.model_dump()
            response = await run_iterative_search(
                turn["query"],
                message_history=messages,
                prior_target=prior_target,
                verbose=False,
            )
            messages = response.messages

        retrieval_mode = response.trace[-1].retrieval_mode if response.trace else "structured"
        return {
            "target": response.target.model_dump(),
            "filters": response.filters,
            "count": response.count,
            "retrieval_mode": retrieval_mode,
            "trace": [step.model_dump() for step in response.trace],
            "results": [r.model_dump() for r in response.results],
            "note": response.note,
            "prior_target": prior_target,
        }

    query = input_dict["query"]
    response = await run_iterative_search(query, verbose=False)
    retrieval_mode = response.trace[-1].retrieval_mode if response.trace else "structured"
    return {
        "target": response.target.model_dump(),
        "filters": response.filters,
        "count": response.count,
        "retrieval_mode": retrieval_mode,
        "trace": [step.model_dump() for step in response.trace],
        "results": [r.model_dump() for r in response.results],
        "note": response.note,
    }


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python evals/playbook/run.py",
        description=(
            "Run evaluations against the HDB comparable-flats agent and publish "
            "results to the Braintrust project 'hdb-compare-agents-eval'."
        ),
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        metavar="NAME",
        help="Braintrust experiment name (e.g. 'baseline', 'planner_v2'). Required.",
    )
    parser.add_argument(
        "--subset",
        default=None,
        metavar="IDS_OR_CATEGORY",
        help=(
            "Comma-separated case IDs or a single category name to run a subset. "
            "E.g. 'easy_001,easy_002' or 'easy'. Omit to run the full dataset."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=str(_DEFAULT_DATASET),
        metavar="PATH",
        help=f"Path to benchmark YAML dataset file. Defaults to {_DEFAULT_DATASET}.",
    )
    return parser


def _load_dataset(path: str) -> list[dict[str, Any]]:
    import yaml
    with open(path) as f:
        cases = yaml.safe_load(f)
    if not isinstance(cases, list):
        raise ValueError(f"Dataset at {path} must be a YAML list of test cases.")
    return cases


def _filter_dataset(cases: list[dict[str, Any]], subset: str | None) -> list[dict[str, Any]]:
    """Apply --subset filter: comma-separated IDs or a single category name."""
    if not subset:
        return cases
    tokens = [t.strip() for t in subset.split(",") if t.strip()]
    if not tokens:
        return cases
    ids = {c.get("id") for c in cases}
    if len(tokens) == 1 and tokens[0] not in ids:
        filtered = [c for c in cases if c.get("category") == tokens[0]]
        if filtered:
            return filtered
    return [c for c in cases if c.get("id") in set(tokens)]


# ---------------------------------------------------------------------------
# Post-Eval failure diagnosis pass
# ---------------------------------------------------------------------------

def _failed_checks_for_row(row: Any) -> list[str]:
    """Return the names of scorers that failed (or 'task_error') for one row.

    Scorers skipped for the case (not in its 'checks' list) appear as None in
    row.scores and are ignored — a skipped check is not counted as failed.
    """
    if getattr(row, "error", None) is not None:
        return ["task_error"]
    scores = getattr(row, "scores", {}) or {}
    return [name for name, s in scores.items() if s is not None and s < 1.0]


def _run_diagnosis_pass(eval_result: Any, experiment_name: str, client: OpenAI | None) -> Path:
    """Walk Eval()'s return value, diagnose every failing case, write the JSON report."""
    records_out: list[dict[str, Any]] = []
    for row in eval_result.results:
        failed = _failed_checks_for_row(row)
        if not failed:
            continue
        case = row.input
        output = row.output or {}
        diagnosis = diagnose_failure(case, output, failed, client=client)
        records_out.append(build_failure_record(case, output, failed, diagnosis))

    out_path = _ROOT / "evals" / "reports" / f"{experiment_name}_failures.json"
    write_failure_report(records_out, out_path)
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    cases = _load_dataset(args.dataset)
    cases = _filter_dataset(cases, args.subset)

    if not cases:
        print(f"No test cases selected. Check --subset value: {args.subset!r}", file=sys.stderr)
        sys.exit(1)

    api_key = settings.braintrust_api_key
    if api_key:
        braintrust.login(api_key=api_key)

    set_proxy_client(_build_proxy_client())

    print(
        f"Running eval: project={_EVAL_PROJECT!r}, "
        f"experiment={args.experiment_name!r}, "
        f"cases={len(cases)}."
    )

    eval_result = Eval(
        _EVAL_PROJECT,
        data=[{"input": case} for case in cases],
        task=task,
        scores=ALL_SCORERS,
        experiment_name=args.experiment_name,
    )

    # Pass client=None so diagnose_failure builds a plain OpenRouter client —
    # the wrapped proxy client can inflate token counts to 200k+ via wrap_openai().
    report_path = _run_diagnosis_pass(eval_result, args.experiment_name, client=None)
    print(f"Failure report written: {report_path}")

    summary_path = write_summary(eval_result, args.experiment_name, report_path)
    print(f"Summary written: {summary_path}")


if __name__ == "__main__":
    main()
