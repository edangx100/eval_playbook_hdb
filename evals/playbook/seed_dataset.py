#!/usr/bin/env python3
"""One-time setup script: pushes hdb_compare_benchmark.yaml to Braintrust as a versioned,
remote dataset. Once seeded, the eval harness can reference cases by name rather than
reading the local YAML. Braintrust tracks dataset history, links experiment results to
exact input cases, and gives CI/teammates access without needing the local file.

Re-run whenever the YAML changes significantly to sync local edits to the platform.

Usage:
    python evals/playbook/seed_dataset.py
    # BRAINTRUST_API_KEY must be set in .env or environment.
"""
import sys
from pathlib import Path

import yaml
import braintrust

from settings import settings


# Both values are read from settings (.env) so they stay in sync with run.py.
_ROOT = Path(__file__).resolve().parents[2]
PROJECT_NAME = settings.eval_project
# Resolve relative paths from project root; absolute paths are used as-is.
DATASET_PATH = _ROOT / settings.eval_dataset_path
DATASET_NAME = "hdb_compare_benchmark"


def main() -> None:
    if not DATASET_PATH.exists():
        print(f"ERROR: {DATASET_PATH} not found", file=sys.stderr)
        sys.exit(1)

    cases = yaml.safe_load(DATASET_PATH.read_text())
    if not cases:
        print("ERROR: no cases loaded from YAML", file=sys.stderr)
        sys.exit(1)

    if not settings.braintrust_api_key:
        print("ERROR: BRAINTRUST_API_KEY not set in .env or environment", file=sys.stderr)
        sys.exit(1)

    dataset = braintrust.init_dataset(
        project=PROJECT_NAME,
        name=DATASET_NAME,
        description=(
            "HDB Compare Agents benchmark — 38 curated test cases across 8 categories: "
            "easy, sparse, broad, street_hint, ambiguous, edge, multi_turn, fallback_stress."
        ),
        api_key=settings.braintrust_api_key,
    )

    inserted = 0
    for case in cases:
        dataset.insert(
            input=case,
            expected=None,
            metadata={"id": case["id"], "category": case.get("category", "unknown")},
            id=case["id"],
        )
        inserted += 1

    dataset.flush()
    print(
        f"Uploaded {inserted} cases to dataset '{DATASET_NAME}' "
        f"in project '{PROJECT_NAME}'"
    )


if __name__ == "__main__":
    main()
