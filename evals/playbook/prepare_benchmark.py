#!/usr/bin/env python3
"""
Prepare a fresh hdb_compare_benchmark.yaml from the live DB.

Auto-generates easy, sparse, broad, and street_hint cases by querying the
hdb_resale table. Ambiguous, edge, multi_turn, and fallback_stress categories
use static templates baked into this script — edit them here when the ground-
truth conversations change.

Why split auto vs static?
  Auto categories (easy/sparse/broad/street_hint) are sensitive to the data
  currently in the DB: which town/flat_type combos are viable, what count
  ranges to expect, which street names exist. These must be regenerated
  whenever the DB is refreshed with new ingestion data.

  Static categories (ambiguous/edge/multi_turn/fallback_stress) are about
  agent behaviour that is independent of data volume — missing fields,
  invalid towns, multi-turn context retention. They are stable across
  data refreshes and are maintained by hand.

Usage:
    # Regenerate the full benchmark from the live DB:
    python evals/playbook/prepare_benchmark.py

    # Custom output path and reproducible sampling:
    python evals/playbook/prepare_benchmark.py --out /tmp/test_bench.yaml --seed 42

    # Adjust case counts per category:
    python evals/playbook/prepare_benchmark.py --easy 10 --sparse 4 --broad 3 --street 6

    # Verify an existing YAML against the current DB (no generation):
    python evals/playbook/prepare_benchmark.py --verify
    python evals/playbook/prepare_benchmark.py --verify evals/datasets/hdb_compare_benchmark.yaml
"""
from __future__ import annotations

import argparse
import io
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml
import sqlalchemy as sa

# Add the project root to sys.path so `from settings import settings` works
# whether the script is run as `python evals/playbook/prepare_benchmark.py`
# or as a module.
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from settings import settings

_THIS_DIR = Path(__file__).parent
_DEFAULT_OUT = _THIS_DIR / "hdb_compare_benchmark.yaml"


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _engine() -> sa.Engine:
    """Create a SQLAlchemy engine from the DATABASE_URL in settings."""
    return sa.create_engine(settings.database_url)


def _db_info(engine: sa.Engine) -> dict[str, Any]:
    """Return the date range and row counts for the header comment in the YAML."""
    sql = sa.text("""
        SELECT
            MIN(month_date)  AS min_date,
            MAX(month_date)  AS max_date,
            COUNT(*)         AS total,
            -- with_embeddings is what the agent actually searches over;
            -- rows without embeddings are excluded from hybrid/vector retrieval.
            COUNT(*) FILTER (WHERE listing_embedding IS NOT NULL) AS with_embeddings
        FROM hdb_resale
    """)
    with engine.connect() as conn:
        row = conn.execute(sql).mappings().fetchone()
    return dict(row)


def _combinations(engine: sa.Engine) -> list[dict[str, Any]]:
    """Return every town/flat_type combination with pre-computed counts at
    multiple time windows (3 / 6 / 9 / 12 / 18 months).

    Computing all windows in one pass avoids N separate round-trips.
    The counts are used to classify each combo as easy / sparse / broad
    without running the agent.
    """
    sql = sa.text("""
        SELECT
            town,
            flat_type,
            -- avg_area drives floor_area_target in generated queries.
            ROUND(AVG(floor_area_sqm)) AS avg_area,
            COUNT(*)                   AS count_all,
            COUNT(*) FILTER (WHERE month_date >= CURRENT_DATE - INTERVAL '18 months') AS count_18mo,
            COUNT(*) FILTER (WHERE month_date >= CURRENT_DATE - INTERVAL '12 months') AS count_12mo,
            COUNT(*) FILTER (WHERE month_date >= CURRENT_DATE - INTERVAL '9 months')  AS count_9mo,
            COUNT(*) FILTER (WHERE month_date >= CURRENT_DATE - INTERVAL '6 months')  AS count_6mo,
            COUNT(*) FILTER (WHERE month_date >= CURRENT_DATE - INTERVAL '3 months')  AS count_3mo
        FROM hdb_resale
        GROUP BY town, flat_type
        ORDER BY town, flat_type
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql).mappings().fetchall()
    return [dict(r) for r in rows]


def _streets(engine: sa.Engine, limit: int = 100) -> list[dict[str, Any]]:
    """Sample street names that have 10–200 transactions in the last 18 months.

    The 10–200 window ensures the street is active enough to test but narrow
    enough to be a meaningful hint. ORDER BY RANDOM() gives a fresh sample
    on each run; use --seed to make sampling reproducible.
    """
    sql = sa.text("""
        SELECT
            town,
            flat_type,
            street_name,
            COUNT(*) FILTER (WHERE month_date >= CURRENT_DATE - INTERVAL '18 months') AS count_18mo
        FROM hdb_resale
        GROUP BY town, flat_type, street_name
        HAVING COUNT(*) FILTER (WHERE month_date >= CURRENT_DATE - INTERVAL '18 months')
               BETWEEN 10 AND 200
        ORDER BY RANDOM()
        LIMIT :lim
    """)
    with engine.connect() as conn:
        rows = conn.execute(sql, {"lim": limit}).mappings().fetchall()
    return [dict(r) for r in rows]


def _count_for_case(engine: sa.Engine, case: dict[str, Any]) -> int | None:
    """Run a raw SQL count matching the expected_target constraints of one case.

    Used only in --verify mode to compare the live DB count against the
    expected_count_range recorded in the YAML.

    Note: this count is the *initial* count before any planner relaxation.
    expected_count_range in the YAML is the *final* count after the planner
    loop exits, so the comparison is approximate — it detects cases that have
    gone badly stale, not minor drift.

    Returns None when the case lacks town or flat_type (e.g. ambiguous cases),
    since there is no useful count to compute.
    """
    t = case.get("expected_target", {})
    if not t.get("town") or not t.get("flat_type"):
        return None

    clauses = ["town = :town", "flat_type = :flat_type"]
    params: dict[str, Any] = {"town": t["town"], "flat_type": t["flat_type"]}

    if t.get("months_back"):
        clauses.append("month_date >= CURRENT_DATE - (:months * INTERVAL '1 month')")
        params["months"] = t["months_back"]

    if t.get("floor_area_target"):
        # ±5 sqm mirrors the default tolerance used by the orchestrator's
        # structured retrieval filter.
        clauses.append("floor_area_sqm BETWEEN :area_lo AND :area_hi")
        params["area_lo"] = float(t["floor_area_target"]) - 5
        params["area_hi"] = float(t["floor_area_target"]) + 5

    sql = sa.text(f"SELECT COUNT(*) FROM hdb_resale WHERE {' AND '.join(clauses)}")
    with engine.connect() as conn:
        return conn.execute(sql, params).scalar()


# ---------------------------------------------------------------------------
# Name formatting helpers
# ---------------------------------------------------------------------------

def _human_town(town: str) -> str:
    """ANG MO KIO → Ang Mo Kio (title-case for readable query strings)."""
    return town.title()


def _human_flat_type(ft: str) -> str:
    """4 ROOM → 4-room, EXECUTIVE → Executive (natural language form)."""
    ft = ft.strip()
    if ft == "EXECUTIVE":
        return "Executive"
    if ft == "MULTI-GENERATION":
        return "Multi-generation"
    if "ROOM" in ft:
        num = ft.split()[0]
        return f"{num}-room"
    return ft.title()


def _human_street(street: str) -> str:
    """FERNVALE RD → Fernvale Rd (title-case; agent handles abbreviations)."""
    return street.title()


# ---------------------------------------------------------------------------
# Case generators — one function per auto-generated category
# ---------------------------------------------------------------------------

# Storey options weighted toward None so most easy cases omit the preference,
# matching how real users query (more often omit storey than specify it).
_STOREY_OPTIONS: list[str | None] = ["low", "mid", "high", None, None, None]

# Query string templates for the easy category. Using lambdas lets us vary
# phrasing across cases so the agent sees natural variation, not a rigid template.
_EASY_TEMPLATES = [
    lambda ft, town, area, storey, months: (
        f"Find {ft} flat in {town}"
        + (f", around {area} sqm" if area else "")
        + (f", {storey} floor" if storey else "")
        + f", last {months} months"
    ),
    lambda ft, town, area, storey, months: (
        f"Looking for {ft} flat in {town}"
        + (f", around {area} sqm" if area else "")
        + (f", {storey} floor" if storey else "")
        + f", last {months} months"
    ),
    lambda ft, town, area, storey, months: (
        f"{ft} resale flat in {town}"
        + (f", {area} sqm" if area else "")
        + f", last {months} months"
        + (f", {storey} floor preferred" if storey else "")
    ),
]


def _easy_cases(combos: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Generate easy (happy-path) cases.

    Selection criterion: count_12mo in [30, 200] — the initial candidate pool
    is already in target range, so the planner should accept in ≤ 2 steps.

    Each case randomly picks storey preference, months_back, and whether to
    include a floor area target, producing natural variation across cases.
    """
    # Filter to combinations that already sit in the target range for a 12-month window.
    pool = [c for c in combos if 30 <= c["count_12mo"] <= 200]
    rng.shuffle(pool)
    cases: list[dict] = []
    seen: set = set()

    for c in pool:
        if len(cases) >= n:
            break
        # Avoid two easy cases with the same town/flat_type pair.
        key = (c["town"], c["flat_type"])
        if key in seen:
            continue
        seen.add(key)

        storey = rng.choice(_STOREY_OPTIONS)
        months = rng.choice([6, 9, 12, 18])
        include_area = rng.random() < 0.6  # 60% of easy cases include a floor area target
        area = round(float(c["avg_area"]), 1) if include_area else None

        tmpl = rng.choice(_EASY_TEMPLATES)
        query = tmpl(_human_flat_type(c["flat_type"]), _human_town(c["town"]), area, storey, months)

        # Build expected_target — only include fields present in the query.
        expected_target: dict[str, Any] = {
            "town": c["town"],
            "flat_type": c["flat_type"],
            "months_back": months,
        }
        if area:
            expected_target["floor_area_target"] = area
        if storey:
            expected_target["storey_preference"] = storey

        # reranking_quality measures whether scoring improved area proximity
        # relative to the raw pool — only meaningful when a floor area target
        # was specified.
        checks = ["target_extraction", "retrieval_quality", "planner_decision", "trace_quality"]
        if area:
            checks.append("reranking_quality")

        cases.append({
            "id": f"easy_{len(cases) + 1:03d}",
            "category": "easy",
            "query": query,
            "expected_target": expected_target,
            "expected_count_range": {"min": 30, "max": 200},
            "checks": checks,
        })

    return cases


def _sparse_cases(combos: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Generate sparse (under-results) cases.

    The agent must relax at least once to reach 30+ results.
    Three pools are tried in order to fill the requested count:

    1. Short 3-month window on an otherwise viable combo.
       count_3mo < 30 ensures the initial query is under-count,
       while count_12mo >= 30 ensures relaxing the time window will help.

    2. Rare combinations (total row count 5–50).
       Even a long time window won't yield 30+ results for these.

    3. 6-month window, optionally combined with a floor area filter.
       Provides variety when pools 1 and 2 are exhausted.
    """
    cases: list[dict] = []
    seen: set = set()

    def _add(c: dict, months: int, include_area: bool = False) -> bool:
        # Deduplicate by (town, flat_type, months) so we don't generate two
        # identical sparse cases that differ only in random seed order.
        key = (c["town"], c["flat_type"], months)
        if key in seen or len(cases) >= n:
            return False
        seen.add(key)

        area = round(float(c["avg_area"]), 1) if include_area else None
        parts = [f"Find {_human_flat_type(c['flat_type'])} flat in {_human_town(c['town'])}"]
        if area:
            parts.append(f"around {area} sqm")
        parts.append(f"last {months} months")

        expected_target: dict[str, Any] = {
            "town": c["town"],
            "flat_type": c["flat_type"],
            "months_back": months,
        }
        if area:
            expected_target["floor_area_target"] = area

        cases.append({
            "id": f"sparse_{len(cases) + 1:03d}",
            "category": "sparse",
            "query": ", ".join(parts),
            "expected_target": expected_target,
            # After relaxation the agent should land in 20–200.
            # Min is 20 rather than 30 to allow for genuinely sparse towns.
            "expected_count_range": {"min": 20, "max": 200},
            "checks": ["target_extraction", "retrieval_quality", "planner_decision", "trace_quality"],
        })
        return True

    # Pool 1: short window
    p3 = [c for c in combos if 5 <= c["count_3mo"] < 30 and c["count_12mo"] >= 30]
    rng.shuffle(p3)
    for c in p3:
        _add(c, 3)

    # Pool 2: rare combos
    p_rare = [c for c in combos if 5 < c["count_all"] < 50]
    rng.shuffle(p_rare)
    for c in p_rare:
        _add(c, 12)

    # Pool 3: 6-month window (with optional area filter for variety)
    p6 = [c for c in combos if 5 <= c["count_6mo"] < 30 and c["count_12mo"] >= 30]
    rng.shuffle(p6)
    for c in p6:
        _add(c, 6, include_area=rng.random() < 0.5)

    return cases[:n]


def _broad_cases(combos: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Generate broad (over-results) cases.

    The agent must tighten (add storey, narrow floor area, or shorten time)
    to bring the candidate pool below 200.

    Selection criterion: count_12mo > 300 — well above the 200 cap, so even
    after the planner applies some tightening there is still meaningful data.

    Half the broad cases omit months_back entirely, simulating users who do
    not specify a time window and generate very large initial pools.
    """
    pool = [c for c in combos if c["count_12mo"] > 300]
    rng.shuffle(pool)
    cases: list[dict] = []
    seen: set = set()

    for c in pool:
        if len(cases) >= n:
            break
        key = (c["town"], c["flat_type"])
        if key in seen:
            continue
        seen.add(key)

        include_months = rng.random() < 0.5
        months = 12 if include_months else None
        ft = _human_flat_type(c["flat_type"])
        town = _human_town(c["town"])

        # Vary phrasing so the agent sees slightly different query structures.
        if months:
            query = f"{ft} flat in {town}, last {months} months"
        else:
            query = f"Find me {ft} flats in {town}"

        expected_target: dict[str, Any] = {"town": c["town"], "flat_type": c["flat_type"]}
        if months:
            expected_target["months_back"] = months

        cases.append({
            "id": f"broad_{len(cases) + 1:03d}",
            "category": "broad",
            "query": query,
            "expected_target": expected_target,
            "expected_count_range": {"min": 30, "max": 200},
            "checks": ["target_extraction", "retrieval_quality", "planner_decision", "trace_quality"],
        })

    return cases


def _street_cases(streets: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Generate street-hint cases.

    Each case includes a street name in the query (via "near X"), which must
    trigger the hybrid retrieval path (vector + BM25) in the orchestrator.
    expected_retrieval_mode: hybrid tells the retrieval_mode scorer what to
    check.

    The street pool is pre-sampled from the DB with ORDER BY RANDOM(), so
    shuffling again here is redundant but harmless — it respects --seed
    consistently even if the pool size changes.
    """
    rng.shuffle(streets)
    cases: list[dict] = []
    seen_streets: set = set()

    for s in streets:
        if len(cases) >= n:
            break
        street = s["street_name"]
        # One case per street name — avoid two cases for the same street
        # even if they differ in flat_type.
        if street in seen_streets:
            continue
        seen_streets.add(street)

        ft = _human_flat_type(s["flat_type"])
        town = _human_town(s["town"])
        query = f"{ft} flat near {_human_street(street)} in {town}, last 18 months"

        cases.append({
            "id": f"street_{len(cases) + 1:03d}",
            "category": "street_hint",
            "query": query,
            "expected_target": {
                "town": s["town"],
                "flat_type": s["flat_type"],
                # street_hint must match what the agent actually extracts — the
                # user-typed form from the query, not the DB-abbreviated form.
                # The query is built with _human_street(street) so the agent
                # sees e.g. "Fernvale Rd" and returns that. Retrieval is
                # unaffected: ILIKE %hint% does substring matching.
                "street_hint": _human_street(street),
                "months_back": 18,
            },
            "expected_retrieval_mode": "hybrid",
            # Min is 10 not 30 — street filtering is narrow and a small result
            # set is acceptable here; the retrieval_mode check is the primary signal.
            "expected_count_range": {"min": 10, "max": 200},
            "checks": ["target_extraction", "retrieval_mode", "retrieval_quality"],
        })

    return cases


# ---------------------------------------------------------------------------
# Static cases — human-curated, not derivable from DB.
#
# These categories test agent behaviour that is data-independent:
#   ambiguous   — missing town / flat_type; planner should clarify.
#   edge        — invalid town, contradictory constraints; near-zero results.
#   multi_turn  — 2-turn conversations testing context retention.
#   fallback    — months_back far beyond data range; stresses planner LLM.
#
# Edit the lists below directly when these ground-truth cases need updating.
# ---------------------------------------------------------------------------

_STATIC_AMBIGUOUS: list[dict] = [
    {
        "id": "ambiguous_001", "category": "ambiguous",
        "query": "Find me 4-room flats with budget under 500000",
        # flat_type is inferrable; town is missing → planner must clarify.
        "expected_target": {"flat_type": "4 ROOM", "price_budget_max": 500000},
        "expected_count_range": {"min": 0, "max": 500},
        "checks": ["target_extraction", "planner_decision"],
    },
    {
        "id": "ambiguous_002", "category": "ambiguous",
        "query": "HDB flat near Ang Mo Kio MRT, good lease remaining, last 12 months",
        # No town or flat_type extractable — MRT proximity is not a DB field.
        "expected_target": {"months_back": 12},
        "expected_count_range": {"min": 0, "max": 500},
        "checks": ["planner_decision"],
    },
    {
        "id": "ambiguous_003", "category": "ambiguous",
        "query": "Show me flats with good lease and sea view, mid floor, recent transactions",
        # Sea view and lease quality are not structured fields; no town or flat_type.
        "expected_target": {"storey_preference": "mid"},
        "expected_count_range": {"min": 0, "max": 500},
        "checks": ["planner_decision"],
    },
    {
        "id": "ambiguous_004", "category": "ambiguous",
        "query": "Something in Bishan, around 95 sqm, last 12 months",
        # Town is clear but flat_type is absent → planner must clarify.
        "expected_target": {"town": "BISHAN", "floor_area_target": 95.0, "months_back": 12},
        "expected_count_range": {"min": 0, "max": 500},
        "checks": ["target_extraction", "planner_decision"],
    },
]

_STATIC_EDGE: list[dict] = [
    {
        "id": "edge_001", "category": "edge",
        "query": "Find 4-room flat in Jurong Island, last 12 months",
        # Jurong Island has no residential HDB stock — should return near zero.
        "expected_target": {"flat_type": "4 ROOM", "months_back": 12},
        "expected_count_range": {"min": 0, "max": 5},
        "checks": ["retrieval_quality", "trace_quality"],
    },
    {
        "id": "edge_002", "category": "edge",
        "query": "Looking for 3-room flat around 200 sqm in Toa Payoh",
        # 200 sqm is far above any 3-room floor area — contradictory constraint.
        "expected_target": {"town": "TOA PAYOH", "flat_type": "3 ROOM", "floor_area_target": 200.0},
        "expected_count_range": {"min": 0, "max": 5},
        "checks": ["retrieval_quality", "trace_quality"],
    },
    {
        "id": "edge_003", "category": "edge",
        "query": "5-room flat in Bukit Timah, last 12 months",
        # Bukit Timah has extremely few 5-room transactions in the DB.
        "expected_target": {"town": "BUKIT TIMAH", "flat_type": "5 ROOM", "months_back": 12},
        "expected_count_range": {"min": 0, "max": 40},
        "checks": ["target_extraction", "retrieval_quality", "trace_quality"],
    },
]

_STATIC_MULTI_TURN: list[dict] = [
    {
        "id": "multiturn_001", "category": "multi_turn",
        "turns": [
            {
                "query": "Find 4-room flats in Sengkang, last 12 months",
                "expected_target": {"town": "SENGKANG", "flat_type": "4 ROOM", "months_back": 12},
                "expected_count_range": {"min": 30, "max": 500},
                "checks": ["target_extraction", "retrieval_quality"],
            },
            {
                # Turn 2 adds constraints without restating town/flat_type —
                # tests that the Target Agent preserves prior context.
                "query": "Actually I want mid floor, around 90 sqm",
                "expected_target_delta": {
                    "updated": {"floor_area_target": 90.0, "storey_preference": "mid"},
                    "preserved": ["town", "flat_type", "months_back"],
                },
                "expected_count_range": {"min": 30, "max": 200},
                "checks": ["target_extraction", "retrieval_quality"],
            },
        ],
    },
    {
        "id": "multiturn_002", "category": "multi_turn",
        "turns": [
            {
                "query": "Show me 3-room flats in Toa Payoh, last 12 months",
                "expected_target": {"town": "TOA PAYOH", "flat_type": "3 ROOM", "months_back": 12},
                "expected_count_range": {"min": 30, "max": 500},
                "checks": ["target_extraction", "retrieval_quality"],
            },
            {
                "query": "I prefer high floor please",
                "expected_target_delta": {
                    "updated": {"storey_preference": "high"},
                    "preserved": ["town", "flat_type", "months_back"],
                },
                "expected_count_range": {"min": 30, "max": 300},
                "checks": ["target_extraction", "retrieval_quality"],
            },
        ],
    },
    {
        "id": "multiturn_003", "category": "multi_turn",
        "turns": [
            {
                "query": "5-room flat in Tampines, 120 sqm, last 12 months",
                "expected_target": {
                    "town": "TAMPINES", "flat_type": "5 ROOM",
                    "floor_area_target": 120.0, "months_back": 12,
                },
                "expected_count_range": {"min": 30, "max": 500},
                "checks": ["target_extraction", "retrieval_quality"],
            },
            {
                # Narrowing the time window — tests that floor_area_target is preserved.
                "query": "Narrow it to last 6 months only",
                "expected_target_delta": {
                    "updated": {"months_back": 6},
                    "preserved": ["town", "flat_type", "floor_area_target"],
                },
                "expected_count_range": {"min": 30, "max": 200},
                "checks": ["target_extraction", "retrieval_quality"],
            },
        ],
    },
    {
        "id": "multiturn_004", "category": "multi_turn",
        "turns": [
            {
                "query": "Find 4-room flat in Bishan, last 18 months",
                "expected_target": {"town": "BISHAN", "flat_type": "4 ROOM", "months_back": 18},
                "expected_count_range": {"min": 30, "max": 500},
                "checks": ["target_extraction", "retrieval_quality"],
            },
            {
                # Tests that budget (price_budget_max) is additive, not replacing prior fields.
                "query": "I prefer low floor and my budget is under 600000",
                "expected_target_delta": {
                    "updated": {"storey_preference": "low", "price_budget_max": 600000},
                    "preserved": ["town", "flat_type", "months_back"],
                },
                "expected_count_range": {"min": 30, "max": 300},
                "checks": ["target_extraction", "retrieval_quality"],
            },
        ],
    },
    {
        "id": "multiturn_005", "category": "multi_turn",
        "turns": [
            {
                "query": "Executive flats in Hougang, last 18 months",
                "expected_target": {"town": "HOUGANG", "flat_type": "EXECUTIVE", "months_back": 18},
                "expected_count_range": {"min": 30, "max": 300},
                "checks": ["target_extraction", "retrieval_quality"],
            },
            {
                "query": "Change to mid floor only",
                "expected_target_delta": {
                    "updated": {"storey_preference": "mid"},
                    "preserved": ["town", "flat_type", "months_back"],
                },
                "expected_count_range": {"min": 30, "max": 200},
                "checks": ["target_extraction", "retrieval_quality"],
            },
        ],
    },
    {
        "id": "multiturn_006", "category": "multi_turn",
        "turns": [
            {
                "query": "3-room flat in Queenstown, last 12 months",
                "expected_target": {"town": "QUEENSTOWN", "flat_type": "3 ROOM", "months_back": 12},
                "expected_count_range": {"min": 30, "max": 500},
                "checks": ["target_extraction", "retrieval_quality"],
            },
            {
                "query": "I want something around 65 sqm",
                "expected_target_delta": {
                    "updated": {"floor_area_target": 65.0},
                    "preserved": ["town", "flat_type", "months_back"],
                },
                "expected_count_range": {"min": 30, "max": 300},
                "checks": ["target_extraction", "retrieval_quality"],
            },
        ],
    },
]

_STATIC_FALLBACK: list[dict] = [
    {
        "id": "fallback_001", "category": "fallback_stress",
        "query": "Find 1-room flat in Toa Payoh, last 48 months",
        # 1-room flats are extremely rare in modern HDB stock; months_back (48)
        # exceeds the DB data range (≈20 months), producing 0 initial results
        # and forcing the planner through multiple relax iterations where it may
        # emit malformed JSON — triggering the deterministic fallback path.
        "expected_target": {"town": "TOA PAYOH", "flat_type": "1 ROOM", "months_back": 48},
        "expected_count_range": {"min": 0, "max": 50},
        "checks": ["target_extraction", "planner_decision", "trace_quality"],
    },
    {
        "id": "fallback_002", "category": "fallback_stress",
        "query": "Multi-generation flat in Woodlands, around 90 sqm, last 48 months",
        "expected_target": {
            "town": "WOODLANDS", "flat_type": "MULTI-GENERATION",
            "floor_area_target": 90.0, "months_back": 48,
        },
        "expected_count_range": {"min": 0, "max": 20},
        "checks": ["target_extraction", "planner_decision", "trace_quality"],
    },
    {
        "id": "fallback_003", "category": "fallback_stress",
        "query": "2-room flat in Central Area, last 60 months",
        "expected_target": {"town": "CENTRAL AREA", "flat_type": "2 ROOM", "months_back": 60},
        "expected_count_range": {"min": 0, "max": 30},
        "checks": ["target_extraction", "planner_decision", "trace_quality"],
    },
]


# ---------------------------------------------------------------------------
# YAML writer
# ---------------------------------------------------------------------------

# Canonical category order for the output file — matches the conceptual
# difficulty progression used in the original hand-crafted benchmark.
_CATEGORY_ORDER = [
    "easy", "sparse", "broad", "street_hint",
    "ambiguous", "edge", "multi_turn", "fallback_stress",
]

# Section header comments injected above each category block.
# Tuple: (section title, body text — may contain \n for multi-line comments).
_CATEGORY_HEADERS: dict[str, tuple[str, str]] = {
    "easy": (
        "EASY / HAPPY-PATH CASES",
        "Well-specified queries: town + flat_type + area + time window.\n"
        "# Agent should accept in ≤ 2 planner steps with count in 30–200.",
    ),
    "sparse": (
        "SPARSE / UNDER-RESULTS CASES",
        "Tight constraints that push the initial count below 30.\n"
        "# Agent must relax at least once.",
    ),
    "broad": (
        "BROAD / OVER-RESULTS CASES",
        "Minimal constraints; initial count >> 200.\n"
        "# Agent must tighten (add storey, narrow sqm, or shorten time).",
    ),
    "street_hint": (
        "STREET HINT CASES",
        "Queries containing 'near X' or a specific street name.\n"
        "# Must trigger hybrid retrieval mode.",
    ),
    "ambiguous": (
        "AMBIGUOUS CASES",
        "Queries missing town or flat_type (or both).\n"
        "# Agent should respond with action 'clarify'.",
    ),
    "edge": (
        "EDGE / FAILURE CASES",
        "Invalid town names, contradictory constraints, unrealistic combos.",
    ),
    "multi_turn": (
        "MULTI-TURN CASES",
        "2-turn conversations: turn 2 refines without restating turn 1 context.\n"
        "# Static templates — edit _STATIC_MULTI_TURN in this script to change.",
    ),
    "fallback_stress": (
        "PLANNER FALLBACK STRESS CASES",
        "months_back > 36 + rare flat_type to induce planner non-conformance.\n"
        "# Static templates — edit _STATIC_FALLBACK in this script to change.",
    ),
}


def _write_benchmark(out_path: Path, all_cases: list[dict], db_info: dict[str, Any]) -> None:
    """Write all cases to a YAML file with header and per-category comments.

    yaml.dump() handles individual case serialisation; category section
    headers are injected as raw comment strings between case blocks, since
    PyYAML does not support comments natively.
    """
    by_cat: dict[str, list[dict]] = {}
    for c in all_cases:
        by_cat.setdefault(c.get("category", "unknown"), []).append(c)

    cat_summary = ", ".join(
        f"{len(by_cat[k])} {k}" for k in _CATEGORY_ORDER if k in by_cat
    )

    buf = io.StringIO()
    buf.write("# HDB Compare Agents — Benchmark Dataset\n")
    buf.write(f"# {len(all_cases)} test cases: {cat_summary}\n")
    buf.write(f"# Generated by evals/playbook/prepare_benchmark.py on {date.today()}\n")
    buf.write(
        f"# Data range in DB: {db_info['min_date']} to {db_info['max_date']}"
        f" ({db_info['with_embeddings']:,} rows with embeddings)\n"
    )
    buf.write("# expected_count_range reflects the FINAL count after the planner loop exits\n")

    for cat in _CATEGORY_ORDER:
        cases = by_cat.get(cat, [])
        if not cases:
            continue

        title, desc = _CATEGORY_HEADERS.get(cat, (cat.upper(), ""))
        buf.write(f"\n# {'=' * 60}\n")
        buf.write(f"# {title} ({len(cases)})\n")
        # desc may be multi-line; each line already starts with "# " from the
        # constant definition, so we prefix just the first line with "# ".
        for line in desc.split("\n"):
            buf.write(f"# {line}\n")
        buf.write(f"# {'=' * 60}\n\n")

        for case in cases:
            # dump([case]) produces a YAML list item "- key: value\n  ..."
            # which matches the format of the hand-crafted benchmark.
            yaml_str = yaml.dump(
                [case],
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
            buf.write(yaml_str + "\n")

    out_path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Wrote {len(all_cases)} cases to {out_path}")


# ---------------------------------------------------------------------------
# Verify mode
# ---------------------------------------------------------------------------

def _verify_dataset(dataset_path: Path, engine: sa.Engine) -> None:
    """Check each single-turn case's raw DB count against its expected_count_range.

    This is a staleness check, not an agent run. It queries the DB directly
    with the expected_target constraints and compares the raw count to the
    expected_count_range recorded in the YAML.

    Because expected_count_range reflects the *final* count after the planner
    loop exits (not the initial raw count), the comparison is approximate:
    stale means the data has drifted enough that even the final post-planner
    range is broken. Minor drift (e.g. a few rows) will not be flagged.

    Exits with code 1 if any case is stale, so this can be used in CI.
    """
    cases: list[dict] = yaml.safe_load(dataset_path.read_text())
    print(f"Verifying {len(cases)} cases in {dataset_path} against live DB…\n")

    ok = stale = skipped = 0
    for case in cases:
        case_id = case.get("id", "?")

        # Multi-turn cases have no single expected_target at the top level.
        if "turns" in case:
            skipped += 1
            continue

        count = _count_for_case(engine, case)
        if count is None:
            # Ambiguous cases with no town/flat_type — nothing useful to count.
            skipped += 1
            continue

        lo = case.get("expected_count_range", {}).get("min", 0)
        hi = case.get("expected_count_range", {}).get("max", 999_999)
        tag = "OK   " if lo <= count <= hi else "STALE"
        if tag == "STALE":
            stale += 1
        else:
            ok += 1
        print(f"  {tag} [{case_id:20s}]  raw_count={count:>5}  expected=[{lo}, {hi}]")

    print(f"\n{ok} ok  |  {stale} stale  |  {skipped} skipped (multi-turn / no town+flat_type)")
    if stale:
        sys.exit(1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python evals/playbook/prepare_benchmark.py",
        description="Prepare a fresh hdb_compare_benchmark.yaml from the live DB.",
    )
    p.add_argument(
        "--out", default=str(_DEFAULT_OUT), metavar="PATH",
        help=f"Output YAML path (default: {_DEFAULT_OUT}).",
    )
    p.add_argument("--easy",   type=int, default=8, metavar="N",
                   help="Easy cases to generate (default: 8).")
    p.add_argument("--sparse", type=int, default=5, metavar="N",
                   help="Sparse cases to generate (default: 5).")
    p.add_argument("--broad",  type=int, default=4, metavar="N",
                   help="Broad cases to generate (default: 4).")
    p.add_argument("--street", type=int, default=5, metavar="N",
                   help="Street-hint cases to generate (default: 5).")
    p.add_argument("--seed", type=int, default=None, metavar="INT",
                   help="Random seed for reproducible sampling.")
    p.add_argument(
        "--verify",
        nargs="?",
        const=str(_DEFAULT_OUT),
        metavar="DATASET_PATH",
        help=(
            "Verify an existing YAML against the live DB instead of generating. "
            f"Defaults to {_DEFAULT_OUT} when no path is given."
        ),
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    rng = random.Random(args.seed)
    engine = _engine()

    if args.verify is not None:
        _verify_dataset(Path(args.verify), engine)
        return

    print("Querying DB…")
    info = _db_info(engine)
    combos = _combinations(engine)
    # Fetch a large street pool so the sampler has plenty to choose from;
    # the pool is then sub-sampled down to args.street cases.
    street_pool = _streets(engine, limit=max(args.street * 10, 100))

    print(
        f"  DB: {info['min_date']} to {info['max_date']}, "
        f"{info['with_embeddings']:,} rows with embeddings"
    )
    print(f"  {len(combos)} town/flat_type combinations, {len(street_pool)} candidate streets")

    easy   = _easy_cases(combos, args.easy, rng)
    sparse = _sparse_cases(combos, args.sparse, rng)
    broad  = _broad_cases(combos, args.broad, rng)
    street = _street_cases(street_pool, args.street, rng)

    print(
        f"  Generated: {len(easy)} easy, {len(sparse)} sparse, "
        f"{len(broad)} broad, {len(street)} street_hint"
    )
    # Warn if the DB didn't have enough viable combos to fill the requested count.
    for label, got, want in [
        ("easy", len(easy), args.easy),
        ("sparse", len(sparse), args.sparse),
        ("broad", len(broad), args.broad),
        ("street_hint", len(street), args.street),
    ]:
        if got < want:
            print(f"  Warning: only {got}/{want} {label} cases found — DB may have fewer viable combos.")

    all_cases = (
        easy + sparse + broad + street
        + _STATIC_AMBIGUOUS + _STATIC_EDGE + _STATIC_MULTI_TURN + _STATIC_FALLBACK
    )

    _write_benchmark(Path(args.out), all_cases, info)


if __name__ == "__main__":
    main()
