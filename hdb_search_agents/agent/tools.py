from __future__ import annotations

from typing import Any, Mapping

from sqlalchemy import text

from hdb_search_agents.agent.models import Stats
from hdb_search_agents.db.queries import (
    _build_filters,
    count_flats,
    fetch_flats,
    get_engine,
    hybrid_fetch_candidates as _hybrid_fetch_candidates,
)


def count_candidates(filters: Mapping[str, Any] | None) -> int:
    """Count candidate rows using the shared query filters."""
    return count_flats(filters)


def fetch_candidates(
    filters: Mapping[str, Any] | None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Fetch candidate rows for reranking or summarization."""
    return fetch_flats(filters, limit=limit)


def price_stats(filters: Mapping[str, Any] | None) -> Stats:
    """Compute price percentiles and summary stats in SQL."""
    # Reuse the same filter builder as the query layer to stay consistent.
    where_sql, params = _build_filters(filters)
    # Let Postgres compute percentiles to avoid pulling all prices into Python.
    sql = (
        "SELECT "
        "COUNT(*) AS count, "
        "percentile_cont(0.5) WITHIN GROUP (ORDER BY resale_price) AS median, "
        "percentile_cont(0.25) WITHIN GROUP (ORDER BY resale_price) AS p25, "
        "percentile_cont(0.75) WITHIN GROUP (ORDER BY resale_price) AS p75, "
        "MIN(resale_price) AS min, "
        "MAX(resale_price) AS max "
        "FROM hdb_resale"
        f"{where_sql}"
    )

    with get_engine().connect() as connection:
        row = connection.execute(text(sql), params).mappings().one()

    count_value = int(row.get("count") or 0)
    if count_value == 0:
        # Return a zeroed Stats payload when the filter set is empty.
        return Stats(median=0, p25=0, p75=0, min=0, max=0, count=0)

    def _to_float(value: Any) -> float:
        # Normalize NULLs from SQL into a numeric value for Stats.
        return 0.0 if value is None else float(value)

    return Stats(
        median=_to_float(row.get("median")),
        p25=_to_float(row.get("p25")),
        p75=_to_float(row.get("p75")),
        min=_to_float(row.get("min")),
        max=_to_float(row.get("max")),
        count=count_value,
    )


def hybrid_fetch_candidates(
    filters: Mapping[str, Any] | None,
    query_text: str,
    limit: int = 200,
    use_bm25: bool | None = None,
    *,
    embed_query_text: str | None = None,
    bm25_query_text: str | None = None,
) -> list[dict[str, Any]]:
    """Fetch candidates using hybrid retrieval (vector similarity + hard filters)."""
    return _hybrid_fetch_candidates(
        filters,
        query_text,
        limit=limit,
        use_bm25=use_bm25,
        embed_query_text=embed_query_text,
        bm25_query_text=bm25_query_text,
    )
