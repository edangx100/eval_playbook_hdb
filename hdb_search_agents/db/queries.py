from __future__ import annotations

import datetime as dt
import json
from functools import lru_cache
from typing import Any, Mapping

import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from settings import Settings

DEFAULT_COLUMNS = [
    "month",
    "town",
    "flat_type",
    "block",
    "street_name",
    "storey_range",
    "floor_area_sqm",
    "flat_model",
    "lease_commence_date",
    "remaining_lease",
    "resale_price",
    "month_date",
    "storey_min",
    "storey_max",
    "storey_mid",
    "remaining_lease_months",
]


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    # Cache the engine so repeated calls reuse pooled connections.
    settings = Settings()
    return create_engine(settings.database_url)


def _to_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(stripped)
        except ValueError:
            return None
    return None


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _month_cutoff(months_back: int, today: dt.date | None = None) -> dt.date:
    # Convert "last N months" into a first-of-month cutoff date.
    if months_back < 1:
        raise ValueError("months_back must be >= 1")
    if today is None:
        today = dt.date.today()
    month = today.month - (months_back - 1)
    year = today.year
    while month <= 0:
        month += 12
        year -= 1
    return dt.date(year, month, 1)


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _build_filters(filters: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if not filters:
        return "", {}

    # Build SQL fragments with bound params to avoid injection.
    clauses: list[str] = []
    params: dict[str, Any] = {}

    if _has_text(filters.get("town")):
        clauses.append("town = :town")
        params["town"] = filters["town"].strip()

    if _has_text(filters.get("flat_type")):
        clauses.append("flat_type = :flat_type")
        params["flat_type"] = filters["flat_type"].strip()

    months_back = _to_int(filters.get("months_back"))
    if months_back:
        clauses.append("month_date >= :month_cutoff")
        params["month_cutoff"] = _month_cutoff(months_back)

    sqm_min = _to_float(filters.get("sqm_min"))
    if sqm_min is not None:
        clauses.append("floor_area_sqm >= :sqm_min")
        params["sqm_min"] = sqm_min

    sqm_max = _to_float(filters.get("sqm_max"))
    if sqm_max is not None:
        clauses.append("floor_area_sqm <= :sqm_max")
        params["sqm_max"] = sqm_max

    lease_min = _to_int(filters.get("remaining_lease_months_min"))
    if lease_min is not None:
        clauses.append("remaining_lease_months >= :remaining_lease_months_min")
        params["remaining_lease_months_min"] = lease_min

    lease_max = _to_int(filters.get("remaining_lease_months_max"))
    if lease_max is not None:
        clauses.append("remaining_lease_months <= :remaining_lease_months_max")
        params["remaining_lease_months_max"] = lease_max

    storey_min = _to_int(filters.get("storey_min"))
    if storey_min is not None:
        clauses.append("storey_min >= :storey_min")
        params["storey_min"] = storey_min

    storey_max = _to_int(filters.get("storey_max"))
    if storey_max is not None:
        clauses.append("storey_max <= :storey_max")
        params["storey_max"] = storey_max

    resale_price_max = _to_int(filters.get("resale_price_max"))
    if resale_price_max is not None:
        clauses.append("resale_price <= :resale_price_max")
        params["resale_price_max"] = resale_price_max

    if _has_text(filters.get("flat_model")):
        clauses.append("flat_model = :flat_model")
        params["flat_model"] = filters["flat_model"].strip()

    if _has_text(filters.get("block")):
        clauses.append("block = :block")
        params["block"] = filters["block"].strip()

    if _has_text(filters.get("street_name")):
        clauses.append("street_name = :street_name")
        params["street_name"] = filters["street_name"].strip()

    # Street hints use partial matching to catch "Compassvale" vs "Compassvale Cres".
    if _has_text(filters.get("street_hint")):
        clauses.append("street_name ILIKE :street_hint")
        params["street_hint"] = f"%{filters['street_hint'].strip()}%"

    if not clauses:
        return "", params

    return " WHERE " + " AND ".join(clauses), params


def _normalize_model_name(model_name: str) -> str:
    # Allow provider/model identifiers; Jina expects the raw model name.
    if "/" in model_name:
        return model_name.rsplit("/", 1)[-1]
    return model_name


def _serialize_embedding(embedding: list[float]) -> str:
    # pgvector accepts JSON-style list syntax for vector inputs.
    return json.dumps(list(embedding), separators=(",", ":"))


def _embed_query_text(query_text: str, settings: Settings) -> str:
    # Normalize and validate the user query before sending it to the embedding API.
    text_value = query_text.strip()
    if not text_value:
        raise ValueError("query_text must be non-empty")
    if settings.jina_api_key.strip().upper() == "REPLACE_ME":
        raise ValueError("JINA_API_KEY is not set; update .env to enable embeddings.")

    # Use the "query" task so the embedding is optimized for search queries.
    model_name = _normalize_model_name(settings.embedding_model_name)
    payload = {
        "model": model_name,
        "task": "retrieval.query",
        "dimensions": settings.embedding_dim,
        "input": [text_value],
    }
    headers = {
        "Authorization": f"Bearer {settings.jina_api_key}",
        "Content-Type": "application/json",
    }
    url = f"{settings.jina_base_url.rstrip('/')}/embeddings"

    # Call Jina's embeddings endpoint and parse the response JSON.
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
    except httpx.TimeoutException as exc:
        raise RuntimeError("Embedding API request timed out. Please try again.") from exc
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"Embedding API error: HTTP {exc.response.status_code}") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Embedding API connection failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Embedding generation failed: {type(exc).__name__}: {exc}") from exc

    # Validate shape/dimension to catch mismatched model settings early.
    data = result.get("data", [])
    if not data:
        raise ValueError("Embedding response missing data for query text.")
    embedding = data[0].get("embedding")
    if not isinstance(embedding, list):
        raise ValueError("Embedding response missing embedding vector.")
    if len(embedding) != settings.embedding_dim:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"expected {settings.embedding_dim}, got {len(embedding)}"
        )
    # Return a pgvector-compatible literal we can bind as a SQL parameter.
    return _serialize_embedding(embedding)


def fetch_flats(filters: Mapping[str, Any] | None, limit: int = 100) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    where_sql, params = _build_filters(filters)
    # Default ordering favors recent and higher-priced sales.
    sql = (
        f"SELECT {', '.join(DEFAULT_COLUMNS)} FROM hdb_resale"
        f"{where_sql} ORDER BY month_date DESC, resale_price DESC LIMIT :limit"
    )
    params["limit"] = limit
    try:
        with get_engine().connect() as connection:
            result = connection.execute(text(sql), params)
            return [dict(row) for row in result.mappings().all()]
    except Exception as exc:
        raise RuntimeError(f"Database query failed: {type(exc).__name__}: {exc}") from exc


def count_flats(filters: Mapping[str, Any] | None) -> int:
    where_sql, params = _build_filters(filters)
    sql = f"SELECT COUNT(*) AS count FROM hdb_resale{where_sql}"
    try:
        with get_engine().connect() as connection:
            result = connection.execute(text(sql), params)
            return int(result.scalar_one())
    except Exception as exc:
        raise RuntimeError(f"Database query failed: {type(exc).__name__}: {exc}") from exc


def bm25_boost_reason(filters: Mapping[str, Any] | None, query_text: str) -> str | None:
    """
    Return the reason BM25 should be enabled for the given query, or None if not triggered.
    """
    if not _has_text(query_text):
        return None

    # Trigger 1: street_hint filter is present
    if filters and _has_text(filters.get("street_hint")):
        return "street_hint present"

    # Trigger 2: Query contains location-related keywords.
    # Proximity words (near, around, area, nearby, close) are intentionally excluded:
    # they are ambiguous ("around 90 sqm") and are already handled by the
    # street_hint extraction path in the orchestrator when a real place follows.
    location_keywords = [
        "lrt", "mrt", "station", "mall", "market", "park",
        "road", "street", "avenue", "crescent", "drive", "walk", "lane",
    ]

    # Trigger 3: Query contains descriptive/model keywords not in structured fields
    descriptive_keywords = [
        "premium", "executive", "apartment", "maisonette",
        "renovated", "new", "old", "upgraded",
        "spacious", "cozy", "bright", "corner",
    ]

    query_lower = query_text.lower()

    for keyword in location_keywords:
        if keyword in query_lower:
            return f"location keyword: {keyword}"

    for keyword in descriptive_keywords:
        if keyword in query_lower:
            return f"descriptive keyword: {keyword}"

    return None


def should_use_bm25(filters: Mapping[str, Any] | None, query_text: str) -> bool:
    """
    Determine whether to activate BM25 lexical boost based on query characteristics.

    BM25 is triggered when:
    1. street_hint is present (partial address/street name queries)
    2. Query contains location keywords (near, around, area, LRT, MRT, mall, park)
    3. Query contains descriptive keywords (premium, executive, renovated, new, old)

    Args:
        filters: SQL filter constraints
        query_text: User's search query text

    Returns:
        True if BM25 should be used, False for vector-only
    """
    return bm25_boost_reason(filters, query_text) is not None


def hybrid_fetch_candidates(
    filters: Mapping[str, Any] | None,
    query_text: str,
    limit: int = 200,
    use_bm25: bool | None = None,
    *,
    embed_query_text: str | None = None,
    bm25_query_text: str | None = None,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval: apply structured SQL filters for correctness, then rank with vector similarity.

    Optionally adds BM25 lexical boost using Reciprocal Rank Fusion (RRF) to combine vector and BM25 rankings.

    Falls back to structured retrieval (fetch_flats) if hybrid retrieval fails.

    Args:
        filters: SQL filter constraints (town, flat_type, sqm range, etc.)
        query_text: User's search query text (defaults for embeddings and BM25)
        limit: Max number of results to return
        use_bm25: If True, force BM25; if False, force vector-only; if None, auto-detect
        embed_query_text: Optional override for vector embedding query text
        bm25_query_text: Optional override for BM25 query text

    Returns:
        List of candidate rows ranked by relevance
    """
    if limit <= 0:
        return []

    # Allow separate text for semantic embeddings vs lexical BM25.
    vector_query_text = embed_query_text or query_text
    bm25_text = bm25_query_text or query_text

    # Auto-detect BM25 usage if not explicitly specified
    if use_bm25 is None:
        use_bm25 = should_use_bm25(filters, bm25_text)

    try:
        if not use_bm25:
            # Vector-only mode (original behavior)
            return _fetch_vector_only(filters, vector_query_text, limit)
        else:
            # Vector + BM25 hybrid mode with RRF
            return _fetch_vector_bm25_rrf(filters, vector_query_text, bm25_text, limit)
    except RuntimeError as exc:
        # Hybrid retrieval failed - fallback to structured retrieval
        print(f"Warning: Hybrid retrieval failed, falling back to structured retrieval: {exc}", flush=True)
        return fetch_flats(filters, limit)


def _fetch_vector_only(
    filters: Mapping[str, Any] | None,
    query_text: str,
    limit: int,
) -> list[dict[str, Any]]:
    """
    Vector-only retrieval with fetch + Python sort workaround.

    PostgreSQL has a limitation where ORDER BY with vector distance operator
    can return 0 results even when matches exist. Workaround: fetch with
    distance filter, then sort in Python.

    Raises RuntimeError if vector retrieval fails (caller should fallback to structured).
    """
    settings = Settings()

    # Normalize query text for consistent processing
    normalized_query = _normalize_query_text(query_text)

    try:
        query_vec = _embed_query_text(normalized_query, settings)
    except (ValueError, RuntimeError) as exc:
        # Embedding generation failed - raise to trigger fallback
        raise RuntimeError(f"Vector embedding failed: {exc}") from exc

    where_sql, params = _build_filters(filters)
    if where_sql:
        where_sql += " AND listing_embedding IS NOT NULL"
    else:
        where_sql = " WHERE listing_embedding IS NOT NULL"

    # Fetch more rows than needed for better coverage, with distance filter
    # NOTE: Cannot ORDER BY vector distance in SQL (PostgreSQL limitation)
    # Filter by distance < 1.0 and sort in Python instead
    fetch_limit = min(limit * 3, 500)

    sql = text(
        f"SELECT {', '.join(DEFAULT_COLUMNS)}, "
        f"listing_embedding <=> '{query_vec}'::vector AS vector_score "
        f"FROM hdb_resale{where_sql} "
        f"AND listing_embedding <=> '{query_vec}'::vector < 1.0 "
        f"LIMIT :fetch_limit"
    )
    params["fetch_limit"] = fetch_limit

    try:
        with get_engine().connect() as connection:
            result = connection.execute(sql, params)
            rows = [dict(row) for row in result.mappings().all()]

            # Sort by vector distance in Python (ascending = closer match)
            rows.sort(key=lambda row: row.get('vector_score', float('inf')))

            # Remove score column and return top results
            clean_rows = [
                {k: v for k, v in row.items() if k != 'vector_score'}
                for row in rows[:limit]
            ]

            return clean_rows
    except Exception as exc:
        # Database query failed - raise to trigger fallback
        raise RuntimeError(f"Vector retrieval query failed: {type(exc).__name__}: {exc}") from exc


def _normalize_query_text(query_text: str) -> str:
    """
    Normalize query text for consistent processing across vector and BM25.

    Handles:
    - Removes excessive whitespace
    - Preserves original casing (BM25 and vector embeddings handle case internally)

    Returns normalized query string.
    """
    # Strip and collapse whitespace
    normalized = " ".join(query_text.strip().split())
    return normalized


def _normalize_bm25_query(query_text: str) -> str:
    """
    Normalize query text specifically for BM25 queries.

    Removes non-descriptive words that don't appear in listing_text and cause
    poor BM25 matching. These words (like "area", "around") reduce result quality
    because BM25 heavily penalizes documents that don't contain them.

    Returns cleaned query string suitable for BM25.
    """
    # Words to remove that break BM25 matching (don't appear in listing_text)
    stopwords_for_bm25 = {"area", "around", "near", "close", "nearby"}

    # Tokenize, filter, and rejoin
    tokens = query_text.lower().split()
    filtered_tokens = [t for t in tokens if t not in stopwords_for_bm25]

    # Return filtered query, or original if all tokens were filtered
    cleaned = " ".join(filtered_tokens) if filtered_tokens else query_text
    return cleaned


def _fetch_vector_bm25_rrf(
    filters: Mapping[str, Any] | None,
    vector_query_text: str,
    bm25_query_text: str,
    limit: int,
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Hybrid retrieval using Reciprocal Rank Fusion (RRF) to combine vector + BM25 rankings.

    RRF formula: score(doc) = sum(w_i / (k + rank_i)) for each ranking method

    Includes fallback logic:
    - If vector search fails, falls back to BM25-only ranking
    - If BM25 search fails, falls back to vector-only ranking
    - If both fail, raises RuntimeError (caller should fallback to structured)

    Args:
        vector_query_text: Text used to generate the embedding for vector ranking
        bm25_query_text: Text used to build the BM25 lexical query
        k: RRF constant (default 60, standard value from literature)
    """
    settings = Settings()
    # Allow .env to tune relative influence of semantic vs lexical ranking.
    vector_weight = settings.rrf_vector_weight
    bm25_weight = settings.rrf_bm25_weight

    # Normalize vector query text for consistent embedding input
    normalized_query = _normalize_query_text(vector_query_text)

    # Normalize BM25 query separately (remove words that don't appear in listing_text)
    normalized_bm25_query = _normalize_query_text(bm25_query_text)
    bm25_query_text = _normalize_bm25_query(normalized_bm25_query)

    # Try to generate embedding - if this fails, we can still try BM25-only
    query_vec = None
    try:
        query_vec = _embed_query_text(normalized_query, settings)
    except (ValueError, RuntimeError) as exc:
        # Embedding generation failed - will fallback to BM25-only below
        print(f"Warning: Vector embedding failed, will try BM25-only: {exc}", flush=True)

    where_sql, params = _build_filters(filters)

    # Build base WHERE clause for both queries
    if where_sql:
        vector_where = where_sql + " AND listing_embedding IS NOT NULL"
        bm25_where = where_sql + " AND listing_text IS NOT NULL"
    else:
        vector_where = " WHERE listing_embedding IS NOT NULL"
        bm25_where = " WHERE listing_text IS NOT NULL"

    # Fetch top results by vector similarity (multiply limit to get good coverage for RRF)
    fetch_limit = min(limit * 3, 500)

    vector_rows = []
    bm25_rows = []

    try:
        with get_engine().connect() as connection:
            # Query 1: Vector similarity ranking (only if embedding succeeded)
            if query_vec is not None:
                try:
                    vector_sql = text(
                        f"SELECT {', '.join(DEFAULT_COLUMNS)}, "
                        f"listing_embedding <=> '{query_vec}'::vector AS vector_score "
                        f"FROM hdb_resale{vector_where} "
                        f"AND listing_embedding <=> '{query_vec}'::vector < 1.0 "
                        f"LIMIT :fetch_limit"
                    )
                    vector_params = params.copy()
                    vector_params["fetch_limit"] = fetch_limit

                    vector_result = connection.execute(vector_sql, vector_params)
                    vector_rows = [dict(row) for row in vector_result.mappings().all()]

                    # Sort by vector distance in Python (ascending = closer match)
                    vector_rows.sort(key=lambda row: row.get('vector_score', float('inf')))
                except Exception as exc:
                    print(f"Warning: Vector retrieval query failed, will try BM25-only: {exc}", flush=True)

            # Query 2: BM25 lexical ranking
            try:
                bm25_sql = text(
                    f"SELECT {', '.join(DEFAULT_COLUMNS)}, "
                    f"listing_text <@> to_bm25query(:query_text, 'idx_hdb_resale_listing_text_bm25') AS bm25_score "
                    f"FROM hdb_resale{bm25_where} "
                    f"AND (listing_text <@> to_bm25query(:query_text, 'idx_hdb_resale_listing_text_bm25')) < 0 "
                    f"ORDER BY bm25_score ASC "      # Sorts best matches first (more negative = better). key fix to avoid random sampling before LIMIT.
                    f"LIMIT :fetch_limit"
                )
                bm25_params = params.copy()
                bm25_params["query_text"] = bm25_query_text  # Use BM25-specific normalization
                bm25_params["fetch_limit"] = fetch_limit

                bm25_result = connection.execute(bm25_sql, bm25_params)
                bm25_rows = [dict(row) for row in bm25_result.mappings().all()]

                # Sort by BM25 score in Python (ascending = better match) to keep ties stable.
                # pg_textsearch <@> operator returns NEGATIVE scores where:
                # - More negative = better match (documents matching more query terms)
                # - Less negative = worse match (documents matching fewer query terms)
                # Therefore sort ASC (most negative first) to rank best matches first
                bm25_rows.sort(key=lambda row: row.get('bm25_score', 0))
            except Exception as exc:
                print(f"Warning: BM25 retrieval query failed: {exc}", flush=True)
    except Exception as exc:
        # Database connection failed entirely
        raise RuntimeError(f"Database connection failed: {type(exc).__name__}: {exc}") from exc

    # Fallback logic: if one ranking method fails, use the other
    if not vector_rows and not bm25_rows:
        # Both failed - return empty
        return []
    elif not vector_rows:
        # Vector failed, BM25 succeeded - return BM25 results only
        bm25_clean = [
            {k: v for k, v in row.items() if k not in ('vector_score', 'bm25_score')}
            for row in bm25_rows[:limit]
        ]
        return bm25_clean
    elif not bm25_rows:
        # BM25 failed, vector succeeded - return vector results only
        vector_clean = [
            {k: v for k, v in row.items() if k not in ('vector_score', 'bm25_score')}
            for row in vector_rows[:limit]
        ]
        return vector_clean

    # Both succeeded - proceed with RRF merge
    # Build unique key for each row (use multiple fields to ensure uniqueness)
    def row_key(row: dict) -> tuple:
        return (
            row.get("month"),
            row.get("town"),
            row.get("flat_type"),
            row.get("block"),
            row.get("street_name"),
            row.get("storey_range"),
            row.get("resale_price"),
        )

    # Compute RRF scores
    rrf_scores = {}

    # Add vector ranking scores
    for rank, row in enumerate(vector_rows, start=1):
        key = row_key(row)
        # Weighted contribution from vector ranking.
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (vector_weight / (k + rank))

    # Add BM25 ranking scores
    for rank, row in enumerate(bm25_rows, start=1):
        key = row_key(row)
        # Weighted contribution from BM25 ranking.
        rrf_scores[key] = rrf_scores.get(key, 0.0) + (bm25_weight / (k + rank))

    # Collect all unique rows
    all_rows = {}
    for row in vector_rows:
        key = row_key(row)
        if key not in all_rows:
            # Remove the score fields before storing
            row_clean = {k: v for k, v in row.items() if k not in ('vector_score', 'bm25_score')}
            all_rows[key] = row_clean

    for row in bm25_rows:
        key = row_key(row)
        if key not in all_rows:
            row_clean = {k: v for k, v in row.items() if k not in ('vector_score', 'bm25_score')}
            all_rows[key] = row_clean

    # Sort by RRF score (higher is better)
    sorted_keys = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)

    # Return top limit results
    return [all_rows[key] for key in sorted_keys[:limit]]
