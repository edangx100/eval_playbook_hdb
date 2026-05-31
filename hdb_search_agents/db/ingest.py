from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import httpx
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # Allow running the script directly from the repo root or db/ dir.
    sys.path.insert(0, str(PROJECT_ROOT))

from settings import Settings

# Accepts "63 years" or "61 years 04 months" from the CSV.
LEASE_RE = re.compile(r"^(?P<years>\d+)\s+years?(?:\s+(?P<months>\d+)\s+month[s]?)?$")


class JinaEmbeddingsClient:
    """Client for Jina embeddings API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        dimensions: int,
        timeout: float = 30.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.dimensions = dimensions
        self.client = httpx.Client(timeout=timeout)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def embed_passages(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        # Batch requests to stay under API payload limits and avoid timeouts.
        embeddings: list[list[float]] = []
        if not texts:
            return embeddings

        with tqdm(total=len(texts), desc="Embedding passages", unit="text") as progress:
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                payload = {
                    "model": self.model_name,
                    "task": "retrieval.passage",
                    "dimensions": self.dimensions,
                    "input": batch,
                }
                for attempt in range(8):
                    response = self.client.post(
                        f"{self.base_url}/embeddings",
                        headers=self.headers,
                        json=payload,
                    )
                    if response.status_code == 429:
                        wait = min(2 ** attempt * 5, 120)
                        tqdm.write(f"Rate limited (429). Waiting {wait}s before retry {attempt + 1}/8...")
                        time.sleep(wait)
                        continue
                    response.raise_for_status()
                    break
                else:
                    response.raise_for_status()

                result = response.json()
                batch_embeddings = [item["embedding"] for item in result.get("data", [])]
                # Defensive checks to catch API/model configuration mismatches early.
                if len(batch_embeddings) != len(batch):
                    raise ValueError(
                        "Embedding count mismatch: "
                        f"expected {len(batch)}, got {len(batch_embeddings)}"
                    )
                if batch_embeddings and len(batch_embeddings[0]) != self.dimensions:
                    raise ValueError(
                        "Embedding dimension mismatch: "
                        f"expected {self.dimensions}, got {len(batch_embeddings[0])}"
                    )
                embeddings.extend(batch_embeddings)
                progress.update(len(batch))

        return embeddings

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "JinaEmbeddingsClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def _normalize_model_name(model_name: str) -> str:
    # Allow provider/model identifiers; Jina expects the raw model name.
    if "/" in model_name:
        return model_name.rsplit("/", 1)[-1]
    return model_name


def _serialize_embedding(embedding: Iterable[float]) -> str:
    # pgvector accepts JSON-style list syntax for vector inputs.
    return json.dumps(list(embedding), separators=(",", ":"))


def parse_remaining_lease(value: str) -> int:
    # Convert remaining lease to total months for filtering/scoring.
    match = LEASE_RE.match(value.strip())
    if not match:
        raise ValueError(f"Unrecognized remaining_lease format: {value}")
    years = int(match.group("years"))
    months = int(match.group("months") or 0)
    return years * 12 + months


def month_cutoff(years: int) -> dt.date:
    # Compute the first day of the cutoff month N years back.
    today = dt.date.today()
    return dt.date(today.year - years, today.month, 1)


def build_listing_text(df: pd.DataFrame) -> pd.Series:
    # Assemble a deterministic, human-readable string for embeddings.
    floor_area = df["floor_area_sqm"].map(
        lambda value: "" if pd.isna(value) else f"{value:g}"
    )
    text = (
        df["town"].fillna("").str.strip()
        + " "
        + df["flat_type"].fillna("").str.strip()
        + " "
        + df["block"].fillna("").str.strip()
        + " "
        + df["street_name"].fillna("").str.strip()
        + " "
        + df["flat_model"].fillna("").str.strip()
        + " storey "
        + df["storey_range"].fillna("").str.strip()
        + " "
        + floor_area
        + "sqm lease "
        + df["remaining_lease"].fillna("").str.strip()
    )
    return text.str.replace(r"\s+", " ", regex=True).str.strip()


def transform_dataframe(df: pd.DataFrame, years: int) -> pd.DataFrame:
    df = df.copy()
    # Normalize month to a first-of-month date for range filtering.
    df["month_date"] = pd.to_datetime(df["month"] + "-01", format="%Y-%m-%d").dt.date

    cutoff = month_cutoff(years)
    df = df[df["month_date"] >= cutoff]

    # Extract storey range like "01 TO 03" into min/max/mid integers.
    storey_parts = df["storey_range"].str.extract(r"(?P<min>\d+)\s+TO\s+(?P<max>\d+)")
    missing_storey = storey_parts["min"].isna() | storey_parts["max"].isna()
    if missing_storey.any():
        bad_values = df.loc[missing_storey, "storey_range"].head(5).tolist()
        print(
            "Dropping rows with unrecognized storey_range values:",
            bad_values,
        )
        df = df.loc[~missing_storey].copy()
        storey_parts = storey_parts.loc[~missing_storey]

    df["storey_min"] = storey_parts["min"].astype(int)
    df["storey_max"] = storey_parts["max"].astype(int)
    df["storey_mid"] = ((df["storey_min"] + df["storey_max"]) / 2).astype(int)

    df["remaining_lease_months"] = df["remaining_lease"].apply(parse_remaining_lease)
    df["listing_text"] = build_listing_text(df)

    ordered_columns = [
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
        "listing_text",
    ]
    return df[ordered_columns]


def generate_embeddings(
    texts: list[str], settings: Settings, batch_size: int
) -> list[str]:
    # Keep the ingest run deterministic by failing fast when config is missing.
    if batch_size <= 0:
        raise ValueError("embed_batch_size must be > 0")
    if settings.jina_api_key.strip().upper() == "REPLACE_ME":
        raise ValueError("JINA_API_KEY is not set; update .env or use --skip-embeddings.")

    model_name = _normalize_model_name(settings.embedding_model_name)
    with JinaEmbeddingsClient(
        api_key=settings.jina_api_key,
        base_url=settings.jina_base_url,
        model_name=model_name,
        dimensions=settings.embedding_dim,
    ) as client:
        embeddings = client.embed_passages(texts, batch_size=batch_size)

    # Serialize as pgvector-compatible strings for SQLAlchemy inserts.
    if len(embeddings) != len(texts):
        raise ValueError(
            f"Expected {len(texts)} embeddings, received {len(embeddings)}"
        )
    return [_serialize_embedding(embedding) for embedding in embeddings]


def ingest_csv(
    csv_path: Path,
    years: int,
    mode: str,
    chunksize: int,
    embed_batch_size: int,
    skip_embeddings: bool,
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    dtype = {
        "month": "string",
        "town": "string",
        "flat_type": "string",
        "block": "string",
        "street_name": "string",
        "storey_range": "string",
        "floor_area_sqm": "float",
        "flat_model": "string",
        "lease_commence_date": "int",
        "remaining_lease": "string",
        "resale_price": "string",
    }

    # Load CSV with explicit types to avoid mixed-type surprises.
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path, dtype=dtype)
    print(f"Rows in CSV: {len(df):,}")

    # Normalize resale_price; some rows include decimal cents.
    df["resale_price"] = pd.to_numeric(df["resale_price"], errors="raise").round(0).astype(int)

    # Apply derived columns + time window filter before loading into Postgres.
    df = transform_dataframe(df, years)
    print(f"Rows after {years}-year filter: {len(df):,}")

    # Settings provide API/database config for embeddings + ingestion.
    settings = Settings()
    if skip_embeddings:
        # Maintain schema compatibility while allowing offline ingests.
        df["listing_embedding"] = None
        print("Skipping embedding generation.")
    else:
        print(
            f"Generating embeddings with {settings.embedding_model_name} "
            f"({settings.embedding_dim} dims)."
        )
        df["listing_embedding"] = generate_embeddings(
            df["listing_text"].tolist(),
            settings,
            embed_batch_size,
        )

    # Use a transaction so truncate + insert are applied atomically.
    engine = create_engine(settings.database_url)

    with engine.begin() as connection:
        print(f"Ingesting data into hdb_resale table (mode={mode})...")
        if mode == "truncate":
            # Truncate wipes existing rows, then reloads fresh data.
            connection.execute(text("TRUNCATE TABLE hdb_resale"))
        # Append skips truncation and adds rows onto the existing table.
        df.to_sql(
            "hdb_resale",
            connection,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=chunksize,
        )

    print("Ingestion complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest HDB resale CSV into Postgres.")
    # Defaults are driven by .env for repeatable local runs.
    settings = Settings()
    default_years = settings.ingest_years
    default_csv_path = Path(settings.ingest_csv_path)
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=default_csv_path,
        help="Path to the HDB resale CSV file.",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=default_years,
        help="Number of years of data to keep (from today).",
    )
    parser.add_argument(
        "--mode",
        choices=("truncate", "append"),
        default="truncate",
        help="Whether to truncate the table before ingesting.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=10000,
        help="Chunk size for pandas to_sql inserts.",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=100,
        help="Batch size for embeddings API calls.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (listing_embedding stays NULL).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest_csv(
        args.csv_path,
        args.years,
        args.mode,
        args.chunksize,
        args.embed_batch_size,
        args.skip_embeddings,
    )


if __name__ == "__main__":
    main()
