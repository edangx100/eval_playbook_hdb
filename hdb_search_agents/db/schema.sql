
-- Task 1.2 Database Schema (No Embeddings Yet)
-- Run:
--   docker compose exec -T postgres psql -U hdb -d hdb < db/schema.sql
-- Verify:
--   docker compose exec -T postgres psql -U hdb -d hdb -c '\d hdb_resale'
--
-- Task 1.3 Basic Data Ingestion (No Embeddings)
-- Test:
--   docker compose exec -T postgres psql -U hdb -d hdb -c "SELECT COUNT(*), MIN(month_date), MAX(month_date) FROM hdb_resale;"
--   docker compose exec -T postgres psql -U hdb -d hdb -c "SELECT storey_range, storey_min, storey_max, storey_mid, remaining_lease, remaining_lease_months FROM hdb_resale LIMIT 5;"
--
-- Task 3.1 Add Vector Columns to Schema
--   docker compose exec -T postgres psql -U hdb -d hdb < db/schema.sql
--   docker compose exec -T postgres psql -U hdb -d hdb -c '\d hdb_resale'
--
-- Task 3.2 Embedding Generation During Ingestion
-- docker compose exec -T postgres psql -U hdb -d hdb -c "SELECT COUNT(*) FILTER (WHERE listing_text IS NOT NULL AND listing_text <> '') AS text_populated, COUNT(*) FILTER (WHERE listing_embedding IS NOT NULL) AS embedding_populated, COUNT(*) AS total_rows FROM hdb_resale;"




-- Embedding vector dimension.
-- Override via:
--   docker compose exec -T postgres psql -U hdb -d hdb -v embedding_dim=1024 < db/schema.sql
--
-- If you want to keep the dimension driven by .env, run schema with:
-- set -a; source .env; set +a
--   docker compose exec -T postgres psql -U hdb -d hdb -v embedding_dim=$EMBEDDING_DIM < db/schema.sql

\set embedding_dim 1024

-- pgvector extension for vector column types + similarity search indexes.
CREATE EXTENSION IF NOT EXISTS vector;

-- pg_textsearch extension for BM25 lexical search and ranking.
CREATE EXTENSION IF NOT EXISTS pg_textsearch;

-- Core resale table (raw + derived fields).
CREATE TABLE IF NOT EXISTS hdb_resale (
    month TEXT NOT NULL,
    town TEXT NOT NULL,
    flat_type TEXT NOT NULL,
    block TEXT NOT NULL,
    street_name TEXT NOT NULL,
    storey_range TEXT NOT NULL,
    floor_area_sqm NUMERIC(10, 2) NOT NULL,
    flat_model TEXT NOT NULL,
    lease_commence_date INTEGER NOT NULL,
    remaining_lease TEXT NOT NULL,
    resale_price INTEGER NOT NULL,
    month_date DATE NOT NULL,
    storey_min INTEGER NOT NULL,
    storey_max INTEGER NOT NULL,
    storey_mid INTEGER NOT NULL,
    remaining_lease_months INTEGER NOT NULL,
    listing_text TEXT,
    listing_embedding VECTOR(:embedding_dim)
);

-- Keep schema idempotent for existing databases.
ALTER TABLE hdb_resale
    ADD COLUMN IF NOT EXISTS listing_text TEXT;
ALTER TABLE hdb_resale
    ADD COLUMN IF NOT EXISTS listing_embedding VECTOR(:embedding_dim);
ALTER TABLE hdb_resale
    ALTER COLUMN listing_embedding TYPE VECTOR(:embedding_dim)
    USING listing_embedding::VECTOR(:embedding_dim);

-- B-tree indexes for common filters.
CREATE INDEX IF NOT EXISTS idx_hdb_resale_town
    ON hdb_resale (town);
CREATE INDEX IF NOT EXISTS idx_hdb_resale_flat_type
    ON hdb_resale (flat_type);
CREATE INDEX IF NOT EXISTS idx_hdb_resale_month_date
    ON hdb_resale (month_date);
CREATE INDEX IF NOT EXISTS idx_hdb_resale_floor_area_sqm
    ON hdb_resale (floor_area_sqm);
CREATE INDEX IF NOT EXISTS idx_hdb_resale_resale_price
    ON hdb_resale (resale_price);

-- HNSW index for fast ANN vector search (cosine similarity).
CREATE INDEX IF NOT EXISTS idx_hdb_resale_listing_embedding_hnsw
    ON hdb_resale USING hnsw (listing_embedding vector_cosine_ops);

-- BM25 index for lexical search and ranking on listing_text.
-- Supports case-insensitive full-text matching for street names, landmarks, and keyword queries.
-- Uses pg_textsearch's bm25 access method with the <@> operator.
-- text_config='english' specifies text search configuration for tokenization.
--
-- Usage example:
--   SELECT *, listing_text <@> to_bm25query('Compassvale', 'idx_hdb_resale_listing_text_bm25') AS score
--   FROM hdb_resale
--   WHERE town = 'SENGKANG'
--   ORDER BY score ASC  -- Lower scores = better match
--   LIMIT 10;
CREATE INDEX IF NOT EXISTS idx_hdb_resale_listing_text_bm25
    ON hdb_resale USING bm25 (listing_text text_bm25_ops)
    WITH (text_config='english');
