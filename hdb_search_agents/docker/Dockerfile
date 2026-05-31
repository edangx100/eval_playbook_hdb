ARG PG_MAJOR=17
ARG PG_TEXTSEARCH_VERSION=0.2.0
ARG PGVECTOR_VERSION=0.7.4

FROM postgres:${PG_MAJOR}

ARG PG_MAJOR=17
ARG PG_TEXTSEARCH_VERSION=0.2.0
ARG PGVECTOR_VERSION=0.7.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        build-essential \
        postgresql-server-dev-${PG_MAJOR} \
    && rm -rf /var/lib/apt/lists/*

# Install pg_textsearch from Timescale's prebuilt release bundle.
RUN set -eu; \
    tmp_dir="/tmp/pg_textsearch"; \
    mkdir -p "$tmp_dir"; \
    curl -fsSL \
      "https://github.com/timescale/pg_textsearch/releases/download/v${PG_TEXTSEARCH_VERSION}/pg_textsearch-pg${PG_MAJOR}-v${PG_TEXTSEARCH_VERSION}.tar.gz" \
      -o /tmp/pg_textsearch.tar.gz; \
    tar -xzf /tmp/pg_textsearch.tar.gz -C "$tmp_dir"; \
    install -m 0755 "$tmp_dir/pg${PG_MAJOR}/pg_textsearch.so" "/usr/lib/postgresql/${PG_MAJOR}/lib/pg_textsearch.so"; \
    install -m 0644 "$tmp_dir/pg${PG_MAJOR}/pg_textsearch.control" "/usr/share/postgresql/${PG_MAJOR}/extension/pg_textsearch.control"; \
    install -m 0644 "$tmp_dir/pg${PG_MAJOR}/pg_textsearch--"*.sql "/usr/share/postgresql/${PG_MAJOR}/extension/"; \
    rm -rf "$tmp_dir" /tmp/pg_textsearch.tar.gz

# Build and install pgvector from source, pinned to a release tag.
RUN set -eu; \
    curl -fsSL \
      "https://github.com/pgvector/pgvector/archive/refs/tags/v${PGVECTOR_VERSION}.tar.gz" \
      -o /tmp/pgvector.tar.gz; \
    tar -xzf /tmp/pgvector.tar.gz -C /tmp; \
    cd "/tmp/pgvector-${PGVECTOR_VERSION}"; \
    make; \
    make install; \
    cd /; \
    rm -rf "/tmp/pgvector-${PGVECTOR_VERSION}" /tmp/pgvector.tar.gz
