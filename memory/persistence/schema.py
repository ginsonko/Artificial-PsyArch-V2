from __future__ import annotations


POSTGRES_SCHEMA_VERSION = "apv21_postgres_memory_schema/v2"


def schema_table_names() -> list[str]:
    return [
        "ap_memory_schema_version",
        "ap_runs",
        "ap_ticks",
        "memory_snapshots",
        "memory_snapshot_items",
        "memory_state_field_items",
        "memory_core_items",
        "memory_posting_tokens",
        "memory_vectors",
        "memory_numeric_features",
        "memory_relation_features",
    "memory_transitions",
    "ap_runtime_state",
    "memory_learning_events",
        "memory_action_feedback_events",
        "memory_asset_refs",
        "memory_index_audit_runs",
        "memory_index_audit_rows",
    ]


def build_postgres_schema_sql(*, vector_dim: int = 64) -> str:
    """
    Return the PostgreSQL-first authoritative memory schema.

    The schema is intentionally append-friendly and white-box: JSONB keeps the
    evolving SA payload inspectable, while token/vector/numeric/relation tables
    expose the derived candidate layers needed for exact audit and rebuild.
    """

    dim = max(16, int(vector_dim))
    return f"""
-- {POSTGRES_SCHEMA_VERSION}
-- APV2.1 authoritative memory schema. Runtime indexes are rebuildable views.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS ap_memory_schema_version (
    schema_id text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now(),
    notes text NOT NULL DEFAULT ''
);

INSERT INTO ap_memory_schema_version(schema_id, notes)
VALUES ('{POSTGRES_SCHEMA_VERSION}', 'PostgreSQL + pgvector authoritative AP memory schema with full-SA state_field_items')
ON CONFLICT (schema_id) DO NOTHING;

CREATE TABLE IF NOT EXISTS ap_runs (
    run_id uuid PRIMARY KEY,
    started_at timestamptz NOT NULL DEFAULT now(),
    run_label text NOT NULL DEFAULT '',
    config jsonb NOT NULL DEFAULT '{{}}'::jsonb
);

CREATE TABLE IF NOT EXISTS ap_ticks (
    run_id uuid NOT NULL REFERENCES ap_runs(run_id) ON DELETE CASCADE,
    tick_index bigint NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    runtime_trace jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    PRIMARY KEY (run_id, tick_index)
);

CREATE TABLE IF NOT EXISTS memory_snapshots (
    run_id uuid NOT NULL REFERENCES ap_runs(run_id) ON DELETE CASCADE,
    memory_id text NOT NULL,
    tick_index bigint NOT NULL,
    memory_kind text NOT NULL,
    source_text text NOT NULL DEFAULT '',
    focus_labels text[] NOT NULL DEFAULT ARRAY[]::text[],
    item_count integer NOT NULL DEFAULT 0,
    state_field_item_count integer NOT NULL DEFAULT 0,
    core_item_count integer NOT NULL DEFAULT 0,
    energy_mass double precision NOT NULL DEFAULT 0,
    snapshot jsonb NOT NULL,
    feature_summary jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, memory_id)
);

ALTER TABLE memory_snapshots
    ADD COLUMN IF NOT EXISTS state_field_item_count integer NOT NULL DEFAULT 0;

CREATE TABLE IF NOT EXISTS memory_snapshot_items (
    run_id uuid NOT NULL,
    memory_id text NOT NULL,
    item_index integer NOT NULL,
    tick_index bigint NOT NULL,
    memory_kind text NOT NULL,
    sa_label text NOT NULL,
    display_text text NOT NULL DEFAULT '',
    family text NOT NULL DEFAULT '',
    source_type text NOT NULL DEFAULT '',
    real_energy double precision NOT NULL DEFAULT 0,
    virtual_energy double precision NOT NULL DEFAULT 0,
    cognitive_pressure double precision NOT NULL DEFAULT 0,
    payload jsonb NOT NULL,
    PRIMARY KEY (run_id, memory_id, item_index),
    FOREIGN KEY (run_id, memory_id) REFERENCES memory_snapshots(run_id, memory_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_state_field_items (
    run_id uuid NOT NULL,
    memory_id text NOT NULL,
    state_field_index integer NOT NULL,
    sa_label text NOT NULL,
    payload jsonb NOT NULL,
    PRIMARY KEY (run_id, memory_id, state_field_index),
    FOREIGN KEY (run_id, memory_id) REFERENCES memory_snapshots(run_id, memory_id) ON DELETE CASCADE
);

-- Legacy/compatibility anchor view. The AP Bn main recognition field is
-- memory_state_field_items; this table keeps old tooling and external-anchor
-- previews stable without redefining "core" as a philosophical exclusion.
CREATE TABLE IF NOT EXISTS memory_core_items (
    run_id uuid NOT NULL,
    memory_id text NOT NULL,
    core_index integer NOT NULL,
    sa_label text NOT NULL,
    payload jsonb NOT NULL,
    PRIMARY KEY (run_id, memory_id, core_index),
    FOREIGN KEY (run_id, memory_id) REFERENCES memory_snapshots(run_id, memory_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_posting_tokens (
    run_id uuid NOT NULL,
    memory_id text NOT NULL,
    memory_kind text NOT NULL,
    token_field text NOT NULL,
    token text NOT NULL,
    weight double precision NOT NULL DEFAULT 1,
    PRIMARY KEY (run_id, memory_id, token_field, token),
    FOREIGN KEY (run_id, memory_id) REFERENCES memory_snapshots(run_id, memory_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_vectors (
    run_id uuid NOT NULL,
    memory_id text NOT NULL,
    memory_kind text NOT NULL,
    vector_space text NOT NULL,
    vector vector({dim}) NOT NULL,
    vector_meta jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    PRIMARY KEY (run_id, memory_id, vector_space),
    FOREIGN KEY (run_id, memory_id) REFERENCES memory_snapshots(run_id, memory_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_numeric_features (
    run_id uuid NOT NULL,
    memory_id text NOT NULL,
    memory_kind text NOT NULL,
    channel text NOT NULL,
    values double precision[] NOT NULL,
    feature_meta jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    PRIMARY KEY (run_id, memory_id, channel),
    FOREIGN KEY (run_id, memory_id) REFERENCES memory_snapshots(run_id, memory_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_relation_features (
    run_id uuid NOT NULL,
    memory_id text NOT NULL,
    memory_kind text NOT NULL,
    relation_token text NOT NULL,
    relation_type text NOT NULL DEFAULT '',
    weight double precision NOT NULL DEFAULT 0,
    payload jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    PRIMARY KEY (run_id, memory_id, relation_token),
    FOREIGN KEY (run_id, memory_id) REFERENCES memory_snapshots(run_id, memory_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_transitions (
    run_id uuid NOT NULL,
    memory_kind text NOT NULL,
    source_memory_id text NOT NULL,
    successor_memory_id text NOT NULL,
    observed_count integer NOT NULL DEFAULT 1,
    last_tick_index bigint NOT NULL DEFAULT 0,
    transition_meta jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    PRIMARY KEY (run_id, memory_kind, source_memory_id, successor_memory_id)
);

CREATE TABLE IF NOT EXISTS ap_runtime_state (
    run_id uuid PRIMARY KEY,
    updated_at timestamptz NOT NULL DEFAULT now(),
    state jsonb NOT NULL DEFAULT '{{}}'::jsonb
);

CREATE TABLE IF NOT EXISTS memory_learning_events (
    run_id uuid NOT NULL,
    event_id text NOT NULL,
    tick_index bigint,
    memory_id text NOT NULL DEFAULT '',
    memory_kind text NOT NULL DEFAULT '',
    event_type text NOT NULL,
    learning_layer text NOT NULL,
    writer text NOT NULL,
    source text NOT NULL DEFAULT '',
    target text NOT NULL DEFAULT '',
    relation text NOT NULL DEFAULT '',
    weight double precision NOT NULL DEFAULT 0,
    payload jsonb NOT NULL,
    PRIMARY KEY (run_id, event_id)
);

CREATE TABLE IF NOT EXISTS memory_action_feedback_events (
    run_id uuid NOT NULL,
    event_id text NOT NULL,
    tick_index bigint NOT NULL,
    action_id text NOT NULL,
    feedback_type text NOT NULL,
    reward_value double precision NOT NULL DEFAULT 0,
    punishment_value double precision NOT NULL DEFAULT 0,
    payload jsonb NOT NULL,
    PRIMARY KEY (run_id, event_id)
);

CREATE TABLE IF NOT EXISTS memory_asset_refs (
    run_id uuid NOT NULL,
    memory_id text NOT NULL,
    asset_id text NOT NULL,
    modality text NOT NULL DEFAULT '',
    uri text NOT NULL DEFAULT '',
    sha256 text NOT NULL DEFAULT '',
    payload jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    PRIMARY KEY (run_id, memory_id, asset_id),
    FOREIGN KEY (run_id, memory_id) REFERENCES memory_snapshots(run_id, memory_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS memory_index_audit_runs (
    audit_id uuid PRIMARY KEY,
    run_id uuid,
    created_at timestamptz NOT NULL DEFAULT now(),
    query_signature text NOT NULL,
    memory_kind text NOT NULL,
    audit_meta jsonb NOT NULL DEFAULT '{{}}'::jsonb
);

CREATE TABLE IF NOT EXISTS memory_index_audit_rows (
    audit_id uuid NOT NULL REFERENCES memory_index_audit_runs(audit_id) ON DELETE CASCADE,
    row_index integer NOT NULL,
    memory_id text NOT NULL,
    source text NOT NULL,
    runtime_rank integer,
    exact_rank integer,
    runtime_score double precision NOT NULL DEFAULT 0,
    exact_score double precision NOT NULL DEFAULT 0,
    payload jsonb NOT NULL DEFAULT '{{}}'::jsonb,
    PRIMARY KEY (audit_id, row_index)
);

CREATE INDEX IF NOT EXISTS idx_memory_snapshots_kind_tick
    ON memory_snapshots(run_id, memory_kind, tick_index DESC);
CREATE INDEX IF NOT EXISTS idx_memory_snapshots_tick_brin
    ON memory_snapshots USING brin(tick_index);
CREATE INDEX IF NOT EXISTS idx_memory_items_label
    ON memory_snapshot_items(run_id, memory_kind, sa_label);
CREATE INDEX IF NOT EXISTS idx_memory_items_payload_gin
    ON memory_snapshot_items USING gin(payload);
CREATE INDEX IF NOT EXISTS idx_memory_posting_lookup
    ON memory_posting_tokens(run_id, memory_kind, token_field, token);
CREATE INDEX IF NOT EXISTS idx_memory_posting_token_gin
    ON memory_posting_tokens USING gin(token gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_memory_transitions_source
    ON memory_transitions(run_id, memory_kind, source_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_learning_tick
    ON memory_learning_events(run_id, tick_index);
CREATE INDEX IF NOT EXISTS idx_memory_action_feedback_action
    ON memory_action_feedback_events(run_id, action_id, feedback_type);

-- Optional after enough vectors exist:
-- CREATE INDEX idx_memory_vectors_hnsw ON memory_vectors
-- USING hnsw (vector vector_cosine_ops)
-- WITH (m = 24, ef_construction = 80);
""".strip()
