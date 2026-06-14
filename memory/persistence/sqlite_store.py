from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from .base import PersistenceWriteResult


SQLITE_SCHEMA_VERSION = "apv21_sqlite_memory_schema/v1"


def _json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads(payload: object) -> dict:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", errors="replace")
    if isinstance(payload, str):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return dict(value) if isinstance(value, dict) else {}
    return {}


def _loads_list(payload: object) -> list:
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8", errors="replace")
    if isinstance(payload, str):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            return []
        return list(value) if isinstance(value, list) else []
    return list(payload) if isinstance(payload, list) else []


def _json_bytes(payload: object) -> bytes:
    return _json(payload).encode("utf-8")


def _loads_blob(blob: object, codec: object) -> dict:
    if blob in (None, b"", ""):
        return {}
    data = bytes(blob) if isinstance(blob, (bytes, bytearray, memoryview)) else bytes(str(blob), "utf-8")
    clean_codec = str(codec or "").strip().lower()
    try:
        if clean_codec == "zlib":
            data = zlib.decompress(data)
        elif clean_codec in {"", "none", "plain"}:
            pass
        else:
            return {}
    except zlib.error:
        return {}
    return _loads(data)


@dataclass
class SqlitePersistenceConfig:
    """
    Embedded local persistence for desktop/product-shell deployments.

    SQLite is the default small-user backend: no Docker, no service process, and
    no credentials. Runtime indexes are still rebuilt by MemoryStore; this file
    only stores authoritative white-box snapshots and derived audit payloads.
    """

    path: str | Path
    run_id: str = "apv21_local_default"
    run_label: str = ""
    vector_dim: int = 64
    resident_hot_snapshots_per_kind: int = 2048
    warm_prefetch_limit: int = 512
    wal_enabled: bool = True
    synchronous: str = "NORMAL"
    busy_timeout_ms: int = 5000
    store_expanded_item_rows: bool = False
    store_derived_index_rows: bool = False
    full_fidelity_snapshot_blob: bool = True
    compressed_snapshot_blob: bool = True
    store_feature_payload_blob: bool = False
    store_vector_blob: bool = True
    vector_json_preview_only: bool = True
    vector_blob_dtype: str = "float16"
    runtime_projection_snapshot_blob: bool = True
    snapshot_compression_level: int = 6
    legacy_json_preview_only: bool = True
    compact_posting_tokens_per_snapshot: int = 18
    store_posting_token_rows: bool = False
    buffered_writes: bool = True
    buffered_flush_limit: int = 1024
    memory_db_budget_bytes: int = 10 * 1024 * 1024 * 1024
    forgetting_enabled: bool = True
    retention_maintenance_interval_writes: int = 512
    retention_prune_batch: int = 256
    hot_layer_snapshots_per_kind: int = 5120
    warm_layer_snapshots_per_kind: int = 32768
    config: dict = field(default_factory=dict)


class SqliteMemoryPersistence:
    backend_name = "sqlite_local"

    def __init__(self, config: SqlitePersistenceConfig) -> None:
        self.config = config
        self.path = Path(config.path)
        self.run_id = str(config.run_id or "apv21_local_default")
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._write_count = 0
        self._error_count = 0
        self._last_error = ""
        self._summary_size_cache_at = 0.0
        self._summary_size_cache: dict[str, int] = {}
        self._pending_records: list[dict] = []
        self._pending_runtime_state: dict | None = None
        self._pending_runtime_state_supplier: dict | None = None
        self._schema_ready = False

    def __getstate__(self) -> dict:
        state = dict(self.__dict__)
        state["_conn"] = None
        state["_lock"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._conn = None
        self._lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is not None:
                return self._conn
            self.path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.path), timeout=max(0.1, self.config.busy_timeout_ms / 1000.0), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute(f"PRAGMA busy_timeout={max(1, int(self.config.busy_timeout_ms))}")
            if self.config.wal_enabled:
                conn.execute("PRAGMA journal_mode=WAL")
            sync = str(self.config.synchronous or "NORMAL").upper()
            if sync not in {"OFF", "NORMAL", "FULL", "EXTRA"}:
                sync = "NORMAL"
            conn.execute(f"PRAGMA synchronous={sync}")
            self._conn = conn
            return conn

    def ensure_schema(self) -> None:
        with self._lock:
            if self._schema_ready:
                return
            conn = self.connect()
            conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS ap_memory_schema_version (
                    schema_id TEXT PRIMARY KEY,
                    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT NOT NULL DEFAULT ''
                );
                INSERT OR IGNORE INTO ap_memory_schema_version(schema_id, notes)
                VALUES ('{SQLITE_SCHEMA_VERSION}', 'SQLite embedded AP memory schema with hash and online learned vectors');

                CREATE TABLE IF NOT EXISTS ap_runs (
                    run_id TEXT PRIMARY KEY,
                    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    run_label TEXT NOT NULL DEFAULT '',
                    config_json TEXT NOT NULL DEFAULT '{{}}'
                );

                CREATE TABLE IF NOT EXISTS ap_ticks (
                    run_id TEXT NOT NULL,
                    tick_index INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    runtime_trace_json TEXT NOT NULL DEFAULT '{{}}',
                    PRIMARY KEY (run_id, tick_index)
                );

                CREATE TABLE IF NOT EXISTS ap_runtime_state (
                    run_id TEXT PRIMARY KEY,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    state_json TEXT NOT NULL DEFAULT '{{}}',
                    state_blob BLOB NOT NULL DEFAULT X'',
                    state_codec TEXT NOT NULL DEFAULT '',
                    state_raw_bytes INTEGER NOT NULL DEFAULT 0,
                    state_stored_bytes INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS memory_snapshots (
                    run_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    tick_index INTEGER NOT NULL,
                    memory_kind TEXT NOT NULL,
                    source_text TEXT NOT NULL DEFAULT '',
                    focus_labels_json TEXT NOT NULL DEFAULT '[]',
                    item_count INTEGER NOT NULL DEFAULT 0,
                    state_field_item_count INTEGER NOT NULL DEFAULT 0,
                    core_item_count INTEGER NOT NULL DEFAULT 0,
                    energy_mass REAL NOT NULL DEFAULT 0,
                    snapshot_json TEXT NOT NULL,
                    feature_summary_json TEXT NOT NULL DEFAULT '{{}}',
                    snapshot_blob BLOB NOT NULL DEFAULT X'',
                    snapshot_codec TEXT NOT NULL DEFAULT '',
                    snapshot_raw_bytes INTEGER NOT NULL DEFAULT 0,
                    snapshot_stored_bytes INTEGER NOT NULL DEFAULT 0,
                    feature_summary_blob BLOB NOT NULL DEFAULT X'',
                    feature_summary_codec TEXT NOT NULL DEFAULT '',
                    feature_summary_raw_bytes INTEGER NOT NULL DEFAULT 0,
                    feature_summary_stored_bytes INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, memory_id)
                );

                CREATE TABLE IF NOT EXISTS memory_snapshot_items (
                    run_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    item_index INTEGER NOT NULL,
                    tick_index INTEGER NOT NULL,
                    memory_kind TEXT NOT NULL,
                    sa_label TEXT NOT NULL,
                    display_text TEXT NOT NULL DEFAULT '',
                    family TEXT NOT NULL DEFAULT '',
                    source_type TEXT NOT NULL DEFAULT '',
                    real_energy REAL NOT NULL DEFAULT 0,
                    virtual_energy REAL NOT NULL DEFAULT 0,
                    cognitive_pressure REAL NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, memory_id, item_index)
                );

                CREATE TABLE IF NOT EXISTS memory_state_field_items (
                    run_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    state_field_index INTEGER NOT NULL,
                    sa_label TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, memory_id, state_field_index)
                );

                CREATE TABLE IF NOT EXISTS memory_core_items (
                    run_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    core_index INTEGER NOT NULL,
                    sa_label TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, memory_id, core_index)
                );

                CREATE TABLE IF NOT EXISTS memory_posting_tokens (
                    run_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    memory_kind TEXT NOT NULL,
                    token_field TEXT NOT NULL,
                    token TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1,
                    PRIMARY KEY (run_id, memory_id, token_field, token)
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memory_peak_fts USING fts5(
                    run_id UNINDEXED,
                    memory_id UNINDEXED,
                    memory_kind UNINDEXED,
                    peak_tokens,
                    tokenize='unicode61'
                );

                CREATE TABLE IF NOT EXISTS memory_vectors (
                    run_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    memory_kind TEXT NOT NULL,
                    vector_space TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    vector_meta_json TEXT NOT NULL DEFAULT '{{}}',
                    vector_blob BLOB NOT NULL DEFAULT X'',
                    vector_codec TEXT NOT NULL DEFAULT '',
                    vector_dim INTEGER NOT NULL DEFAULT 0,
                    vector_raw_bytes INTEGER NOT NULL DEFAULT 0,
                    vector_stored_bytes INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (run_id, memory_id, vector_space)
                );

                CREATE TABLE IF NOT EXISTS memory_numeric_features (
                    run_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    memory_kind TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    values_json TEXT NOT NULL,
                    feature_meta_json TEXT NOT NULL DEFAULT '{{}}',
                    PRIMARY KEY (run_id, memory_id, channel)
                );

                CREATE TABLE IF NOT EXISTS memory_relation_features (
                    run_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    memory_kind TEXT NOT NULL,
                    relation_token TEXT NOT NULL,
                    relation_type TEXT NOT NULL DEFAULT '',
                    weight REAL NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL DEFAULT '{{}}',
                    PRIMARY KEY (run_id, memory_id, relation_token)
                );

                CREATE TABLE IF NOT EXISTS memory_transitions (
                    run_id TEXT NOT NULL,
                    memory_kind TEXT NOT NULL,
                    source_memory_id TEXT NOT NULL,
                    successor_memory_id TEXT NOT NULL,
                    observed_count INTEGER NOT NULL DEFAULT 1,
                    last_tick_index INTEGER NOT NULL DEFAULT 0,
                    transition_meta_json TEXT NOT NULL DEFAULT '{{}}',
                    PRIMARY KEY (run_id, memory_kind, source_memory_id, successor_memory_id)
                );

                CREATE TABLE IF NOT EXISTS memory_asset_refs (
                    run_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    asset_id TEXT NOT NULL,
                    modality TEXT NOT NULL DEFAULT '',
                    uri TEXT NOT NULL DEFAULT '',
                    sha256 TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{{}}',
                    PRIMARY KEY (run_id, memory_id, asset_id)
                );

                CREATE TABLE IF NOT EXISTS memory_retention (
                    run_id TEXT NOT NULL,
                    memory_id TEXT NOT NULL,
                    memory_kind TEXT NOT NULL,
                    memory_tier TEXT NOT NULL DEFAULT 'hot',
                    importance_score REAL NOT NULL DEFAULT 0,
                    forget_protected INTEGER NOT NULL DEFAULT 0,
                    retention_reason TEXT NOT NULL DEFAULT '',
                    source_domain TEXT NOT NULL DEFAULT '',
                    last_recalled_at REAL NOT NULL DEFAULT 0,
                    recall_count INTEGER NOT NULL DEFAULT 0,
                    last_updated_at REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (run_id, memory_id)
                );

                CREATE INDEX IF NOT EXISTS idx_sqlite_memory_snapshots_kind_tick
                    ON memory_snapshots(run_id, memory_kind, tick_index DESC, memory_id DESC);
                CREATE INDEX IF NOT EXISTS idx_sqlite_memory_items_label
                    ON memory_snapshot_items(run_id, memory_kind, sa_label);
                CREATE INDEX IF NOT EXISTS idx_sqlite_memory_posting_lookup
                    ON memory_posting_tokens(run_id, memory_kind, token_field, token);
                CREATE INDEX IF NOT EXISTS idx_sqlite_memory_vectors_space
                    ON memory_vectors(run_id, memory_kind, vector_space);
                CREATE INDEX IF NOT EXISTS idx_sqlite_memory_retention_tier
                    ON memory_retention(run_id, memory_kind, memory_tier, importance_score DESC, last_recalled_at DESC);
                CREATE INDEX IF NOT EXISTS idx_sqlite_memory_retention_prune
                    ON memory_retention(run_id, forget_protected, memory_tier, importance_score ASC, last_recalled_at ASC);
                """
            )
            self._ensure_column("memory_snapshots", "snapshot_blob", "BLOB NOT NULL DEFAULT X''")
            self._ensure_column("memory_snapshots", "snapshot_codec", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("memory_snapshots", "snapshot_raw_bytes", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("memory_snapshots", "snapshot_stored_bytes", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("memory_snapshots", "feature_summary_blob", "BLOB NOT NULL DEFAULT X''")
            self._ensure_column("memory_snapshots", "feature_summary_codec", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("memory_snapshots", "feature_summary_raw_bytes", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("memory_snapshots", "feature_summary_stored_bytes", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("memory_vectors", "vector_blob", "BLOB NOT NULL DEFAULT X''")
            self._ensure_column("memory_vectors", "vector_codec", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("memory_vectors", "vector_dim", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("memory_vectors", "vector_raw_bytes", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("memory_vectors", "vector_stored_bytes", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("memory_retention", "memory_tier", "TEXT NOT NULL DEFAULT 'hot'")
            self._ensure_column("memory_retention", "importance_score", "REAL NOT NULL DEFAULT 0")
            self._ensure_column("memory_retention", "forget_protected", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("memory_retention", "retention_reason", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("memory_retention", "source_domain", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column("memory_retention", "last_recalled_at", "REAL NOT NULL DEFAULT 0")
            self._ensure_column("memory_retention", "recall_count", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column("memory_retention", "last_updated_at", "REAL NOT NULL DEFAULT 0")
            conn.execute(
                """
                INSERT OR IGNORE INTO ap_runs(run_id, run_label, config_json)
                VALUES (?, ?, ?)
                """,
                (
                    self.run_id,
                    str(self.config.run_label or ""),
                    _json(
                        {
                            "schema": "apv21_sqlite_persistence_config/v1",
                            "resident_hot_snapshots_per_kind": int(self.config.resident_hot_snapshots_per_kind),
                            "warm_prefetch_limit": int(self.config.warm_prefetch_limit),
                            "wal_enabled": bool(self.config.wal_enabled),
                            "synchronous": str(self.config.synchronous or "NORMAL"),
                            "store_expanded_item_rows": bool(self.config.store_expanded_item_rows),
                            "store_derived_index_rows": bool(self.config.store_derived_index_rows),
                            "full_fidelity_snapshot_blob": bool(self.config.full_fidelity_snapshot_blob),
                            "compressed_snapshot_blob": bool(self.config.compressed_snapshot_blob),
                            "store_feature_payload_blob": bool(self.config.store_feature_payload_blob),
                            "store_vector_blob": bool(self.config.store_vector_blob),
                            "vector_json_preview_only": bool(self.config.vector_json_preview_only),
                            "vector_blob_dtype": str(self.config.vector_blob_dtype or ""),
                            "runtime_projection_snapshot_blob": bool(self.config.runtime_projection_snapshot_blob),
                            "snapshot_compression_level": int(self.config.snapshot_compression_level),
                            "legacy_json_preview_only": bool(self.config.legacy_json_preview_only),
                            "compact_posting_tokens_per_snapshot": int(self.config.compact_posting_tokens_per_snapshot),
                            "store_posting_token_rows": bool(self.config.store_posting_token_rows),
                            "buffered_writes": bool(self.config.buffered_writes),
                            "memory_db_budget_bytes": int(self.config.memory_db_budget_bytes),
                            "forgetting_enabled": bool(self.config.forgetting_enabled),
                            "retention_maintenance_interval_writes": int(self.config.retention_maintenance_interval_writes),
                            "retention_prune_batch": int(self.config.retention_prune_batch),
                            "hot_layer_snapshots_per_kind": int(self.config.hot_layer_snapshots_per_kind),
                            "warm_layer_snapshots_per_kind": int(self.config.warm_layer_snapshots_per_kind),
                            **dict(self.config.config or {}),
                        }
                    ),
                ),
            )
            conn.commit()
            self._schema_ready = True

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        conn = self.connect()
        existing = {str(row["name"]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if str(column) in existing:
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def write_snapshot(
        self,
        *,
        snapshot: dict,
        features: dict,
        vector: list[float],
        energy_profile: dict[str, float],
        energy_mass: float,
        numeric_features: dict[str, list[float]],
        relation_features: dict,
        previous_memory_id: str,
        transition_edges: list[dict] | None = None,
        learned_vector: list[float] | None = None,
    ) -> PersistenceWriteResult:
        if self.config.buffered_writes:
            try:
                record = {
                    "snapshot": dict(snapshot or {}),
                    "features": dict(features or {}),
                    "vector": list(vector or []),
                    "learned_vector": list(learned_vector or []),
                    "energy_profile": dict(energy_profile or {}),
                    "energy_mass": float(energy_mass or 0.0),
                    "numeric_features": {str(key): list(value or []) for key, value in dict(numeric_features or {}).items()},
                    "relation_features": dict(relation_features or {}),
                    "previous_memory_id": str(previous_memory_id or ""),
                    "transition_edges": [dict(edge) for edge in list(transition_edges or []) if isinstance(edge, dict)],
                }
                with self._lock:
                    self._pending_records.append(record)
                    pending = len(self._pending_records)
                self._write_count += 1
                if pending >= max(1, int(self.config.buffered_flush_limit)):
                    self.flush()
                return PersistenceWriteResult(ok=True, backend=self.backend_name, operation="write_snapshot_buffered", rows_written=1)
            except Exception as exc:
                self._error_count += 1
                self._last_error = str(exc)
                return PersistenceWriteResult(ok=False, backend=self.backend_name, operation="write_snapshot_buffered", rows_written=0, error=str(exc))
        try:
            rows = self._write_snapshot_tx(
                snapshot=snapshot,
                features=features,
                vector=vector,
                learned_vector=learned_vector,
                energy_profile=energy_profile,
                energy_mass=energy_mass,
                numeric_features=numeric_features,
                relation_features=relation_features,
                previous_memory_id=previous_memory_id,
                transition_edges=transition_edges,
            )
        except Exception as exc:
            self._error_count += 1
            self._last_error = str(exc)
            try:
                conn = self.connect()
                conn.rollback()
            except Exception:
                pass
            return PersistenceWriteResult(ok=False, backend=self.backend_name, operation="write_snapshot", rows_written=0, error=str(exc))
        self._write_count += 1
        if self._should_run_retention_maintenance():
            self.enforce_retention_budget()
        return PersistenceWriteResult(ok=True, backend=self.backend_name, operation="write_snapshot", rows_written=rows)

    def flush(self) -> dict:
        with self._lock:
            records = list(self._pending_records)
            self._pending_records.clear()
            runtime_state = dict(self._pending_runtime_state or {})
            self._pending_runtime_state = None
            runtime_supplier = dict(self._pending_runtime_state_supplier or {})
            self._pending_runtime_state_supplier = None
        if not runtime_state and runtime_supplier:
            supplier = runtime_supplier.get("supplier")
            if callable(supplier):
                try:
                    runtime_state = dict(supplier() or {})
                except Exception as exc:
                    self._error_count += 1
                    self._last_error = str(exc)
                    runtime_state = {}
        if not records:
            if runtime_state:
                runtime_result = self._write_runtime_state_tx(runtime_state)
                return {
                    "backend": self.backend_name,
                    "flushed": 0,
                    "rows_written": int(runtime_result.rows_written or 0),
                    "error_count": int(self._error_count),
                    "runtime_state_flushed": bool(runtime_result.ok),
                }
            return {"backend": self.backend_name, "flushed": 0, "rows_written": 0, "error_count": int(self._error_count)}
        rows_written = 0
        flushed = 0
        for record in records:
            try:
                rows_written += self._write_snapshot_tx(
                    snapshot=dict(record.get("snapshot", {}) or {}),
                    features=dict(record.get("features", {}) or {}),
                    vector=list(record.get("vector", []) or []),
                    learned_vector=list(record.get("learned_vector", []) or []),
                    energy_profile=dict(record.get("energy_profile", {}) or {}),
                    energy_mass=float(record.get("energy_mass", 0.0) or 0.0),
                    numeric_features={str(key): list(value or []) for key, value in dict(record.get("numeric_features", {}) or {}).items()},
                    relation_features=dict(record.get("relation_features", {}) or {}),
                    previous_memory_id=str(record.get("previous_memory_id", "") or ""),
                    transition_edges=[dict(edge) for edge in list(record.get("transition_edges", []) or []) if isinstance(edge, dict)],
                )
                flushed += 1
            except Exception as exc:
                self._error_count += 1
                self._last_error = str(exc)
        runtime_state_flushed = False
        runtime_rows_written = 0
        if runtime_state:
            runtime_result = self._write_runtime_state_tx(runtime_state)
            runtime_state_flushed = bool(runtime_result.ok)
            runtime_rows_written = int(runtime_result.rows_written or 0)
        budget_trace = {}
        if self._should_run_retention_maintenance():
            budget_trace = self.enforce_retention_budget()
        return {
            "backend": self.backend_name,
            "flushed": flushed,
            "rows_written": rows_written + runtime_rows_written,
            "error_count": int(self._error_count),
            "last_error": self._last_error,
            "runtime_state_flushed": runtime_state_flushed,
            "retention_budget": budget_trace,
        }

    def discard_pending(self, *, reason: str = "") -> dict:
        """
        Drop buffered writes that have not reached the authoritative database.

        Teacher-guided or speculative runs can fail after producing many
        low-grain snapshots. Those traces are valid audit material, but they are
        not positive long-term AP memory until the action chain closes
        successfully. This method keeps the persistence boundary explicit
        without changing recall or action policy.
        """

        with self._lock:
            discarded = len(self._pending_records)
            self._pending_records.clear()
            had_runtime_state = self._pending_runtime_state is not None or self._pending_runtime_state_supplier is not None
            self._pending_runtime_state = None
            self._pending_runtime_state_supplier = None
        return {
            "backend": self.backend_name,
            "discarded_snapshots": int(discarded),
            "discarded_runtime_state": bool(had_runtime_state),
            "reason": str(reason or ""),
            "policy": "unclosed_or_failed_teacher_demo_trace_is_audit_only_not_authoritative_memory",
        }

    def queue_runtime_state_supplier(self, supplier, *, reason: str = "") -> bool:
        if not callable(supplier):
            return False
        with self._lock:
            self._pending_runtime_state_supplier = {
                "supplier": supplier,
                "reason": str(reason or ""),
            }
        return True

    def load_recent_snapshots(self, *, memory_kind: str | None = None, limit_per_kind: int | None = None) -> list[dict]:
        self.flush()
        self.ensure_schema()
        cap = max(1, int(limit_per_kind or self.config.resident_hot_snapshots_per_kind))
        params: list[object] = [self.run_id]
        kind_filter = ""
        if memory_kind is not None:
            kind_filter = "AND s.memory_kind = ?"
            params.append(str(memory_kind or ""))
        with self._lock:
            rows = self.connect().execute(
                f"""
                WITH ranked AS (
                    SELECT
                        s.snapshot_json,
                        s.snapshot_blob,
                        s.snapshot_codec,
                        s.memory_kind,
                        s.tick_index,
                        s.memory_id,
                        row_number() OVER (
                            PARTITION BY s.memory_kind
                            ORDER BY
                                CASE COALESCE(r.memory_tier, 'hot')
                                    WHEN 'core' THEN 0
                                    WHEN 'hot' THEN 1
                                    WHEN 'warm' THEN 2
                                    ELSE 3
                                END ASC,
                                COALESCE(r.forget_protected, 0) DESC,
                                CASE WHEN COALESCE(r.forget_protected, 0) = 1 THEN COALESCE(r.importance_score, 0) ELSE 0 END DESC,
                                CASE WHEN COALESCE(r.forget_protected, 0) = 1 THEN COALESCE(r.last_recalled_at, 0) ELSE 0 END DESC,
                                s.tick_index DESC,
                                s.memory_id DESC
                        ) AS rn
                    FROM memory_snapshots s
                    LEFT JOIN memory_retention r
                      ON r.run_id = s.run_id AND r.memory_id = s.memory_id
                    WHERE s.run_id = ? {kind_filter}
                )
                SELECT snapshot_json, snapshot_blob, snapshot_codec
                FROM ranked
                WHERE rn <= ?
                ORDER BY memory_kind ASC, tick_index ASC, memory_id ASC
                """,
                tuple([*params, cap]),
            ).fetchall()
        snapshots = []
        for row in rows:
            payload = self._snapshot_from_row(row)
            if payload:
                snapshots.append(payload)
        return snapshots

    def snapshot_by_id(self, memory_id: str) -> dict | None:
        clean = str(memory_id or "")
        if not clean:
            return None
        self.flush()
        self.ensure_schema()
        with self._lock:
            row = self.connect().execute(
                "SELECT snapshot_json, snapshot_blob, snapshot_codec FROM memory_snapshots WHERE run_id = ? AND memory_id = ?",
                (self.run_id, clean),
            ).fetchone()
        if not row:
            return None
        payload = self._snapshot_from_row(row)
        if payload:
            self._touch_recalled([clean])
        return payload or None

    def _snapshot_from_row(self, row: sqlite3.Row) -> dict:
        try:
            blob = row["snapshot_blob"]
            codec = row["snapshot_codec"]
        except (KeyError, IndexError):
            blob = b""
            codec = ""
        payload = _loads_blob(blob, codec)
        if payload:
            return payload
        return _loads(row["snapshot_json"])

    def exact_posting_candidates(
        self,
        *,
        memory_kind: str,
        tokens_by_field: dict[str, list[str]],
        limit: int = 64,
    ) -> list[dict]:
        fts_rows = self._peak_fts_candidates(memory_kind=memory_kind, tokens_by_field=tokens_by_field, limit=limit)
        if fts_rows:
            return fts_rows
        clauses = []
        params: list[object] = [self.run_id, str(memory_kind or "")]
        for field_name, tokens in dict(tokens_by_field or {}).items():
            clean_tokens = [str(token or "").strip() for token in list(tokens or []) if str(token or "").strip()]
            if not clean_tokens:
                continue
            placeholders = ",".join("?" for _ in clean_tokens)
            clauses.append(f"(token_field = ? AND token IN ({placeholders}))")
            params.append(str(field_name or ""))
            params.extend(clean_tokens)
        if not clauses:
            return []
        params.append(max(1, int(limit)))
        sql = f"""
        SELECT
            p.memory_id,
            count(*) AS matched,
            sum(p.weight) AS posting_score,
            COALESCE(r.memory_tier, 'cold') AS memory_tier,
            COALESCE(r.importance_score, 0) AS importance_score,
            COALESCE(r.forget_protected, 0) AS forget_protected
        FROM memory_posting_tokens p
        LEFT JOIN memory_retention r
          ON r.run_id = p.run_id AND r.memory_id = p.memory_id
        WHERE p.run_id = ? AND p.memory_kind = ? AND ({' OR '.join(clauses)})
        GROUP BY p.memory_id
        ORDER BY posting_score DESC, matched DESC, forget_protected DESC, importance_score DESC, p.memory_id ASC
        LIMIT ?
        """
        self.ensure_schema()
        with self._lock:
            rows = self.connect().execute(sql, tuple(params)).fetchall()
        result = [
            {
                "memory_id": str(row["memory_id"]),
                "matched": int(row["matched"] or 0),
                "posting_score": float(row["posting_score"] or 0.0),
                "memory_tier": str(row["memory_tier"] or "cold"),
                "importance_score": float(row["importance_score"] or 0.0),
                "forget_protected": bool(row["forget_protected"]),
                "candidate_sources": ["sqlite_exact_posting"],
            }
            for row in rows
        ]
        self._touch_recalled([str(row.get("memory_id", "") or "") for row in result])
        return result

    def _peak_fts_candidates(
        self,
        *,
        memory_kind: str,
        tokens_by_field: dict[str, list[str]],
        limit: int,
    ) -> list[dict]:
        query_pairs: list[tuple[str, str, str]] = []
        seen_ids: set[str] = set()
        for field_name, tokens in dict(tokens_by_field or {}).items():
            for token in list(tokens or []):
                clean = str(token or "").strip()
                if not clean:
                    continue
                token_id = self._peak_token_id(str(field_name or ""), clean)
                if token_id in seen_ids:
                    continue
                seen_ids.add(token_id)
                query_pairs.append((str(field_name or ""), clean, token_id))
        if not query_pairs:
            return []
        self.flush()
        self.ensure_schema()
        match_expr = " OR ".join(token_id for _, _, token_id in query_pairs)
        fetch_limit = max(1, int(limit)) * 4
        sql = """
        SELECT
            memory_peak_fts.memory_id,
            memory_peak_fts.peak_tokens,
            rank,
            COALESCE(r.memory_tier, 'cold') AS memory_tier,
            COALESCE(r.importance_score, 0) AS importance_score,
            COALESCE(r.forget_protected, 0) AS forget_protected
        FROM memory_peak_fts
        LEFT JOIN memory_retention r
          ON r.run_id = memory_peak_fts.run_id AND r.memory_id = memory_peak_fts.memory_id
        WHERE memory_peak_fts MATCH ?
          AND memory_peak_fts.run_id = ?
          AND memory_peak_fts.memory_kind = ?
        ORDER BY rank, forget_protected DESC, importance_score DESC
        LIMIT ?
        """
        try:
            with self._lock:
                rows = self.connect().execute(sql, (match_expr, self.run_id, str(memory_kind or ""), fetch_limit)).fetchall()
        except sqlite3.Error:
            return []
        query_id_set = {token_id for _, _, token_id in query_pairs}
        original_by_id: dict[str, list[dict]] = {}
        for field_name, token, token_id in query_pairs:
            original_by_id.setdefault(token_id, []).append({"field": field_name, "token": token})
        candidates = []
        for row in rows:
            peak_ids = set(str(row["peak_tokens"] or "").split())
            matched_ids = sorted(query_id_set & peak_ids)
            if not matched_ids:
                continue
            matched_tokens: dict[str, list[str]] = {}
            for token_id in matched_ids[:16]:
                for original in original_by_id.get(token_id, [])[:2]:
                    field = str(original.get("field", "") or "")
                    matched_tokens.setdefault(field, []).append(str(original.get("token", "") or ""))
            matched = len(matched_ids)
            rank = float(row["rank"] or 0.0)
            candidates.append(
                {
                    "memory_id": str(row["memory_id"]),
                    "matched": int(matched),
                    "posting_score": float(matched),
                    "candidate_sources": ["sqlite_peak_fts"],
                    "matched_tokens": matched_tokens,
                    "fts_rank": rank,
                    "memory_tier": str(row["memory_tier"] or "cold"),
                    "importance_score": float(row["importance_score"] or 0.0),
                    "forget_protected": bool(row["forget_protected"]),
                }
            )
        candidates.sort(
            key=lambda item: (
                -int(item.get("matched", 0) or 0),
                float(item.get("fts_rank", 0.0) or 0.0),
                -float(item.get("importance_score", 0.0) or 0.0),
                str(item.get("memory_id", "")),
            )
        )
        result = candidates[: max(1, int(limit))]
        self._touch_recalled([str(row.get("memory_id", "") or "") for row in result])
        return result

    def successor_edges(
        self,
        *,
        memory_kind: str,
        source_memory_ids: list[str],
        limit_per_source: int = 8,
    ) -> list[dict]:
        clean_sources = [str(memory_id or "").strip() for memory_id in list(source_memory_ids or []) if str(memory_id or "").strip()]
        if not clean_sources:
            return []
        self.flush()
        self.ensure_schema()
        placeholders = ",".join("?" for _ in clean_sources)
        cap = max(1, int(limit_per_source or 8))
        sql = f"""
        WITH ranked AS (
            SELECT
                memory_kind,
                source_memory_id,
                successor_memory_id,
                observed_count,
                last_tick_index,
                row_number() OVER (
                    PARTITION BY source_memory_id
                    ORDER BY observed_count DESC, last_tick_index DESC, successor_memory_id ASC
                ) AS rn
            FROM memory_transitions
            WHERE run_id = ? AND memory_kind = ? AND source_memory_id IN ({placeholders})
        )
        SELECT memory_kind, source_memory_id, successor_memory_id, observed_count, last_tick_index
        FROM ranked
        WHERE rn <= ?
        ORDER BY source_memory_id ASC, rn ASC
        """
        params: list[object] = [self.run_id, str(memory_kind or ""), *clean_sources, cap]
        with self._lock:
            rows = self.connect().execute(sql, tuple(params)).fetchall()
        return [
            {
                "memory_kind": str(row["memory_kind"]),
                "source_memory_id": str(row["source_memory_id"]),
                "successor_memory_id": str(row["successor_memory_id"]),
                "observed_count": int(row["observed_count"] or 0),
                "last_tick_index": int(row["last_tick_index"] or 0),
            }
            for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self.flush()
            if self._conn is None:
                return
            self._conn.close()
            self._conn = None

    def summary(self) -> dict:
        size_summary = self._database_size_summary()
        retention_summary = self._retention_summary()
        return {
            "backend": self.backend_name,
            "enabled": True,
            "path": str(self.path),
            "exists": self.path.exists(),
            "run_id": self.run_id,
            "write_count": int(self._write_count),
            "error_count": int(self._error_count),
            "last_error": self._last_error,
            "pending_buffered_writes": len(self._pending_records),
            **size_summary,
            "retention": retention_summary,
            "resident_policy": {
                "load_all_history_into_memory": False,
                "resident_hot_snapshots_per_kind": int(self.config.resident_hot_snapshots_per_kind),
                "warm_prefetch_limit": int(self.config.warm_prefetch_limit),
                "hot_layer_snapshots_per_kind": int(self.config.hot_layer_snapshots_per_kind),
                "warm_layer_snapshots_per_kind": int(self.config.warm_layer_snapshots_per_kind),
                "memory_db_budget_bytes": int(self.config.memory_db_budget_bytes),
                "forgetting_enabled": bool(self.config.forgetting_enabled),
                "meaning": "sqlite_is_authoritative_local_history;runtime_indexes_are_bounded_rebuildable_working_memory",
            },
            "deployment_profile": "embedded_desktop_default",
        }

    def write_runtime_state(self, *, state: dict) -> PersistenceWriteResult:
        self.ensure_schema()
        payload = self._jsonable(dict(state or {}))
        if self.config.buffered_writes:
            with self._lock:
                self._pending_runtime_state = payload
            self._write_count += 1
            return PersistenceWriteResult(ok=True, backend=self.backend_name, operation="write_runtime_state_buffered", rows_written=1)
        return self._write_runtime_state_tx(payload)

    def load_runtime_state(self) -> dict | None:
        self.ensure_schema()
        with self._lock:
            if self._pending_runtime_state is not None:
                return dict(self._pending_runtime_state)
            supplier = dict(self._pending_runtime_state_supplier or {})
        if supplier:
            runtime_supplier = supplier.get("supplier")
            if callable(runtime_supplier):
                try:
                    pending = dict(runtime_supplier() or {})
                except Exception:
                    pending = {}
                if pending:
                    with self._lock:
                        self._pending_runtime_state = dict(pending)
                    return pending
        with self._lock:
            row = self.connect().execute(
                "SELECT state_json, state_blob, state_codec FROM ap_runtime_state WHERE run_id = ?",
                (self.run_id,),
            ).fetchone()
        if not row:
            return None
        payload = _loads_blob(row["state_blob"], row["state_codec"])
        if not payload:
            payload = _loads(row["state_json"])
        return payload if isinstance(payload, dict) else None

    def _write_runtime_state_tx(self, payload: dict) -> PersistenceWriteResult:
        blob, codec, raw_bytes, stored_bytes = self._encode_payload(payload)
        try:
            with self._lock:
                self.connect().execute(
                    """
                    INSERT INTO ap_runtime_state(run_id, updated_at, state_json, state_blob, state_codec, state_raw_bytes, state_stored_bytes)
                    VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        updated_at = excluded.updated_at,
                        state_json = excluded.state_json,
                        state_blob = excluded.state_blob,
                        state_codec = excluded.state_codec,
                        state_raw_bytes = excluded.state_raw_bytes,
                        state_stored_bytes = excluded.state_stored_bytes
                    """,
                    (
                        self.run_id,
                        _json(payload),
                        blob,
                        codec,
                        raw_bytes,
                        stored_bytes,
                    ),
                )
                self.connect().commit()
            return PersistenceWriteResult(ok=True, backend=self.backend_name, operation="write_runtime_state", rows_written=1)
        except Exception as exc:
            self._error_count += 1
            self._last_error = str(exc)
            return PersistenceWriteResult(ok=False, backend=self.backend_name, operation="write_runtime_state", rows_written=0, error=self._last_error)

    def _retention_summary(self) -> dict:
        try:
            self.flush()
            self.ensure_schema()
            with self._lock:
                rows = self.connect().execute(
                    """
                    SELECT memory_tier, count(*) AS count, sum(forget_protected) AS protected_count
                    FROM memory_retention
                    WHERE run_id = ?
                    GROUP BY memory_tier
                    ORDER BY memory_tier ASC
                    """,
                    (self.run_id,),
                ).fetchall()
        except Exception as exc:
            return {
                "schema_id": "apv21_memory_retention_summary/v1",
                "available": False,
                "error": str(exc),
            }
        tiers = {
            str(row["memory_tier"]): {
                "count": int(row["count"] or 0),
                "protected": int(row["protected_count"] or 0),
            }
            for row in rows
        }
        return {
            "schema_id": "apv21_memory_retention_summary/v1",
            "available": True,
            "tiers": tiers,
            "budget_bytes": int(self.config.memory_db_budget_bytes),
            "forgetting_enabled": bool(self.config.forgetting_enabled),
            "meaning": "core_skill_facts_and_rewarded_paradigms_are_protected;ordinary_trace_memory_is_budget_managed",
        }

    def _retention_profile(self, snapshot: dict, *, energy_mass: float) -> dict:
        items = [item for item in list((snapshot or {}).get("items", []) or []) if isinstance(item, dict)]
        labels = [str(item.get("sa_label", "") or "") for item in items]
        families = [str(item.get("family", "") or "") for item in items]
        source_types = [str(item.get("source_type", "") or "") for item in items]
        focus_labels = [str(label or "") for label in list((snapshot or {}).get("focus_labels", []) or []) if str(label or "")]
        meta_rows = [dict(item.get("anchor_meta", {}) or {}) for item in items if isinstance(item.get("anchor_meta", {}), dict)]
        all_text = " ".join([*labels, *families, *source_types, *focus_labels])
        domains = []
        for meta in meta_rows:
            domain = str(meta.get("domain", "") or meta.get("source_domain", "") or "")
            if domain and domain not in domains:
                domains.append(domain)
        for label in labels:
            if label.startswith("skill_domain::"):
                domain = label.split("::", 1)[1]
                if domain and domain not in domains:
                    domains.append(domain)
        reward = 0.0
        punishment = 0.0
        for meta in meta_rows:
            reward += float(meta.get("feedback_reward", 0.0) or meta.get("reward_value", 0.0) or 0.0)
            punishment += float(meta.get("feedback_punishment", 0.0) or meta.get("punishment_value", 0.0) or 0.0)
        protected_markers = (
            "core_skill_atom",
            "protected_skill_fact",
            "skill_package",
            "skill_domain",
            "math_paradigm",
            "attention_process",
            "desktop_safety",
            "dialogue_category",
            "action_feedback",
            "multiplication_table",
            "addition_fact",
            "subtraction_fact",
            "division_trial_quotient",
            "learned_semantic_frequency",
        )
        is_protected = any(marker in all_text for marker in protected_markers) or reward >= 0.14
        if punishment >= max(0.18, reward + 0.05):
            is_protected = False
        memory_kind = str((snapshot or {}).get("memory_kind", "") or "")
        base = float(energy_mass or 0.0)
        importance = base + reward * 4.0 - punishment * 3.0 + min(1.2, len(focus_labels) * 0.04)
        if is_protected:
            importance += 3.0
        if any("user_manual_feedback" in str(meta) for meta in meta_rows):
            importance += 1.0
        if memory_kind == "focus" and not is_protected:
            importance *= 0.82
        tier = "core" if is_protected else "hot"
        reason_bits = []
        if is_protected:
            reason_bits.append("protected_skill_or_rewarded_paradigm")
        if reward:
            reason_bits.append("rewarded")
        if punishment:
            reason_bits.append("punished")
        if not reason_bits:
            reason_bits.append("ordinary_runtime_memory")
        return {
            "memory_tier": tier,
            "importance_score": round(max(0.0, float(importance)), 6),
            "forget_protected": bool(is_protected),
            "retention_reason": ";".join(reason_bits)[:240],
            "source_domain": ",".join(domains[:4])[:160],
        }

    def _should_run_retention_maintenance(self) -> bool:
        if not bool(self.config.forgetting_enabled):
            return False
        if int(self.config.memory_db_budget_bytes or 0) <= 0:
            return False
        interval = max(1, int(self.config.retention_maintenance_interval_writes or 512))
        return int(self._write_count or 0) > 0 and int(self._write_count or 0) % interval == 0

    def enforce_retention_budget(self) -> dict:
        self.ensure_schema()
        if not bool(self.config.forgetting_enabled):
            return {"schema_id": "apv21_retention_budget/v1", "enabled": False, "reason": "forgetting_disabled"}
        budget = int(self.config.memory_db_budget_bytes or 0)
        if budget <= 0:
            return {"schema_id": "apv21_retention_budget/v1", "enabled": False, "reason": "no_budget"}
        size = self._database_size_summary()
        current = int(size.get("ap_table_bytes", 0) or 0)
        if current <= budget:
            return {"schema_id": "apv21_retention_budget/v1", "enabled": True, "over_budget": False, "bytes": current, "budget_bytes": budget, "deleted": 0}
        deleted_total = 0
        attempts = []
        for tier in ("cold", "warm", "hot"):
            if current <= budget:
                break
            ids = self._retention_prune_candidates(tier=tier, limit=max(1, int(self.config.retention_prune_batch or 256)))
            if not ids:
                attempts.append({"tier": tier, "candidate_count": 0, "deleted": 0})
                continue
            deleted = self._delete_memories(ids)
            deleted_total += deleted
            attempts.append({"tier": tier, "candidate_count": len(ids), "deleted": deleted})
            size = self._database_size_summary()
            current = int(size.get("ap_table_bytes", 0) or 0)
        return {
            "schema_id": "apv21_retention_budget/v1",
            "enabled": True,
            "over_budget": current > budget,
            "bytes": current,
            "budget_bytes": budget,
            "deleted": deleted_total,
            "attempts": attempts,
            "policy": "bounded_indexed_prune;never_delete_forget_protected_core_memories",
        }

    def _retention_prune_candidates(self, *, tier: str, limit: int) -> list[str]:
        sql = """
        SELECT memory_id
        FROM memory_retention
        WHERE run_id = ?
          AND forget_protected = 0
          AND memory_tier = ?
        ORDER BY importance_score ASC, last_recalled_at ASC, last_updated_at ASC, memory_id ASC
        LIMIT ?
        """
        with self._lock:
            rows = self.connect().execute(sql, (self.run_id, str(tier or ""), max(1, int(limit)))).fetchall()
        return [str(row["memory_id"]) for row in rows if str(row["memory_id"] or "")]

    def _delete_memories(self, memory_ids: list[str]) -> int:
        clean_ids = [str(memory_id or "").strip() for memory_id in list(memory_ids or []) if str(memory_id or "").strip()]
        if not clean_ids:
            return 0
        tables = (
            "memory_snapshot_items",
            "memory_state_field_items",
            "memory_core_items",
            "memory_posting_tokens",
            "memory_vectors",
            "memory_numeric_features",
            "memory_relation_features",
            "memory_asset_refs",
            "memory_peak_fts",
            "memory_snapshots",
            "memory_retention",
        )
        placeholders = ",".join("?" for _ in clean_ids)
        with self._lock:
            conn = self.connect()
            for table in tables:
                conn.execute(
                    f"DELETE FROM {table} WHERE run_id = ? AND memory_id IN ({placeholders})",
                    (self.run_id, *clean_ids),
                )
            conn.execute(
                f"DELETE FROM memory_transitions WHERE run_id = ? AND (source_memory_id IN ({placeholders}) OR successor_memory_id IN ({placeholders}))",
                (self.run_id, *clean_ids, *clean_ids),
            )
            conn.commit()
        self._summary_size_cache_at = 0.0
        return len(clean_ids)

    def _touch_recalled(self, memory_ids: list[str]) -> None:
        clean_ids = [str(memory_id or "").strip() for memory_id in list(memory_ids or []) if str(memory_id or "").strip()]
        if not clean_ids:
            return
        placeholders = ",".join("?" for _ in clean_ids)
        now = time.time()
        try:
            with self._lock:
                conn = self.connect()
                conn.execute(
                    f"""
                    UPDATE memory_retention
                    SET last_recalled_at = ?,
                        recall_count = recall_count + 1,
                        memory_tier = CASE
                            WHEN forget_protected = 1 THEN 'core'
                            WHEN memory_tier = 'cold' THEN 'warm'
                            ELSE memory_tier
                        END
                    WHERE run_id = ? AND memory_id IN ({placeholders})
                    """,
                    (now, self.run_id, *clean_ids),
                )
                conn.commit()
        except sqlite3.Error:
            pass

    def _write_snapshot_tx(
        self,
        *,
        snapshot: dict,
        features: dict,
        vector: list[float],
        energy_profile: dict[str, float],
        energy_mass: float,
        numeric_features: dict[str, list[float]],
        relation_features: dict,
        previous_memory_id: str,
        transition_edges: list[dict] | None = None,
        learned_vector: list[float] | None = None,
    ) -> int:
        self.ensure_schema()
        memory_id = str(snapshot.get("memory_id", "") or "")
        memory_kind = str(snapshot.get("memory_kind", "") or "")
        tick_index = int(snapshot.get("tick_index", 0) or 0)
        if not memory_id or not memory_kind:
            raise ValueError("snapshot must contain memory_id and memory_kind")
        stored_snapshot = self._snapshot_payload_for_storage(snapshot)
        snapshot_preview = self._snapshot_preview_for_legacy_json(stored_snapshot)
        feature_payload = self._feature_payload_for_storage(
            features=features,
            energy_profile=energy_profile,
            energy_mass=energy_mass,
            numeric_features=numeric_features,
            relation_features=relation_features,
            previous_memory_id=previous_memory_id,
            vector=vector,
            learned_vector=learned_vector,
        )
        feature_preview = self._feature_preview_for_legacy_json(feature_payload)
        snapshot_blob, snapshot_codec, snapshot_raw_bytes, snapshot_stored_bytes = self._encode_payload(stored_snapshot)
        if self.config.store_feature_payload_blob:
            feature_blob_payload = feature_payload
            feature_blob, feature_codec, feature_raw_bytes, feature_stored_bytes = self._encode_payload(feature_blob_payload)
        else:
            feature_blob, feature_codec, feature_raw_bytes, feature_stored_bytes = b"", "", 0, 0
        rows_written = 0
        conn = self.connect()
        with self._lock:
            conn.execute(
                "INSERT OR IGNORE INTO ap_ticks(run_id, tick_index) VALUES (?, ?)",
                (self.run_id, tick_index),
            )
            rows_written += 1
            conn.execute(
                """
                INSERT OR IGNORE INTO memory_snapshots(
                    run_id, memory_id, tick_index, memory_kind, source_text,
                    focus_labels_json, item_count, state_field_item_count, core_item_count, energy_mass,
                    snapshot_json, feature_summary_json,
                    snapshot_blob, snapshot_codec, snapshot_raw_bytes, snapshot_stored_bytes,
                    feature_summary_blob, feature_summary_codec, feature_summary_raw_bytes, feature_summary_stored_bytes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    memory_id,
                    tick_index,
                    memory_kind,
                    str(snapshot_preview.get("source_text", "") or ""),
                    _json(list(snapshot_preview.get("focus_labels", []) or [])),
                    len(snapshot.get("items", []) or []),
                    len(snapshot.get("state_field_items", []) or snapshot.get("items", []) or []),
                    len(snapshot.get("core_items", []) or []),
                    float(energy_mass or 0.0),
                    _json(snapshot_preview),
                    _json(feature_preview),
                    snapshot_blob,
                    snapshot_codec,
                    snapshot_raw_bytes,
                    snapshot_stored_bytes,
                    feature_blob,
                    feature_codec,
                    feature_raw_bytes,
                    feature_stored_bytes,
                ),
            )
            rows_written += 1
            retention = self._retention_profile(stored_snapshot, energy_mass=energy_mass)
            now = time.time()
            conn.execute(
                """
                INSERT INTO memory_retention(
                    run_id, memory_id, memory_kind, memory_tier, importance_score,
                    forget_protected, retention_reason, source_domain,
                    last_recalled_at, recall_count, last_updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(run_id, memory_id)
                DO UPDATE SET
                    memory_kind = excluded.memory_kind,
                    memory_tier = CASE
                        WHEN memory_retention.forget_protected = 1 AND memory_retention.memory_tier = 'core' THEN memory_retention.memory_tier
                        ELSE excluded.memory_tier
                    END,
                    importance_score = MAX(memory_retention.importance_score, excluded.importance_score),
                    forget_protected = MAX(memory_retention.forget_protected, excluded.forget_protected),
                    retention_reason = excluded.retention_reason,
                    source_domain = excluded.source_domain,
                    last_updated_at = excluded.last_updated_at
                """,
                (
                    self.run_id,
                    memory_id,
                    memory_kind,
                    str(retention.get("memory_tier", "hot") or "hot"),
                    float(retention.get("importance_score", 0.0) or 0.0),
                    1 if retention.get("forget_protected") else 0,
                    str(retention.get("retention_reason", "") or ""),
                    str(retention.get("source_domain", "") or ""),
                    now,
                    now,
                ),
            )
            rows_written += 1
            if self.config.store_expanded_item_rows:
                expanded_items = [item for item in list(stored_snapshot.get("items", []) or []) if isinstance(item, dict)]
                for idx, item in enumerate(expanded_items):
                    if not isinstance(item, dict):
                        continue
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO memory_snapshot_items(
                            run_id, memory_id, item_index, tick_index, memory_kind,
                            sa_label, display_text, family, source_type,
                            real_energy, virtual_energy, cognitive_pressure, payload_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            self.run_id,
                            memory_id,
                            idx,
                            tick_index,
                            memory_kind,
                            str(item.get("sa_label", "") or ""),
                            str(item.get("display_text", "") or ""),
                            str(item.get("family", "") or ""),
                            str(item.get("source_type", "") or ""),
                            float(item.get("real_energy", 0.0) or 0.0),
                            float(item.get("virtual_energy", 0.0) or 0.0),
                            float(item.get("cognitive_pressure", 0.0) or 0.0),
                            _json(item),
                        ),
                    )
                    rows_written += 1
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO memory_state_field_items(
                            run_id, memory_id, state_field_index, sa_label, payload_json
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            self.run_id,
                            memory_id,
                            idx,
                            str(item.get("sa_label", "") or ""),
                            _json(item),
                        ),
                    )
                    rows_written += 1
                for idx, item in enumerate(list(stored_snapshot.get("core_items", []) or [])):
                    if not isinstance(item, dict):
                        continue
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO memory_core_items(
                            run_id, memory_id, core_index, sa_label, payload_json
                        )
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            self.run_id,
                            memory_id,
                            idx,
                            str(item.get("sa_label", "") or ""),
                            _json(item),
                        ),
                    )
                    rows_written += 1
            if self.config.store_derived_index_rows:
                posting_tokens = self._posting_tokens(features, energy_profile=energy_profile)
                peak_token_ids = self._peak_token_ids(posting_tokens)
                if peak_token_ids:
                    conn.execute("DELETE FROM memory_peak_fts WHERE run_id = ? AND memory_id = ?", (self.run_id, memory_id))
                    conn.execute(
                        "INSERT INTO memory_peak_fts(run_id, memory_id, memory_kind, peak_tokens) VALUES (?, ?, ?, ?)",
                        (self.run_id, memory_id, memory_kind, " ".join(peak_token_ids)),
                    )
                    rows_written += 1
                if self.config.store_posting_token_rows:
                    for field_name, tokens in posting_tokens.items():
                        for token in tokens:
                            conn.execute(
                                "INSERT OR IGNORE INTO memory_posting_tokens(run_id, memory_id, memory_kind, token_field, token, weight) VALUES (?, ?, ?, ?, ?, ?)",
                                (self.run_id, memory_id, memory_kind, field_name, token, 1.0),
                            )
                            rows_written += 1
            hash_vector = self._fixed_vector(vector)
            hash_blob, hash_codec, hash_raw_bytes, hash_stored_bytes = self._encode_vector_payload(hash_vector)
            conn.execute(
                """
                INSERT OR IGNORE INTO memory_vectors(
                    run_id, memory_id, memory_kind, vector_space,
                    vector_json, vector_meta_json, vector_blob, vector_codec,
                    vector_dim, vector_raw_bytes, vector_stored_bytes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.run_id,
                    memory_id,
                    memory_kind,
                    "hash_vector",
                    _json(self._vector_json_payload(hash_vector, codec=hash_codec)),
                    _json(self._vector_meta_payload(hash_vector, codec=hash_codec, raw_bytes=hash_raw_bytes, stored_bytes=hash_stored_bytes)),
                    hash_blob,
                    hash_codec,
                    len(hash_vector),
                    hash_raw_bytes,
                    hash_stored_bytes,
                ),
            )
            rows_written += 1
            learned = list(learned_vector or [])
            if learned and any(abs(float(value or 0.0)) > 1e-12 for value in learned):
                learned_fixed = self._fixed_vector(learned)
                learned_blob, learned_codec, learned_raw_bytes, learned_stored_bytes = self._encode_vector_payload(learned_fixed)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO memory_vectors(
                        run_id, memory_id, memory_kind, vector_space,
                        vector_json, vector_meta_json, vector_blob, vector_codec,
                        vector_dim, vector_raw_bytes, vector_stored_bytes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self.run_id,
                        memory_id,
                        memory_kind,
                        "online_learned_vector",
                        _json(self._vector_json_payload(learned_fixed, codec=learned_codec)),
                        _json(
                            self._vector_meta_payload(
                                learned_fixed,
                                codec=learned_codec,
                                raw_bytes=learned_raw_bytes,
                                stored_bytes=learned_stored_bytes,
                                extra={"source": "OnlineEmbeddingStore.learned_vector"},
                            )
                        ),
                        learned_blob,
                        learned_codec,
                        len(learned_fixed),
                        learned_raw_bytes,
                        learned_stored_bytes,
                    ),
                )
                rows_written += 1
            if self.config.store_derived_index_rows:
                for channel, values in dict(numeric_features or {}).items():
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_numeric_features(run_id, memory_id, memory_kind, channel, values_json, feature_meta_json) VALUES (?, ?, ?, ?, ?, ?)",
                        (self.run_id, memory_id, memory_kind, str(channel), _json([float(value or 0.0) for value in list(values or [])]), _json({})),
                    )
                    rows_written += 1
                relation_weights = dict((relation_features or {}).get("relation_token_weights", {}) or {})
                for token, weight in relation_weights.items():
                    conn.execute(
                        "INSERT OR IGNORE INTO memory_relation_features(run_id, memory_id, memory_kind, relation_token, relation_type, weight, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (self.run_id, memory_id, memory_kind, str(token), self._relation_type(str(token)), float(weight or 0.0), _json({})),
                    )
                    rows_written += 1
                transition_rows: list[dict] = []
                if previous_memory_id:
                    transition_rows.append(
                        {
                            "memory_kind": memory_kind,
                            "source_memory_id": str(previous_memory_id),
                            "successor_memory_id": memory_id,
                            "transition_meta": {},
                        }
                    )
                for edge in list(transition_edges or []):
                    if not isinstance(edge, dict):
                        continue
                    source_id = str(edge.get("source_memory_id", "") or "").strip()
                    successor_id = str(edge.get("successor_memory_id", memory_id) or "").strip()
                    edge_kind = str(edge.get("memory_kind", "") or "").strip()
                    if not source_id or not successor_id or not edge_kind:
                        continue
                    transition_rows.append(
                        {
                            "memory_kind": edge_kind,
                            "source_memory_id": source_id,
                            "successor_memory_id": successor_id,
                            "transition_meta": dict(edge.get("transition_meta", {}) or {}),
                        }
                    )
                seen_edges: set[tuple[str, str, str]] = set()
                for edge in transition_rows:
                    edge_key = (
                        str(edge["memory_kind"]),
                        str(edge["source_memory_id"]),
                        str(edge["successor_memory_id"]),
                    )
                    if edge_key in seen_edges:
                        continue
                    seen_edges.add(edge_key)
                    conn.execute(
                        """
                        INSERT INTO memory_transitions(
                            run_id, memory_kind, source_memory_id, successor_memory_id,
                            observed_count, last_tick_index, transition_meta_json
                        )
                        VALUES (?, ?, ?, ?, 1, ?, ?)
                        ON CONFLICT(run_id, memory_kind, source_memory_id, successor_memory_id)
                        DO UPDATE SET observed_count = observed_count + 1,
                                      last_tick_index = excluded.last_tick_index,
                                      transition_meta_json = excluded.transition_meta_json
                        """,
                        (
                            self.run_id,
                            str(edge["memory_kind"]),
                            str(edge["source_memory_id"]),
                            str(edge["successor_memory_id"]),
                            tick_index,
                            _json(dict(edge.get("transition_meta", {}) or {})),
                        ),
                    )
                    rows_written += 1
            for ref in list(stored_snapshot.get("asset_refs", []) or []):
                if not isinstance(ref, dict):
                    continue
                asset_id = str(ref.get("asset_id", "") or "")
                if not asset_id:
                    continue
                conn.execute(
                    "INSERT OR IGNORE INTO memory_asset_refs(run_id, memory_id, asset_id, modality, uri, sha256, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        self.run_id,
                        memory_id,
                        asset_id,
                        str(ref.get("modality", "") or ""),
                        str(ref.get("uri", ref.get("path", "")) or ""),
                        str(ref.get("sha256", "") or ""),
                        _json(ref),
                    ),
                )
                rows_written += 1
            conn.commit()
        return rows_written

    def _snapshot_payload_for_storage(self, snapshot: dict) -> dict:
        if self.config.full_fidelity_snapshot_blob:
            payload = self._jsonable(dict(snapshot or {}))
            if self.config.runtime_projection_snapshot_blob:
                return self._runtime_projection_snapshot(payload)
            return payload
        return self._compact_snapshot_for_storage(snapshot)

    def _runtime_projection_snapshot(self, snapshot: dict) -> dict:
        """
        Persist the AP-runtime memory, not the full white-box audit mirror.

        Runtime can rebuild state-field/core/relation/sequence/prediction/numeric
        views from `items` during warm-load. Keeping those hot-memory views in
        SQLite duplicates the same cognitive event several times and makes the
        desktop DB look much larger than the actual skill memory.
        """

        keep = {
            "memory_id",
            "tick_index",
            "memory_kind",
            "successor_boundary",
            "focus_labels",
            "source_text",
            "items",
            "asset_refs",
            "vector_spaces",
        }
        projected = {key: snapshot.get(key) for key in keep if key in snapshot}
        projected["items"] = [
            self._runtime_projection_item(item)
            for item in list(projected.get("items", []) or [])
            if isinstance(item, dict)
        ]
        projected["asset_refs"] = [
            self._runtime_projection_asset_ref(ref)
            for ref in list(projected.get("asset_refs", []) or [])
            if isinstance(ref, dict)
        ]
        projected["storage_policy"] = {
            "schema_id": "apv21_runtime_projection_snapshot/v1",
            "db_role": "ap_runtime_operational_memory",
            "audit_payload_external": True,
            "rebuildable_views_omitted": [
                "state_field_items",
                "anchor_items",
                "core_items",
                "sequence_features",
                "relation_features",
                "prediction_payload_items",
                "action_feedback_items",
                "numeric_features",
            ],
        }
        return projected

    def _runtime_projection_item(self, item: dict) -> dict:
        keep = {
            "sa_label",
            "display_text",
            "family",
            "source_type",
            "real_energy",
            "virtual_energy",
            "cognitive_pressure",
            "tick_index",
            "position",
            "modality",
            "is_focus",
            "attention_gain",
            "fatigue",
            "last_seen_tick",
            "last_updated_tick",
            "numeric_features",
        }
        projected = {key: item.get(key) for key in keep if key in item}
        meta = item.get("anchor_meta", {})
        if isinstance(meta, dict):
            meta_keep = {
                "schema_id",
                "event_type",
                "token",
                "action_id",
                "source",
                "source_event_type",
                "self_generated",
                "position",
                "current_glyph_index",
                "current_glyph_role",
                "cursor",
                "cursor_before",
                "cursor_after",
                "cursor_index",
                "visible_text",
                "visible_text_before",
                "visible_text_after",
                "visible_length",
                "last_visible_token",
                "previous_prefix",
                "variant_text",
                "expected_text",
                "reply_trace_hash",
                "canonical_memory_hash",
                "process_anchor_role",
                "prediction_payload_priority",
                "readout_pattern_id",
                "readout_semantic_role",
                "semantic_frame_role",
                "dynamic_slot_role",
                "action_param_reason",
                "action_param_kind",
                "expected_token",
                "candidate_token",
                "target_token",
                "feedback_outcome",
                "feedback_reward",
                "feedback_punishment",
                "feedback_correctness",
                "feedback_reference_token",
                "feedback_type",
                "reward_value",
                "punishment_value",
                "modality",
                "asset_id",
                "uri",
                "path",
                "numeric_features",
                "features",
                "not_answer_table",
                "not_regex_route",
                "not_full_sentence_macro",
                "teaching_protocol",
                "category",
            }
            projected_meta = {
                key: meta.get(key)
                for key in meta_keep
                if key in meta and meta.get(key) not in (None, "", [], {})
            }
            if projected_meta:
                projected["anchor_meta"] = projected_meta
        return projected

    def _runtime_projection_asset_ref(self, ref: dict) -> dict:
        return {
            key: ref.get(key)
            for key in ("asset_id", "modality", "uri", "path", "sha256", "mime")
            if key in ref and ref.get(key) not in (None, "")
        }

    def _feature_payload_for_storage(
        self,
        *,
        features: dict,
        energy_profile: dict[str, float],
        energy_mass: float,
        numeric_features: dict[str, list[float]],
        relation_features: dict,
        previous_memory_id: str,
        vector: list[float],
        learned_vector: list[float] | None,
    ) -> dict:
        return self._jsonable(
            {
                "schema_id": "apv21_sqlite_feature_payload/v1",
                "features": dict(features or {}),
                "energy_profile": dict(energy_profile or {}),
                "energy_mass": float(energy_mass or 0.0),
                "numeric_features": {str(key): list(value or []) for key, value in dict(numeric_features or {}).items()},
                "relation_features": dict(relation_features or {}),
                "previous_memory_id": str(previous_memory_id or ""),
                "vector_spaces": {
                    "hash_vector": self._fixed_vector(vector),
                    "online_learned_vector": self._fixed_vector(list(learned_vector or [])) if learned_vector else [],
                },
                "storage_policy": {
                    "authoritative_payload": "compressed_full_fidelity_event",
                    "derived_indexes_rebuildable": True,
                    "legacy_json_columns": "preview_only" if self.config.legacy_json_preview_only else "full_legacy_copy",
                },
            }
        )

    def _feature_preview_blob_payload(self, preview: dict) -> dict:
        return self._jsonable(
            {
                "schema_id": "apv21_sqlite_feature_preview_blob/v1",
                "feature_preview": dict(preview or {}),
                "storage_policy": {
                    "full_feature_payload_stored": False,
                    "full_feature_payload_rebuildable_from": "snapshot_blob",
                    "reason": "features/postings/vectors/relations_are_derived_working_indexes;authoritative_memory_is_full_fidelity_snapshot_blob",
                },
            }
        )

    def _encode_payload(self, payload: dict) -> tuple[bytes, str, int, int]:
        raw = _json_bytes(self._jsonable(payload))
        if not self.config.compressed_snapshot_blob:
            return raw, "plain", len(raw), len(raw)
        level = max(1, min(9, int(self.config.snapshot_compression_level or 6)))
        compressed = zlib.compress(raw, level)
        if len(compressed) < len(raw):
            return compressed, "zlib", len(raw), len(compressed)
        return raw, "plain", len(raw), len(raw)

    def _snapshot_preview_for_legacy_json(self, snapshot: dict) -> dict:
        if not self.config.legacy_json_preview_only:
            return self._jsonable(snapshot)
        items = [item for item in list(snapshot.get("items", []) or []) if isinstance(item, dict)]
        state_field_items = [item for item in list(snapshot.get("state_field_items", []) or []) if isinstance(item, dict)]
        core_items = [item for item in list(snapshot.get("core_items", []) or []) if isinstance(item, dict)]
        return {
            "schema_id": "apv21_sqlite_snapshot_preview/v1",
            "memory_id": str(snapshot.get("memory_id", "") or ""),
            "tick_index": int(snapshot.get("tick_index", 0) or 0),
            "memory_kind": str(snapshot.get("memory_kind", "") or ""),
            "source_text": str(snapshot.get("source_text", "") or "")[:160],
            "focus_labels": list(snapshot.get("focus_labels", []) or [])[:12],
            "item_count": len(items),
            "state_field_item_count": len(state_field_items or items),
            "core_item_count": len(core_items),
            "item_labels_preview": [str(item.get("sa_label", "") or "") for item in items[:12]],
            "state_field_labels_preview": [str(item.get("sa_label", "") or "") for item in (state_field_items or items)[:12]],
            "core_labels_preview": [str(item.get("sa_label", "") or "") for item in core_items[:8]],
            "asset_ref_count": len(list(snapshot.get("asset_refs", []) or [])),
            "vector_spaces": sorted((snapshot.get("vector_spaces", {}) or {}).keys()),
            "authoritative_payload": "snapshot_blob",
        }

    def _feature_preview_for_legacy_json(self, payload: dict) -> dict:
        if not self.config.legacy_json_preview_only:
            return self._jsonable(payload)
        features = dict((payload or {}).get("features", {}) or {})
        return {
            "schema_id": "apv21_sqlite_feature_preview/v2",
            "counts": {
                "labels": len(list(features.get("labels", []) or [])),
                "displays": len(list(features.get("displays", []) or [])),
                "bigrams": len(list(features.get("bigrams", []) or [])),
                "sequence_bigrams": len(list(features.get("sequence_bigrams", []) or [])),
                "focus_labels": len(list(features.get("focus_labels", []) or [])),
                "relation_tokens": len(list(features.get("relation_tokens", []) or [])),
            },
            "top_labels": list(features.get("labels", []) or [])[:4],
            "top_focus": list(features.get("focus_labels", []) or [])[:4],
            "energy_mass": float((payload or {}).get("energy_mass", 0.0) or 0.0),
            "relation_count": len(list(((payload or {}).get("relation_features", {}) or {}).get("relation_tokens", []) or [])),
            "full_stored": bool(self.config.store_feature_payload_blob),
            "source": "feature_summary_blob" if self.config.store_feature_payload_blob else "snapshot_blob",
            "policy": "full_feature_payload_blob" if self.config.store_feature_payload_blob else "preview_only;rebuild_from_full_snapshot_blob",
        }

    def _encode_vector_payload(self, values: list[float]) -> tuple[bytes, str, int, int]:
        fixed = self._fixed_vector(values)
        if not self.config.store_vector_blob:
            raw = _json_bytes(fixed)
            return b"", "json_only", len(raw), 0
        dtype = str(self.config.vector_blob_dtype or "float16").strip().lower()
        if dtype in {"float16", "f16", "half"}:
            try:
                blob = struct.pack("<" + "e" * len(fixed), *[float(value or 0.0) for value in fixed])
                return blob, "float16-le", len(fixed) * 4, len(blob)
            except (struct.error, OverflowError):
                pass
        blob = struct.pack("<" + "f" * len(fixed), *[float(value or 0.0) for value in fixed])
        return blob, "float32-le", len(fixed) * 4, len(blob)

    def _vector_json_payload(self, values: list[float], *, codec: str) -> object:
        fixed = self._fixed_vector(values)
        if not self.config.vector_json_preview_only or codec in {"", "json_only"}:
            return fixed
        return {
            "d": len(fixed),
            "s": "blob",
            "c": str(codec or ""),
            "p": [round(float(value or 0.0), 4) for value in fixed[:6]],
        }

    def _vector_meta_payload(
        self,
        values: list[float],
        *,
        codec: str,
        raw_bytes: int,
        stored_bytes: int,
        extra: dict | None = None,
    ) -> dict:
        fixed = self._fixed_vector(values)
        payload = {
            "dim": len(fixed),
            "codec": str(codec or ""),
            "raw": int(raw_bytes),
            "stored": int(stored_bytes),
            "loss": "float16" if str(codec or "") == "float16-le" else "none_or_json",
            "source": "snapshot_blob.vector_spaces",
        }
        payload.update(dict(extra or {}))
        return payload

    def _jsonable(self, value: object) -> object:
        if isinstance(value, dict):
            return {str(key): self._jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(item) for item in value]
        if isinstance(value, set):
            return [self._jsonable(item) for item in sorted(value, key=lambda item: str(item))]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (bytes, bytearray, memoryview)):
            return bytes(value).hex()
        return str(value)

    def _compact_snapshot_for_storage(self, snapshot: dict) -> dict:
        keep = {
            "memory_id",
            "tick_index",
            "memory_kind",
            "successor_boundary",
            "focus_labels",
            "source_text",
            "numeric_features",
            "vector_spaces",
        }
        compact = {key: snapshot.get(key) for key in keep if key in snapshot}
        compact["items"] = [self._compact_item(item) for item in list(snapshot.get("items", []) or []) if isinstance(item, dict)]
        compact["asset_refs"] = [self._compact_asset_ref(ref) for ref in list(snapshot.get("asset_refs", []) or []) if isinstance(ref, dict)]
        return compact

    def _compact_item(self, item: dict) -> dict:
        keep = {
            "sa_label",
            "display_text",
            "family",
            "source_type",
            "real_energy",
            "virtual_energy",
            "cognitive_pressure",
            "tick_index",
            "position",
            "modality",
            "is_focus",
            "attention_gain",
            "fatigue",
            "last_seen_tick",
            "last_updated_tick",
            "numeric_features",
        }
        compact = {key: item.get(key) for key in keep if key in item}
        anchor_meta = item.get("anchor_meta", {})
        if isinstance(anchor_meta, dict):
            compact_meta = {}
            for key in (
                "schema_id",
                "event_type",
                "token",
                "action_id",
                "source",
                "source_event_type",
                "self_generated",
                "position",
                "current_glyph_index",
                "current_glyph_role",
                "cursor",
                "cursor_before",
                "cursor_after",
                "cursor_index",
                "visible_length",
                "last_visible_token",
                "process_anchor_role",
                "prediction_payload_priority",
                "readout_pattern_id",
                "readout_semantic_role",
                "semantic_frame_role",
                "dynamic_slot_role",
                "action_param_reason",
                "action_param_kind",
                "expected_token",
                "candidate_token",
                "target_token",
                "feedback_outcome",
                "feedback_reward",
                "feedback_punishment",
                "feedback_correctness",
                "feedback_reference_token",
                "feedback_type",
                "reward_value",
                "punishment_value",
                "modality",
                "asset_id",
                "uri",
                "path",
                "numeric_features",
                "features",
            ):
                value = anchor_meta.get(key)
                if value not in (None, "", [], {}):
                    compact_meta[key] = value
            if compact_meta:
                compact["anchor_meta"] = compact_meta
        return compact

    def _compact_asset_ref(self, ref: dict) -> dict:
        return {
            key: ref.get(key)
            for key in ("asset_id", "modality", "uri", "path", "sha256", "bytes", "mime")
            if key in ref and ref.get(key) not in (None, "")
        }

    def _feature_summary(self, features: dict) -> dict:
        return {
            "labels": list(features.get("labels", []) or [])[:128],
            "displays": list(features.get("displays", []) or [])[:128],
            "bigrams": list(features.get("bigrams", []) or [])[:128],
            "sequence_bigrams": list(features.get("sequence_bigrams", []) or [])[:128],
            "focus_labels": list(features.get("focus_labels", []) or [])[:64],
            "relation_tokens": list(features.get("relation_tokens", []) or [])[:128],
        }

    def _posting_tokens(self, features: dict, *, energy_profile: dict[str, float] | None = None) -> dict[str, list[str]]:
        """
        Compact wave-peak index for warm/cold recall.

        Full snapshot content lives in the compressed snapshot blob. Posting rows
        are only the small "peak" handles used to jump to candidate memories, so
        this must never expand every derived feature into SQLite.
        """

        total_cap = max(4, int(self.config.compact_posting_tokens_per_snapshot or 18))
        energy = {str(key): float(value or 0.0) for key, value in dict(energy_profile or {}).items()}

        def _ranked(values: object, *, by_energy: bool = False, weights: dict[str, float] | None = None) -> list[str]:
            rows = self._unique(values)
            if weights:
                rows.sort(key=lambda token: (-float(weights.get(token, 0.0) or 0.0), token))
            elif by_energy and energy:
                rows.sort(key=lambda token: (-float(energy.get(token, 0.0) or 0.0), token))
            return rows

        relation_weights = {
            str(key): float(value or 0.0)
            for key, value in dict(features.get("relation_token_weights", {}) or {}).items()
        }
        sources = [
            ("focus", _ranked(features.get("focus_labels", []), by_energy=True), 4),
            ("label", _ranked(features.get("labels", []), by_energy=True), 8),
            ("relation", _ranked(features.get("relation_tokens", []), weights=relation_weights), 3),
            ("sequence", _ranked(features.get("sequence_bigrams", [])), 2),
            ("bigram", _ranked(features.get("bigrams", [])), 2),
            ("display", _ranked(features.get("displays", [])), 2),
        ]
        result: dict[str, list[str]] = {field: [] for field, _, _ in sources}
        used_total = 0
        seen_pairs: set[tuple[str, str]] = set()
        for field_name, tokens, field_cap in sources:
            for token in tokens:
                if used_total >= total_cap or len(result[field_name]) >= int(field_cap):
                    break
                clean = str(token or "").strip()
                if not clean:
                    continue
                key = (field_name, clean)
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                result[field_name].append(clean)
                used_total += 1
        return {field: tokens for field, tokens in result.items() if tokens}

    def _peak_token_ids(self, tokens_by_field: dict[str, list[str]]) -> list[str]:
        seen = set()
        rows = []
        for field_name, tokens in dict(tokens_by_field or {}).items():
            for token in list(tokens or []):
                clean = str(token or "").strip()
                if not clean:
                    continue
                token_id = self._peak_token_id(str(field_name or ""), clean)
                if token_id in seen:
                    continue
                seen.add(token_id)
                rows.append(token_id)
        return rows

    def _peak_token_id(self, field_name: str, token: str) -> str:
        raw = f"{str(field_name or '')}\x1f{str(token or '')}".encode("utf-8", errors="replace")
        return "p" + hashlib.blake2b(raw, digest_size=10).hexdigest()

    def _unique(self, values: object) -> list[str]:
        seen = set()
        rows = []
        for value in list(values or []):
            clean = str(value or "").strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            rows.append(clean)
        return rows

    def _fixed_vector(self, values: list[float]) -> list[float]:
        dim = max(16, int(self.config.vector_dim))
        fixed = [float(value or 0.0) for value in list(values or [])[:dim]]
        if len(fixed) < dim:
            fixed.extend([0.0] * (dim - len(fixed)))
        return fixed

    def _relation_type(self, token: str) -> str:
        if token.startswith("rel::"):
            parts = token.split("::")
            return parts[1] if len(parts) > 1 else ""
        return ""

    def _database_size_summary(self) -> dict[str, int]:
        now = time.monotonic()
        if self._summary_size_cache and now - self._summary_size_cache_at < 2.0:
            return dict(self._summary_size_cache)
        database_bytes = 0
        wal_bytes = 0
        shm_bytes = 0
        try:
            database_bytes = int(self.path.stat().st_size) if self.path.exists() else 0
            wal = Path(str(self.path) + "-wal")
            shm = Path(str(self.path) + "-shm")
            wal_bytes = int(wal.stat().st_size) if wal.exists() else 0
            shm_bytes = int(shm.stat().st_size) if shm.exists() else 0
        except OSError:
            pass
        table_bytes = database_bytes + wal_bytes + shm_bytes
        self._summary_size_cache = {
            "database_bytes": database_bytes,
            "ap_table_bytes": table_bytes,
            "sqlite_wal_bytes": wal_bytes,
            "sqlite_shm_bytes": shm_bytes,
        }
        self._summary_size_cache_at = now
        return dict(self._summary_size_cache)
