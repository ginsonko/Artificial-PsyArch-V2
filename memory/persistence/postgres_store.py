from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field

from .base import PersistenceWriteResult
from .schema import build_postgres_schema_sql


def _json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _loads(payload: object) -> dict:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str):
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            return {}
        return dict(value) if isinstance(value, dict) else {}
    return {}


@dataclass
class PostgresPersistenceConfig:
    """
    PostgreSQL-first persistence configuration.

    AP should not require all historical memory in RAM. The resident limits here
    describe the intended warm-load policy for real deployments: PostgreSQL
    keeps the authoritative history, while MemoryStore rebuilds only a bounded
    hot window and taredacted-test-key warm set into runtime indexes.
    """

    dsn: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    run_label: str = ""
    vector_dim: int = 64
    synchronous_commit: bool = False
    resident_hot_snapshots_per_kind: int = 4096
    warm_prefetch_limit: int = 512


class PostgresMemoryPersistence:
    """
    Authoritative PostgreSQL/pgvector writer for APV2.1 memory.

    The implementation imports psycopg lazily so the prototype can still run its
    in-memory tests on machines without PostgreSQL client packages installed.
    Real deployments attach this adapter and call ensure_schema() before use.
    """

    backend_name = "postgresql_pgvector"

    def __init__(self, config: PostgresPersistenceConfig) -> None:
        self.config = config
        self.run_id = str(config.run_id)
        self._conn = None
        self._write_count = 0
        self._error_count = 0
        self._last_error = ""
        self._summary_size_cache_at = 0.0
        self._summary_size_cache: dict[str, int] = {}

    def connect(self):
        if self._conn is not None:
            return self._conn
        try:
            import psycopg  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local deployment
            raise RuntimeError("psycopg is required for PostgresMemoryPersistence") from exc
        conn = psycopg.connect(self.config.dsn)
        conn.autocommit = False
        self._conn = conn
        return conn

    def ensure_schema(self) -> None:
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(build_postgres_schema_sql(vector_dim=self.config.vector_dim))
            cur.execute(
                "INSERT INTO ap_runs(run_id, run_label, config) VALUES (%s, %s, %s::jsonb) ON CONFLICT (run_id) DO NOTHING",
                (
                    self.run_id,
                    self.config.run_label,
                    _json(
                        {
                            "schema": "apv21_postgres_persistence_config/v1",
                            "resident_hot_snapshots_per_kind": int(self.config.resident_hot_snapshots_per_kind),
                            "warm_prefetch_limit": int(self.config.warm_prefetch_limit),
                            "synchronous_commit": bool(self.config.synchronous_commit),
                        }
                    ),
                ),
            )
            if not self.config.synchronous_commit:
                cur.execute("SET LOCAL synchronous_commit TO off")
        conn.commit()

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
        learned_vector: list[float] | None = None,
    ) -> PersistenceWriteResult:
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
            )
        except Exception as exc:  # pragma: no cover - requires live PostgreSQL
            self._error_count += 1
            self._last_error = str(exc)
            if self._conn is not None:
                self._conn.rollback()
            return PersistenceWriteResult(ok=False, backend=self.backend_name, operation="write_snapshot", rows_written=0, error=str(exc))
        self._write_count += 1
        return PersistenceWriteResult(ok=True, backend=self.backend_name, operation="write_snapshot", rows_written=rows)

    def write_runtime_state(self, *, state: dict) -> PersistenceWriteResult:
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ap_runtime_state (
                        run_id uuid PRIMARY KEY,
                        updated_at timestamptz NOT NULL DEFAULT now(),
                        state jsonb NOT NULL DEFAULT '{}'::jsonb
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO ap_runtime_state(run_id, updated_at, state)
                    VALUES (%s, now(), %s::jsonb)
                    ON CONFLICT (run_id)
                    DO UPDATE SET updated_at = EXCLUDED.updated_at, state = EXCLUDED.state
                    """,
                    (self.run_id, _json(dict(state or {}))),
                )
            conn.commit()
            return PersistenceWriteResult(ok=True, backend=self.backend_name, operation="write_runtime_state", rows_written=1)
        except Exception as exc:  # pragma: no cover - requires live PostgreSQL
            self._error_count += 1
            self._last_error = str(exc)
            if self._conn is not None:
                self._conn.rollback()
            return PersistenceWriteResult(ok=False, backend=self.backend_name, operation="write_runtime_state", rows_written=0, error=str(exc))

    def load_runtime_state(self) -> dict | None:
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT state
                    FROM ap_runtime_state
                    WHERE run_id = %s
                    """,
                    (self.run_id,),
                )
                row = cur.fetchone()
        except Exception:
            return None
        if not row:
            return None
        payload = row[0] if not isinstance(row, dict) else row.get("state", {})
        return _loads(payload) if not isinstance(payload, dict) else dict(payload)

    def queue_runtime_state_supplier(self, supplier, *, reason: str = "") -> bool:
        # PostgreSQL runtime state is written synchronously on flush; the
        # desktop scheduler only relies on this hook existing.
        return callable(supplier)

    def load_recent_snapshots(self, *, memory_kind: str | None = None, limit_per_kind: int | None = None) -> list[dict]:
        """
        Load a bounded hot window from PostgreSQL.

        This is the restart path for AP working memory. It intentionally does
        not read the whole database; cold history remains in PostgreSQL until a
        timefelt/replay/audit path asks for it.
        """

        cap = max(1, int(limit_per_kind or self.config.resident_hot_snapshots_per_kind))
        conn = self.connect()
        params: list[object] = [self.run_id]
        kind_filter = ""
        if memory_kind is not None:
            kind_filter = "AND memory_kind = %s"
            params.append(str(memory_kind or ""))
        params.append(cap)
        sql = f"""
        WITH ranked AS (
            SELECT
                snapshot,
                memory_kind,
                tick_index,
                row_number() OVER (PARTITION BY memory_kind ORDER BY tick_index DESC, memory_id DESC) AS rn
            FROM memory_snapshots
            WHERE run_id = %s {kind_filter}
        )
        SELECT snapshot
        FROM ranked
        WHERE rn <= %s
        ORDER BY memory_kind ASC, tick_index ASC
        """
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = [_loads(row[0]) for row in cur.fetchall()]
        return [row for row in rows if row]

    def snapshot_by_id(self, memory_id: str) -> dict | None:
        clean = str(memory_id or "")
        if not clean:
            return None
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT snapshot FROM memory_snapshots WHERE run_id = %s AND memory_id = %s",
                (self.run_id, clean),
            )
            row = cur.fetchone()
        if not row:
            return None
        payload = _loads(row[0])
        return payload or None

    def exact_posting_candidates(
        self,
        *,
        memory_kind: str,
        tokens_by_field: dict[str, list[str]],
        limit: int = 64,
    ) -> list[dict]:
        """
        PostgreSQL-side white-box posting candidate query.

        This is a cold-history candidate source and audit helper. Runtime Bn can
        still use in-memory posting/ANN for fixed tick budgets.
        """

        clauses = []
        params: list[object] = [self.run_id, str(memory_kind or "")]
        for field_name, tokens in dict(tokens_by_field or {}).items():
            clean_tokens = [str(token or "").strip() for token in list(tokens or []) if str(token or "").strip()]
            if not clean_tokens:
                continue
            clauses.append("(token_field = %s AND token = ANY(%s))")
            params.extend([str(field_name or ""), clean_tokens])
        if not clauses:
            return []
        params.append(max(1, int(limit)))
        sql = f"""
        SELECT memory_id, count(*) AS matched, sum(weight) AS posting_score
        FROM memory_posting_tokens
        WHERE run_id = %s AND memory_kind = %s AND ({' OR '.join(clauses)})
        GROUP BY memory_id
        ORDER BY posting_score DESC, matched DESC, memory_id ASC
        LIMIT %s
        """
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        return [
            {
                "memory_id": str(row[0]),
                "matched": int(row[1] or 0),
                "posting_score": float(row[2] or 0.0),
                "candidate_sources": ["postgres_exact_posting"],
            }
            for row in rows
        ]

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
        sql = """
        SELECT source_memory_id, successor_memory_id, observed_count, last_tick_index
        FROM (
            SELECT
                source_memory_id,
                successor_memory_id,
                observed_count,
                last_tick_index,
                row_number() OVER (
                    PARTITION BY source_memory_id
                    ORDER BY observed_count DESC, last_tick_index DESC, successor_memory_id ASC
                ) AS rn
            FROM memory_transitions
            WHERE run_id = %s AND memory_kind = %s AND source_memory_id = ANY(%s)
        ) ranked
        WHERE rn <= %s
        ORDER BY source_memory_id ASC, rn ASC
        """
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(sql, (self.run_id, str(memory_kind or ""), clean_sources, max(1, int(limit_per_source or 8))))
            rows = cur.fetchall()
        return [
            {
                "source_memory_id": str(row[0]),
                "successor_memory_id": str(row[1]),
                "observed_count": int(row[2] or 0),
                "last_tick_index": int(row[3] or 0),
            }
            for row in rows
        ]

    def close(self) -> None:
        if self._conn is None:
            return
        self._conn.close()
        self._conn = None

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
        learned_vector: list[float] | None = None,
    ) -> int:
        conn = self.connect()
        memory_id = str(snapshot.get("memory_id", "") or "")
        memory_kind = str(snapshot.get("memory_kind", "") or "")
        tick_index = int(snapshot.get("tick_index", 0) or 0)
        if not memory_id or not memory_kind:
            raise ValueError("snapshot must contain memory_id and memory_kind")
        rows_written = 0
        with conn.cursor() as cur:
            if not self.config.synchronous_commit:
                cur.execute("SET LOCAL synchronous_commit TO off")
            cur.execute(
                "INSERT INTO ap_ticks(run_id, tick_index) VALUES (%s, %s) ON CONFLICT (run_id, tick_index) DO NOTHING",
                (self.run_id, tick_index),
            )
            rows_written += 1
            cur.execute(
                """
                INSERT INTO memory_snapshots(
                    run_id, memory_id, tick_index, memory_kind, source_text,
                    focus_labels, item_count, state_field_item_count, core_item_count, energy_mass,
                    snapshot, feature_summary
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                ON CONFLICT (run_id, memory_id) DO NOTHING
                """,
                (
                    self.run_id,
                    memory_id,
                    tick_index,
                    memory_kind,
                    str(snapshot.get("source_text", "") or ""),
                    list(snapshot.get("focus_labels", []) or []),
                    len(snapshot.get("items", []) or []),
                    len(snapshot.get("state_field_items", []) or []),
                    len(snapshot.get("core_items", []) or []),
                    float(energy_mass or 0.0),
                    _json(snapshot),
                    _json(
                        {
                            "features": self._feature_summary(features),
                            "energy_profile_preview": dict(list((energy_profile or {}).items())[:64]),
                        }
                    ),
                ),
            )
            rows_written += 1
            for idx, item in enumerate(list(snapshot.get("items", []) or [])):
                if not isinstance(item, dict):
                    continue
                cur.execute(
                    """
                    INSERT INTO memory_snapshot_items(
                        run_id, memory_id, item_index, tick_index, memory_kind,
                        sa_label, display_text, family, source_type,
                        real_energy, virtual_energy, cognitive_pressure, payload
                    )
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT (run_id, memory_id, item_index) DO NOTHING
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
            # Full-SA state field is the Bn main recognition view. It includes
            # external input, feelings, emotions, action/control nodes, and any
            # other high-energy SA without prefix-based philosophical exclusion.
            for idx, item in enumerate(list(snapshot.get("state_field_items", []) or [])):
                if not isinstance(item, dict):
                    continue
                cur.execute(
                    "INSERT INTO memory_state_field_items(run_id, memory_id, state_field_index, sa_label, payload) VALUES (%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING",
                    (self.run_id, memory_id, idx, str(item.get("sa_label", "") or ""), _json(item)),
                )
                rows_written += 1
            # `core_items` is retained as the legacy external-anchor/compat view.
            # It is not the philosophical cognition core after the full-SA
            # correction; future tooling should prefer state_field_items.
            for idx, item in enumerate(list(snapshot.get("core_items", []) or [])):
                if not isinstance(item, dict):
                    continue
                cur.execute(
                    "INSERT INTO memory_core_items(run_id, memory_id, core_index, sa_label, payload) VALUES (%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING",
                    (self.run_id, memory_id, idx, str(item.get("sa_label", "") or ""), _json(item)),
                )
                rows_written += 1
            for field_name, tokens in self._posting_tokens(features).items():
                for token in tokens:
                    cur.execute(
                        "INSERT INTO memory_posting_tokens(run_id, memory_id, memory_kind, token_field, token, weight) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                        (self.run_id, memory_id, memory_kind, field_name, token, 1.0),
                    )
                    rows_written += 1
            cur.execute(
                "INSERT INTO memory_vectors(run_id, memory_id, memory_kind, vector_space, vector, vector_meta) VALUES (%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING",
                (self.run_id, memory_id, memory_kind, "hash_vector", self._vector_literal(vector), _json({"dim": len(vector or [])})),
            )
            rows_written += 1
            learned = list(learned_vector or [])
            if learned and any(abs(float(value or 0.0)) > 1e-12 for value in learned):
                cur.execute(
                    "INSERT INTO memory_vectors(run_id, memory_id, memory_kind, vector_space, vector, vector_meta) VALUES (%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING",
                    (
                        self.run_id,
                        memory_id,
                        memory_kind,
                        "online_learned_vector",
                        self._vector_literal(learned),
                        _json({"dim": len(learned), "source": "OnlineEmbeddingStore.learned_vector"}),
                    ),
                )
                rows_written += 1
            for channel, values in dict(numeric_features or {}).items():
                cur.execute(
                    "INSERT INTO memory_numeric_features(run_id, memory_id, memory_kind, channel, values, feature_meta) VALUES (%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING",
                    (self.run_id, memory_id, memory_kind, str(channel), [float(value or 0.0) for value in list(values or [])], _json({})),
                )
                rows_written += 1
            relation_weights = dict((relation_features or {}).get("relation_token_weights", {}) or {})
            for token, weight in relation_weights.items():
                cur.execute(
                    "INSERT INTO memory_relation_features(run_id, memory_id, memory_kind, relation_token, relation_type, weight, payload) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING",
                    (self.run_id, memory_id, memory_kind, str(token), self._relation_type(str(token)), float(weight or 0.0), _json({})),
                )
                rows_written += 1
            if previous_memory_id:
                cur.execute(
                    """
                    INSERT INTO memory_transitions(run_id, memory_kind, source_memory_id, successor_memory_id, observed_count, last_tick_index)
                    VALUES (%s,%s,%s,%s,1,%s)
                    ON CONFLICT (run_id, memory_kind, source_memory_id, successor_memory_id)
                    DO UPDATE SET observed_count = memory_transitions.observed_count + 1,
                                  last_tick_index = EXCLUDED.last_tick_index
                    """,
                    (self.run_id, memory_kind, str(previous_memory_id), memory_id, tick_index),
                )
                rows_written += 1
            for ref in list(snapshot.get("asset_refs", []) or []):
                if not isinstance(ref, dict):
                    continue
                asset_id = str(ref.get("asset_id", "") or "")
                if not asset_id:
                    continue
                cur.execute(
                    "INSERT INTO memory_asset_refs(run_id, memory_id, asset_id, modality, uri, sha256, payload) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING",
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

    def _feature_summary(self, features: dict) -> dict:
        return {
            "labels": list(features.get("labels", []) or [])[:128],
            "displays": list(features.get("displays", []) or [])[:128],
            "bigrams": list(features.get("bigrams", []) or [])[:128],
            "sequence_bigrams": list(features.get("sequence_bigrams", []) or [])[:128],
            "focus_labels": list(features.get("focus_labels", []) or [])[:64],
            "relation_tokens": list(features.get("relation_tokens", []) or [])[:128],
        }

    def _posting_tokens(self, features: dict) -> dict[str, list[str]]:
        return {
            "label": self._unique(features.get("labels", [])),
            "display": self._unique(features.get("displays", [])),
            "bigram": self._unique(features.get("bigrams", [])),
            "sequence": self._unique(features.get("sequence_bigrams", [])),
            "focus": self._unique(features.get("focus_labels", [])),
            "relation": self._unique(features.get("relation_tokens", [])),
        }

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

    def _vector_literal(self, values: list[float]) -> str:
        fixed = [float(value or 0.0) for value in list(values or [])[: max(16, int(self.config.vector_dim))]]
        if len(fixed) < max(16, int(self.config.vector_dim)):
            fixed.extend([0.0] * (max(16, int(self.config.vector_dim)) - len(fixed)))
        return "[" + ",".join(str(float(value)) for value in fixed) + "]"

    def _relation_type(self, token: str) -> str:
        if token.startswith("rel::"):
            parts = token.split("::")
            return parts[1] if len(parts) > 1 else ""
        return ""

    def summary(self) -> dict:
        size_summary = self._database_size_summary()
        return {
            "backend": self.backend_name,
            "enabled": True,
            "run_id": self.run_id,
            "write_count": int(self._write_count),
            "error_count": int(self._error_count),
            "last_error": self._last_error,
            **size_summary,
            "resident_policy": {
                "load_all_history_into_memory": False,
                "resident_hot_snapshots_per_kind": int(self.config.resident_hot_snapshots_per_kind),
                "warm_prefetch_limit": int(self.config.warm_prefetch_limit),
                "meaning": "postgres_is_authoritative_history;runtime_indexes_are_bounded_rebuildable_working_memory",
            },
        }

    def _database_size_summary(self) -> dict[str, int]:
        now = time.monotonic()
        if self._summary_size_cache and now - self._summary_size_cache_at < 5.0:
            return dict(self._summary_size_cache)
        try:
            conn = self.connect()
            with conn.cursor() as cur:
                cur.execute("SELECT pg_database_size(current_database())")
                database_bytes = int((cur.fetchone() or [0])[0] or 0)
                cur.execute(
                    """
                    SELECT COALESCE(sum(pg_total_relation_size(c.oid)), 0)
                    FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = current_schema()
                      AND c.relkind IN ('r','i','m')
                      AND (c.relname LIKE 'memory_%' OR c.relname LIKE 'ap_%')
                    """
                )
                ap_table_bytes = int((cur.fetchone() or [0])[0] or 0)
        except Exception:
            return {"database_bytes": 0, "ap_table_bytes": 0}
        self._summary_size_cache = {
            "database_bytes": database_bytes,
            "ap_table_bytes": ap_table_bytes,
        }
        self._summary_size_cache_at = now
        return dict(self._summary_size_cache)
