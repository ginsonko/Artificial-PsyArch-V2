from __future__ import annotations

import os
import uuid


def _env_bool(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return max(minimum, int(default))
    try:
        return max(minimum, int(raw))
    except ValueError:
        return max(minimum, int(default))


def postgres_config_from_env():
    """
    Build PostgreSQL persistence config from explicit environment variables.

    DSN is intentionally not committed to repo config. AP memory can contain
    private local traces, so deployments should opt in with APV21_PG_DSN.
    """

    dsn = str(os.environ.get("APV21_PG_DSN", "") or "").strip()
    if not dsn:
        return None
    # Local import avoids a config -> memory.persistence -> pg_health -> config
    # cycle when small diagnostic scripts import the config package first.
    from memory.persistence.postgres_store import PostgresPersistenceConfig

    return PostgresPersistenceConfig(
        dsn=dsn,
        run_id=str(os.environ.get("APV21_PG_RUN_ID", "") or "").strip() or str(uuid.uuid4()),
        run_label=str(os.environ.get("APV21_PG_RUN_LABEL", "") or "").strip(),
        vector_dim=_env_int("APV21_PG_VECTOR_DIM", 64, minimum=16),
        synchronous_commit=_env_bool("APV21_PG_SYNC_COMMIT", False),
        resident_hot_snapshots_per_kind=_env_int("APV21_PG_HOT_PER_KIND", 4096, minimum=8),
        warm_prefetch_limit=_env_int("APV21_PG_WARM_PREFETCH", 512, minimum=8),
    )
