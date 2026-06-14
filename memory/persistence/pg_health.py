from __future__ import annotations

import importlib.util
import shutil
import subprocess
from typing import Any

from config.persistence import postgres_config_from_env
from .postgres_store import PostgresMemoryPersistence, PostgresPersistenceConfig


def _check_docker_daemon() -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "docker_not_found"
    try:
        completed = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as exc:
        return False, f"docker_probe_error:{exc}"
    if completed.returncode != 0:
        return False, (completed.stderr or completed.stdout or "docker_daemon_not_ready").strip()
    return True, (completed.stdout or "").strip()


def _base_report(config: PostgresPersistenceConfig | None) -> dict[str, Any]:
    docker_daemon_ok, docker_daemon_detail = _check_docker_daemon()
    psycopg_available = importlib.util.find_spec("psycopg") is not None
    checks = {
        "dsn_configured": bool(config and str(config.dsn or "").strip()),
        "psycopg_available": bool(psycopg_available),
        "psql_available": shutil.which("psql") is not None,
        "pg_isready_available": shutil.which("pg_isready") is not None,
        "docker_available": shutil.which("docker") is not None,
        "docker_daemon_available": bool(docker_daemon_ok),
        "connection_ok": False,
        "schema_ok": False,
        "extension_vector_ok": False,
        "extension_pg_trgm_ok": False,
    }
    report = {
        "schema_id": "apv21_postgres_environment_health/v1",
        "ok": False,
        "ready_for_real_smoke": False,
        "checks": checks,
        "blockers": [],
        "hints": [],
        "details": {
            "docker_daemon": docker_daemon_detail,
            "run_id": str(config.run_id) if config is not None else "",
            "run_label": str(config.run_label) if config is not None else "",
        },
    }
    if not checks["dsn_configured"]:
        report["blockers"].append("APV21_PG_DSN is not set")
        report["hints"].append("Set APV21_PG_DSN=postgresql://user:password@host:5432/dbname")
    if not checks["psycopg_available"]:
        report["blockers"].append("psycopg is not installed")
        report["hints"].append("Install psycopg in the active Python environment before real PostgreSQL smoke tests")
    if not checks["docker_daemon_available"] and checks["docker_available"]:
        report["hints"].append("Docker client exists but daemon is not running; start Docker only if you plan to host PostgreSQL locally")
    if not checks["psql_available"]:
        report["hints"].append("psql is optional for Python smoke tests but useful for manual PostgreSQL diagnosis")
    return report


def check_postgres_environment(
    config: PostgresPersistenceConfig | None = None,
    *,
    connect: bool = False,
    ensure_schema: bool = False,
) -> dict[str, Any]:
    """
    Return a JSON-friendly PostgreSQL readiness report.

    The function separates local tooling, Python dependency, connection, schema,
    and extension checks so failures tell us exactly what is missing.
    """

    cfg = config if config is not None else postgres_config_from_env()
    report = _base_report(cfg)
    checks = report["checks"]
    if not connect:
        report["ok"] = bool(checks["dsn_configured"] and checks["psycopg_available"])
        report["ready_for_real_smoke"] = False
        return report
    if cfg is None or not checks["psycopg_available"]:
        return report
    persistence = PostgresMemoryPersistence(cfg)
    try:
        if ensure_schema:
            persistence.ensure_schema()
            checks["schema_ok"] = True
        conn = persistence.connect()
        checks["connection_ok"] = True
        with conn.cursor() as cur:
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')")
            checks["extension_vector_ok"] = bool(cur.fetchone()[0])
            cur.execute("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm')")
            checks["extension_pg_trgm_ok"] = bool(cur.fetchone()[0])
    except Exception as exc:
        report["blockers"].append(f"postgres_connection_or_schema_failed:{exc}")
        report["hints"].append("Check APV21_PG_DSN, database availability, credentials, and pgvector/pg_trgm extension privileges")
        if persistence._conn is not None:
            persistence._conn.rollback()
    finally:
        try:
            persistence.close()
        except Exception:
            pass
    if checks["connection_ok"] and (not ensure_schema or checks["schema_ok"]) and checks["extension_vector_ok"] and checks["extension_pg_trgm_ok"]:
        report["ok"] = True
        report["ready_for_real_smoke"] = True
    else:
        if checks["connection_ok"] and not checks["extension_vector_ok"]:
            report["blockers"].append("pgvector extension is not available/enabled")
        if checks["connection_ok"] and not checks["extension_pg_trgm_ok"]:
            report["blockers"].append("pg_trgm extension is not available/enabled")
    return report
