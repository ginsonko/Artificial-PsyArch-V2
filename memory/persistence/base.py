from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PersistenceWriteResult:
    ok: bool
    backend: str
    operation: str
    rows_written: int = 0
    error: str = ""


class MemoryPersistenceAdapter(Protocol):
    """
    Authoritative memory sink for AP snapshots.

    This adapter deliberately receives already-built white-box payloads instead
    of owning cognition itself. SQLite/PostgreSQL are durable truth layers;
    posting, ANN, numeric, relation and online-learning stores remain derived
    runtime views that can be rebuilt from these writes.
    """

    backend_name: str

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
        ...

    def summary(self) -> dict:
        ...

    def write_runtime_state(self, *, state: dict) -> PersistenceWriteResult:
        ...

    def load_runtime_state(self) -> dict | None:
        ...

    def discard_pending(self, *, reason: str = "") -> dict:
        ...


class NullMemoryPersistence:
    backend_name = "none"

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
        return PersistenceWriteResult(ok=True, backend=self.backend_name, operation="write_snapshot", rows_written=0)

    def summary(self) -> dict:
        return {
            "backend": self.backend_name,
            "enabled": False,
            "meaning": "runtime_only_memory;no_authoritative_persistence_attached",
        }

    def write_runtime_state(self, *, state: dict) -> PersistenceWriteResult:
        return PersistenceWriteResult(ok=True, backend=self.backend_name, operation="write_runtime_state", rows_written=0)

    def load_runtime_state(self) -> dict | None:
        return None

    def discard_pending(self, *, reason: str = "") -> dict:
        return {
            "backend": self.backend_name,
            "discarded_snapshots": 0,
            "discarded_runtime_state": False,
            "reason": str(reason or ""),
        }

    def load_recent_snapshots(self, *, memory_kind: str | None = None, limit_per_kind: int = 128) -> list[dict]:
        return []

    def snapshot_by_id(self, memory_id: str) -> dict | None:
        return None
