from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any


class MultimodalAssetStore:
    """
    Bounded in-memory evidence asset store.

    The state pool and memory snapshots only carry compact asset refs. Payloads
    stay here so high-fidelity media evidence can be resolved by the
    observatory without turning SA rows into a media warehouse.
    """

    def __init__(
        self,
        *,
        max_assets: int = 128,
        store_name: str = "multimodal_asset_store",
        root_dir: str | None = None,
        persist_payloads: bool = False,
        keep_hot_payloads: bool = True,
    ) -> None:
        self.max_assets = max(1, int(max_assets))
        self.store_name = str(store_name or "multimodal_asset_store")
        self.root_dir = Path(root_dir).resolve() if root_dir else None
        self.persist_payloads = bool(persist_payloads and self.root_dir is not None)
        self.keep_hot_payloads = bool(keep_hot_payloads)
        if self.persist_payloads and self.root_dir is not None:
            self.root_dir.mkdir(parents=True, exist_ok=True)
        self._records: OrderedDict[str, dict] = OrderedDict()
        self._next_id = 1

    def put_bytes(
        self,
        *,
        asset_type: str,
        modality: str,
        tick_index: int,
        payload: bytes,
        encoding: str,
        scope: str = "global",
        focus_distance: float | None = None,
        fidelity_level: str = "mid",
        summary_features: dict | None = None,
        retention_policy: dict | None = None,
    ) -> dict:
        data = bytes(payload or b"")
        digest = hashlib.sha256(data).hexdigest()
        asset_id = f"asset-{int(tick_index):06d}-{self._next_id:06d}-{digest[:12]}"
        self._next_id += 1
        payload_path = self._write_payload_file(
            asset_id=asset_id,
            modality=str(modality or "unknown"),
            encoding=str(encoding or "bytes"),
            payload=data,
        )
        ref = {
            "schema_id": "multimodal_evidence_asset_ref/v1",
            "asset_id": asset_id,
            "asset_type": str(asset_type or "unknown_asset"),
            "modality": str(modality or "unknown"),
            "tick_index": int(tick_index),
            "scope": str(scope or "global"),
            "focus_distance": None if focus_distance is None else float(focus_distance),
            "fidelity_level": str(fidelity_level or "mid"),
            "payload_ref": {
                "store": self.store_name,
                "encoding": str(encoding or "bytes"),
                "sha256": digest,
                "byte_length": len(data),
                "storage_tier": "disk_hot_cache" if payload_path else "memory_hot_cache",
                "path": str(payload_path) if payload_path else "",
            },
            "summary_features": deepcopy(dict(summary_features or {})),
            "retention_policy": deepcopy(dict(retention_policy or {})),
        }
        record = deepcopy(ref)
        record["schema_id"] = "multimodal_evidence_asset/v1"
        if self.keep_hot_payloads or not payload_path:
            record["_payload_bytes"] = data
        else:
            record["_payload_bytes"] = b""
        self._records[asset_id] = record
        self._records.move_to_end(asset_id)
        self._write_metadata_file(asset_id=asset_id, modality=str(modality or "unknown"), record=record)
        self._evict_oldest()
        return deepcopy(ref)

    def put_json(
        self,
        *,
        asset_type: str,
        modality: str,
        tick_index: int,
        payload: dict,
        scope: str = "global",
        focus_distance: float | None = None,
        fidelity_level: str = "low",
        summary_features: dict | None = None,
        retention_policy: dict | None = None,
    ) -> dict:
        data = json.dumps(payload or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return self.put_bytes(
            asset_type=asset_type,
            modality=modality,
            tick_index=tick_index,
            payload=data,
            encoding="json",
            scope=scope,
            focus_distance=focus_distance,
            fidelity_level=fidelity_level,
            summary_features=summary_features,
            retention_policy=retention_policy,
        )

    def get(self, asset_id: str, *, include_payload: bool = False) -> dict | None:
        clean = str(asset_id or "")
        if not clean:
            return None
        record = self._records.get(clean)
        if record is None:
            record = self._load_metadata_by_id(clean)
        if record is None:
            return None
        self._records.move_to_end(clean)
        result = deepcopy(record)
        payload = result.pop("_payload_bytes", b"")
        if include_payload:
            if not payload:
                payload = self._read_payload_from_ref(result)
            result["payload_bytes"] = payload
        return result

    def refs_for_tick(self, tick_index: int) -> list[dict]:
        tick = int(tick_index)
        return [
            self._ref_from_record(record)
            for record in self._records.values()
            if int(record.get("tick_index", -1) or -1) == tick
        ]

    def summary(self) -> dict:
        return {
            "schema_id": "multimodal_asset_store_summary/v1",
            "store": self.store_name,
            "hot_asset_count": len(self._records),
            "max_hot_assets": self.max_assets,
            "persistent": bool(self.persist_payloads),
            "root_dir": str(self.root_dir) if self.root_dir is not None else "",
            "latest_asset_ids": list(self._records.keys())[-8:],
        }

    def _ref_from_record(self, record: dict[str, Any]) -> dict:
        clean = deepcopy(record)
        clean.pop("_payload_bytes", None)
        clean["schema_id"] = "multimodal_evidence_asset_ref/v1"
        return clean

    def _evict_oldest(self) -> None:
        while len(self._records) > self.max_assets:
            self._records.popitem(last=False)

    def _write_payload_file(self, *, asset_id: str, modality: str, encoding: str, payload: bytes) -> Path | None:
        if not self.persist_payloads or self.root_dir is None:
            return None
        extension = self._extension_for_encoding(encoding)
        folder = self.root_dir / self._safe_segment(modality)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self._safe_segment(asset_id)}.{extension}"
        path.write_bytes(payload)
        return path.resolve()

    def _write_metadata_file(self, *, asset_id: str, modality: str, record: dict) -> None:
        if not self.persist_payloads or self.root_dir is None:
            return
        folder = self.root_dir / self._safe_segment(modality)
        folder.mkdir(parents=True, exist_ok=True)
        metadata = deepcopy(record)
        metadata.pop("_payload_bytes", None)
        path = folder / f"{self._safe_segment(asset_id)}.asset.json"
        path.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")

    def _load_metadata_by_id(self, asset_id: str) -> dict | None:
        if self.root_dir is None:
            return None
        matches = list(self.root_dir.glob(f"*/*{self._safe_segment(asset_id)}.asset.json"))
        if not matches:
            matches = list(self.root_dir.glob(f"**/{self._safe_segment(asset_id)}.asset.json"))
        if not matches:
            return None
        try:
            record = json.loads(matches[0].read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(record, dict):
            return None
        record["_payload_bytes"] = b""
        self._records[asset_id] = record
        self._records.move_to_end(asset_id)
        self._evict_oldest()
        return record

    def _read_payload_from_ref(self, record: dict) -> bytes:
        payload_ref = dict((record or {}).get("payload_ref", {}) or {})
        path_text = str(payload_ref.get("path", "") or "")
        if not path_text:
            return b""
        path = Path(path_text)
        if not path.exists() or not path.is_file():
            return b""
        data = path.read_bytes()
        expected = str(payload_ref.get("sha256", "") or "")
        if expected and hashlib.sha256(data).hexdigest() != expected:
            return b""
        return data

    def _extension_for_encoding(self, encoding: str) -> str:
        clean = str(encoding or "bytes").strip().lower()
        if clean in {"png", "jpg", "jpeg", "webp", "wav", "json"}:
            return "jpg" if clean == "jpeg" else clean
        return "bin"

    def _safe_segment(self, value: str) -> str:
        clean = str(value or "unknown")
        allowed = []
        for char in clean:
            if char.isalnum() or char in {"-", "_", "."}:
                allowed.append(char)
            else:
                allowed.append("_")
        return "".join(allowed).strip("._") or "unknown"
