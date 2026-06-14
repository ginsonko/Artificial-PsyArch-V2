from __future__ import annotations


def round4(value: float) -> float:
    return round(float(value), 4)


def flatten_payload_values(values: object, *, limit: int = 4096) -> list[float]:
    rows: list[float] = []

    def walk(value: object) -> None:
        if len(rows) >= limit:
            return
        if isinstance(value, dict):
            for key in sorted(value):
                walk(value[key])
                if len(rows) >= limit:
                    break
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
                if len(rows) >= limit:
                    break
            return
        try:
            rows.append(round4(float(value)))
        except (TypeError, ValueError):
            rows.append(0.0)

    walk(values)
    return rows[: max(1, int(limit))]


def make_reconstruction_payload(
    *,
    modality: str,
    channel: str,
    scope: str,
    fidelity_level: str,
    summary_vector: list[float],
    payload_shape: list[int],
    payload_values: object,
    compression: str = "none",
    energy_binding: str = "real_energy",
    sampling_precision: float = 1.0,
    payload_limit: int = 4096,
) -> dict:
    return {
        "schema_id": "reconstruction_payload/v1",
        "modality": str(modality or ""),
        "channel": str(channel or ""),
        "scope": str(scope or ""),
        "fidelity_level": str(fidelity_level or ""),
        "summary_vector": [round4(float(value or 0.0)) for value in list(summary_vector or [])[:64]],
        "payload_shape": [int(value) for value in list(payload_shape or [])],
        "payload_values": flatten_payload_values(payload_values, limit=payload_limit),
        "compression": str(compression or "none"),
        "energy_binding": str(energy_binding or "real_energy"),
        "sampling_precision": round4(float(sampling_precision)),
    }


def payload_summary_vector(payload: dict, *, limit: int = 64) -> list[float]:
    if not isinstance(payload, dict):
        return []
    summary = payload.get("summary_vector", [])
    if not isinstance(summary, (list, tuple)):
        return []
    rows = []
    for value in list(summary)[: max(1, int(limit))]:
        try:
            rows.append(round4(float(value)))
        except (TypeError, ValueError):
            rows.append(0.0)
    return rows
