# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sensors.vision_sensor_v1 import VisionSensorV1


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "dynamic_vision_acceptance"


def _round4(value: float) -> float:
    return round(float(value), 4)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(text or ""), encoding="utf-8")


def _sensor() -> VisionSensorV1:
    return VisionSensorV1(
        patch_budget=32,
        focus_patch_budget=16,
        raw_state_budget=256,
        reconstruction_patch_budget=1024,
        edge_candidate_gain=2.0,
        edge_priority_gain=1.5,
        attention_boost_enabled=True,
        attention_boost_max_extra_raw_budget=256,
        attention_boost_max_extra_focus_budget=12,
        attention_boost_min_radius_scale=0.26,
        attention_boost_edge_gain=1.45,
        attention_boost_gaze_sigma_scale=0.48,
        dynamic_track_window=6,
        dynamic_candidate_limit_background=16,
        dynamic_candidate_limit_focus=32,
        dynamic_track_limit=48,
        dynamic_summary_limit=6,
        dynamic_match_threshold=0.44,
        dynamic_track_forget_ticks=4,
    )


def _png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _blank(size: tuple[int, int] = (192, 192), color: tuple[int, int, int] = (18, 18, 18)) -> Image.Image:
    return Image.new("RGB", size, color=color)


def _single_rect_image(x0: int, y0: int, x1: int, y1: int, *, size: tuple[int, int] = (192, 192)) -> Image.Image:
    image = _blank(size=size)
    draw = ImageDraw.Draw(image)
    draw.rectangle((x0, y0, x1, y1), fill=(240, 240, 240))
    return image


def _multi_rect_image(rects: list[tuple[int, int, int, int, tuple[int, int, int]]], *, size: tuple[int, int] = (192, 192)) -> Image.Image:
    image = _blank(size=size)
    draw = ImageDraw.Draw(image)
    for x0, y0, x1, y1, color in rects:
        draw.rectangle((x0, y0, x1, y1), fill=color)
    return image


def _shift_image(image: Image.Image, *, dx: int, dy: int) -> Image.Image:
    shifted = _blank(size=image.size)
    shifted.paste(image, (dx, dy))
    return shifted


def _extract_tick(packet: dict[str, Any], tick_index: int, note: str) -> dict[str, Any]:
    summary = dict(packet.get("dynamic_track_summary", {}) or {})
    stream = dict(packet.get("stream_state", {}) or {})
    tracks = list(packet.get("dynamic_tracks", []) or [])
    dynamic_motion = list(packet.get("dynamic_motion_samples", []) or [])
    return {
        "tick_index": int(tick_index),
        "note": str(note or ""),
        "budget_used": int(packet.get("budget_used", 0) or 0),
        "shape_candidate_count": int(stream.get("shape_candidate_count", 0) or 0),
        "dynamic_track_count": int(summary.get("track_count", 0) or 0),
        "dynamic_object_count": int(summary.get("object_count", 0) or 0),
        "dynamic_salience_mean": _round4(float(summary.get("dynamic_salience_mean", 0.0) or 0.0)),
        "global_motion_dx": _round4(float(summary.get("global_motion_dx", 0.0) or 0.0)),
        "global_motion_dy": _round4(float(summary.get("global_motion_dy", 0.0) or 0.0)),
        "global_motion_speed": _round4(float(summary.get("global_motion_speed", 0.0) or 0.0)),
        "top_tracks": [
            {
                "track_id": str(item.get("track_id", "") or ""),
                "speed": _round4(float(item.get("speed", 0.0) or 0.0)),
                "dynamic_objectness": _round4(float(item.get("dynamic_objectness", 0.0) or 0.0)),
                "motion_coherence": _round4(float(item.get("motion_coherence", 0.0) or 0.0)),
                "boundary_motion_contrast": _round4(float(item.get("boundary_motion_contrast", 0.0) or 0.0)),
                "temporal_persistence": _round4(float(item.get("temporal_persistence", 0.0) or 0.0)),
                "coords": dict(item.get("coords", {}) or {}),
            }
            for item in tracks[:4]
        ],
        "dynamic_motion_preview": [
            {
                "sa_label": str(item.get("sa_label", "") or ""),
                "energy": _round4(float(item.get("energy", 0.0) or 0.0)),
                "motion_speed": _round4(float(((item.get("attributes", {}) or {}).get("motion_speed", 0.0)) or 0.0)),
                "dynamic_objectness": _round4(float(((item.get("attributes", {}) or {}).get("dynamic_objectness", 0.0)) or 0.0)),
            }
            for item in dynamic_motion[:4]
        ],
    }


def _run_sequence(sensor: VisionSensorV1, frames: list[Image.Image], *, source_prefix: str, note_prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for tick_index, image in enumerate(frames):
        packet = sensor.ingest_image_bytes(_png_bytes(image), tick_index=tick_index, source_type=f"{source_prefix}::{tick_index}")
        rows.append(_extract_tick(packet, tick_index, f"{note_prefix}{tick_index}"))
    return rows


def _mean(values: list[float]) -> float:
    return _round4(sum(values) / max(1, len(values)))


def _static_test() -> dict[str, Any]:
    sensor = _sensor()
    frame = _single_rect_image(64, 64, 120, 120)
    rows = _run_sequence(sensor, [frame for _ in range(20)], source_prefix="dyn_static", note_prefix="静态第")
    tail = rows[-10:]
    mean_speed = _mean([float(row["global_motion_speed"]) for row in tail])
    false_motion_peak = max(float(row["dynamic_salience_mean"]) for row in tail)
    passed = mean_speed <= 0.03 and false_motion_peak <= 0.38
    return {
        "case_id": "A_static_zero_false_motion",
        "passed": bool(passed),
        "metrics": {
            "tail_mean_global_motion_speed": mean_speed,
            "tail_peak_dynamic_salience": _round4(false_motion_peak),
            "tail_mean_track_count": _mean([float(row["dynamic_track_count"]) for row in tail]),
        },
        "rows": rows,
    }


def _translation_test() -> dict[str, Any]:
    sensor = _sensor()
    base = _multi_rect_image(
        [
            (18, 24, 52, 64, (220, 220, 220)),
            (84, 36, 132, 80, (180, 180, 180)),
            (42, 112, 158, 150, (250, 250, 250)),
        ]
    )
    frames = [_shift_image(base, dx=i * 4, dy=0) for i in range(12)]
    rows = _run_sequence(sensor, frames, source_prefix="dyn_translation", note_prefix="平移第")
    tail = rows[-8:]
    mean_dx = _mean([float(row["global_motion_dx"]) for row in tail])
    mean_speed = _mean([float(row["global_motion_speed"]) for row in tail])
    mean_objects = _mean([float(row["dynamic_object_count"]) for row in tail])
    passed = mean_dx >= 0.01 and mean_speed >= 0.01 and mean_objects <= 1.5
    return {
        "case_id": "B_global_translation_baseline",
        "passed": bool(passed),
        "metrics": {
            "tail_mean_global_motion_dx": mean_dx,
            "tail_mean_global_motion_speed": mean_speed,
            "tail_mean_dynamic_object_count": mean_objects,
        },
        "rows": rows,
    }


def _single_object_test() -> dict[str, Any]:
    sensor = _sensor()
    frames = [_single_rect_image(20 + i * 6, 72, 52 + i * 6, 110) for i in range(12)]
    rows = _run_sequence(sensor, frames, source_prefix="dyn_single_object", note_prefix="单对象第")
    tail = rows[-8:]
    persistence_peak = max(
        max((float(track.get("temporal_persistence", 0.0) or 0.0) for track in row["top_tracks"]), default=0.0)
        for row in tail
    )
    objectness_peak = max(
        max((float(track.get("dynamic_objectness", 0.0) or 0.0) for track in row["top_tracks"]), default=0.0)
        for row in tail
    )
    speed_peak = max(
        max((float(track.get("speed", 0.0) or 0.0) for track in row["top_tracks"]), default=0.0)
        for row in tail
    )
    passed = persistence_peak >= 0.45 and objectness_peak >= 0.36 and speed_peak >= 0.015
    return {
        "case_id": "C_single_object_relative_motion",
        "passed": bool(passed),
        "metrics": {
            "tail_peak_temporal_persistence": _round4(persistence_peak),
            "tail_peak_dynamic_objectness": _round4(objectness_peak),
            "tail_peak_track_speed": _round4(speed_peak),
        },
        "rows": rows,
    }


def _multi_object_test() -> dict[str, Any]:
    sensor = _sensor()
    frames: list[Image.Image] = []
    for i in range(12):
        frames.append(
            _multi_rect_image(
                [
                    (16 + i * 4, 30, 44 + i * 4, 58, (240, 240, 240)),
                    (126, 94 + i * 3, 156, 126 + i * 3, (210, 210, 210)),
                ]
            )
        )
    rows = _run_sequence(sensor, frames, source_prefix="dyn_multi_object", note_prefix="多对象第")
    tail = rows[-8:]
    dual_track_rows = 0
    speed_divergences: list[float] = []
    for row in tail:
        top_tracks = list(row.get("top_tracks", []) or [])
        if len(top_tracks) >= 2:
            dual_track_rows += 1
            speed_divergences.append(abs(float(top_tracks[0].get("speed", 0.0) or 0.0) - float(top_tracks[1].get("speed", 0.0) or 0.0)))
    mean_divergence = _mean(speed_divergences) if speed_divergences else 0.0
    passed = dual_track_rows >= 3 and mean_divergence >= 0.005
    return {
        "case_id": "D_multi_object_motion_separation",
        "passed": bool(passed),
        "metrics": {
            "tail_dual_track_rows": int(dual_track_rows),
            "tail_mean_speed_divergence": mean_divergence,
        },
        "rows": rows,
    }


def _stress_test() -> dict[str, Any]:
    sensor = _sensor()
    rows: list[dict[str, Any]] = []
    speeds: list[float] = []
    for tick_index in range(180):
        frame = _multi_rect_image(
            [
                (18 + (tick_index * 3) % 96, 18 + (tick_index % 12), 42 + (tick_index * 3) % 96, 44 + (tick_index % 12), (230, 230, 230)),
                (104, 24 + (tick_index * 2) % 92, 136, 60 + (tick_index * 2) % 92, (180, 180, 180)),
                (44 + (tick_index * 5) % 108, 126, 80 + (tick_index * 5) % 108, 154, (250, 250, 250)),
            ]
        )
        packet = sensor.ingest_image_bytes(_png_bytes(frame), tick_index=tick_index, source_type=f"dyn_stress::{tick_index}")
        row = _extract_tick(packet, tick_index, "压力")
        rows.append(row)
        speeds.append(float(row["global_motion_speed"]))
    tail = rows[-40:]
    track_counts = [float(row["dynamic_track_count"]) for row in tail]
    saliences = [float(row["dynamic_salience_mean"]) for row in tail]
    passed = max(track_counts, default=0.0) <= 48 and statistics.mean(saliences or [0.0]) <= 0.9
    return {
        "case_id": "H_scale_and_pressure_stability",
        "passed": bool(passed),
        "metrics": {
            "tail_mean_track_count": _round4(statistics.mean(track_counts or [0.0])),
            "tail_max_track_count": _round4(max(track_counts, default=0.0)),
            "tail_mean_dynamic_salience": _round4(statistics.mean(saliences or [0.0])),
            "overall_mean_global_motion_speed": _round4(statistics.mean(speeds or [0.0])),
        },
        "rows": rows,
    }


@dataclass(frozen=True)
class AcceptanceCase:
    case_id: str
    result: dict[str, Any]


def main() -> None:
    output_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        AcceptanceCase("A_static_zero_false_motion", _static_test()),
        AcceptanceCase("B_global_translation_baseline", _translation_test()),
        AcceptanceCase("C_single_object_relative_motion", _single_object_test()),
        AcceptanceCase("D_multi_object_motion_separation", _multi_object_test()),
        AcceptanceCase("H_scale_and_pressure_stability", _stress_test()),
    ]
    summary = {
        "schema_id": "dynamic_vision_acceptance/v1",
        "schema_version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "case_count": len(cases),
        "passed_count": sum(1 for case in cases if bool(case.result.get("passed", False))),
        "all_passed": all(bool(case.result.get("passed", False)) for case in cases),
        "cases": [case.result for case in cases],
    }
    _write_json(output_dir / "summary.json", summary)

    lines = [
        "# V2 动态视觉对象层验收结果",
        "",
        f"- 生成时间: {summary['generated_at']}",
        f"- 用例总数: {summary['case_count']}",
        f"- 通过数: {summary['passed_count']}",
        f"- 是否全部通过: {'是' if summary['all_passed'] else '否'}",
        "",
    ]
    for case in cases:
        lines.append(f"## {case.case_id}")
        lines.append(f"- 通过: {'是' if case.result.get('passed', False) else '否'}")
        for key, value in dict(case.result.get("metrics", {}) or {}).items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    _write_text(output_dir / "report.md", "\n".join(lines))
    print(str(output_dir))


if __name__ == "__main__":
    main()
