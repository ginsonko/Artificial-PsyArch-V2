# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import base64
import json
import math
import random
import struct
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


SCENE_CONCEPTS = [
    {
        "id": "apple",
        "text": "apple 苹果 红色 苹果",
        "rgb": (212, 54, 52),
        "accent": (88, 168, 92),
        "shape": "apple",
        "audio_freqs": (420.0, 510.0),
    },
    {
        "id": "banana",
        "text": "banana 香蕉 黄色 香蕉",
        "rgb": (232, 208, 68),
        "accent": (132, 96, 44),
        "shape": "banana",
        "audio_freqs": (690.0, 860.0),
    },
    {
        "id": "pear",
        "text": "pear 梨 绿色 梨",
        "rgb": (118, 194, 76),
        "accent": (96, 72, 40),
        "shape": "pear",
        "audio_freqs": (560.0, 660.0),
    },
]


LONG_RUN_PRESETS: dict[str, dict[str, Any]] = {
    "heavy": {
        "vision_patch_budget": 40,
        "vision_focus_patch_budget": 20,
        "vision_raw_state_budget": 64,
        "vision_reconstruction_patch_budget": 1024,
        "hearing_window_budget": 24,
        "text_sensor_budget": 20,
    },
    "reduced_boost": {
        "vision_patch_budget": 20,
        "vision_focus_patch_budget": 10,
        "vision_raw_state_budget": 64,
        "vision_reconstruction_patch_budget": 768,
        "vision_attention_boost_max_extra_raw_budget": 64,
        "vision_attention_boost_max_extra_focus_budget": 6,
        "vision_attention_boost_edge_gain": 1.22,
        "hearing_window_budget": 18,
        "text_sensor_budget": 20,
        "memory_candidate_limit": 192,
        "memory_ann_top_k": 56,
    },
    "lighter": {
        "vision_patch_budget": 16,
        "vision_focus_patch_budget": 8,
        "vision_raw_state_budget": 80,
        "vision_reconstruction_patch_budget": 640,
        "vision_attention_boost_max_extra_raw_budget": 48,
        "vision_attention_boost_max_extra_focus_budget": 4,
        "vision_attention_boost_edge_gain": 1.18,
        "hearing_window_budget": 18,
        "text_sensor_budget": 20,
        "memory_candidate_limit": 176,
        "memory_ann_top_k": 48,
    },
}


def _encode_png_bytes(image: Image.Image) -> bytes:
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _render_scene_image(concept: dict[str, Any], *, variant: int, width: int = 256, height: int = 256) -> bytes:
    rng = random.Random((variant + 1) * 917 + len(str(concept.get("id", ""))) * 113)
    bg = Image.new("RGB", (width, height), color=(18, 22, 30))
    draw = ImageDraw.Draw(bg)
    draw.rectangle((16, 16, width - 16, height - 16), outline=(74, 88, 106), width=2)
    draw.line((24, height - 40, width - 24, height - 40), fill=(220, 224, 232), width=3)

    cx = width * (0.50 + rng.uniform(-0.08, 0.08))
    cy = height * (0.50 + rng.uniform(-0.05, 0.05))
    scale = 0.92 + rng.uniform(-0.06, 0.06)
    rgb = tuple(int(v) for v in concept["rgb"])
    accent = tuple(int(v) for v in concept["accent"])
    shape = str(concept.get("shape", "") or "")

    if shape == "apple":
        w = 118 * scale
        h = 122 * scale
        draw.ellipse((cx - w * 0.50, cy - h * 0.48, cx + w * 0.50, cy + h * 0.50), fill=rgb)
        draw.ellipse((cx - w * 0.12, cy - h * 0.20, cx + w * 0.10, cy + h * 0.02), fill=(236, 150, 150))
        draw.rectangle((cx - 6, cy - h * 0.70, cx + 6, cy - h * 0.38), fill=(106, 74, 42))
        draw.polygon([(cx + 6, cy - h * 0.66), (cx + 34, cy - h * 0.74), (cx + 20, cy - h * 0.48)], fill=accent)
    elif shape == "banana":
        box = (cx - 84 * scale, cy - 58 * scale, cx + 84 * scale, cy + 64 * scale)
        draw.pieslice(box, start=212, end=332, fill=rgb)
        inner = (box[0] + 26 * scale, box[1] + 20 * scale, box[2] - 18 * scale, box[3] - 12 * scale)
        draw.pieslice(inner, start=212, end=332, fill=(18, 22, 30))
        draw.rectangle((box[0] + 12 * scale, cy + 30 * scale, box[0] + 22 * scale, cy + 48 * scale), fill=accent)
        draw.rectangle((box[2] - 16 * scale, cy - 36 * scale, box[2] - 6 * scale, cy - 18 * scale), fill=accent)
    else:
        w = 104 * scale
        h = 132 * scale
        draw.ellipse((cx - w * 0.42, cy - h * 0.20, cx + w * 0.42, cy + h * 0.48), fill=rgb)
        draw.polygon([(cx - w * 0.28, cy - h * 0.06), (cx, cy - h * 0.58), (cx + w * 0.28, cy - h * 0.06)], fill=rgb)
        draw.rectangle((cx - 5, cy - h * 0.72, cx + 5, cy - h * 0.52), fill=(106, 74, 42))
        draw.polygon([(cx + 4, cy - h * 0.68), (cx + 28, cy - h * 0.76), (cx + 16, cy - h * 0.54)], fill=accent)

    text = str(concept.get("id", "") or "")
    draw.text((22, 22), text, fill=(232, 236, 244))
    return _encode_png_bytes(bg)


def _render_scene_audio(concept: dict[str, Any], *, variant: int, duration_sec: float = 0.42, sample_rate: int = 16000) -> bytes:
    rng = random.Random((variant + 1) * 613 + len(str(concept.get("id", ""))) * 71)
    f1, f2 = concept["audio_freqs"]
    amp = 9600 + int(rng.uniform(-800, 800))
    total = int(sample_rate * duration_sec)
    frames = bytearray()
    for i in range(total):
        t = i / max(1, sample_rate)
        env = 0.45 + 0.55 * math.sin(math.pi * i / max(1, total - 1))
        wobble = 1.0 + 0.03 * math.sin(2 * math.pi * (2.0 + rng.uniform(0.0, 0.8)) * t)
        sample = (
            math.sin(2 * math.pi * (f1 * wobble) * t) * 0.62
            + math.sin(2 * math.pi * (f2 * wobble) * t) * 0.30
            + math.sin(2 * math.pi * ((f1 + f2) * 0.5) * t) * 0.12
        )
        pcm = int(amp * env * sample)
        frames += struct.pack("<h", pcm)
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
    return buf.getvalue()


def _mk_item(
    *,
    text: str = "",
    image_bytes: bytes | None = None,
    audio_bytes: bytes | None = None,
    source_type: str,
    scene_id: str = "",
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "text": str(text or ""),
        "source_type": str(source_type or "long_multimodal_scene"),
    }
    if scene_id:
        item["scene_id"] = str(scene_id)
    if image_bytes is not None:
        item["image_b64"] = base64.b64encode(image_bytes).decode("ascii")
    if audio_bytes is not None:
        item["audio_b64"] = base64.b64encode(audio_bytes).decode("ascii")
    return item


def _build_block(
    *,
    concept: dict[str, Any],
    length: int,
    block_index: int,
    mode: str,
) -> list[dict[str, Any]]:
    image_bytes = _render_scene_image(concept, variant=block_index)
    audio_bytes = _render_scene_audio(concept, variant=block_index)
    text = str(concept.get("text", "") or "")
    scene_id = f"{concept['id']}::{block_index}"
    rows: list[dict[str, Any]] = []
    for _ in range(max(1, int(length))):
        include_text = mode in {"multi", "text", "text_vision", "text_audio"}
        include_image = mode in {"multi", "vision", "text_vision", "vision_audio"}
        include_audio = mode in {"multi", "audio", "text_audio", "vision_audio"}
        rows.append(
            _mk_item(
                text=text if include_text else "",
                image_bytes=image_bytes if include_image else None,
                audio_bytes=audio_bytes if include_audio else None,
                source_type=f"long_multimodal_scene::{mode}",
                scene_id=scene_id,
            )
        )
    return rows


def _build_idle(length: int) -> list[dict[str, Any]]:
    return [_mk_item(text="", source_type="long_multimodal_scene::idle") for _ in range(max(1, int(length)))]


def build_dataset(
    *,
    ticks: int,
    tick_interval_ms: int,
    vision_budget: int,
    vision_reconstruction_budget: int,
    preset: str,
    output_path: Path,
) -> None:
    rng = random.Random(20260524)
    block_templates = [
        ("multi", 14, 18),
        ("vision", 10, 14),
        ("audio", 10, 14),
        ("text", 8, 12),
        ("text_vision", 10, 14),
        ("text_audio", 10, 14),
        ("vision_audio", 10, 14),
    ]
    items: list[dict[str, Any]] = []
    block_index = 0
    concept_index = 0
    while len(items) < ticks:
        concept = SCENE_CONCEPTS[concept_index % len(SCENE_CONCEPTS)]
        mode, lo, hi = block_templates[block_index % len(block_templates)]
        block_len = rng.randint(lo, hi)
        items.extend(_build_block(concept=concept, length=block_len, block_index=block_index, mode=mode))
        if len(items) >= ticks:
            break
        idle_len = rng.randint(3, 7)
        items.extend(_build_idle(idle_len))
        block_index += 1
        concept_index += 1
    items = items[: max(1, int(ticks))]
    preset_name = str(preset or "heavy").strip().lower()
    preset_overrides = dict(LONG_RUN_PRESETS.get(preset_name, LONG_RUN_PRESETS["heavy"]))
    payload = {
        "label": f"长程多模态连续场景_{ticks}tick_{preset_name}",
        "mode": "multimodal",
        "tick_interval_ms": int(max(0, tick_interval_ms)),
        "config_overrides": {
            **preset_overrides,
            "vision_patch_budget": int(max(4, preset_overrides.get("vision_patch_budget", vision_budget))),
            "vision_focus_patch_budget": int(max(4, preset_overrides.get("vision_focus_patch_budget", max(8, min(int(vision_budget // 2), 64))))),
            "vision_reconstruction_patch_budget": int(max(16, preset_overrides.get("vision_reconstruction_patch_budget", vision_reconstruction_budget))),
            "observatory_tick_list_limit": min(max(int(ticks), 256), 2048),
        },
        "items": items,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 AP 二期长程多模态连续场景数据集")
    parser.add_argument("--ticks", type=int, default=512, help="tick 数量")
    parser.add_argument("--tick-interval-ms", type=int, default=0, help="tick 间隔毫秒")
    parser.add_argument("--vision-budget", type=int, default=48, help="认知视觉 patch 预算")
    parser.add_argument("--vision-reconstruction-budget", type=int, default=1024, help="视觉重建预算")
    parser.add_argument("--preset", default="heavy", choices=sorted(LONG_RUN_PRESETS.keys()), help="长程配置预设")
    parser.add_argument("--output", default="config/generated_long_multimodal_dataset.json", help="输出路径")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (Path(__file__).resolve().parents[1] / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_dataset(
        ticks=max(1, int(args.ticks)),
        tick_interval_ms=max(0, int(args.tick_interval_ms)),
        vision_budget=max(4, int(args.vision_budget)),
        vision_reconstruction_budget=max(16, int(args.vision_reconstruction_budget)),
        preset=str(args.preset or "heavy"),
        output_path=output_path,
    )
    print(str(output_path))


if __name__ == "__main__":
    main()
