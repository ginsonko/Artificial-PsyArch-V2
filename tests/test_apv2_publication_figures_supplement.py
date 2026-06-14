from __future__ import annotations

import json
from pathlib import Path

from scripts.build_apv2_publication_figures_supplement import build


ROOT = Path(__file__).resolve().parents[1]


def materialize(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT / path


def test_publication_figures_and_supplement_are_generated(tmp_path: Path) -> None:
    output_dir = tmp_path / "figures_pack"
    supplement_path = tmp_path / "APV2_MainPaper_Supplement_Index_20260611.md"

    manifest = build(output_dir=output_dir, supplement_path=supplement_path)

    manifest_path = materialize(manifest["manifest_path"])
    assert manifest_path.exists()
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["figure_count"] == 5
    assert loaded["scope"] == "AP-Core architecture/runtime publication figures and supplement index"
    assert "GL learning proof" in loaded["not_scope"]

    for figure in loaded["figures"]:
        svg = materialize(figure["svg_path"])
        png = materialize(figure["png_path"])
        assert svg.exists()
        assert png.exists()
        assert figure["svg_bytes"] > 0
        assert figure["png_bytes"] > 0
        assert len(figure["svg_sha256"]) == 64
        assert len(figure["png_sha256"]) == 64

    text = supplement_path.read_text(encoding="utf-8")
    assert "APV2 Main Paper Supplement Index" in text
    assert "APV2-BottomLoop-ParamSensitivity-1" in text
    assert "ShortTermSlot-OrderAblation-1" in text
    assert "LongRun-InterruptionRecovery-1" in text
    assert "RhythmSuccessor-Replay-1" in text
    assert "PersistenceBackend-Reload-1" in text
    assert "ArtifactFreeze-1" in text
    assert "DPP/Skill37/product shell are adjacent evidence lines" in text
    assert "not AP-Core proof" in text


def test_publication_figures_do_not_use_conceptual_version_split(tmp_path: Path) -> None:
    output_dir = tmp_path / "figures_pack"
    supplement_path = tmp_path / "supplement.md"
    manifest = build(output_dir=output_dir, supplement_path=supplement_path)

    combined = supplement_path.read_text(encoding="utf-8")
    for figure in manifest["figures"]:
        combined += materialize(figure["svg_path"]).read_text(encoding="utf-8", errors="ignore")

    assert "APV2.1" not in combined
    assert "APV2.2" not in combined
    assert "student_side_llm" not in combined
    if "hidden solver" in combined:
        assert "Forbidden substitutes" in combined
        assert "Do not replace learning" in combined
