from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_v2_multimodal_teaching_dolphin_probe import MultimodalConcept, _render_markdown_report

RUN_ROOT = REPO_ROOT / "outputs" / "multimodal_teaching_dolphin_probe" / "20260523_210518"
DOC_PATH = REPO_ROOT / "docs" / "V2_多模态教学与海豚训练综合实验报告_2026-05-23.md"


def main() -> None:
    summary = json.loads((RUN_ROOT / "summary.json").read_text(encoding="utf-8"))
    assets = dict(summary.get("assets", {}) or {})
    concepts: list[MultimodalConcept] = []
    for concept_id, asset in assets.items():
        concepts.append(
            MultimodalConcept(
                concept_id=str(concept_id),
                text_label=str(asset.get("text_label", concept_id) or concept_id),
                zh_text=str(asset.get("zh_text", "") or ""),
                rgb=(0, 0, 0),
                shape="regenerated",
                spoken_text=str(asset.get("spoken_text", asset.get("text_label", concept_id)) or concept_id),
                fallback_freqs=(0.0, 0.0),
                tts_voice=str(asset.get("tts_voice", "Microsoft Zira Desktop") or "Microsoft Zira Desktop"),
                tts_rate=int(asset.get("tts_rate", 0) or 0),
            )
        )
    report = _render_markdown_report(
        output_root=RUN_ROOT,
        concepts=concepts,
        assets=assets,
        training=dict(summary.get("training", {}) or {}),
        probes=list(summary.get("probes", []) or []),
        switching_rows=list(summary.get("switching_rows", []) or []),
        observatory_showcase=dict(summary.get("observatory_showcase", {}) or {}),
    )
    (RUN_ROOT / "report.md").write_text(report, encoding="utf-8")
    DOC_PATH.write_text(report, encoding="utf-8")
    print(str(RUN_ROOT / "report.md"))
    print(str(DOC_PATH))


if __name__ == "__main__":
    main()
