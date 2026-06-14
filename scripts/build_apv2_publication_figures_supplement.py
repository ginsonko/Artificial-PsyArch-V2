from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_BASE = ROOT / "outputs"
DEFAULT_SUPPLEMENT = ROOT / "docs" / "APV2_MainPaper_Supplement_Index_20260611.md"
P1_FIGURE_SOURCE = ROOT / "outputs" / "apv2_p1_hardening_materials_20260611_000013" / "figures"


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    title: str
    source: str
    draw: Callable[[plt.Axes], None]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def display_path(path: str | Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def add_box(ax: plt.Axes, xy: tuple[float, float], text: str, width: float = 0.19, height: float = 0.11,
            face: str = "#f7fafc", edge: str = "#2d3748", fontsize: int = 9) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.018",
        linewidth=1.25,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color="#1a202c", wrap=True)


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float],
              color: str = "#4a5568", rad: float = 0.0, lw: float = 1.4) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def style_axes(ax: plt.Axes, title: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.96, title, fontsize=12, fontweight="bold", color="#111827", va="top")


def draw_runtime_loop(ax: plt.Axes) -> None:
    style_axes(ax, "F1. APV2 runtime loop")
    positions = {
        "Sensors\nstate input": (0.12, 0.66),
        "Dual-energy\nstate pool": (0.32, 0.66),
        "Fast B/C\nrecall": (0.52, 0.66),
        "Attention\nselection": (0.72, 0.66),
        "Short-term\nnarrative slot": (0.72, 0.40),
        "Slow recall\nreadback": (0.52, 0.40),
        "Action\ncompetition": (0.32, 0.40),
        "Feedback\nwriteback": (0.12, 0.40),
    }
    for i, (label, xy) in enumerate(positions.items()):
        add_box(ax, xy, label, face="#edf2f7" if i % 2 == 0 else "#ebf8ff")
    chain = list(positions.values())
    for start, end in zip(chain[:4], chain[1:4]):
        add_arrow(ax, (start[0] + 0.095, start[1]), (end[0] - 0.095, end[1]))
    add_arrow(ax, (0.72, 0.605), (0.72, 0.455))
    add_arrow(ax, (0.625, 0.40), (0.615, 0.40))
    add_arrow(ax, (0.425, 0.40), (0.415, 0.40))
    add_arrow(ax, (0.225, 0.40), (0.215, 0.40))
    add_arrow(ax, (0.12, 0.455), (0.12, 0.605), rad=-0.25)
    add_arrow(ax, (0.72, 0.455), (0.39, 0.62), color="#2563eb", rad=0.18)
    ax.text(0.50, 0.20, "All materials enter as SA; action feedback returns to the next tick.",
            ha="center", fontsize=9, color="#374151")


def draw_state_pool_slot(ax: plt.Axes) -> None:
    style_axes(ax, "F2. State pool vs short-term narrative slot")
    add_box(ax, (0.24, 0.66), "State pool\nunordered bounded\nenergy field", width=0.26, height=0.17, face="#fef3c7")
    add_box(ax, (0.24, 0.37), "Fast system\nSA energy matching\nB/C recall", width=0.26, height=0.15, face="#fde68a")
    add_box(ax, (0.72, 0.72), "Tick attention\npacket", width=0.24, height=0.12, face="#e0f2fe")
    add_box(ax, (0.72, 0.52), "Short-term slot\nordered narrative\npacket", width=0.28, height=0.16, face="#dbeafe")
    add_box(ax, (0.72, 0.30), "Slow system\norder + continuity\nreadback", width=0.28, height=0.15, face="#bfdbfe")
    add_arrow(ax, (0.24, 0.575), (0.24, 0.455))
    add_arrow(ax, (0.72, 0.66), (0.72, 0.61))
    add_arrow(ax, (0.72, 0.44), (0.72, 0.38))
    add_arrow(ax, (0.61, 0.52), (0.38, 0.62), color="#2563eb", rad=0.10)
    add_arrow(ax, (0.37, 0.37), (0.58, 0.30), color="#4b5563", rad=-0.10)
    ax.text(0.50, 0.16, "The slot injects virtual-energy inner-sense SA each tick.",
            ha="center", fontsize=9, color="#374151")


def draw_residual_recall(ax: plt.Axes) -> None:
    style_axes(ax, "F3. Residual B recall absorption")
    xs = [0.12, 0.34, 0.56, 0.78]
    labels = [
        "Mixed query\nA+B+C+E",
        "Round 1\nwinner AB",
        "Round 2\nwinner C",
        "Round 3\nwinner E",
    ]
    colors = ["#f8fafc", "#dcfce7", "#dbeafe", "#fce7f3"]
    for x, label, color in zip(xs, labels, colors):
        add_box(ax, (x, 0.62), label, width=0.18, height=0.14, face=color)
    for i in range(3):
        add_arrow(ax, (xs[i] + 0.09, 0.62), (xs[i + 1] - 0.09, 0.62))
    add_box(ax, (0.34, 0.36), "Matched SA\nmass absorbed", width=0.22, height=0.12, face="#ecfccb")
    add_box(ax, (0.56, 0.36), "Residual query\nmakes C salient", width=0.22, height=0.12, face="#e0f2fe")
    add_box(ax, (0.78, 0.36), "Residual mass\nkeeps declining", width=0.22, height=0.12, face="#fae8ff")
    add_arrow(ax, (0.34, 0.55), (0.34, 0.43), color="#16a34a")
    add_arrow(ax, (0.56, 0.55), (0.56, 0.43), color="#2563eb")
    add_arrow(ax, (0.78, 0.55), (0.78, 0.43), color="#9333ea")
    ax.text(0.50, 0.17, "One winner per round; matched SA are weakened only for residual scoring.",
            ha="center", fontsize=9, color="#374151")


def draw_successor_rhythm(ax: plt.Axes) -> None:
    style_axes(ax, "F4. Successor lag and rhythm replay")
    add_box(ax, (0.18, 0.70), "Current B\nobject", face="#f8fafc")
    add_box(ax, (0.45, 0.78), "Lag 1 peak\nkernel 1.00", face="#dcfce7")
    add_box(ax, (0.45, 0.58), "Lag 2 drop\nkernel 0.42", face="#fef3c7")
    add_box(ax, (0.45, 0.38), "Lag tail\nkernel decays", face="#fee2e2")
    add_arrow(ax, (0.275, 0.70), (0.355, 0.78), color="#16a34a")
    add_arrow(ax, (0.275, 0.69), (0.355, 0.58), color="#ca8a04")
    add_arrow(ax, (0.275, 0.68), (0.355, 0.38), color="#dc2626")
    add_box(ax, (0.78, 0.70), "Periodic focus\npulses", face="#e0f2fe")
    add_box(ax, (0.78, 0.50), "rhythmfelt\nphase expectation", width=0.22, face="#dbeafe")
    add_box(ax, (0.78, 0.30), "short_term_slot\nrhythm row", width=0.22, face="#bfdbfe")
    add_arrow(ax, (0.78, 0.64), (0.78, 0.56), color="#2563eb")
    add_arrow(ax, (0.78, 0.44), (0.78, 0.36), color="#2563eb")
    ax.plot([0.67, 0.56], [0.50, 0.53], color="#2563eb", linewidth=1.4, linestyle=(0, (4, 3)))
    ax.text(0.61, 0.46, "phase gates\nsuccessor salience", ha="center", va="top", fontsize=7, color="#2563eb")
    ax.text(0.50, 0.14, "Clear successor peaks support replay; weak peaks favor reread or aggregation.",
            ha="center", fontsize=9, color="#374151")


def draw_evidence_layer(ax: plt.Axes) -> None:
    style_axes(ax, "F5. Evidence layer split")
    layers = [
        ("AP-Core runtime", "P0/P1 bottom-loop\nfigures, ablations,\npersistence"),
        ("AP-Core tasks", "KeySuite / STP\nstrict task evidence"),
        ("GL learning", "learning protocol,\nDPP, Skill37"),
        ("Product shell", "desktop pet / UI\nintegration demos"),
    ]
    y_values = [0.76, 0.58, 0.40, 0.22]
    colors = ["#dcfce7", "#dbeafe", "#fef3c7", "#fee2e2"]
    for (name, desc), y, color in zip(layers, y_values, colors):
        add_box(ax, (0.25, y), name, width=0.26, height=0.10, face=color)
        add_box(ax, (0.65, y), desc, width=0.36, height=0.12, face="#f8fafc", fontsize=8)
        add_arrow(ax, (0.38, y), (0.47, y), color="#4b5563")
    ax.text(0.50, 0.09, "Adjacent evidence can inform the paper, but it must not replace AP-Core proof.",
            ha="center", fontsize=9, color="#374151")


def build_specs() -> list[FigureSpec]:
    return [
        FigureSpec("F1", "APV2 runtime loop", str(P1_FIGURE_SOURCE / "apv2_runtime_loop.mmd"), draw_runtime_loop),
        FigureSpec("F2", "State pool vs short-term narrative slot", str(P1_FIGURE_SOURCE / "state_pool_vs_short_term_slot.mmd"), draw_state_pool_slot),
        FigureSpec("F3", "Residual B recall absorption", str(P1_FIGURE_SOURCE / "residual_b_recall_absorption.mmd"), draw_residual_recall),
        FigureSpec("F4", "Successor lag and rhythm replay", str(P1_FIGURE_SOURCE / "successor_lag_rhythm_replay.mmd"), draw_successor_rhythm),
        FigureSpec("F5", "Evidence layer split", str(P1_FIGURE_SOURCE / "evidence_layer_split.mmd"), draw_evidence_layer),
    ]


def render_figure(spec: FigureSpec, figures_dir: Path) -> dict[str, object]:
    svg_path = figures_dir / f"{spec.figure_id.lower()}_{spec.title.lower().replace(' ', '_')}.svg"
    png_path = figures_dir / f"{spec.figure_id.lower()}_{spec.title.lower().replace(' ', '_')}.png"
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=180)
    fig.patch.set_facecolor("white")
    spec.draw(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(svg_path, format="svg", bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, format="png", bbox_inches="tight", facecolor="white", dpi=220)
    plt.close(fig)
    return {
        "figure_id": spec.figure_id,
        "title": spec.title,
        "source": display_path(spec.source),
        "svg_path": display_path(svg_path),
        "png_path": display_path(png_path),
        "svg_bytes": svg_path.stat().st_size,
        "png_bytes": png_path.stat().st_size,
        "svg_sha256": sha256_file(svg_path),
        "png_sha256": sha256_file(png_path),
    }


def supplement_text(manifest: dict[str, object]) -> str:
    figures = manifest["figures"]
    figure_rows = "\n".join(
        f"| {f['figure_id']} | {f['title']} | `{f['svg_path']}` | `{f['png_path']}` |"
        for f in figures
    )
    return f"""# APV2 Main Paper Supplement Index

Date: 2026-06-11

This index connects the concise APV2 architecture/runtime main paper to the long technical report and local evidence artifacts. It is an index, not a second copy of the full report.

## 1. Source Relationship

| item | path | role |
|---|---|---|
| Main paper draft | `docs/APV2_MainPaper_ArchitectureRuntime_Draft_20260611.md` | concise architecture/runtime manuscript |
| Master technical report | `docs/APV21_PublicPaper_InitialDraft_v1_0n_20260610.md` | long-form source, appendix material, full evidence context |
| P0 report | `outputs/apv2_bottom_loop_p0_materials_20260610_234817/apv2_bottom_loop_p0_materials_report.md` | parameter sensitivity, order ablation, defaults, tick trace |
| P1 report | `outputs/apv2_p1_hardening_materials_20260611_000013/apv2_p1_hardening_materials_report.md` | long-run recovery, rhythm replay, persistence reload, artifact freeze |

## 2. Figure Inventory

| id | title | svg | png |
|---|---|---|---|
{figure_rows}

Manifest: `{display_path(manifest['manifest_path'])}`

## 3. Main Paper To Technical Report Map

| main paper section | technical report support | evidence artifact | boundary note |
|---|---|---|---|
| Abstract | Summary and APV2 bottom-loop additions | P0/P1 reports | Claims runtime mechanisms, not full open-world mastery |
| 1. Introduction | Chapter 1.0-1.8 | technical report chapter 1 | APV2 is positioned as continuous cognition runtime |
| 2. Minimal objects | Chapter 2.2-2.8 | technical report chapter 2 | External fields are ordinary SA, not AP-native feelings |
| 3. Runtime architecture | Chapter 3.0-3.5 | F1/F2 figures | GL and product shell stay outside AP-Core proof |
| 4. Core mechanisms | Chapter 4.1-4.10.1 | P0/P1 reports, F3/F4 | Residual recall and successor lag are mechanisms, not macro routes |
| 5. P0/P1 evidence | Chapter 6.12 and recent P0/P1 reports | P0/P1 JSON/report artifacts | Evidence supports AP-Core bottom-loop dynamics |
| 6. Related systems | Chapter 7 and 8.2-8.5 | technical report discussion | LLMs are complementary carriers/teachers/tools |
| 7. Evidence boundary | Chapter 1.6, 5, 6.11, 9 | F5 evidence layer figure | DPP/Skill37/product shell are adjacent evidence lines |
| 8. Conclusion | Chapter 8 and 9 | technical report synthesis | Keep the claim positive but bounded |

## 4. P0/P1 Evidence Map

| evidence | result | manuscript use |
|---|---|---|
| `APV2-BottomLoop-ParamSensitivity-1` | `16/16 pass` | bottom-loop mechanisms remain qualitatively stable under conservative parameter perturbations |
| `ShortTermSlot-OrderAblation-1` | full-order margin `18.2466`, without-order margin `9.0539` | order is a soft bias, not a hard gate |
| `LongRun-InterruptionRecovery-1` | interruptions `2`, resumptions `2`, final slot virtual mass `1.3715` | short-term narrative traces can recover after controlled interruption |
| `RhythmSuccessor-Replay-1` | lag 1 `1.0`, lag 2 `0.42`, lag 4 `0.172` | successor shaping has a next-tick peak and decaying tail |
| `PersistenceBackend-Reload-1` | warm-load loaded `3`, JSONL SHA-256 recorded | MemoryStore crosses a real local file persistence boundary |
| `ArtifactFreeze-1` | local manifest `12` entries | local pre-public traceability exists |

## 5. Boundary Notes

| line | correct use |
|---|---|
| AP-Core runtime | Use P0/P1 to support bottom-loop mechanism claims |
| AP-Core tasks | Use KeySuite/STP only as adjacent taredacted-test-key context unless Paper 2 expands it |
| GL learning | Use DPP/Skill37 only after GL-side teacher-off/cold retest and no-leakage audit |
| Product shell | Use desktop pet or UI demos as product/integration evidence, not AP-Core proof |
| Cognitive feelings | Treat them as process-grounded SA generated from internal process quantities |
| Forbidden substitutes | Do not replace learning with answer tables, regex routes, hidden solvers, student-side LLM, or full-sentence macros |

## 6. Next Supplement Work

1. Convert this index into a venue-specific appendix after choosing a target format.
2. Add static line-number anchors if the technical report is frozen.
3. Add public artifact freeze commit/tag/hash when the release package is ready.
4. Add GL learning evidence only as a separate appendix or companion paper once GL validation is complete.
"""


def build(output_dir: Path, supplement_path: Path) -> dict[str, object]:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")
    figure_records = [render_figure(spec, figures_dir) for spec in build_specs()]
    manifest_path = output_dir / "figure_manifest.json"
    manifest: dict[str, object] = {
        "schema_id": "apv2_publication_figures_supplement/v1",
        "generated_at": generated_at,
        "scope": "AP-Core architecture/runtime publication figures and supplement index",
        "not_scope": "GL learning proof / DPP / Skill37 / product-shell proof",
        "figure_count": len(figure_records),
        "figures": figure_records,
        "supplement_path": display_path(supplement_path),
        "manifest_path": display_path(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    supplement_path.parent.mkdir(parents=True, exist_ok=True)
    supplement_path.write_text(supplement_text(manifest), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--supplement-path", type=Path, default=DEFAULT_SUPPLEMENT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_BASE / f"apv2_publication_figures_supplement_{stamp}"
    else:
        output_dir = args.output_dir
    manifest = build(output_dir=output_dir, supplement_path=args.supplement_path)
    print(json.dumps({
        "output_dir": str(output_dir),
        "manifest_path": manifest["manifest_path"],
        "supplement_path": manifest["supplement_path"],
        "figure_count": manifest["figure_count"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
