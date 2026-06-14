"""Build two publication-grade evidence figures for the APV2 paper / press article.

- F-EV1: four-layer evidence panorama (mechanism / controlled baseline / task / third-party).
- F-EV2: RepeatMap v0.5 controlled pilot — holdout accuracy vs token cost across routes.

Labels are in English so the figures work for both the Chinese and English press
versions; each version provides its own surrounding caption.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _box(ax, xy, w, h, text, face, edge="#2d3748", fontsize=10, fontcolor="#1a202c", weight="normal"):
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.3, edgecolor=edge, facecolor=face,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=fontcolor, weight=weight, wrap=True)


def build_evidence_panorama(out_dir: Path) -> dict:
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.965, "APV2 Evidence Stack: From Mechanism to Cross-Implementation",
            ha="center", va="center", fontsize=15, weight="bold", color="#1a202c")
    ax.text(0.5, 0.918, "A white-box predictive-action runtime, evidenced in four separable layers",
            ha="center", va="center", fontsize=10.5, color="#4a5568", style="italic")

    # Four stacked layers (widest = foundation).
    layers = [
        {
            "y": 0.78, "w": 0.92, "face": "#e8f0fe", "edge": "#3b6fb6",
            "title": "Layer 1 — AP-Core runtime mechanism  (this paper's main claim)",
            "body": "P0/P1/P2: param sensitivity 16/16 · short-term-slot order ablation · interruption recovery ·\n"
                    "successor lag/rhythm · JSONL persistence reload · residual-depth stress · slot grid 108/108 ·\n"
                    "pressure dynamics: text_commit → reread / replace / replay",
        },
        {
            "y": 0.575, "w": 0.78, "face": "#e9f7ef", "edge": "#2f9e63",
            "title": "Layer 2 — Controlled pilot vs real LLM / agent",
            "body": "RepeatMap v0.5 (1424 real calls): AP-style 1.00 @ 0 token · real Claude no-mem ~0.22-0.30 ·\n"
                    "real GPT+memory/tool 0.83-0.85 · LBF1 four-route · LongRun v0.2 re-adaptation",
        },
        {
            "y": 0.40, "w": 0.62, "face": "#fef6e7", "edge": "#d39a23",
            "title": "Layer 3 — AP-Core task evidence",
            "body": "Canonical-KeySuite 8/8 · STP-v2 process anchors:\ncross-surface transfer 1.000, sham-feeling falsification",
        },
        {
            "y": 0.245, "w": 0.46, "face": "#f3eafb", "edge": "#8a4fbf",
            "title": "Layer 4 — Third-party reproduction",
            "body": "ACG-j Rust clean-room:\n8/8 commands, 84/84 lib tests",
        },
    ]
    for layer in layers:
        _box(ax, (0.5, layer["y"]), layer["w"], 0.135, "", layer["face"], edge=layer["edge"])
        ax.text(0.5, layer["y"] + 0.042, layer["title"], ha="center", va="center",
                fontsize=10.5, weight="bold", color="#1a202c")
        ax.text(0.5, layer["y"] - 0.022, layer["body"], ha="center", va="center",
                fontsize=8.4, color="#2d3748")

    # Boundary / provenance footer.
    _box(ax, (0.5, 0.085), 0.92, 0.085, "", "#f7fafc", edge="#718096")
    ax.text(0.5, 0.11, "Boundary & provenance", ha="center", va="center", fontsize=9.5, weight="bold", color="#1a202c")
    ax.text(0.5, 0.065,
            "No answer table · no regex route · no hidden solver · no student-side LLM · no full-sentence macro.\n"
            "Local artifact freeze with SHA-256; GL open-world dialogue base and product shell are separate, adjacent layers.\n"
            "Pressure dynamics: high pressure shifts action competition toward reread / replace / replay.",
            ha="center", va="center", fontsize=8.2, color="#4a5568")

    svg = out_dir / "f_ev1_evidence_panorama.svg"
    png = out_dir / "f_ev1_evidence_panorama.png"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return {"figure_id": "F-EV1", "title": "APV2 evidence stack", "svg": svg, "png": png}


def build_baseline_figure(out_dir: Path) -> dict:
    # RepeatMap v0.5 fixed: holdout accuracy (mean of alpha/beta) vs token cost.
    routes = [
        {"name": "AP-style\n(G1A)", "holdout": 1.0000, "tokens": 0, "color": "#2f9e63"},
        {"name": "strict_core\nbridge (G1B)", "holdout": 1.0000, "tokens": 0, "color": "#3b6fb6"},
        {"name": "fixed\nheuristic (G2)", "holdout": 0.2250, "tokens": 0, "color": "#a0aec0"},
        {"name": "real Claude\nno-memory (G3)", "holdout": 0.2584, "tokens": 2928528, "color": "#d39a23"},
        {"name": "real GPT +\nmemory/tool (G4)", "holdout": 0.8417, "tokens": 1001624, "color": "#8a4fbf"},
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4))

    # Left: holdout accuracy bars.
    names = [r["name"] for r in routes]
    holdouts = [r["holdout"] for r in routes]
    colors = [r["color"] for r in routes]
    bars = ax1.bar(range(len(routes)), holdouts, color=colors, edgecolor="#2d3748", linewidth=0.8, width=0.62)
    ax1.set_xticks(range(len(routes)))
    ax1.set_xticklabels(names, fontsize=8.6)
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Holdout accuracy (alpha/beta mean)", fontsize=10)
    ax1.set_title("Unknown-mapping feedback learning (RepeatMap v0.5)", fontsize=11, weight="bold")
    ax1.axhline(0.0833, color="#e53e3e", linestyle="--", linewidth=1.0)
    ax1.text(len(routes) - 0.5, 0.10, "chance 1/12", ha="right", fontsize=7.8, color="#e53e3e")
    for bar, h in zip(bars, holdouts):
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.02, f"{h:.2f}", ha="center", fontsize=9, weight="bold")
    ax1.grid(axis="y", linestyle=":", alpha=0.4)

    # Right: token cost (log) vs holdout, showing AP-style at zero cost.
    ax2.set_title("Cost vs capability (same task)", fontsize=11, weight="bold")
    # Manual label offsets (dx, dy in points) to avoid overlap at the zero-token corner.
    label_offsets = {
        "AP-style\n(G1A)": (10, 10),
        "strict_core\nbridge (G1B)": (10, -16),
        "fixed\nheuristic (G2)": (10, 8),
        "real Claude\nno-memory (G3)": (-10, -16),
        "real GPT +\nmemory/tool (G4)": (-95, 10),
    }
    for r in routes:
        x = max(r["tokens"], 1)  # log axis: place zero-token routes at 1
        ax2.scatter(x, r["holdout"], s=150, color=r["color"], edgecolor="#2d3748", linewidth=0.9, zorder=3)
        dx, dy = label_offsets.get(r["name"], (8, 8))
        ax2.annotate(r["name"].replace("\n", " "), (x, r["holdout"]),
                     textcoords="offset points", xytext=(dx, dy), fontsize=8.2)
    ax2.set_xscale("log")
    ax2.set_xlabel("LLM token cost (log scale; 0-token routes shown at 1)", fontsize=10)
    ax2.set_ylabel("Holdout accuracy", fontsize=10)
    ax2.set_ylim(0, 1.18)
    ax2.set_xlim(0.5, 1e7)
    ax2.grid(True, linestyle=":", alpha=0.4)
    ax2.text(0.6, 1.12, "AP-style: top-left = high accuracy at zero token cost",
             fontsize=8.6, color="#2f9e63", weight="bold")

    fig.suptitle("APV2 controlled pilot: not a benchmark, a mechanism/cost/audit comparison (G4 is a strong, non-strawman baseline)",
                 fontsize=10.5, color="#4a5568", y=0.015)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    svg = out_dir / "f_ev2_baseline_cost_vs_capability.svg"
    png = out_dir / "f_ev2_baseline_cost_vs_capability.png"
    fig.savefig(svg, bbox_inches="tight")
    fig.savefig(png, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return {"figure_id": "F-EV2", "title": "Controlled pilot cost vs capability", "svg": svg, "png": png}


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "outputs" / f"apv2_press_evidence_figures_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    figs = [build_evidence_panorama(out_dir), build_baseline_figure(out_dir)]
    manifest = {
        "schema_id": "apv2_press_evidence_figures/v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "figures": [
            {
                "figure_id": f["figure_id"],
                "title": f["title"],
                "svg": f["svg"].relative_to(ROOT).as_posix(),
                "png": f["png"].relative_to(ROOT).as_posix(),
                "svg_sha256": _sha256(f["svg"]),
                "png_sha256": _sha256(f["png"]),
            }
            for f in figs
        ],
    }
    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "figure_count": len(figs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
