from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "docs" / "APV2_MainPaper_ArchitectureRuntime_Draft_20260611.md"


REQUIRED_SNIPPETS = [
    "APV2-BottomLoop-ParamSensitivity-1",
    "16/16 pass",
    "ShortTermSlot-OrderAblation-1",
    "18.2466",
    "9.0539",
    "LongRun-InterruptionRecovery-1",
    "1.3715",
    "RhythmSuccessor-Replay-1",
    "0.172",
    "PersistenceBackend-Reload-1",
    "19a0d88ba4ce8eacb01fe488ae72207a427c50f85bd8cc7bf4a30f74a50e60d7",
    "ArtifactFreeze-1",
    "12 entries",
    "ResidualDepth-Stress-1",
    "8 个 winner",
    "14.3126",
    "0.7448",
    "LongRun-Stability-1",
    "interruptions 4",
    "resumptions 5",
    "ShortTermSlot-Grid-1",
    "108/108 pass",
    "163-file",
    "capacity",
    "focus_merge_limit",
    "residual before",
    "residual after",
]

REQUIRED_HEADINGS = [
    "# APV2: 面向持续认知的白箱预测-行动闭环 Runtime",
    "## 摘要",
    "## Abstract",
    "## 1. 引言",
    "## 2. APV2 的最小对象",
    "## 3. Runtime 架构",
    "## 4. 核心机制",
    "## 5. P0/P1/P2 机制证据",
    "## 6. 与 LLM Agent 和传统认知架构的关系",
    "## 7. 证据边界与后续路线",
    "## 8. 结论",
    "## References",
]

CONCEPTUAL_VERSION_PATTERNS = [
    r"\bAPV2\.1\b",
    r"\bAPV2\.2\b",
    r"\bV2\.1\b",
    r"\bV2\.2\b",
]

FORBIDDEN_SUBSTITUTE_PATTERNS = [
    r"关键词硬门",
    r"regex route",
    r"answer table",
    r"student_side_llm",
    r"hidden solver",
    r"full-sentence macro",
    r"整句动作宏",
]


def _line_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def main() -> int:
    if not DRAFT.exists():
        print(f"missing draft: {DRAFT}")
        return 1

    text = DRAFT.read_text(encoding="utf-8")
    failures: list[str] = []

    for snippet in REQUIRED_SNIPPETS:
        if snippet not in text:
            failures.append(f"missing required snippet: {snippet}")

    for heading in REQUIRED_HEADINGS:
        if heading not in text:
            failures.append(f"missing required heading: {heading}")

    for match in re.finditer(r"\]\((\.\./outputs/[^)]+\.(?:png|svg))\)|`(\.\./outputs/[^`]+\.(?:png|svg))`", text):
        raw_path = match.group(1) or match.group(2)
        target = (DRAFT.parent / raw_path).resolve()
        if not target.exists():
            line = _line_for_offset(text, match.start())
            failures.append(f"missing referenced figure at line {line}: {raw_path}")

    if "English title:" not in text:
        failures.append("missing English title")

    if "## Keywords" not in text:
        failures.append("missing English keywords heading")

    figure_captions = re.findall(r"^Figure\s+\d+\.", text, flags=re.MULTILINE)
    if len(figure_captions) < 5:
        failures.append(f"expected at least 5 Figure captions, found {len(figure_captions)}")

    citation_keys = set(re.findall(r"\[([A-Za-z][A-Za-z0-9]+(?:19|20)\d{2}[A-Za-z0-9]*)\]", text))
    reference_text = text.split("## References", 1)[1] if "## References" in text else ""
    reference_keys = set(re.findall(r"^\[([A-Za-z][A-Za-z0-9]+(?:19|20)\d{2}[A-Za-z0-9]*)\]", reference_text, flags=re.MULTILINE))
    missing_references = sorted(citation_keys - reference_keys)
    unused_references = sorted(reference_keys - citation_keys)
    if missing_references:
        failures.append(f"citation keys missing references: {missing_references}")
    if unused_references:
        failures.append(f"reference keys not cited in body: {unused_references}")

    for pattern in CONCEPTUAL_VERSION_PATTERNS:
        for match in re.finditer(pattern, text):
            line = _line_for_offset(text, match.start())
            failures.append(f"conceptual version split term at line {line}: {match.group(0)}")

    for pattern in FORBIDDEN_SUBSTITUTE_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            line = _line_for_offset(text, match.start())
            window = text[max(0, match.start() - 60): match.end() + 80]
            boundary_framed = any(
                marker in window
                for marker in ("不是", "不把", "不应", "禁区", "边界", "替代", "避免")
            )
            if not boundary_framed:
                failures.append(f"unframed forbidden substitute term at line {line}: {match.group(0)}")

    defensive_markers = ["不是", "不直接", "不替代", "不能", "不应", "不把"]
    for i, paragraph in enumerate(re.split(r"\n\s*\n", text), start=1):
        marker_count = sum(paragraph.count(marker) for marker in defensive_markers)
        if marker_count >= 6:
            failures.append(f"over-defensive paragraph #{i}: {marker_count} negative-boundary markers")

    if failures:
        print("APV2 mainpaper draft check: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("APV2 mainpaper draft check: PASS")
    print(f"checked: {DRAFT}")
    print(f"chars: {len(text)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
