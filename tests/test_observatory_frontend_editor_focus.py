from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_INDEX_PATH = REPO_ROOT / "observatory_v2" / "web_static" / "index.html"


class ObservatoryFrontendEditorFocusTests(unittest.TestCase):
    def test_editor_focus_and_row_duplication_hooks_exist(self) -> None:
        html = WEB_INDEX_PATH.read_text(encoding="utf-8")
        required_fragments = [
            'id="focusVisibleSingleRuleBtn"',
            'id="clearRulePinBtn"',
            'id="focusVisibleSingleTunerBtn"',
            'id="clearTunerPinBtn"',
            'function pinSingleVisibleRuleCard(',
            'function clearPinnedRuleCard(',
            'function pinSingleVisibleTunerCard(',
            'function clearPinnedTunerCard(',
            'function enhanceRulesEditorDom(',
            'function enhanceTunerEditorDom(',
            'function effectTypeHint(',
            'function tunerTargetHint(',
            'data-role="duplicate-condition"',
            'data-role="duplicate-effect"',
            'data-role="duplicate-adjustment"',
            'data-role="pin-rule-card"',
            'data-role="pin-tuner-card"',
            'data-role="effect-display-text"',
            'data-role="effect-reason"',
            'data-role="effect-message"',
            'pinnedRuleId',
            'pinnedTunerId',
            'function effectFieldMode(',
            'function formulaFieldMode(',
            'function formulaKindHint(',
            'function applyEffectRowMode(',
            'function applyFormulaPanelMode(',
            'function renderReferenceBarChart(',
            'function topEntriesFromHist(',
            'data-effect-field="channel"',
            'data-effect-field="when_channel"',
            'data-effect-field="threshold"',
            'data-effect-field="action_id"',
            'data-effect-field="sa_label"',
            'data-effect-field="display_meta"',
            'data-effect-field="formula"',
            'data-formula-key="kind"',
            'data-formula-key="metric"',
            'data-formula-key="metrics"',
            'data-formula-key="value"',
            'data-formula-key="base"',
            'data-formula-key="factor"',
            'data-formula-key="min"',
            'data-formula-key="max"',
            'mode-hidden',
            'mode-muted',
            'field-inline-note',
            'editor-reference-bar',
            'editor-reference-chart',
            'editor-reference-caption',
        ]
        for fragment in required_fragments:
            self.assertIn(fragment, html)

    def test_type_and_formula_modes_cover_main_semantics(self) -> None:
        html = WEB_INDEX_PATH.read_text(encoding="utf-8")
        semantic_fragments = [
            "if (clean === 'set_emotion_floor')",
            "if (clean === 'inject_sa')",
            "if (clean === 'add_action_drive')",
            "if (clean === 'append_rule_log')",
            "hidden.add('when_channel')",
            "hidden.add('threshold')",
            "hidden.add('action_id')",
            "hidden.add('sa_label')",
            "hidden.add('display_meta')",
            "notes.display_meta = '主要填写 reason；display_text / message 对该类型通常不会被读取。'",
            "if (clean === 'constant')",
            "if (clean === 'metric')",
            "if (clean === 'mul')",
            "if (clean === 'affine')",
            "if (clean === 'max_metric')",
            "hintBits.push(`当前公式摘要：${formulaSummary(collectFormulaFromRow(panel))}`)",
            "hintBits.push(`说明：${formulaKindHint(kind)}`)",
            "state.pinnedRuleId = cleanId;",
            "state.pinnedTunerId = cleanId;",
            "已定位并聚焦规则卡",
            "已定位并聚焦调参档",
            "topFamilyBars",
            "topEffectTypeBars",
            "topFormulaBars",
            "topChannelBars",
            "topTargetBars",
            "topMetricBars",
            "familyHist,",
            "effectTypeHist,",
            "formulaKindHist,",
            "channelHist",
            "targetHist,",
            "metricHist",
        ]
        for fragment in semantic_fragments:
            self.assertIn(fragment, html)


if __name__ == "__main__":
    unittest.main()
