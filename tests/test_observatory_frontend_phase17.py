from __future__ import annotations

import re
import shutil
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_INDEX_PATH = REPO_ROOT / "observatory_v2" / "web_static" / "index.html"


class ObservatoryFrontendPhase17Tests(unittest.TestCase):
    def test_structure_panels_and_bulk_controls_exist(self) -> None:
        html = WEB_INDEX_PATH.read_text(encoding="utf-8")
        required_fragments = [
            "data-main=\"sidecar_cards\"",
            "data-main=\"runtime_cards\"",
            "id=\"mainViewSidecarCards\"",
            "id=\"mainViewRuntimeCards\"",
            "data-right=\"live_cards\"",
            "id=\"rightViewLiveCards\"",
            "id=\"toggleAutoRefreshBtn\"",
            "id=\"autoRefreshModeBox\"",
            "id=\"cacheTickBox\"",
            "id=\"cacheOverviewBox\"",
            "id=\"cacheHitBox\"",
            "id=\"refreshIntervalSelect\"",
            "id=\"refreshHealthBox\"",
            "id=\"enableVisibleRulesBtn\"",
            "id=\"disableVisibleRulesBtn\"",
            "id=\"enableVisibleTunerBtn\"",
            "id=\"disableVisibleTunerBtn\"",
            "id=\"liveFreshnessBox\"",
            "id=\"manifestMetaBody\"",
            "id=\"manifestIndexDigestBody\"",
            "id=\"manifestConfigBody\"",
            "id=\"manifestTraceBody\"",
            "id=\"manifestNotesBody\"",
            "id=\"ruleFamilyFilter\"",
            "id=\"ruleEnabledFilter\"",
            "id=\"ruleAuditFilter\"",
            "id=\"rulesFilterHint\"",
            "id=\"rulesWarningsBox\"",
            "id=\"ruleTemplateSelect\"",
            "id=\"addRuleTemplateBtn\"",
            "id=\"ruleTemplateNote\"",
            "id=\"rulesReferencePanel\"",
            "输入待同步",
            "表单已同步",
            "rulesWarningLevelFilter",
            "rulesWarningCodeFilter",
            "rulesWarningPathFilter",
            "rulesWarningResetBtn",
            "id=\"tunerTargetFilter\"",
            "id=\"tunerEnabledFilter\"",
            "id=\"tunerAuditFilter\"",
            "id=\"tunerFilterHint\"",
            "id=\"tunerWarningsBox\"",
            "id=\"tunerTemplateSelect\"",
            "id=\"addTunerTemplateBtn\"",
            "id=\"tunerTemplateNote\"",
            "id=\"tunerReferencePanel\"",
            "data-role=\"quick-add-condition\"",
            "data-role=\"quick-add-effect\"",
            "data-role=\"quick-add-adjustment\"",
            "id=\"runtimePoolDigestBox\"",
            "id=\"runtimeIndexDigestBox\"",
            "id=\"runtimeShortTermDigestBox\"",
            "id=\"runtimeExecutorDigestBox\"",
            "id=\"runtimeLearningDigestBox\"",
            "id=\"runtimeTeacherDigestBox\"",
            "id=\"runtimeVideoDigestBox\"",
            "id=\"sidecarSensorDigestBox\"",
            "id=\"sidecarRecallDigestBox\"",
            "id=\"sidecarMemoryDigestBox\"",
            "id=\"sidecarActionDigestBox\"",
            "id=\"liveFreshnessDigestBox\"",
            "id=\"liveFocusDigestBox\"",
            "id=\"liveRunDigestBox\"",
            "id=\"liveMetaDigestBox\"",
            "id=\"researchLogicDigestBox\"",
            "id=\"researchCapacityDigestBox\"",
            "id=\"researchHotspotDigestBox\"",
            "id=\"researchRiskDigestBox\"",
            "tunerWarningLevelFilter",
            "tunerWarningCodeFilter",
            "tunerWarningPathFilter",
            "tunerWarningResetBtn",
            "id=\"sidecarRecallBody\"",
            "id=\"sidecarPredictBody\"",
            "id=\"sidecarSensorDetailBody\"",
            "id=\"sidecarCompetitionBody\"",
            "id=\"runtimeAnchorBody\"",
            "id=\"runtimeRecentExternalBody\"",
            "id=\"runtimeLearningBody\"",
            "id=\"runtimeVideoBody\"",
            "id=\"liveFocusHotBody\"",
            "id=\"liveRunStatusBody\"",
            "id=\"tickResearchBody\"",
            "id=\"tickShortTermDigestBody\"",
            "id=\"auditPanelTitle\"",
            "id=\"auditDiffSummaryList\"",
            "id=\"auditDiffDetailList\"",
            "id=\"auditCompareFilter\"",
            "id=\"auditCompareHint\"",
            "id=\"collapseVisibleRulesBtn\"",
            "id=\"expandVisibleRulesBtn\"",
            "id=\"collapseVisibleTunerBtn\"",
            "id=\"expandVisibleTunerBtn\"",
            "折叠可见档位",
            "展开可见档位",
            "id=\"clearRuleSimulationBtn\"",
            "id=\"startAutonomousSessionBtn\"",
            "id=\"pauseAutonomousSessionBtn\"",
            "id=\"resumeAutonomousSessionBtn\"",
            "id=\"recoverAutonomousSessionBtn\"",
            "id=\"stopAutonomousSessionBtn\"",
            "id=\"startWebcamBtn\"",
            "id=\"startMicrophoneBtn\"",
            "id=\"autonomousSessionStatusBox\"",
            "id=\"externalTeacherStatusBox\"",
            "id=\"streamSourceStatusBox\"",
            "session_phase",
            "session_health",
            "session_recover_hint",
            "session_focus",
            "session_actions",
            "phase=",
            "health=",
            "function formatAutonomousSessionSummary(",
            "id=\"webcamDeviceIndex\"",
            "id=\"webcamMaxFrames\"",
            "id=\"webcamFrameWidth\"",
            "id=\"webcamFrameHeight\"",
            "id=\"microphoneDeviceIndex\"",
            "id=\"microphoneMaxWindows\"",
            "id=\"microphoneTickWindowMs\"",
            "id=\"microphoneSampleRate\"",
            "id=\"microphoneChannels\"",
            "id=\"autonomousTeacherModeSelect\"",
            "id=\"autonomousTeacherEndpointInput\"",
            "id=\"autonomousSessionTextHint\"",
            "id=\"autonomousSessionMaxTicks\"",
            "id=\"autonomousSessionIntervalMs\"",
            "id=\"autonomousRecoverRunId\"",
            "function renderSidecarCards()",
            "function renderRuntimeCards()",
            "function renderLiveCards()",
            "function renderCacheStatus()",
            "decoder_attempts",
            "teacher_layer_summary",
            "runtimeLearningDigestBox",
            "async function controlAutonomousSession(",
            "/api/runs/webcam-stream/start",
            "/api/runs/microphone-stream/start",
            "/api/autonomous-session/recover",
            "webcamDeviceIndex",
            "microphoneDeviceIndex",
            "autonomousRecoverRunId",
            "function extractRulesAuditResult()",
            "function summarizeAuditDiff()",
            "function renderAuditDiffCards()",
            "function formatTs(",
            "function boolText(",
            "async function triggerRefresh(",
            "function applyVisibleRuleEnabled(enabled)",
            "function applyVisibleTunerEnabled(enabled)",
            "function setVisibleRuleCollapsed(collapsed)",
            "function setVisibleTunerCollapsed(collapsed)",
            "function updateFreshnessState()",
            "function buildRuleFamilyOptions()",
            "function buildTunerTargetOptions()",
            "function buildEditorAuditIndexes()",
            "function buildRuleAuditState(",
            "function buildTunerAuditState(",
            "function renderEditorBadgeRow(",
            "function normalizeWarningLevel(",
            "function collectWarningCodes(",
            "function renderWarningBox(",
            "function warningSuggestionText(",
            "function renderEditorWarnings(",
            "function renderJumpChip(",
            "function bindJumpChips(",
            "function jumpToEditorCard(",
            "function scrollPendingEditorCard(",
            "function renderVisibleMainView()",
            "function renderVisibleRightView()",
            "function renderManifestJsonView()",
            "function renderRuntimeJsonView()",
            "function renderLiveJsonView()",
            "RULE_TEMPLATES",
            "TUNER_TEMPLATES",
            "function addRuleFromTemplate(",
            "function addTunerFromTemplate(",
            "function renderRuleTemplateControls(",
            "function renderTunerTemplateControls(",
            "function renderRulesReferencePanel(",
            "function renderTunerReferencePanel(",
            "function bindRuleTemplateControls(",
            "function bindTunerTemplateControls(",
            "latestRuleSimulation",
            "collapsedRuleIds",
            "collapsedTunerIds",
            "rules_payload: state.rulesPayload",
            "tuner_payload: state.tunerPayload",
            "latest_run_label",
            "latest_run_path",
            "<th>建议</th>",
        ]
        for fragment in required_fragments:
            self.assertIn(fragment, html)

    def test_embedded_script_compiles_under_node(self) -> None:
        html = WEB_INDEX_PATH.read_text(encoding="utf-8")
        matches = re.findall(r"<script>([\s\S]*?)</script>", html)
        self.assertTrue(matches, "index.html should contain at least one inline script block")

        node = shutil.which("node")
        if not node:
            self.skipTest("node is not available in PATH")

        for index, script in enumerate(matches):
            result = subprocess.run(
                [
                    node,
                    "-e",
                    "const fs=require('fs'); const src=fs.readFileSync(0,'utf8'); new Function(src);",
                ],
                input=script,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"inline script {index} failed to compile under Node:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )

    def test_inner_visual_sequence_handler_is_unique_and_contour_bundle_is_used(self) -> None:
        html = WEB_INDEX_PATH.read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"function\s+playInnerVisionSequence\s*\(", html)), 1)
        self.assertIn("composite_data_url", html)
        self.assertIn("pruneImageAssetCache(", html)
        self.assertIn("tick: clone(frame.tick || {})", html)
        self.assertIn("sidecar: clone(frame.sidecar || {})", html)
        self.assertIn("(((packet.contour_reconstruction || {}).composite_data_url) || '')", html)
        self.assertIn("id=\"innerVisionContourLayerSelect\"", html)
        self.assertIn("innerVisionContourLayer: 'cognitive'", html)
        self.assertIn("bindInnerSelect('innerVisionContourLayerSelect', 'innerVisionContourLayer'", html)
        self.assertIn("motion_composite_data_url", html)
        self.assertIn("function drawMotionContourLayer(", html)
        self.assertIn("播放代理音频", html)
        self.assertIn("播放刺激原音", html)
        self.assertIn("proxy_preview_wav_b64", html)
        self.assertIn("selectedInnerAudioLabel: ''", html)
        self.assertIn("innerTextShowSensoryHints: false", html)
        self.assertIn("id=\"innerTextSensoryToggle\"", html)
        self.assertIn("inner-audio-object-cloud", html)
        self.assertIn("function isAssetRef(", html)
        self.assertIn("function resolveRunTextAsset(", html)
        self.assertIn("function expandAudioPlaybackNeighborhood(", html)
        self.assertIn("局部回听 =", html)
        self.assertIn("/api/runs/${encodeURIComponent(cleanRunId)}/assets/${encodedPath}", html)
        self.assertIn("extractMemoryRows(sidecar, 'vision').length", html)
        self.assertIn("normalizeVisionRow(item, 'c_star')", html)
        self.assertIn("renderInnerSummary([", html)
        self.assertIn("inner-summary-list", html)
        self.assertIn("async function drawMaskedContourComponent(", html)
        self.assertIn("bundle.composite_data_url || bundle.outline_data_url || bundle.silhouette_data_url", html)
        self.assertIn("innerVisionRenderTickOverride", html)
        self.assertIn("认知轮廓=${Number(digest.cognitiveContourCount || 0)}", html)
        self.assertIn("召回轮廓=${Number(digest.recalledAssetCount || 0)}", html)


    def test_hybrid_cognitive_visual_mode_and_component_mask_pipeline_exist(self) -> None:
        html = WEB_INDEX_PATH.read_text(encoding="utf-8")
        self.assertIn("value=\"hybrid\"", html)
        self.assertIn("认知混合", html)
        self.assertIn("function pruneComponentMaskCache(", html)
        self.assertIn("async function getSelectedComponentBinaryMask(", html)
        self.assertIn("function drawBinaryMaskLayer(", html)
        self.assertIn("cognitiveOnly: true", html)
        self.assertIn("includeStateColorField: Boolean(opts.includeStateColorField)", html)


if __name__ == "__main__":
    unittest.main()
