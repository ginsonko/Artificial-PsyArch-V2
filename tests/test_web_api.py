# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

from observatory_v2.app import ObservatoryV2App
from observatory_v2.config import load_config
from observatory_v2.web import create_server

REPO_ROOT = Path(__file__).resolve().parents[1]


class WebApiTests(unittest.TestCase):
    def test_health_and_demo_api(self) -> None:
        config = load_config()
        with tempfile.TemporaryDirectory() as tmpdir:
            app = ObservatoryV2App(config=config, repo_root_value=REPO_ROOT, outputs_root_override=tmpdir)
            server = create_server(app, host="127.0.0.1", port=0)
            host, port = server.server_address[:2]
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(f"http://{host}:{port}/api/health", timeout=5) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(payload["ok"])
                self.assertIn("server_meta", payload)
                self.assertEqual(payload["server_meta"]["service"], "observatory_v2")
                self.assertGreater(int(payload["server_meta"]["process_id"]), 0)

                with urllib.request.urlopen(f"http://{host}:{port}/api/config", timeout=5) as resp:
                    config_payload = json.loads(resp.read().decode("utf-8"))
                self.assertIn("repo_root", config_payload)
                self.assertIn("outputs_root_resolved", config_payload)
                self.assertIn("server_meta", config_payload)

                req = urllib.request.Request(
                    f"http://{host}:{port}/api/runs/demo/start",
                    data=json.dumps({"tick_count": 2, "tick_interval_ms": 5, "label": "api test"}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(result["ok"])
                self.assertTrue(app.wait_for_idle(timeout_sec=10.0))
                time.sleep(0.05)

                with urllib.request.urlopen(f"http://{host}:{port}/api/runs/latest", timeout=5) as resp:
                    latest = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(latest["manifest"]["status"], "completed")

                req_text = urllib.request.Request(
                    f"http://{host}:{port}/api/runs/text/start",
                    data=json.dumps({"texts": ["今天 天气 不错", "我 想 出门"], "label": "web text api", "tick_interval_ms": 0}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_text, timeout=5) as resp:
                    text_result = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(text_result["ok"])
                self.assertTrue(app.wait_for_idle(timeout_sec=10.0))
                with urllib.request.urlopen(f"http://{host}:{port}/api/runs/{text_result['run_id']}/ticks/1/sidecar", timeout=5) as resp:
                    sidecar = json.loads(resp.read().decode("utf-8"))
                self.assertIn("state_pool_sidecar", sidecar)
                self.assertIn("hot_anchor_cache", sidecar["state_pool_sidecar"])
                self.assertIn("sandbox_result", sidecar)
                self.assertIn("runtime_controls", sidecar)
                self.assertIn("logic_feedback", sidecar)

                req_multi = urllib.request.Request(
                    f"http://{host}:{port}/api/runs/multimodal/start",
                    data=json.dumps({"items": [{"text": "今天 天气 不错"}, {"text": "我 想 出门"}], "label": "web multimodal api", "tick_interval_ms": 0}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_multi, timeout=5) as resp:
                    multi_result = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(multi_result["ok"])
                self.assertTrue(app.wait_for_idle(timeout_sec=10.0))

                with patch("cv2.VideoCapture") as fake_capture, patch("cv2.imencode") as fake_encode:
                    from PIL import Image
                    from io import BytesIO

                    def _png_bytes(color: tuple[int, int, int]) -> bytes:
                        image = Image.new("RGB", (16, 16), color=color)
                        buf = BytesIO()
                        image.save(buf, format="PNG")
                        return buf.getvalue()

                    class _Cap:
                        def __init__(self) -> None:
                            self.index = 0
                            self.frames = [_png_bytes((10, 10, 10)), _png_bytes((240, 240, 240))]

                        def isOpened(self) -> bool:
                            return True

                        def read(self) -> tuple[bool, object]:
                            if self.index >= len(self.frames):
                                return False, None
                            frame = self.frames[self.index]
                            self.index += 1
                            return True, frame

                        def get(self, prop: int) -> float:
                            if prop == 5:
                                return 10.0
                            if prop == 7:
                                return 2.0
                            return 0.0

                        def release(self) -> None:
                            return None

                    class _Encoded:
                        def __init__(self, data: bytes) -> None:
                            self.data = data

                        def tobytes(self) -> bytes:
                            return self.data

                    fake_capture.return_value = _Cap()
                    fake_encode.side_effect = lambda _ext, frame: (True, _Encoded(frame))
                    req_video = urllib.request.Request(
                        f"http://{host}:{port}/api/runs/video-stream/start",
                        data=json.dumps({"video_b64": "ZmFrZS12aWRlbw==", "frame_stride": 1, "max_frames": 2}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req_video, timeout=5) as resp:
                        video_result = json.loads(resp.read().decode("utf-8"))
                    self.assertTrue(video_result["ok"])
                    self.assertTrue(app.wait_for_idle(timeout_sec=10.0))

                with urllib.request.urlopen(f"http://{host}:{port}/api/runtime/export", timeout=5) as resp:
                    exported = json.loads(resp.read().decode("utf-8"))
                self.assertIn("runtime", exported)
                self.assertIn("sandbox", exported)
                self.assertIn("memory_index_summary", exported.get("export_meta", {}))

                with urllib.request.urlopen(f"http://{host}:{port}/api/runtime/summary", timeout=5) as resp:
                    runtime_summary = json.loads(resp.read().decode("utf-8"))
                self.assertIn("export_meta", runtime_summary)
                self.assertIn("state_pool_summary", runtime_summary)
                self.assertIn("short_term_summary", runtime_summary)
                self.assertIn("memory_index_summary", runtime_summary.get("export_meta", {}))
                self.assertIn("executor_status", runtime_summary.get("export_meta", {}))
                self.assertIn("action_learning_summary", runtime_summary)

                with urllib.request.urlopen(f"http://{host}:{port}/api/executor/status", timeout=5) as resp:
                    executor_status = json.loads(resp.read().decode("utf-8"))
                self.assertIn("enabled", executor_status)
                self.assertIn("recent_events", executor_status)

                session_start_req = urllib.request.Request(
                    f"http://{host}:{port}/api/autonomous-session/start",
                    data=json.dumps({"text_hint": "web api session", "max_ticks": 2, "tick_interval_ms": 0}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(session_start_req, timeout=5) as resp:
                    session_start_payload = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(session_start_payload["ok"])

                with urllib.request.urlopen(f"http://{host}:{port}/api/autonomous-session/status", timeout=5) as resp:
                    session_status = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(session_status.get("active"))
                self.assertIn("autonomous_tick_meta", session_status)
                self.assertTrue(app.wait_for_idle(timeout_sec=10.0))

                manual_req = urllib.request.Request(
                    f"http://{host}:{port}/api/executor/manual-action",
                    data=json.dumps({"action_name": "noop", "params": {}}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(manual_req, timeout=5) as resp:
                    manual_result = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(manual_result["ok"])
                self.assertIn("result", manual_result)

                with urllib.request.urlopen(f"http://{host}:{port}/api/runs/{text_result['run_id']}/overview", timeout=5) as resp:
                    overview = json.loads(resp.read().decode("utf-8"))
                self.assertIn("tick_count", overview)
                self.assertIn("memory_index_summary", overview)
                self.assertIn("rollup", overview)

                with urllib.request.urlopen(f"http://{host}:{port}/api/runs/{text_result['run_id']}/rollup", timeout=5) as resp:
                    rollup = json.loads(resp.read().decode("utf-8"))
                self.assertIn("series_tail", rollup)
                self.assertIn("emotion_dissonance", rollup["series_tail"])
                self.assertIn("rules_fired_count", rollup["series_tail"])

                with urllib.request.urlopen(f"http://{host}:{port}/api/runs/overview-batch?limit=4&run_id={text_result['run_id']}", timeout=5) as resp:
                    batch = json.loads(resp.read().decode("utf-8"))
                self.assertIn("runs", batch)
                self.assertGreaterEqual(len(batch["runs"]), 1)
                self.assertEqual(batch["runs"][0]["run_id"], text_result["run_id"])
                self.assertIn("rollup", batch["runs"][0])
                self.assertIn("__hydrated", batch["runs"][0])

                with urllib.request.urlopen(f"http://{host}:{port}/api/rules", timeout=5) as resp:
                    rules_payload = json.loads(resp.read().decode("utf-8"))
                self.assertIn("rules", rules_payload)

                save_rules_req = urllib.request.Request(
                    f"http://{host}:{port}/api/rules",
                    data=json.dumps(rules_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(save_rules_req, timeout=5) as resp:
                    saved_rules = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(saved_rules["ok"])
                self.assertIn("warnings", saved_rules)
                self.assertIn("stats", saved_rules)

                simulate_rules_req = urllib.request.Request(
                    f"http://{host}:{port}/api/rules/simulate",
                    data=json.dumps(
                        {
                            "tick_index": 1,
                            "state_top": [{"sa_label": "text::今天", "energy": 1.2}],
                            "state_pool_summary": {"state_pool_size": 3, "residual_summary": {"count": 2, "total_unresolved_mass": 0.4}},
                            "bn_list": [{"memory_id": "mem_1", "score": 0.8}],
                            "c_star": {"items": [{"sa_label": "text::冷", "energy": 0.7}]},
                            "runtime_metrics": {"logic_ms": 12.0},
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(simulate_rules_req, timeout=5) as resp:
                    simulated_rules = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(simulated_rules["ok"])
                self.assertIn("emotion_channels", simulated_rules["result"])

                validate_rules_req = urllib.request.Request(
                    f"http://{host}:{port}/api/rules/validate",
                    data=json.dumps({"schema_id": "innate_rules_v2", "rules": [{"rule_id": "dup"}, {"rule_id": "dup"}]}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(validate_rules_req, timeout=5) as resp:
                    validated_rules = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(validated_rules["ok"])
                self.assertIn("warnings", validated_rules)
                self.assertIn("stats", validated_rules)

                with urllib.request.urlopen(f"http://{host}:{port}/api/tuner", timeout=5) as resp:
                    tuner_payload = json.loads(resp.read().decode("utf-8"))
                self.assertIn("profiles", tuner_payload)

                save_tuner_req = urllib.request.Request(
                    f"http://{host}:{port}/api/tuner",
                    data=json.dumps(tuner_payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(save_tuner_req, timeout=5) as resp:
                    saved_tuner = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(saved_tuner["ok"])
                self.assertIn("warnings", saved_tuner)
                self.assertIn("stats", saved_tuner)

                validate_tuner_req = urllib.request.Request(
                    f"http://{host}:{port}/api/tuner/validate",
                    data=json.dumps({"schema_id": "auto_tuner_v2", "profiles": [{"profile_id": "dup"}, {"profile_id": "dup"}]}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(validate_tuner_req, timeout=5) as resp:
                    validated_tuner = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(validated_tuner["ok"])
                self.assertIn("warnings", validated_tuner)
                self.assertIn("stats", validated_tuner)

                with urllib.request.urlopen(f"http://{host}:{port}/api/runs/{text_result['run_id']}/ticks", timeout=5) as resp:
                    ticks_payload = json.loads(resp.read().decode("utf-8"))
                self.assertIn("ticks", ticks_payload)

                for tick_index in (0, 1):
                    with urllib.request.urlopen(f"http://{host}:{port}/api/runs/{multi_result['run_id']}/ticks/{tick_index}/sidecar", timeout=5) as resp:
                        multimodal_sidecar = json.loads(resp.read().decode("utf-8"))
                    image_packet = multimodal_sidecar.get("image_packet", {}) or {}
                    preview_data = ((image_packet.get("preview_image") or {}).get("data_url"))
                    if preview_data is None:
                        continue
                    self.assertTrue(isinstance(preview_data, str) or isinstance(preview_data, dict))
                    if isinstance(preview_data, str) and preview_data.startswith("data:image/"):
                        self.assertTrue(preview_data.startswith("data:image/"))
                    elif isinstance(preview_data, dict):
                        rel_path = str(preview_data.get("rel_path", "") or "")
                        self.assertTrue(rel_path)
                        with urllib.request.urlopen(f"http://{host}:{port}/api/runs/{multi_result['run_id']}/assets/{rel_path}", timeout=5) as asset_resp:
                            asset_payload = json.loads(asset_resp.read().decode("utf-8"))
                        self.assertTrue(str(asset_payload.get("text", "")).startswith("data:image/"))
                    break

                with urllib.request.urlopen(f"http://{host}:{port}/api/executor/screen-preview", timeout=5) as resp:
                    preview = json.loads(resp.read().decode("utf-8"))
                self.assertIn("enabled", preview)

                req_forget = urllib.request.Request(
                    f"http://{host}:{port}/api/runtime/forget",
                    data=json.dumps(
                        {
                            "keep_latest": 2,
                            "strategy": "score_prune",
                            "min_reality_weight": 0.5,
                            "min_total_item_energy": 0.5,
                            "protect_memory_kinds": ["teacher_feedback"],
                            "max_memory_count": 3,
                            "dry_run": True,
                        }
                    ).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_forget, timeout=5) as resp:
                    forget = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(forget["ok"])
                self.assertTrue(forget["dry_run"])
                self.assertEqual(forget["strategy"], "score_prune")
                self.assertIn("kind_histogram_before", forget)

                bundle_dir = Path(tmpdir) / "memory_bundle"
                export_bundle_req = urllib.request.Request(
                    f"http://{host}:{port}/api/runtime/memory-bundle/export",
                    data=json.dumps({"directory": str(bundle_dir)}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(export_bundle_req, timeout=5) as resp:
                    bundle_export = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(bundle_export["ok"])
                self.assertTrue((bundle_dir / "memory_store_v2.json").exists())
                self.assertTrue((bundle_dir / "bundle_meta.json").exists())

                inspect_bundle_req = urllib.request.Request(
                    f"http://{host}:{port}/api/runtime/memory-bundle/inspect",
                    data=json.dumps({"directory": str(bundle_dir)}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(inspect_bundle_req, timeout=5) as resp:
                    bundle_inspect = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(bundle_inspect["ok"])
                self.assertEqual(bundle_inspect["bundle_format"], "layered_v2")
                self.assertIn("index_summary", bundle_inspect)
                self.assertIn("vector_bundle", bundle_inspect)
                self.assertIn("spacetime_bundle", bundle_inspect)
                self.assertIn("vector_meta", bundle_inspect["files"])
                self.assertIn("spacetime_meta", bundle_inspect["files"])

                import_bundle_req = urllib.request.Request(
                    f"http://{host}:{port}/api/runtime/memory-bundle/import",
                    data=json.dumps({"directory": str(bundle_dir)}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(import_bundle_req, timeout=5) as resp:
                    bundle_import = json.loads(resp.read().decode("utf-8"))
                self.assertTrue(bundle_import["ok"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
