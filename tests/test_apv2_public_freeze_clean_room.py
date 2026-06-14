from __future__ import annotations

import json
from pathlib import Path

from scripts.build_apv2_public_freeze_candidate import build
from scripts.run_apv2_clean_room_rerun import copy_package, find_manifest, package_root_from_manifest


def _all_paths(root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]


def test_public_freeze_candidate_manifest_zip_and_boundaries(tmp_path: Path) -> None:
    result = build(output_base=tmp_path)

    package_dir = Path(result["package_dir"])
    manifest_path = Path(result["manifest_path"])
    zip_path = Path(result["zip_path"])
    report_path = Path(result["report_path"])

    assert package_dir.exists()
    assert manifest_path.exists()
    assert zip_path.exists()
    assert report_path.exists()
    assert result["file_count"] > 40
    assert len(result["zip_sha256"]) == 64

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_id"] == "apv2_public_freeze_candidate/v1"
    assert manifest["scope"] == "AP-Core architecture/runtime main-paper reproducibility candidate"
    assert "GL learning proof" in manifest["not_scope"]
    assert manifest["repository_status"]["kind"] in {"not_git_repository", "git_repository_unqueried_by_freeze_script"}
    assert manifest["boundary"]["ap_core_scope_only"] is True
    assert manifest["boundary"]["student_side_llm"] is False

    commands = manifest["rerun_commands"]["required"]
    assert any("check_apv2_mainpaper_runtime_draft.py" in command for command in commands)
    assert any("test_apv2_bottom_loop_p0_materials.py" in command for command in commands)
    assert any("test_apv2_p1_hardening_materials.py" in command for command in commands)
    assert any("test_apv2_p2_stress_mechanism_evidence.py" in command for command in commands)
    assert any("test_apv2_public_freeze_clean_room.py" in command for command in commands)

    for entry in manifest["files"]:
        assert entry["bytes"] > 0
        assert len(entry["sha256"]) == 64

    report = report_path.read_text(encoding="utf-8")
    assert "APV2 Public Freeze Candidate Report" in report
    assert "third-party replication" in report


def test_public_freeze_candidate_excludes_sensitive_and_noisy_files(tmp_path: Path) -> None:
    result = build(output_base=tmp_path)
    package_dir = Path(result["package_dir"])
    paths = _all_paths(package_dir)
    joined = "\n".join(paths)

    assert ".env.dpp1.local" not in joined
    assert "__pycache__" not in joined
    assert ".pytest_cache" not in joined
    assert not any(path.endswith(".pyc") for path in paths)
    assert not any(path.endswith(".log") for path in paths)
    assert not any(path.startswith("tmp_") for path in paths)
    assert not any(path.startswith("GL_TaskBuilder/") for path in paths)
    assert not any(path.startswith("StrongestNurturingSystem/") for path in paths)


def test_clean_room_helpers_find_and_copy_package(tmp_path: Path) -> None:
    result = build(output_base=tmp_path)
    freeze_output = Path(result["output_dir"])
    manifest_path = find_manifest(freeze_output)
    package_source = package_root_from_manifest(manifest_path)
    stage = tmp_path / "stage_copy"

    copy_package(package_source, stage)

    assert (stage / "public_freeze_manifest.json").exists()
    assert (stage / "scripts" / "check_apv2_mainpaper_runtime_draft.py").exists()
    assert (stage / "tests" / "test_apv2_public_freeze_clean_room.py").exists()
    assert (stage / "tests" / "test_apv2_p2_stress_mechanism_evidence.py").exists()
    copied = _all_paths(stage)
    assert not any("__pycache__" in path for path in copied)
    assert not any(path.endswith(".pyc") for path in copied)
