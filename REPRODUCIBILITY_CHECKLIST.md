# APV2 Reproducibility Checklist

本清单用于帮助读者从公开仓库、manifest、Word/PDF 文档和实验输出追溯 APV2 发布版主张。

## 1. Identify The Release

- [ ] Confirm the release tag: `apv2-release-20260614-final-longreport`.
- [ ] Read `RELEASE_NOTES_20260614.md`.
- [ ] Read this repository's `PUBLIC_STAGING_MANIFEST.json`.

## 2. Verify File Integrity

- [ ] Check every file listed in `PUBLIC_STAGING_MANIFEST.json` by bytes and SHA-256.
- [ ] If using the three zip packages, compare them against `release_repos_20260614/PUBLIC_REPO_STAGING_SUMMARY.json`.
- [ ] For Word/PDF release documents, compare against `paper_artifacts/release_20260614/release_manifest_20260614.json`.

## 3. Read The Evidence In Layers

- [ ] AP-Core: runtime mechanism tests and bottom-loop evidence.
- [ ] GL: teacher-off/no-leakage learning validation and open Chinese dialogue evidence.
- [ ] Reproduction artifacts: frozen outputs, manifests, third-party Rust rerun reports.
- [ ] Product/Canvas/Desktop: controlled application-interface evidence and safety contracts.

## 4. Run Local Checks

Suggested commands for this repository:

- `python -m pytest tests/test_apv22_apcore_dynamics.py -q`
- `python experiments/apv22_apcore_dynamics.py`
- `python scripts/check_apv2_mainpaper_runtime_draft.py`

## 5. Audit Boundaries

- [ ] Student-side LLM/provider is disabled in GL student-side evidence.
- [ ] No answer table, regex answer route, hidden solver, keyword hard gate, or full-sentence action macro is used as AP-native learning evidence.
- [ ] Teacher/examiner roles are separated from learner-side test-time behavior.
- [ ] Claims are mapped to AP-Core, GL, third-party reproduction, or controlled application evidence lines.

## 6. Cite The Release

Cite the repository URL, release tag, and relevant manifest SHA-256 entries. If using the third-party Rust reproduction, cite `ACG-j/artificial_psyarch` and the APV2 reproduction report together.
