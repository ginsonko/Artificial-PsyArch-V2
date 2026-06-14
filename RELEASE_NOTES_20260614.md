# APV2 Release Notes 2026-06-14

本仓库是 APV2 发布包的一部分，角色是：AP-Core runtime、机制实验、论文主文与长篇技术报告。

This repository is one part of the APV2 public release package. Its role is: AP-Core runtime、机制实验、论文主文与长篇技术报告.

## Release Tag

`apv2-release-20260614-final-longreport`

## Evidence Included

- AP-Core 底层循环机制: 参数敏感性、短期叙事槽顺序消融、打断恢复、节奏后继、持久化重载、残差深度、压力动力学和 online learned vector 消融。
- 发布版短主文、长篇技术报告、新闻稿、仓库说明和 journal-style 图表。
- 与 GL 开放中文对话仓库和 reproduction artifact 仓库共同形成 APV2 发布证据链。

## Reproduction Anchors

- Repository manifest: `PUBLIC_STAGING_MANIFEST.json`
- Cross-repository zip summary: `release_repos_20260614/PUBLIC_REPO_STAGING_SUMMARY.json`
- Paper artifact manifest: `paper_artifacts/release_20260614/release_manifest_20260614.json`
- Main AP-Core repository: `https://github.com/ginsonko/Artificial-PsyArch-V2`
- GL repository: `https://github.com/ginsonko/APV2-GL-OpenWorld-Chinese`
- Artifact repository: `https://github.com/ginsonko/APV2-Reproduction-Artifacts`
- Third-party reference implementation: `https://github.com/ACG-j/artificial_psyarch`

## Suggested Quick Checks

- `python -m pytest tests/test_apv22_apcore_dynamics.py -q`
- `python experiments/apv22_apcore_dynamics.py`
- `python scripts/check_apv2_mainpaper_runtime_draft.py`

## Evidence Layering

AP-Core、GL、第三方复现和 Product/Canvas/Desktop 证据线在发布材料中分层呈现。AP-Core 负责底层机制验证；GL 负责学习协议和开放中文对话验证；artifact 仓库负责冻结证据与复现锚点；Product/Canvas/Desktop 证据线展示受控应用接口和外部效度。
