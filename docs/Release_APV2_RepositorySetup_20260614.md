# APV2 发布仓库说明

本次发布使用 3 个公开仓库:

- `https://github.com/ginsonko/Artificial-PsyArch-V2`
- `https://github.com/ginsonko/APV2-GL-OpenWorld-Chinese`
- `https://github.com/ginsonko/APV2-Reproduction-Artifacts`

统一冻结 tag: `apv2-release-20260614-final`

| 仓库 | 主要内容 | 版本锚定 |
|---|---|---|
| `Artificial-PsyArch-V2` | AP-Core runtime、机制实验、主论文和补充材料 | Git tag + `PUBLIC_STAGING_MANIFEST.json` |
| `APV2-GL-OpenWorld-Chinese` | GL 学习协议、开放中文对话验证、Fresh300/Skill38 证据 | Git tag + `PUBLIC_STAGING_MANIFEST.json` |
| `APV2-Reproduction-Artifacts` | 发布稿、实验输出、第三方复现与冻结 artifact | Git tag + `PUBLIC_STAGING_MANIFEST.json` |

外发 zip 包的 SHA-256 不写入会被重新打包的仓库内部文件，统一由外层 `release_repos_20260614/PUBLIC_REPO_STAGING_SUMMARY.json` 记录。这样可以避免 zip 内文件记录自身 zip hash 造成的递归漂移。

许可证: `APV2 Public Research License v2026-06-14`。这是 source-available public research license, 不是 OSI open-source license。它允许公开阅读、clone、fork、本地运行、非商业研究复验和合理引用，同时保留商业使用、模型训练、数据再打包、产品部署和派生系统公开分发等权限边界。

建议 GitHub 仓库简介、Release notes 和 README 都统一使用这句话:

```text
Source-available public research preview under the APV2 Public Research License v2026-06-14.
```

## 1. AP-Core 主仓库

仓库名: `Artificial-PsyArch-V2`

简介:

```text
APV2 core runtime: a white-box predictive-action architecture for continuous cognition, with AP-Core mechanism experiments and paper artifacts.
```

推荐 Topics:

```text
artificial-cognition, cognitive-architecture, white-box-ai, agent-memory, continual-learning, predictive-processing
```

README 首页第一句:

```text
APV2 is a white-box predictive-action runtime for continuous cognition: state atoms, dual-energy state pools, short-term narrative slots, residual recall, successor prediction, cognitive feelings, action feedback, and persistence.
```

## 2. GL 开放中文对话仓库

仓库名: `APV2-GL-OpenWorld-Chinese`

简介:

```text
GL learning and validation layer for APV2 open-world Chinese dialogue: teacher-off curricula, Fresh300 evaluations, no-leakage audits, and skill packages.
```

推荐 Topics:

```text
chinese-dialogue, teacher-off, no-leakage, curriculum-learning, open-world-dialogue, apv2
```

README 首页第一句:

```text
This repository contains the GL learning protocol and validation artifacts for APV2 open-world Chinese dialogue, including teacher-off Fresh300 and Skill38 evidence.
```

## 3. 实验复现与冻结锚定仓库

仓库名: `APV2-Reproduction-Artifacts`

简介:

```text
Frozen APV2 experiment artifacts, manifests, hashes, rerun logs, and third-party reproduction materials.
```

推荐 Topics:

```text
reproducibility, artifacts, experiment-manifest, audit-trace, apv2, rust-reproduction
```

README 首页第一句:

```text
This repository anchors APV2 release artifacts: frozen experiment outputs, manifests, SHA-256 records, rerun commands, and third-party reproduction summaries.
```

## 4. 第三方复现仓库

保持引用:

```text
https://github.com/ACG-j/artificial_psyarch
```

这不是官方 AP-Core 主仓库，而是第三方独立实现和复现证据。
