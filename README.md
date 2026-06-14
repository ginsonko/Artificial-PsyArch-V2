# Artificial-PsyArch-V2 / APV2 白箱预测-行动闭环

APV2 是一个面向持续认知的白箱预测-行动 runtime。它把外界输入、短期叙事、历史记忆、后继预测、认知感受、情绪慢量、行动竞争和行动反馈组织成同一条持续 tick 循环。

APV2 is a white-box predictive-action runtime for continuous cognition. It organizes state atoms, dual-energy state pools, short-term narrative slots, residual recall, successor prediction, process-grounded cognitive feelings, action feedback, and persistence into an auditable loop.

## 发布锚点 / Release Anchors

- Official release tag: `apv2-release-20260614-final-cn-pdf`
- Repository manifest: `PUBLIC_STAGING_MANIFEST.json`
- Release documents: `paper_artifacts/release_20260614/`
- Companion GL repository: `APV2-GL-OpenWorld-Chinese`
- Companion artifact repository: `APV2-Reproduction-Artifacts`

## 核心能力 / Core Evidence

- AP-Core runtime loop: 状态池、短期叙事槽、残差 B 召回、C/C* 后继预测、认知感受、情绪慢量、行动反馈和持久化。
- Mechanism experiments: 参数敏感性、短期槽顺序消融、打断恢复、节奏后继、持久化重载、残差召回压力、压力动力学和 online learned vector 消融。
- Auditable outputs: 精选实验报告、图表、manifest 和发布版 Word 文档。

## 仓库内容 / Contents

- `core/`, `memory/`, `config/`: AP-Core runtime 底座。
- `experiments/`: AP-Core 机制实验入口。
- `tests/`: 对应的可复跑测试。
- `docs/`: 主论文、补充索引、设计说明和最终报告。
- `outputs/`: 精选机制证据输出和图表。
- `paper_artifacts/release_20260614/`: Word 版发布论文、新闻稿和 manifest。

## 快速验证 / Quick Validation

```bash
python -m pytest tests/test_apv22_apcore_dynamics.py -q
python experiments/apv22_apcore_dynamics.py
python scripts/check_apv2_mainpaper_runtime_draft.py
```

## 边界 / Boundary

本仓库证明 AP-Core runtime 机制: 状态池、短期叙事槽、残差召回、后继预测、压力动力学、online learned vector、行动反馈与持久化。开放中文对话学习验证位于 `APV2-GL-OpenWorld-Chinese` 仓库; 冻结 artifact 位于 `APV2-Reproduction-Artifacts` 仓库。

## License / 许可证

本仓库按 `APV2 Public Research License v2026-06-14` 发布为 source-available public research preview。允许公开阅读、克隆、本地复验、非商业研究评估和合理引用；商业使用、模型训练或评测数据复用、数据集/技能包再打包、产品部署或声称官方衍生版本需要另行授权。完整条款见 `LICENSE`。

This repository is source-available for public research preview under `APV2 Public Research License v2026-06-14`. Public reading, cloning, local reproduction, non-commercial research evaluation, and citation are permitted; commercial use, model-training/evaluation-data reuse, dataset or skill-package repackaging, product deployment, and official-release claims require separate permission. See `LICENSE`.
