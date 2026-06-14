# Artificial-PsyArch-V2 / APV2 白箱预测-行动闭环

        APV2 是一个面向持续认知的白箱预测-行动 runtime。它把外界输入、短期叙事、历史记忆、后继预测、认知感受、情绪慢量、行动竞争和行动反馈组织成同一条持续 tick 循环。

        APV2 is a white-box predictive-action runtime for continuous cognition. It organizes state atoms, dual-energy state pools, short-term narrative slots, residual recall, successor prediction, process-grounded cognitive feelings, action feedback, and persistence into an auditable loop.

        ## 仓库内容 / Contents

        - `core/`, `memory/`, `config/`: 最新 AP-Core runtime 底座。
        - `experiments/`: AP-Core 机制实验，包括底层动力学、online learned vector、压力动力学等。
        - `tests/`: 对应的可复跑测试。
        - `docs/`: 主论文、补充索引、设计和最终报告。
        - `outputs/`: 精选机制证据输出和图表，不包含本地临时数据库。
        - `paper_artifacts/release_20260614/`: Word 版发布论文、新闻稿和 manifest。

        ## 快速验证 / Quick validation

        ```bash
        python -m pytest tests/test_apv22_apcore_dynamics.py -q
        python experiments/apv22_apcore_dynamics.py
        python scripts/check_apv2_mainpaper_runtime_draft.py
        ```

        ## 边界 / Boundary

        本仓库证明 AP-Core runtime 机制: 状态池、短期叙事槽、残差召回、后继预测、压力动力学、online learned vector、行动反馈与持久化。开放中文对话学习验证位于 `APV2-GL-OpenWorld-Chinese` 仓库; 冻结 artifact 位于 `APV2-Reproduction-Artifacts` 仓库。
