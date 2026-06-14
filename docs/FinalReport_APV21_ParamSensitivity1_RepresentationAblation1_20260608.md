# APV2.1 ParamSensitivity-1 / Representation-Ablation-1 最终汇总报告

日期: 2026-06-08  
执行流程: 设计 -> 审查完善 -> 通过落地 -> 严谨验收测试 -> 最终汇总报告

## 1. 本轮目标

本轮回应两个高杀伤力评审质疑:

1. STP-v2 过程锚点结果是否只是单点调参?
2. AP 是否只是因为使用任务特化结构化表示而占优?

本轮没有把这两个问题写成防御性声明, 而是直接做成两个可复核实验:

- `ParamSensitivity-1`
- `Representation-Ablation-1`

## 2. 设计

### 2.1 ParamSensitivity-1

固定 STP-v2 v0.4 的 D1-trained process-anchor policy 和 private examiner, 扫描:

- threshold offset
- feeling gain
- anchor noise sigma
- repair weight bias
- trigger weight bias

验收标准不是“每个参数都满分”, 而是:

- base config 通过;
- core basin 有稳定通过率;
- extreme region 出现可解释失败边界;
- public records 无 private key / answer key / solver / LLM 泄漏。

### 2.2 Representation-Ablation-1

比较五个表示层级:

| 组别 | 含义 |
|---|---|
| R1 structured process SA | 当前 STP-v2 numeric process anchors |
| R2 surface text token | D1 表面 token/marker 基线 |
| R3 generic process-event bridge | 从公开过程事件标签重构 anchors |
| R4 domain surface adapter | 分域表面适配器, 非 AP-native 上界 |
| R5 shuffled process bridge | 同分布但 case-level 打乱的过程事件反证 |

该实验的关键不是否认表示设计重要, 而是分离“白箱表示工程优势”“外界表面关键词优势”和“过程来源正确的内源事件优势”。

## 3. 审查完善

已形成设计与自审文档:

- `docs/Design_APV21_ParamSensitivity1_RepresentationAblation1_20260608.md`
- `docs/Review_APV21_ParamSensitivity1_RepresentationAblation1_Design_20260608.md`

自审中特别锁定四个风险:

1. ParamSensitivity 不能只报告最好点。
2. process labels 不能变成 private answer labels。
3. domain adapter 必须标为 non AP-native。
4. 表示消融结论不能写成开放世界胜负。

## 4. 通过落地

新增 runner:

- `scripts/run_param_sensitivity_1.py`
- `scripts/run_representation_ablation_1.py`

新增测试:

- `tests/test_param_sensitivity_1.py`
- `tests/test_representation_ablation_1.py`

生成 artifact:

### ParamSensitivity-1

目录: `outputs/param_sensitivity_1_20260608/`

| 文件 | 作用 |
|---|---|
| `summary.json` | 机器可读汇总 |
| `records.json` | 配置级 public records |
| `private_examiner_key.json` | 复现用 private examiner |
| `ParamSensitivity1_report_zh.md` | 中文报告 |
| `param_sensitivity_1_showcase_zh.html` | 展示页 |
| `artifact_manifest.json` | 本地 hash manifest |

### Representation-Ablation-1

目录: `outputs/representation_ablation_1_20260608/`

| 文件 | 作用 |
|---|---|
| `summary.json` | 机器可读汇总 |
| `records.json` | public records |
| `private_examiner_key.json` | 复现用 private examiner |
| `RepresentationAblation1_report_zh.md` | 中文报告 |
| `representation_ablation_1_showcase_zh.html` | 展示页 |
| `artifact_manifest.json` | 本地 hash manifest |

## 5. 严谨验收测试

### 5.1 单项测试

```powershell
python -m pytest tests\test_param_sensitivity_1.py -q
```

结果:

```text
4 passed in 71.19s
```

```powershell
python -m pytest tests\test_representation_ablation_1.py -q
```

结果:

```text
4 passed in 1.18s
```

### 5.2 组合回归

```powershell
python -m pytest tests\test_stpv2_process_anchor_ablation_v03.py tests\test_stpv2_process_anchor_transfer_v04.py tests\test_param_sensitivity_1.py tests\test_representation_ablation_1.py -q
```

结果:

```text
18 passed, 1 warning in 110.80s
```

warning 来自 legacy `audioop` deprecation, 与本轮实验无关。

## 6. 结果

### 6.1 ParamSensitivity-1

schema: `apv21_param_sensitivity_1/v0.1`  
records: `4,860`  
validation: `true`

| 指标 | 值 |
|---|---:|
| base pass rate | 1.000 |
| base macro accuracy | 1.000 |
| core basin n | 1,215 |
| core basin pass rate | 0.9407 |
| core basin macro accuracy | 0.9845 |
| extreme region n | 3,645 |
| extreme region pass rate | 0.8211 |
| extreme failure count | 652 |
| extreme min macro accuracy | 0.500 |

结论: STP-v2 过程锚点存在稳定参数盆地, 不是单点阈值偶然; 同时极端区域有明确失败边界, 说明实验不是自证式永远通过。

### 6.2 Representation-Ablation-1

schema: `apv21_representation_ablation_1/v0.1`  
records: `2,700`  
validation: `true`

| 组别 | macro | D2 | D3 | FP | FN | AP-native |
|---|---:|---:|---:|---:|---:|---|
| R1 structured process SA | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | true |
| R2 surface text token | 0.5926 | 0.500 | 0.500 | 0.000 | 0.4074 | false |
| R3 process-event bridge | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | true |
| R4 domain surface adapter | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | false |
| R5 shuffled process bridge | 0.5222 | 0.5167 | 0.5222 | 0.1833 | 0.2945 | false |

结论: 结构化表示确实重要, 但 AP 的优势不只是“外界关键词写得好”。R3 只从公开过程事件 bridge 重构 anchors, 仍能达到 R1 水平; R5 同分布但错位后失败, 说明关键是 case-level 过程来源正确。R4 高分说明分域规则也能强, 但它是 non AP-native 工程上界, 成本和可迁移性不同。

## 7. 论文与答疑稿更新

已更新:

- `docs/APV21_PublicPaper_InitialDraft_v1_0m_20260608.md`
- `docs/APV21_VideoFAQ_PublicInitialDraft_v1_0m_20260608.md`

更新内容:

1. 摘要层补入 ParamSensitivity-1 与 Representation-Ablation-1 核心数字。
2. 第 9 章将两项从 future route 升级为 local controlled appendix 已完成 / 待 public freeze。
3. 答疑稿新增“是不是单点调参”和“是不是结构化表示占便宜”的实测回答。

## 8. 当前可支持的新结论

现在可以更有底气地说:

1. STP-v2 过程锚点不仅有因果贡献和跨表面迁移, 还存在稳定参数盆地。
2. 表示设计是 AP 白箱能力工程的一部分, 但过程来源正确的内源事件比外界表面关键词更关键。
3. process-event bridge 能恢复 structured process SA 的效果, 说明下一步接真实 runtime trace / GL trace 是合理路线。
4. domain-specific adapter 可以高分, 但必须标为非 AP-native 上界, 不能混入 AP-Core 证据。

## 9. 下一步

最稳路线:

1. 把这两个实验纳入 public artifact freeze 清单。
2. 做 `OOD-Generalization-1 D1-D3`, 直接沿用 STP-v2 process-event bridge。
3. 等 GL/桌宠真实 trace 成熟后, 做 R3 的 runtime bridge 版, 检验真实接口 trace 能否自然产生同类 process events。
4. 再做 LongRun-Stability-1, 回应持续认知长期稳定性质疑。

