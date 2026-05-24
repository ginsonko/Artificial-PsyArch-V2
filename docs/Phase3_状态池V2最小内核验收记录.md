# AP 二期 Phase 3 状态池 V2 最小内核验收记录

日期：2026-05-19

## 本阶段目标

对应执行手册中的：

1. 阶段 3：状态池 V2 最小内核

本阶段目标不是实现完整 HDB-V2，也不是实现 IESM、短期记忆或后继优势全链，而是先把：

`SA 主体状态池 + 热锚点缓存 + 残留桶 + 轻量句柄环 + 可观测 sidecar`

这一层真正落地，并验证它满足以下边界：

1. 主能量仍只有一份，不出现“第二状态池”
2. 旁路结构是有界的、可失效的、事件驱动更新的
3. `R_state` 可以从更完整的现状摘要中读头，而不只是全局 top
4. 长跑下不会明显膨胀

## 本阶段已落地内容

1. [state_pool_v2.py](<H:\AP原型实验第二期\core\state_pool_v2.py>) 升级为 Phase 3 最小内核
2. [config.py](<H:\AP原型实验第二期\observatory_v2\config.py>) 新增状态池旁路参数
3. `app_config.schema.json` 新增旁路参数校验
4. `runtime_config.json` 新增默认旁路上限
5. [app.py](<H:\AP原型实验第二期\observatory_v2\app.py>) 把状态池旁路摘要写入 `summary / sidecar / metrics`
6. [test_state_pool_phase3.py](<H:\AP原型实验第二期\tests\test_state_pool_phase3.py>) 新增 Phase 3 单元测试
7. [test_text_sensor_phase2.py](<H:\AP原型实验第二期\tests\test_text_sensor_phase2.py>) 与 [test_web_api.py](<H:\AP原型实验第二期\tests\test_web_api.py>) 同步补 sidecar 断言

## 当前 Phase 3 的旁路结构口径

### 1. 热锚点缓存 `hot_anchor_cache`

作用：

1. 提供“当前最热对象摘要”
2. 供 `head_anchor` 读取
3. 供白箱回放观察锚点效应

注意：

1. 它不存第二份真实能量
2. 它只是从主 SA 表按 `anchor_score` 排序得到的缓存摘要
3. 有固定上限 `state_pool_anchor_cache_limit`

### 2. 残留桶 `residual_bucket`

作用：

1. 收纳“本 tick 没进入主 SA 流的截断单元”
2. 收纳“被疲劳压制但仍值得保留的未解释质量”
3. 为 `head_residual` 提供查询头

当前最小实现来源：

1. 文本预算截断部分
2. 疲劳抑制部分

注意：

1. 它存的是 `unresolved_mass`，不是主能量
2. 有独立衰减和上限
3. 只保留摘要，不展开成完整对象壳体系

### 3. 轻量句柄环 `handle_ring`

作用：

1. 为最近若干 tick 提供在线白箱句柄
2. 保留输入预览、选中 SA、锚点、残留更新等最小归因锚点
3. 为后续 Phase 5/6 的旁路叙事重建留接口

注意：

1. 它不是日志替代品
2. 它不是第二个短期记忆池
3. 只是非常小的在线窗口

## 已验证内容

### 1. 自动化测试

执行：

```bash
python -m unittest discover -s tests -v
```

结果：

1. `14` 项测试全部通过
2. Phase 3 专项测试全部通过

新增关键测试包括：

1. 残留桶能记录截断单元且保持有界
2. 热锚点缓存和句柄环不会无限增长
3. `R_state` 暴露 `head_residual`，但主表仍使用 `energy`，旁路使用 `unresolved_mass`

### 2. 500 tick 长跑探针

探针结果：

1. `tick_done = 500`
2. `state_pool_size = 30`
3. `anchor_count = 12`
4. `residual_count = 1`
5. `handle_count = 32`

说明：

1. 主表和旁路都未随 tick 线性失控
2. 旁路结构保持在配置上限附近

### 3. 5000 tick 长跑探针

本地探针结果：

1. `tick_done = 5000`
2. `status = completed`
3. `elapsed_ms = 62403.89`
4. `state_pool_size = 18`
5. `anchor_count = 12`
6. `residual_count = 1`
7. `handle_count = 32`

可见：

1. 5k tick 下没有出现明显膨胀
2. `anchor / residual / handle` 都严格被边界约束
3. 当前最小文本链下，Phase 3 的状态池层已经具备长期运行基础

## 当前可观测结果

从 `summary.jsonl` 可以直接看到：

1. `state_pool_summary`
2. `state_pool_sidecar_summary.anchor_top`
3. `state_pool_sidecar_summary.residual_top`

从 `sidecar.jsonl` 可以直接看到：

1. `state_pool_sidecar.hot_anchor_cache`
2. `state_pool_sidecar.residual_bucket`
3. `state_pool_sidecar.handle_ring`
4. `state_pool_sidecar.last_pool_result`

从 `metrics.jsonl` 可以直接看到：

1. `state_pool_anchor_count`
2. `state_pool_residual_count`
3. `state_pool_size`

## 当前阶段结论

Phase 3 已经成立，且满足执行手册中最关键的边界要求：

1. 状态池主竞争面仍然是 SA 主表
2. 旁路结构没有长成第二状态池
3. `R_state` 已经不再只看单一全局 top
4. 长跑下大小稳定
5. 观测台和离线日志都能直接看到旁路证据

## 当前仍存在的限制

1. 目前旁路仍是最小文本版本，还没有接入 `C* / Bn / provenance_buffer`
2. 残留桶目前只接了文本截断和疲劳压制来源
3. 还没有动作目标壳、奖励/惩罚绑定壳、预测壳
4. 还没有短期记忆池与后继优势
5. 还没有真实 HDB-V2 混合召回

## 为什么现在先停在这里

因为执行手册要求：

1. 先把状态池主核和轻量旁路层做稳
2. 先证明它不会在长期运行中膨胀
3. 再进入 HDB-V2 主链

如果在这一步之前直接跳去做 Bn / `C_i` / `C*`：

1. 旁路接口会反复返工
2. 长跑稳定性问题会被记忆召回链掩盖
3. 后续白箱调试会更难切责任

## 下一步建议

最推荐的下一步是继续按执行手册进入：

1. Phase 4：SA Registry 与多尺度竞争
2. Phase 5：HDB-V2 最小主链

在这之前，不建议先跳去做视觉、听觉、Agent 或完整前端重构。
