# Phase17C 观测台 Digest 控制台补强验收记录

日期：2026-05-19

## 1. 本轮目标

在不改变 AP 二期核心哲学、不退回 JSON-first 观测方式的前提下，继续把观测台从“结构化表格 + JSON 面板”推进到更接近研究控制台的形态：

1. `Sidecar 结构` 不再只靠细表阅读，新增第一眼摘要层
2. `Live 结构` 不再只显示 ring 与 run 列表，新增运行态 digest
3. `长期研究专题` 不再只给统计结论表，新增研究判断型 digest
4. 保持原有白箱细表不丢失，形成“摘要卡 -> 细表 -> JSON”的三层阅读路径

## 2. 本轮实际完成

### 2.1 Sidecar 结构新增 digest 卡

新增 4 张 digest 卡：

1. `感受器采样摘要`
2. `召回与预测主线`
3. `focus / exact 摘要`
4. `规则与动作摘要`

对应真实字段口径：

- `input_item`
- `sensor_packet.sampling_summary`
- `sensor_packet.fatigue_summary`
- `sensor_packet.sa_flow`
- `bn_list`
- `c_i_list`
- `c_star.items`
- `focus_memory`
- `exact_memory`
- `rules_result`
- `sandbox_result`
- `state_pool_sidecar.last_pool_result`

目标效果：

1. 第一眼就能看到当前 tick 是什么输入
2. 第一眼就能看到最强一级召回、最强预测分支、C* 主峰
3. 第一眼就能看到 focus memory 与 exact memory 的区别
4. 第一眼就能看到规则、情绪、动作驱动力有没有形成闭环

### 2.2 Live 结构新增 digest 卡

新增 4 张 digest 卡：

1. `运行新鲜度`
2. `近期主线摘要`
3. `Run 混合摘要`
4. `最新 Run 摘要`

对应真实字段口径：

- `latestLive.status`
- `latestLive.server_time_ms`
- `latestLive.latest_tick`
- `latestLive.recent_ticks`
- `latestLive.known_runs`
- `state.freshness.tickStallCount`
- `state.latestTick.generated_at_ms`

目标效果：

1. 第一眼知道 live 当前是 `running` 还是 `completed`
2. 第一眼知道最近 focus 主线是什么
3. 第一眼知道当前 run 混合态，比如最近是否全 completed
4. 第一眼知道最新 run 的 label、进度、路径

### 2.3 长期研究专题新增 digest 卡

新增 4 张 digest 卡：

1. `逻辑耗时判断`
2. `规模与容量判断`
3. `规则与来源热点`
4. `风险与提醒摘要`

对应真实字段口径：

- `getRunCompareSeries(8)` 的近期 run 汇总
- `aggregateHistogramFromRuns('rules_fired_histogram')`
- `aggregateHistogramFromRuns('candidate_source_histogram')`
- `state.rulesWarnings`
- `state.tunerWarnings`

目标效果：

1. 不只给数字，而是直接给“当前处于轻负载 / 可接受 / 偏高”的判断
2. 直接看出 memory/state 的增长跨度与增量
3. 直接看出长期最热规则与最热来源
4. 直接看到 warning 风险概况

## 3. 新增前端锚点

本轮新增 DOM 锚点如下：

### Sidecar digest

- `sidecarSensorDigestBox`
- `sidecarRecallDigestBox`
- `sidecarMemoryDigestBox`
- `sidecarActionDigestBox`

### Live digest

- `liveFreshnessDigestBox`
- `liveFocusDigestBox`
- `liveRunDigestBox`
- `liveMetaDigestBox`

### Research digest

- `researchLogicDigestBox`
- `researchCapacityDigestBox`
- `researchHotspotDigestBox`
- `researchRiskDigestBox`

## 4. 新增/调整的脚本逻辑

### 新增辅助函数

- `topHistogramEntry(...)`
- `textPreview(...)`
- `joinLines(...)`
- `shallowCount(...)`
- `recentTickRangeText(...)`
- `dominantStatusText(...)`
- `formatRunProgress(...)`

用途：

1. 统一 digest 文案拼装逻辑
2. 避免不同区域各自写一套简化规则
3. 保证“主线 / 热点 / 风险”摘要口径尽量一致

### 改造渲染函数

- `renderSidecarCards()`
- `renderLiveCards()`
- `renderResearchPanels()`

原则：

1. 不另起新视图状态
2. 仍复用现有 `renderAll()` 主链
3. 摘要层与明细表共享同一数据源
4. 不引入新的重型前端缓存结构

## 5. 验收结果

### 5.1 自动化测试

通过：

```bash
python -m unittest tests.test_observatory_frontend_phase17 -v
python -m unittest discover -s tests -v
```

结果：

1. 前端专项 2 项通过
2. 全量 32 项通过

### 5.2 HTTP 验收

对 `http://127.0.0.1:8766/` 拉源码，确认以下锚点存在：

- `sidecarSensorDigestBox`
- `sidecarRecallDigestBox`
- `sidecarMemoryDigestBox`
- `sidecarActionDigestBox`
- `liveFreshnessDigestBox`
- `liveFocusDigestBox`
- `liveRunDigestBox`
- `liveMetaDigestBox`
- `researchLogicDigestBox`
- `researchCapacityDigestBox`
- `researchHotspotDigestBox`
- `researchRiskDigestBox`

### 5.3 真页验收

用新开页签 + 时间戳 query 的方式验收真实页面，确认：

1. 页面标题正常：`AP 二期观测台 V2`
2. `Sidecar 结构` / `Live 结构` / `长期研究专题` 均存在
3. 切到对应页签后 digest 卡不再是空盒子，而是有真实文本

真页样例信号：

#### Sidecar digest 实测样例

- `Bn 6 / top=算了 不说了 / score=0.68`
- `C_i 6 / top=c_local::mem_000006 / virtual=0.68`
- `C* 16 SA / top=天 / energy=11.06`
- `rules 3 / top=稳定一级召回触发正确感`

#### Live digest 实测样例

- `status=completed / latest_run=phase2_text_min_loop_20260519_073954_242`
- `recent_ticks 22 / range=0 -> 2`
- `top focus=不 (14)`
- `progress=3 / 3`

#### Research digest 实测样例

- `当前逻辑耗时处于很轻的实验级负载`
- `memory 0 -> 12`
- `top_rule=rule::bn_correctness (12)`
- `logic_peak_guard=稳定`

## 6. 本轮意义

这一轮的意义不是“又加了几张卡”，而是进一步把观测台的阅读路径从：

`先看表 -> 再自己总结`

推进到：

`先看 digest 抓主线 -> 再下钻细表 -> 最后必要时看 JSON`

这会直接提升：

1. 长跑观测效率
2. 规则与调参迭代效率
3. 白箱叙事能力
4. 后续做更复杂多模态实验时的可读性上限

## 7. 当前仍可继续补强的点

虽然本轮 digest 层已经明显更完整，但还可以继续做：

1. `Sidecar 结构` 的 recall / predict 行做点击联动，支持直接跳到对应 tick 或 memory 文本链
2. `Live 结构` 把 freshness digest 和 refresh health 进一步合并成更像值班监控卡的形态
3. `长期研究专题` 后续可增加“异常 run 自动标红”或“热点规则变化趋势”卡
4. 若未来前端数据量继续扩大，可考虑把部分 digest 预计算下沉到后端 summary 层

## 8. 结论

本轮已经把观测台继续从“结构化原型页面”往“研究控制台”推进了一步，而且没有牺牲现有白箱细节、自动化覆盖和长跑稳定性。

当前结果符合本轮目标，可作为后续继续做：

1. 更深入的长跑研究可视化
2. 更强的规则 / 调参交互
3. 更专业的多模态实验观测

的稳定基础。
