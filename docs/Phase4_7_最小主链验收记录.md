# AP 二期 Phase 4-7 最小主链验收记录

日期：2026-05-19

## 覆盖阶段

1. 阶段 4：SA Registry 与多尺度竞争
2. 阶段 5：HDB-V2 最小主链
3. 阶段 6：`A_focus`、短期记忆与后继优势
4. 阶段 7：IESM V2 最小可用集

## 本阶段目标

在 Phase 3 的状态池稳定底座上，先把真正的最小认知主链跑通：

`文本输入 -> SA 竞争 -> 状态池 -> R_state -> Bn -> C_i -> C* -> rules -> A_focus -> 短期记忆 -> 长期记忆`

目标不是一步到位实现论文中的全部强版本，而是先把：

1. 高阶 SA 与基础 SA 共存
2. 最小一级召回 Bn 成立
3. 最小二级预测包 `C_i` 和单主综合预测包 `C*` 成立
4. `A_focus` 与短期链成立
5. 最小规则注入成立

## 已落地内容

1. [sa_registry_v2.py](<H:\AP原型实验第二期\core\sa_registry_v2.py>)
2. [memory_store_v2.py](<H:\AP原型实验第二期\memory\memory_store_v2.py>)
3. [short_term_memory_v2.py](<H:\AP原型实验第二期\memory\short_term_memory_v2.py>)
4. [rules_engine_v2.py](<H:\AP原型实验第二期\iesm\rules_engine_v2.py>)
5. [runtime_v2.py](<H:\AP原型实验第二期\core\runtime_v2.py>)
6. [app.py](<H:\AP原型实验第二期\observatory_v2\app.py>) 已切到统一 runtime 主循环

## 当前正式口径

### Phase 4：SA Registry

当前版本支持：

1. 基础文本单位 SA
2. 少量内置 phrase prototype
3. 基础 SA 与高阶 phrase SA 共存

当前特意保留的边界：

1. 高阶 SA 会参与召回与状态池
2. 但 `A_focus` 仍默认优先读最近原始单位链
3. 这样既保住显意识连续输出，又不丢高阶结构缓存

### Phase 5：最小 HDB 主链

当前最小版采用：

1. `R_state` 多头读出
2. 基于标签与权重重叠的最小 Bn 召回
3. 基于时间邻域与局部后继偏置的 `C_i`
4. 聚合成单个 `C*`

当前还不是完整版本：

1. 还没有 ANN / posting / beam 重建的正式工程版
2. 还没有时空数据库和向量库
3. 但语义角色已经成立

### Phase 6：`A_focus` 与短期记忆

当前已具备：

1. `A_focus` 从更新后的状态池读出
2. 最近几轮 `A_focus` 进入短期记忆
3. 二级预测分支会读取最近焦点尾部做最小后继偏置

### Phase 7：IESM 最小可用集

当前最小规则层已具备：

1. 基于 residual 的违和感
2. 基于 Bn 稳定命中的正确感雏形
3. 基于 `C*` 的期待感雏形
4. 违和带动压力
5. 规则可向状态池注入属性项
6. 规则可输出行动驱动力摘要

## 已验证内容

### 自动化测试

执行：

```bash
python -m unittest discover -s tests -v
```

结果：

1. `19` 项测试全部通过
2. 新增 `test_runtime_chain_phase4_7.py`

关键验证点：

1. phrase prototype 能命中
2. Bn 至少能召回已有历史
3. `C*` 会生成
4. 短期链会积累
5. 规则层有对象级输出

## 当前可观测结果

从 `summary.jsonl` 可以看到：

1. `competition_summary`
2. `bn_preview`
3. `c_star_preview`
4. `rules_preview`
5. `short_term_preview`

从 `sidecar.jsonl` 可以看到：

1. `competition_packet`
2. `bn_list`
3. `c_i_list`
4. `c_star`
5. `rules_result`
6. `short_term_snapshot`
7. `focus_memory`
8. `exact_memory`

## 当前限制

1. 召回仍是最小白箱实现，不是最终复杂版 HDB-V2
2. 规则层仍只有最小集
3. 后继优势目前还是轻量偏置，不是论文里更完整的连续场版本
4. 还没有视觉/听觉接入主循环

## 阶段结论

Phase 4-7 最小主链已经成立。

也就是说，现在系统已经不再只是：

`外源文本 -> 状态池 -> A_focus`

而是已经升级为：

`外源文本 -> SA 竞争 -> 现状读出 -> 一级召回 -> 二级预测 -> 规则调制 -> 焦点短链 -> 记忆回写`

这已经是二期真正主线的第一版工程落地。
