# AP二期 先天规则编码底层可追踪条件总表与建议规则表

版本：Draft v0.1
生成日期：2026-05-18
适用范围：AP 二期 HDB-V2 / IESM V2 / 后续规则编辑器、审计台、实验脚本
文档性质：冷保存设计底表 / 规则字典 / 能力边界清单 / 迁移与扩展规划

---

## 1. 文档目标

这份文档不是为了先给出“某几条灵感规则”，而是为了把 **AP 二期先天规则系统到底能读什么、能判什么、能改什么、一期已经真实实现了什么、哪些东西虽能追踪但还没用、二期最值得补哪些拟人规则** 这几层问题一次性拉平。

它要服务的不是单次讨论，而是后续这几件事：

1. 给未来的大模型或人类开发者一个可复用的规则边界词典。
2. 为未来的图形化规则编辑器、规则 DSL、规则审计台提供枚举基础。
3. 避免后续规则继续走“想到什么加什么”的经验主义路线。
4. 让二期 IESM 与 HDB-V2 的主线流程、状态池、注意力、短期记忆、传感器、奖惩与行动系统形成统一口径。

本文件重点回答五个问题：

1. 一期先天规则引擎现在到底已经支持哪些条件与动作。
2. 一期默认 58 条规则分别做了什么。
3. 哪些条件虽然已经可追踪，但默认规则还没有动用。
4. 到二期 HDB-V2 / 感受器 / 短期记忆 / 现状读出 `R_state` / `Bn` / `C_i` / `C*` 之后，规则层还应该新增哪些条件。
5. 从拟人持续认知闭环角度，建议优先补哪些先天规则。

---

## 2. 本次真实核对范围

本表不是纯概念脑补，而是基于一期原型当前实现做的代码核对。主要核对文件如下：

- `H:\PA原型测试\Artificial-PsyArch\innate_script\_rules_engine.py`
- `H:\PA原型测试\Artificial-PsyArch\innate_script\config\innate_rules.yaml`
- `H:\PA原型测试\Artificial-PsyArch\innate_script\main.py`
- `H:\PA原型测试\Artificial-PsyArch\observatory\_app.py`
- `H:\PA原型测试\Artificial-PsyArch\action\main.py`
- `H:\PA原型测试\Artificial-PsyArch\attention\main.py`
- `H:\PA原型测试\Artificial-PsyArch\emotion\main.py`
- `H:\PA原型测试\Artificial-PsyArch\innate_script\docs\先天规则编辑指南.md`

因此文档里会明确区分三类内容：

- `一期已实现 / 已接线`：代码里真的能跑。
- `一期已可追踪但默认未用`：引擎或上下文已经能提供，但默认 YAML 没拿它写规则。
- `二期建议新增`：为了 HDB-V2 / 现状读出 / 传感器 / 后继优势 / 属性实例化 / 视觉听觉等新口径，需要新增追踪与新规则。

---

## 3. 一期 IESM 的真实能力边界

- 默认规则数：**58**
- phase 分布：`{'directives': 33, 'cfs': 23, 'emotion_post': 2}`
- 顶层 when 分布：`{'state_window': 2, 'metric': 12, 'all': 18, 'cfs': 26}`
- then 动作分布：`{'emit_script': 2, 'cfs_emit': 22, 'pool_bind_attribute': 15, 'branch': 5, 'emotion_update': 25, 'action_trigger': 8}`
- 指标预设数：**54**
- 指标别名数：**82**

### 3.1 一期 when 真实支持

- `any`
- `all`
- `not`
- `cfs`
- `state_window`
- `timer`
- `metric`

### 3.2 一期 then 真实支持

- `cfs_emit`：运行态追加 CFS 信号，可选同时绑定属性刺激元；可被同 tick 后续规则继续读取。（yes/native / used_by_default）
- `focus`：产出 focus_directives，后续由注意力与 ActionManager 消费。（yes/native / unused_by_default）
- `emit_script`：仅做可观测触发记录，不直接改变主系统状态。（yes/native / used_by_default）
- `emotion_update`：输出递质通道增量，交由 emotion.update_emotion_state 消费。（yes/native / used_by_default）
- `action_trigger`：输出结构化行动驱动，交由 ActionManager 转为内部 trigger。（yes/native / used_by_default）
- `pool_energy`：输出状态池能量变更 effect，交由 observatory 安全执行。（yes/native / unused_by_default）
- `pool_bind_attribute`：输出绑定属性 effect，交由 observatory 安全执行。（yes/native / used_by_default）
- `delay`：在 runtime_state 中登记延时动作队列，到期后再执行。（yes/native / unused_by_default）
- `branch`：在规则内部做 if/else/on_error 分支，可引用刚算出的变量。（yes/native / used_by_default）
- `log`：追加审计日志。（yes/native / unused_by_default）

### 3.3 一期 selector 真实支持

- `mode=all`：全量对象选择。选全部活跃候选对象，默认排除 context-only 类型。
- `mode=specific_item`：指定 item_id。定向操作某个状态池对象实例。
- `mode=specific_ref`：指定 ref_object_id/ref_object_type。面向某个语义对象/结构对象，而非某个瞬时 item。
- `mode=contains_text`：文本包含匹配。在 display/detail/feature/attribute 文本中做轻量语义文本门控。
- `mode=top_n`：按总能量 top_n。常见于对象级感受只看高显著若干对象。
- `mode=has_attribute`：具有某属性。历史兼容默认 runtime 口径，也可显式 scope。
- `mode=has_packet_attribute`：具有 packet 属性。只看记忆/结构本身携带的属性。
- `mode=has_runtime_attribute`：具有 runtime 属性。只看运行时绑定属性。
- `mode=has_any_attribute`：具有任意口径属性。packet+runtime 并集。
- `attribute_names + require_all`：属性 names require_all。支持多属性交集选择。
- `ref_object_types=[...]`：按 ref_object_types 过滤。类型级裁剪，是性能和口径控制的重要手段。
- `selector.where[field]={op,value}`：数值 where 过滤。在选对象时叠加第二层数值门控，避免 when.metric 过于单薄。

### 3.4 一期 metric 真实支持的求值模式

- `state`：当前值
- `prev_state`：上一 tick 值
- `delta`：变化量
- `avg_rate`：近窗口平均变化率
- `changed`：对 `state` 语义增强为“与上一 tick 是否不同”
- `prev_gate`：先看上一 tick 是否满足另一层约束，再决定当前条件是否生效

### 3.5 一期 metric compare 真实支持

- `exists`
- `changed`
- `>=`
- `>`
- `<=`
- `<`
- `==`
- `!=`
- `between`

---

## 4. 二期单 tick 中先天规则的推荐介入位置

为了和当前 HDB-V2 草案统一口径，二期建议把先天规则系统理解为一个**横切式调制层**，而不是单一阶段的外挂。它至少应在一个 tick 内参与以下位置：

1. `状态池维护后`：可读取当前完整状态、疲劳、近因、属性实例、情绪基础态。
2. `内外源刺激合流并写入短期记忆时`：可决定是否生成某些认知感受节点、是否给新记忆打标签。
3. `现状读出 R_state 构建时`：可根据注意力、情绪、任务、时间感受调节各查询头预算。
4. `Bn 一级召回后`：可对期待/压力/熟悉度/违和等做对象级或全局级判定。
5. `Pred2 / C_i / C* 构建后`：可对预测自洽性、后继优势、回音稳定性、行动前景做规则计算。
6. `emotion_post`：在奖励/惩罚覆盖后再做一轮递质后处理，避免奖惩与规则顺序反咬。
7. `action`：把 focus/action_trigger 交给 ActionManager 竞争执行。

因此，二期 IESM 不能只盯着“当前显著对象是否大于阈值”，它还要盯：

- 现状读出器在这一拍到底读了什么。
- 一级召回/二级预测的结构是否稳定。
- 近期短期主线是否正在形成连续语序。
- 视觉/听觉焦点是否应该本能转移。
- 当前对象是否因为疲劳、近因锚点、期待验证而改变竞争力。

---

## 5. 底层可追踪条件总表

### 5.1 阅读说明

下表不是“只列出现在默认 YAML 里出现过的条件”，而是按底层能力做族群化整理。一个条件族往往意味着：

- 有一类可直接读取的字段，或
- 有一类按统一数学规则派生出来的量，或
- 二期应当新增并标准化输出到规则上下文的数据。

表中字段说明：

- `family_id`：稳定编号，便于以后做规则编辑器、搜索、审计。
- `scope`：作用域，区分 global / pool / item / retrieval / sensor / V2 query 等。
- `path_or_formula`：一期真实字段路径或二期推荐字段名/公式名。
- `ap1_status`：一期是否已经原生可用。
- `ap1_usage`：一期默认规则是否已经动用。
- `v2_priority`：二期建议优先级。

### 5.2 条件族总表

| family_id | 中文名 | scope | path / 公式 | compare_ops | modes | 一期状态 | 一期默认使用 | 二期优先级 | 说明 |
|---|---|---|---|---|---|---|---|---|---|
| META-001 | tick 序号 | global/meta | `tick_index` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/derived | unused_by_default | high | 所有定时、节律、冷却、阶段依赖规则的最基础参照量。 |
| META-002 | tick 阶段 | global/meta | `allowed_phases / phase = cfs|directives|emotion_post` | ==,!=,in | state | yes/native | used_by_engine | high | 用于把规则分配到同一 tick 内的不同执行相位，避免竞态与时序混乱。 |
| META-003 | 输入角色标记 | stimulus/item | `input_is_user / input_is_assistant / input_is_system` | >=,==,changed | state,delta | yes/native | used_by_default | high | 区分外界来源，决定是否允许工具行动、是否加强礼貌性回复、是否忽略系统噪声。 |
| META-004 | 输入种类标记 | stimulus/item | `input_is_message / input_is_reply / input_is_session_restore / input_stream_kind` | >=,==,changed | state,delta | yes/native | used_by_default | high | 区分普通消息、回复、会话恢复、未来多模态传感器等通道。 |
| META-005 | 输入是否为空 | stimulus/global | `stimulus.input_is_empty / stimulus.input_has_text` | >=,==,changed | state,delta,avg_rate | yes/native | used_by_default | medium | 用于空闲/无外界刺激时的内源主导、无聊感、发散探索等规则。 |
| POOL-001 | 状态池实能量总量 | pool/global | `pool.total_er` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | high | 状态池级全局统计量，用于驱动复杂度、专注度、内外平衡、整体压力和全局调参规则。 |
| POOL-002 | 状态池虚能量总量 | pool/global | `pool.total_ev` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | high | 状态池级全局统计量，用于驱动复杂度、专注度、内外平衡、整体压力和全局调参规则。 |
| POOL-003 | 状态池总能量 | pool/global | `pool.total_energy` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | high | 状态池级全局统计量，用于驱动复杂度、专注度、内外平衡、整体压力和全局调参规则。 |
| POOL-004 | 状态池对象数 | pool/global | `pool.item_count` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | high | 状态池级全局统计量，用于驱动复杂度、专注度、内外平衡、整体压力和全局调参规则。 |
| POOL-005 | 状态池认知压总量 | pool/global | `pool.total_cp_delta` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | high | 状态池级全局统计量，用于驱动复杂度、专注度、内外平衡、整体压力和全局调参规则。 |
| POOL-006 | 状态池认知压大小总量 | pool/global | `pool.total_cp_abs` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | high | 状态池级全局统计量，用于驱动复杂度、专注度、内外平衡、整体压力和全局调参规则。 |
| POOL-007 | 状态池能量聚集度 | pool/global | `pool.energy_concentration` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | high | 状态池级全局统计量，用于驱动复杂度、专注度、内外平衡、整体压力和全局调参规则。 |
| POOL-008 | 状态池有效波峰数量 | pool/global | `pool.effective_peak_count` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | high | 状态池级全局统计量，用于驱动复杂度、专注度、内外平衡、整体压力和全局调参规则。 |
| POOL-009 | 状态池综合复杂度 | pool/global | `pool.complexity_score` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/derived | used_by_default | high | 状态池级全局统计量，用于驱动复杂度、专注度、内外平衡、整体压力和全局调参规则。 |
| POOL-010 | 状态池核心复杂度 | pool/global | `pool.core_complexity_score` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/derived | used_by_default | high | 状态池级全局统计量，用于驱动复杂度、专注度、内外平衡、整体压力和全局调参规则。 |
| POOL-011 | 状态池上下文对象数 | pool/global | `pool.context_item_count` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native_context | unused_by_default | high | 状态池级全局统计量，用于驱动复杂度、专注度、内外平衡、整体压力和全局调参规则。 |
| POOL-012 | 状态池上下文预算上限 | pool/global | `pool.context_item_limit` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native_context | unused_by_default | high | 状态池级全局统计量，用于驱动复杂度、专注度、内外平衡、整体压力和全局调参规则。 |
| POOL-013 | 状态池核心能量聚集度 | pool/global | `pool.core_energy_concentration` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/derived | unused_by_default | high | 状态池级全局统计量，用于驱动复杂度、专注度、内外平衡、整体压力和全局调参规则。 |
| POOL-014 | 状态池核心有效波峰数 | pool/global | `pool.core_effective_peak_count` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/derived | unused_by_default | high | 状态池级全局统计量，用于驱动复杂度、专注度、内外平衡、整体压力和全局调参规则。 |
| CAM-001 | 当前注意记忆体大小 | cam/global | `cam.size` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | high | 衡量显意识窗口拥挤程度，可直接参与繁、简、聚焦、发散规则。 |
| CAM-002 | 当前注意记忆体聚集度 | cam/global | `cam.energy_concentration` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | medium | 衡量显意识是否被一个峰主导还是处于多峰竞争态。 |
| MAP-001 | 记忆赋能池条目数 | memory_activation/global | `memory_activation.item_count` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | high | 衡量当前回忆/赋能链条是否已形成足够可用的激活记忆。 |
| MAP-002 | 记忆赋能池总虚能量 | memory_activation/global | `memory_activation.total_ev` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | high | 用于衡量整体预测势能、联想势能和回忆拥挤程度。 |
| RET-001 | 刺激级最佳匹配分数 | retrieval/stimulus | `retrieval.stimulus.best_match_score` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | high | 当前现状与历史记忆最近邻的总体贴合度，是把握感和熟悉感的重要来源。 |
| RET-002 | 刺激级把握感综合分数 | retrieval/stimulus | `retrieval.stimulus.grasp_score` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/derived | used_by_default | high | 对最佳匹配、残余比例、EV 稳定性、结构支持等做的综合判定，直接适合驱动 grasp / familiarity 规则。 |
| RET-003 | 刺激级目标分数表 | retrieval/stimulus | `retrieval.stimulus.match_scores[target_id]` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | high | 允许规则针对某个具体记忆目标或别名对象做定向判断，而不是只看全局 best。 |
| RET-004 | 结构级最佳匹配分数 | retrieval/structure | `retrieval.structure.best_match_score` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | medium | 衡量结构层而非刺激层的匹配质量，用于更高层概念规则。 |
| RET-005 | 结构级目标分数表 | retrieval/structure | `retrieval.structure.match_scores[group_id]` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | medium | 可将规则定向绑定到某类 group/结构范式上。 |
| RET-006 | 刺激级剩余比例 | retrieval/stimulus | `stimulus.residual_ratio` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | high | 衡量当前现状中仍有多少部分未被现有记忆解释，是新奇、困惑、好奇、恐怖谷的重要信号。 |
| EMO-001 | 奖励信号状态 | emotion/global | `emotion.rwd` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | high | 全局奖励态，用于期待、愉悦、动机增强与正向学习速率调节。 |
| EMO-002 | 惩罚信号状态 | emotion/global | `emotion.pun` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | high | 全局惩罚态，用于压力、抑制、规避与错误学习调节。 |
| NT-001 | 情绪递质状态 | emotion/nt_channel | `emotion.nt.{channel}` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | high | 多巴胺、肾上腺素、血清素、催产素、皮质醇、专注等通道的直接规则入口。 |
| NT-002 | 情绪递质变化量 | emotion/nt_channel | `emotion.nt.{channel}` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | high | 多巴胺、肾上腺素、血清素、催产素、皮质醇、专注等通道的直接规则入口。 |
| NT-003 | 情绪递质变化率 | emotion/nt_channel | `emotion.nt.{channel}` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | high | 多巴胺、肾上腺素、血清素、催产素、皮质醇、专注等通道的直接规则入口。 |
| NT-004 | 情绪递质发生变化 | emotion/nt_channel | `emotion.nt.{channel}` | >=,>,<=,<,==,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | high | 多巴胺、肾上腺素、血清素、催产素、皮质醇、专注等通道的直接规则入口。 |
| ITEM-001 | 对象实能量 | pool/item | `item.er` | >=,>,<=,<,==,between,exists,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | high | 单对象层的基本能量与可见性入口，是所有具象认知感受、对象级期待/压力、对象级动作驱动的核心条件。 |
| ITEM-002 | 对象虚能量 | pool/item | `item.ev` | >=,>,<=,<,==,between,exists,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | high | 单对象层的基本能量与可见性入口，是所有具象认知感受、对象级期待/压力、对象级动作驱动的核心条件。 |
| ITEM-003 | 对象总能量 | pool/item | `item.total_energy` | >=,>,<=,<,==,between,exists,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | high | 单对象层的基本能量与可见性入口，是所有具象认知感受、对象级期待/压力、对象级动作驱动的核心条件。 |
| ITEM-004 | 对象认知压 | pool/item | `item.cp_delta` | >=,>,<=,<,==,between,exists,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | high | 单对象层的基本能量与可见性入口，是所有具象认知感受、对象级期待/压力、对象级动作驱动的核心条件。 |
| ITEM-005 | 对象认知压大小 | pool/item | `item.cp_abs` | >=,>,<=,<,==,between,exists,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | high | 单对象层的基本能量与可见性入口，是所有具象认知感受、对象级期待/压力、对象级动作驱动的核心条件。 |
| ITEM-006 | 对象疲劳度 | pool/item | `item.fatigue` | >=,>,<=,<,==,between,exists,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | high | 单对象层的基本能量与可见性入口，是所有具象认知感受、对象级期待/压力、对象级动作驱动的核心条件。 |
| ITEM-007 | 对象近因增益 | pool/item | `item.recency_gain` | >=,>,<=,<,==,between,exists,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | high | 单对象层的基本能量与可见性入口，是所有具象认知感受、对象级期待/压力、对象级动作驱动的核心条件。 |
| ITEM-008 | 对象存在性 | pool/item | `item.exists` | >=,>,<=,<,==,between,exists,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | high | 单对象层的基本能量与可见性入口，是所有具象认知感受、对象级期待/压力、对象级动作驱动的核心条件。 |
| ITEM-009 | 对象 salience_score | pool/item | `item.salience_score` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 可直接作为对象显著性代理，适合未来替代 top_n 的简单排序逻辑。 |
| ITEM-010 | 对象 delta_er | pool/item | `item.delta_er` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 当没有完整历史时可作为 ER 变化量回退。 |
| ITEM-011 | 对象 delta_ev | pool/item | `item.delta_ev` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 当没有完整历史时可作为 EV 变化量回退。 |
| ITEM-012 | 对象 delta_cp_delta | pool/item | `item.delta_cp_delta` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 支持认知压带符号变化触发。 |
| ITEM-013 | 对象 delta_cp_abs | pool/item | `item.delta_cp_abs` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 支持违和/正确感对称通道。 |
| ITEM-014 | 对象 er_change_rate | pool/item | `item.er_change_rate` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 对象级 ER 变化率的即时近似。 |
| ITEM-015 | 对象 ev_change_rate | pool/item | `item.ev_change_rate` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 对象级 EV 变化率的即时近似。 |
| ITEM-016 | 对象 cp_delta_rate | pool/item | `item.cp_delta_rate` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 对象级认知压变化率的即时近似。 |
| ITEM-017 | 对象 cp_abs_rate | pool/item | `item.cp_abs_rate` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 对象级 |CP| 变化率的即时近似。 |
| ITEM-018 | 对象更新时间 | pool/item | `item.updated_at / last_update_tick` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 用于短时锚点、衰减、最近焦点偏置与新鲜度门控。 |
| ITEM-019 | 对象更新次数 | pool/item | `item.update_count` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 用于熟悉度、重复感、固化候选强度。 |
| ITEM-020 | 对象创建时间 | pool/item | `item.created_at` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 可用于幼年记忆保护期、短期对象和长期对象分流。 |
| ITEM-021 | 对象类型 | pool/item | `item.ref_object_type` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | medium | st/sg/sa/input/em 等类型区分，是 selector 的基本入口。 |
| ITEM-022 | 对象主显示文本 | pool/item | `item.display / display_text` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | medium | 文本 contains_text 选择器与人类审计的基础字段。 |
| ITEM-023 | 对象细节显示 | pool/item | `item.display_detail` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | medium | 适合做更细粒度文本命中或前端解释。 |
| ITEM-024 | 对象语义签名 | pool/item | `item.semantic_signature` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native | unused_by_default | medium | 可作为未来 rule-level 相似选择或互斥条件。 |
| ITEM-025 | 对象别名 ref_alias_ids | pool/item | `item.ref_alias_ids` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native | used_implicitly | medium | 支持同语义对象跨 SA/ST/别名的聚焦与验证归并。 |
| ITEM-026 | 对象上下文拥有者 | pool/item | `item.context_owner_structure_id / context_owner_id` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 可用于‘同一事件壳内’规则。 |
| ITEM-027 | 对象上下文路径 | pool/item | `item.context_path_ids` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 适合未来层级传播规则。 |
| ITEM-028 | 对象源记忆 id | pool/item | `item.memory_id / source_memory_created_at` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 可区分热记忆、老记忆、刚写入记忆。 |
| ITEM-029 | 对象残余来源 | pool/item | `item.residual_origin_kind / residual_origin_entry_id` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 适合‘残余未解释对象’专门规则。 |
| ITEM-030 | 对象目标引用 | pool/item | `item.target_ref_object_id / target_item_id` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 用于动作、期待、压力、时间感受等面向某目标的规则。 |
| ITEM-031 | 对象时间桶信息 | pool/item | `item.time_bucket_ref_object_id / time_bucket_center_sec / time_basis / time_bucket_unit` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | used_by_default | medium | 时间感受规则的关键旁路结构。 |
| ITEM-032 | 对象验证锚点 | pool/item | `item.verification_anchor_*` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | used_by_default | medium | 期待/压力验证分支的关键桥梁。 |
| ITEM-033 | 对象自带属性名列表 | pool/item | `item.packet_attribute_names / runtime_attribute_names / all_attribute_names` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native | used_by_default | medium | 属性型 selector 的稳定键来源。 |
| ITEM-034 | 对象绑定属性数量 | pool/item | `item.bound_attribute_count` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 适合做‘属性饱和度’、复杂度与异常绑定监控。 |
| ITEM-035 | 对象绑定的 CSA id | pool/item | `item.bound_csa_item_id` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 可用于未来对象壳/绑定关系可视化。 |
| ITEM-036 | 对象状态位 | pool/item | `item.status` | ==,!=,contains,exists,>=,>,<=,<,between,changed | state,prev_state,delta,avg_rate | yes/native_summary | unused_by_default | medium | 为将来 inactive/decaying/shadowed 等状态预留。 |

### 5.3 二期新增条件族建议（V2 专属）

| family_id | 中文名 | scope | 推荐字段 | 用途 |
|---|---|---|---|---|
| V2-001 | R_state 总预算长度 | query/global | `R_state.total_slots` | 固定预算读出器的真实负载大小。 |
| V2-002 | R_state 各头占比 | query/global | `R_state.head_{core,input,predict,task,time,sensory}.ratio` | 判断当前召回更受哪一类上下文驱动。 |
| V2-003 | R_state 增量量 | query/global | `R_state.delta_added_sa_count / delta_replaced_sa_count` | 用于固定增量缓存策略与环境复杂度估计。 |
| V2-004 | 短期原文窗口长度 | short_memory/global | `STM.raw_window_token_count` | 决定后继优势与精确短句保持能力。 |
| V2-005 | 短期记忆命中率 | short_memory/global | `STM.cache_hit_ratio` | 用于动态决定是否扩大原文窗口影响力。 |
| V2-006 | Bn 数量 | retrieval/v2 | `retrieval_v2.bn_count` | 当前第一层召回命中的稳定记忆对象数量。 |
| V2-007 | Bn 最大分数 | retrieval/v2 | `retrieval_v2.bn_top_score` | 当前最强历史记忆命中度。 |
| V2-008 | Bn 分数熵/离散度 | retrieval/v2 | `retrieval_v2.bn_score_entropy` | 用于判断是单峰回忆还是多峰竞争。 |
| V2-009 | 一级匹配实能量总量 | retrieval/v2 | `retrieval_v2.bn_total_er` | 直接反映现状刺激流被一级匹配分支吸收了多少现实度。 |
| V2-010 | 一级匹配虚能量总量 | retrieval/v2 | `retrieval_v2.bn_total_ev` | 反映现状的想象/预测牵引有多强。 |
| V2-011 | 二级预测对象数 | retrieval/v2 | `retrieval_v2.pred2_count` | 衡量未来综合预测场的分叉程度。 |
| V2-012 | 综合预测包 C* 峰数 | prediction/v2 | `prediction.cstar_peak_count` | 用于判断未来图景是否明确。 |
| V2-013 | 综合预测包 C* 一致度 | prediction/v2 | `prediction.cstar_coherence` | 用于期待/压力是否收敛。 |
| V2-014 | 综合预测包 C* 与现状一致度 | prediction/v2 | `prediction.cstar_alignment_to_state` | 一致度低时更可能产生违和感、好奇或不确定。 |
| V2-015 | 抽象回声能量 | echo/v2 | `echo.total_energy` | 衡量抽象结构在当前 tick 是否足够成形。 |
| V2-016 | 抽象回声尖锐度 | echo/v2 | `echo.peak_sharpness` | 区分模糊云团还是清晰抽象。 |
| V2-017 | 后继优势中心 | sequence/v2 | `succession.center_lag` | 当前后继优势以哪个短期位置为中心。 |
| V2-018 | 后继优势衰减曲线 | sequence/v2 | `succession.kernel_strength(lag)` | 用于语序保持和自然思维跳跃。 |
| V2-019 | 视焦点位置 | sensor/vision | `vision.focus_xy / focus_abs_xy / focus_depth` | 实现视焦点转移、注视、追踪与具身视觉。 |
| V2-020 | 视野采样预算 | sensor/vision | `vision.sample_budget_per_tick` | 固定视觉复杂度的重要约束。 |
| V2-021 | 视觉显著性热点数 | sensor/vision | `vision.salient_patch_count` | 亮点过多时可触发聚焦/扫描行为。 |
| V2-022 | 视觉对象相对坐标 | sensor/vision | `vision.object.rel_x/rel_y/rel_z/angle/elevation/distance` | 视觉对象的时空定位基础。 |
| V2-023 | 音频采样预算 | sensor/audio | `audio.sample_budget_per_tick` | 固定听觉复杂度的重要约束。 |
| V2-024 | 音频瞬时响度峰值 | sensor/audio | `audio.max_amplitude` | 用于惊觉、声音注意转移。 |
| V2-025 | 音频主频 / 音高 / 节奏 | sensor/audio | `audio.freq / pitch / beat_interval` | 支持音乐、语调、节奏感与歌曲召回。 |
| V2-026 | 属性实例显著性 | attribute_instance/v2 | `attr_instance.salience` | 决定属性是否独立进入状态池。 |
| V2-027 | 属性通道数值 | attribute_instance/v2 | `attr_channel.value` | 例如违和/正确共用一条有符号通道。 |
| V2-028 | 结构候选竞争得分 | sa_competition/v2 | `sa_competition.candidate_score` | 多尺度 SA/结晶 SA 竞争的核心可追踪量。 |
| V2-029 | 结构候选疲劳值 | sa_competition/v2 | `sa_competition.candidate_fatigue` | 整体识别与逐步拆解切换的关键量。 |
| V2-030 | 查询缓存命中率 | cache/v2 | `query_cache.hit_ratio` | 固定增量算法是否真正有效的重要指标。 |
| V2-031 | 教师奖励/惩罚局部别名命中率 | teacher/v2 | `teacher_alias.hit_ratio` | 用于判断教师监督是否被系统正确接住。 |
| V2-032 | 现实度通道 | reality/v2 | `item.reality_score / source_realness` | 区分实际看见、回忆、想象、语言描述。 |

### 5.4 条件族示例说明

- `META-001 tick 序号`：
  - 每 10 tick 触发一次探索冲动
  - 在 session_restore 后前 5 tick 禁止高风险行动
- `META-002 tick 阶段`：
  - 只允许奖励兑现规则在 emotion_post 阶段运行
- `META-003 输入角色标记`：
  - 只有用户输入才能触发天气查询 stub
- `META-004 输入种类标记`：
  - session_restore 阶段降低教师别名缓存写入
- `META-005 输入是否为空`：
  - 连续空输入时提升内源性联想预算
- `POOL-001 状态池实能量总量`：
  - 状态池实能量总量 高于阈值时切入聚焦模式
  - 状态池实能量总量 长期下降时触发疲劳恢复
- `POOL-002 状态池虚能量总量`：
  - 状态池虚能量总量 高于阈值时切入聚焦模式
  - 状态池虚能量总量 长期下降时触发疲劳恢复
- `POOL-003 状态池总能量`：
  - 状态池总能量 高于阈值时切入聚焦模式
  - 状态池总能量 长期下降时触发疲劳恢复
- `POOL-004 状态池对象数`：
  - 状态池对象数 高于阈值时切入聚焦模式
  - 状态池对象数 长期下降时触发疲劳恢复
- `POOL-005 状态池认知压总量`：
  - 状态池认知压总量 高于阈值时切入聚焦模式
  - 状态池认知压总量 长期下降时触发疲劳恢复
- `POOL-006 状态池认知压大小总量`：
  - 状态池认知压大小总量 高于阈值时切入聚焦模式
  - 状态池认知压大小总量 长期下降时触发疲劳恢复
- `POOL-007 状态池能量聚集度`：
  - 状态池能量聚集度 高于阈值时切入聚焦模式
  - 状态池能量聚集度 长期下降时触发疲劳恢复
- `POOL-008 状态池有效波峰数量`：
  - 状态池有效波峰数量 高于阈值时切入聚焦模式
  - 状态池有效波峰数量 长期下降时触发疲劳恢复
- `POOL-009 状态池综合复杂度`：
  - 状态池综合复杂度 高于阈值时切入聚焦模式
  - 状态池综合复杂度 长期下降时触发疲劳恢复
- `POOL-010 状态池核心复杂度`：
  - 状态池核心复杂度 高于阈值时切入聚焦模式
  - 状态池核心复杂度 长期下降时触发疲劳恢复
- `POOL-011 状态池上下文对象数`：
  - 状态池上下文对象数 高于阈值时切入聚焦模式
  - 状态池上下文对象数 长期下降时触发疲劳恢复
- `POOL-012 状态池上下文预算上限`：
  - 状态池上下文预算上限 高于阈值时切入聚焦模式
  - 状态池上下文预算上限 长期下降时触发疲劳恢复
- `POOL-013 状态池核心能量聚集度`：
  - 状态池核心能量聚集度 高于阈值时切入聚焦模式
  - 状态池核心能量聚集度 长期下降时触发疲劳恢复

这里特意只展开前若干个典型例子，避免正文爆得太散。后续若要把整张表喂给规则编辑器，建议把 `examples` 再单独抽成 JSON 资源。

---

## 6. 一期 selector / 目标对象面完整说明

### 6.1 一期真实 selector 总表

| selector_id | 名称 | 表达式 | 一期状态 | 说明 |
|---|---|---|---|---|
| SEL-001 | 全量对象选择 | `mode=all` | yes/native | 选全部活跃候选对象，默认排除 context-only 类型。 |
| SEL-002 | 指定 item_id | `mode=specific_item` | yes/native | 定向操作某个状态池对象实例。 |
| SEL-003 | 指定 ref_object_id/ref_object_type | `mode=specific_ref` | yes/native | 面向某个语义对象/结构对象，而非某个瞬时 item。 |
| SEL-004 | 文本包含匹配 | `mode=contains_text` | yes/native | 在 display/detail/feature/attribute 文本中做轻量语义文本门控。 |
| SEL-005 | 按总能量 top_n | `mode=top_n` | yes/native | 常见于对象级感受只看高显著若干对象。 |
| SEL-006 | 具有某属性 | `mode=has_attribute` | yes/native | 历史兼容默认 runtime 口径，也可显式 scope。 |
| SEL-007 | 具有 packet 属性 | `mode=has_packet_attribute` | yes/native | 只看记忆/结构本身携带的属性。 |
| SEL-008 | 具有 runtime 属性 | `mode=has_runtime_attribute` | yes/native | 只看运行时绑定属性。 |
| SEL-009 | 具有任意口径属性 | `mode=has_any_attribute` | yes/native | packet+runtime 并集。 |
| SEL-010 | 属性 names require_all | `attribute_names + require_all` | yes/native | 支持多属性交集选择。 |
| SEL-011 | 按 ref_object_types 过滤 | `ref_object_types=[...]` | yes/native | 类型级裁剪，是性能和口径控制的重要手段。 |
| SEL-012 | 数值 where 过滤 | `selector.where[field]={op,value}` | yes/native | 在选对象时叠加第二层数值门控，避免 when.metric 过于单薄。 |

### 6.2 selector 的真实工程意义

一期 selector 设计很重要，因为它已经隐含了你提出的那种对象口径：

- `任意对象`：`mode=all`
- `某个特定对象`：`specific_item` / `specific_ref`
- `含有某特征对象`：`contains_text` / `has_attribute` / `ref_object_types`
- `满足其它条件的对象`：`selector.where` 作为对象选择期的数值子门控
- `由其它规则传入的对象`：通过 `capture_as` + `{{{match_item_id}}}` / `{{{match_ref_object_id}}}` 在后续动作和 branch 中继续传递

也就是说，一期 IESM 虽然还不是一个完整“可视化对象查询语言”，但它已经有了相当明显的雏形。二期真正需要做的不是从零发明，而是：

1. 把 selector 的对象口径正式制度化。
2. 把 `R_state / Bn / C_i / C* / 属性实例 / 视觉对象 / 音频对象 / 行动节点` 这些新对象也变成 selector 能选的族。
3. 把文本 contains_text 从临时调试手段，逐步升级为“只在输入型或审计型场景使用”，真正运行期更多用结构化属性和对象 family。

---

## 7. 一期可执行动作总表

| action_id | 动作名 | 一期状态 | 默认使用 | 一句话说明 |
|---|---|---|---|---|
| ACT-001 | `cfs_emit` | yes/native | used_by_default | 运行态追加 CFS 信号，可选同时绑定属性刺激元；可被同 tick 后续规则继续读取。 |
| ACT-002 | `focus` | yes/native | unused_by_default | 产出 focus_directives，后续由注意力与 ActionManager 消费。 |
| ACT-003 | `emit_script` | yes/native | used_by_default | 仅做可观测触发记录，不直接改变主系统状态。 |
| ACT-004 | `emotion_update` | yes/native | used_by_default | 输出递质通道增量，交由 emotion.update_emotion_state 消费。 |
| ACT-005 | `action_trigger` | yes/native | used_by_default | 输出结构化行动驱动，交由 ActionManager 转为内部 trigger。 |
| ACT-006 | `pool_energy` | yes/native | unused_by_default | 输出状态池能量变更 effect，交由 observatory 安全执行。 |
| ACT-007 | `pool_bind_attribute` | yes/native | used_by_default | 输出绑定属性 effect，交由 observatory 安全执行。 |
| ACT-008 | `delay` | yes/native | unused_by_default | 在 runtime_state 中登记延时动作队列，到期后再执行。 |
| ACT-009 | `branch` | yes/native | used_by_default | 在规则内部做 if/else/on_error 分支，可引用刚算出的变量。 |
| ACT-010 | `log` | yes/native | unused_by_default | 追加审计日志。 |

### 7.1 一期动作的真实消费路径

- `cfs_emit`：在 `evaluate_rules(...)` 内直接往运行态 `cfs_signals` 追加，本 tick 后续规则可继续读取。
- `focus` / `focus_directives`：由注意力模块读取；ActionManager 也会把 focus 指令折算成 `attention_focus` 内部 trigger。
- `emotion_update`：由 `emotion.update_emotion_state(...)` 消费。
- `action_trigger`：由 `ActionManager._triggers_from_action_triggers(...)` 转为内部 trigger，再参与行动竞争。
- `pool_energy` / `pool_bind_attribute`：由 `observatory._apply_innate_pool_effects(...)` 安全执行到 StatePool。
- `delay`：只在 IESM 的 `runtime_state.scheduled_actions` 里记账，不直接碰外部模块。
- `emit_script` / `log`：偏审计、观测、联调用。

### 7.2 二期对动作面的建议扩展

二期最值得新增的动作不是“随便让规则能做一切”，而是做几类 **足够强但仍可审计** 的结构化动作：

1. `query_budget_update`：临时调整 `R_state` 各头预算。
2. `sensor_focus_move`：移动视觉焦点或未来的具身感受器焦点。
3. `predictive_bias_update`：调整后继优势核、查询头权重、属性实例化阈值等可微调参数。
4. `short_memory_tag`：为新写入短期记忆打上“主线片段 / 情绪片段 / 惩罚片段 / 教师片段”等标签。
5. `attribute_instance_emit`：在显著属性超过阈值时，生成属性实例 SA 进入状态池。
6. `teacher_guardrail`：对高风险行动增加一层规则侧安全闸。

这样仍然是结构化白箱，不会走向一个“规则里直接写 Python 任意执行”的失控系统。

---

## 8. 一期 58 条默认规则逐条映射

下面这一节的目标不是简单把 YAML 复制过来，而是把每条规则放到认知闭环里解释清楚。

| 序号 | rule_id | phase | 顶层 when | 主要动作 | 语义作用 | V2 迁移建议 |
|---|---|---|---|---|---|---|
| 1 | `state_window_fast_cp_rise` | directives | state_window | `emit_script` | 状态窗口快速上升记录 | 保留，迁移到 V2 作为底层调试钩子，不必直接参与主算法。 |
| 2 | `state_window_fast_cp_drop` | directives | state_window | `emit_script` | 状态窗口快速下降记录 | 保留，服务正确感/缓解事件观测。 |
| 3 | `cfs_dissonance_from_cp_abs` | cfs | metric | `cfs_emit` | 高 |CP| 生成违和感 | 升级为 signed 通道的一侧。 |
| 4 | `cfs_punish_signal_from_cp_abs_high` | cfs | metric | `pool_bind_attribute` | 高违和桥接为惩罚线索 | 保留，但将来可由 Bn/C* 父链归因。 |
| 5 | `cfs_correct_event_from_cp_abs_drop` | cfs | metric | `cfs_emit, pool_bind_attribute, pool_bind_attribute` | 违和显著回落生成正确事件与奖励线索 | 保留，并与 signed 通道对称化。 |
| 6 | `cfs_surprise_from_er_surge_cp_positive` | cfs | metric | `cfs_emit` | ER 突增且 CP 正时生成惊 | 可继续使用，未来适配视觉/音频突发。 |
| 7 | `cfs_expectation_from_reward_pred` | cfs | metric | `cfs_emit, branch` | 预测+奖励线索生成期待，并做渐变验证/不验 | 直接保留，迁移到 V2 一级/二级召回。 |
| 8 | `cfs_expectation_from_teacher_reward_runtime_state_fallback` | cfs | all | `cfs_emit, branch` | 教师奖励落在运行态对象时的期待补盲 | 保留，后期随着教师淡出逐步减弱。 |
| 9 | `cfs_pressure_from_punish_pred` | cfs | metric | `cfs_emit, branch` | 预测+惩罚线索生成压力 | 直接保留，迁移到 V2 一级/二级召回。 |
| 10 | `cfs_pressure_from_teacher_punish_runtime_state_fallback` | cfs | all | `cfs_emit, branch` | 教师惩罚运行态补盲 | 保留，后期淡出。 |
| 11 | `cfs_pressure_from_runtime_punish_pred_fallback` | cfs | all | `cfs_emit, branch` | 运行态惩罚预测补盲 | 保留，但可并入统一 fallback 族。 |
| 12 | `cfs_grasp_from_stimulus_match` | cfs | metric | `cfs_emit` | 高 grasp_score 生成把握感 | 保留，并接入 V2 的 R_state / Bn。 |
| 13 | `cfs_complexity_from_cam_size` | cfs | metric | `cfs_emit, pool_bind_attribute` | 复杂度生成繁感 | 保留，未来更多看 pool/C* 多峰度。 |
| 14 | `cfs_simplicity_from_low_complexity` | cfs | all | `cfs_emit, pool_bind_attribute` | 低复杂度生成简单感 | 保留。 |
| 15 | `cfs_relief_from_cp_abs_drop` | cfs | all | `cfs_emit` | 认知压下降生成缓解 | 保留。 |
| 16 | `cfs_reassurance_from_settled_relief` | cfs | all | `cfs_emit, pool_bind_attribute` | 缓解稳定后生成安心 | 保留，并可接 C* 自洽度。 |
| 17 | `cfs_repetition_from_item_fatigue` | cfs | metric | `cfs_emit, pool_bind_attribute` | 对象疲劳高生成重复感 | 保留，与传感器底噪疲劳共用哲学。 |
| 18 | `cfs_familiarity_from_grasp_and_low_residual` | cfs | all | `cfs_emit, pool_bind_attribute` | 高把握低残余生成熟悉感 | 保留。 |
| 19 | `cfs_deja_vu_from_familiarity_without_memory` | cfs | all | `cfs_emit, pool_bind_attribute` | 熟悉但没记忆解释时生成既视感 | 保留，未来可更依赖 Bn 稀疏错位。 |
| 20 | `cfs_fear_unknown_from_high_cp_low_grasp` | cfs | all | `cfs_emit, pool_bind_attribute` | 高压低把握生成未知恐惧 | 保留。 |
| 21 | `cfs_uncanny_valley_from_high_match_high_residual` | cfs | all | `cfs_emit, pool_bind_attribute` | 高匹配高残余生成恐怖谷 | 保留，非常关键。 |
| 22 | `cfs_curiosity_from_manageable_residual` | cfs | all | `cfs_emit, pool_bind_attribute` | 可控残余生成好奇 | 保留。 |
| 23 | `cfs_boredom_from_repetition_low_novelty` | cfs | all | `cfs_emit, pool_bind_attribute` | 重复且低新奇生成无聊 | 保留。 |
| 24 | `cfs_uncertainty_from_cp_low_grasp_many_peaks` | cfs | all | `cfs_emit, pool_bind_attribute` | 低把握多峰竞争生成不确定 | 保留，二期可更精细。 |
| 25 | `cfs_agency_readiness_from_expectation_grasp` | cfs | all | `cfs_emit, pool_bind_attribute` | 期待+把握生成行动就绪 | 保留，衔接主动性。 |
| 26 | `nt_update_from_dissonance` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 27 | `nt_update_from_correct_event` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 28 | `nt_update_from_surprise` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 29 | `nt_update_from_expectation` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 30 | `nt_update_from_pressure` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 31 | `nt_update_from_expectation_verified` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 32 | `nt_update_from_expectation_unverified` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 33 | `nt_update_from_pressure_verified` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 34 | `nt_update_from_pressure_unverified` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 35 | `nt_update_from_complexity` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 36 | `nt_update_from_repetition` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 37 | `nt_update_from_grasp` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 38 | `nt_update_from_simplicity` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 39 | `nt_update_from_relief` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 40 | `nt_update_from_reassurance` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 41 | `nt_update_from_familiarity` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 42 | `nt_update_from_deja_vu` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 43 | `nt_update_from_fear_unknown` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 44 | `nt_update_from_uncanny_valley` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 45 | `nt_update_from_curiosity` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 46 | `nt_update_from_boredom` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 47 | `nt_update_from_uncertainty` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 48 | `nt_update_from_agency_readiness` | directives | cfs | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 49 | `nt_update_from_reward_state` | emotion_post | metric | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 50 | `nt_update_from_punish_state` | emotion_post | metric | `emotion_update` | 默认规则语义见 note，属于一期现有链条的一部分 | 二期按同名语义迁移，必要时重写触发条件。 |
| 51 | `innate_action_attention_focus_from_core_cfs` | directives | cfs | `action_trigger` | 核心认知感受触发聚焦 | 保留，但未来更建议 focus/action_trigger 合并审计。 |
| 52 | `innate_action_weather_stub_from_query_weather` | directives | all | `action_trigger` | 明确天气查询触发工具行动 | 一期 demo 规则，二期保留为环境测试模板，不作为通用心智规则。 |
| 53 | `innate_action_weather_stub_from_weather_question` | directives | all | `action_trigger` | 隐式天气问句触发中强工具行动 | 同上。 |
| 54 | `innate_action_weather_stub_from_weather_only` | directives | all | `action_trigger` | 轻微提及天气触发弱工具行动 | 同上。 |
| 55 | `innate_action_recall_from_expect_pressure` | directives | all | `action_trigger` | 期待/压力驱动回忆 | 保留且很重要，二期 recall 会更核心。 |
| 56 | `innate_action_recall_from_time_feeling_bucket_gain` | directives | metric | `action_trigger` | 时间感受对象获得能量时触发回忆 | 保留，但时间感受可能由时间维度与属性实例共同承担。 |
| 57 | `innate_action_attention_focus_mode_from_complexity_high` | directives | cfs | `action_trigger` | 复杂度高时切聚焦模式 | 保留。 |
| 58 | `innate_action_attention_diverge_mode_from_complexity_low` | directives | cfs | `action_trigger` | 复杂度低时切发散模式 | 保留。 |

### 8.1 按主题重组一期默认规则

为了更适合后续编辑器分类，一期 58 条规则可以按主题重组为：

1. **状态窗口触发观察**：2 条
2. **对象级认知感受生成**：23 条 CFS 规则
3. **认知感受到递质更新**：25 条 NT 更新规则
4. **行动 / 聚焦 / 回忆 / demo 工具触发**：8 条

这个分组非常有启发性，因为它说明了一期其实已经在形成一条清晰的逻辑：

`状态/对象变化 -> 认知感受 -> 递质/行动/聚焦 -> 下一拍状态改变`

而二期并不是推翻这条链，只是要把条件层从“少量对象指标 + retrieval 摘要”扩展到“完整现状读出 + 一级记忆匹配 + 二级预测 + 传感器 + 短期主线 + 属性实例”。

---

## 9. 已实现但默认未充分使用的能力

### 9.1 指标预设已实现但默认未使用

当前默认规则实际用到的预设是：`['complexity_score', 'core_complexity_score', 'cp_abs_state', 'er_rate', 'er_state', 'ev_state', 'fatigue_state', 'got_cp_abs', 'got_er', 'got_total_energy', 'grasp_score', 'map_item_count', 'nt_state', 'pool_effective_peak_count', 'punish_state', 'reward_state', 'stimulus_match_score', 'stimulus_residual_ratio', 'total_energy_state']`。

当前默认规则**没有**用到、但引擎已经支持的预设有：`['cam_concentration', 'cam_size', 'cp_abs_rate', 'cp_rate', 'cp_state', 'ev_rate', 'got_cp', 'got_ev', 'map_total_ev', 'nt_changed', 'nt_delta', 'nt_rate', 'pool_concentration', 'pool_concentration_rate', 'pool_cp_abs_rate', 'pool_cp_abs_total', 'pool_cp_rate', 'pool_cp_total', 'pool_er_rate', 'pool_er_total', 'pool_ev_rate', 'pool_ev_total', 'pool_item_count', 'pool_total_energy', 'pool_total_energy_got', 'pool_total_energy_rate', 'punish_got', 'punish_rate', 'recency_state', 'reward_got', 'reward_rate', 'stimulus_match_score_target', 'structure_match_score', 'structure_match_score_target', 'total_energy_rate']`。

其中我认为价值最高的未充分利用预设有：

- `cam_size` -> `cam.size`：当前注意记忆体大小（CAM 条目数）
- `cam_concentration` -> `cam.energy_concentration`：CAM 能量聚集度（Herfindahl 指数）
- `cp_rate` -> `item.cp_delta`：认知压变化率（CP 近 N tick 平均）
- `cp_abs_rate` -> `item.cp_abs`：认知压大小变化率（|CP| 近 N tick 平均）
- `er_rate` -> `item.er`：实能量变化率（ER 近 N tick 平均）
- `ev_rate` -> `item.ev`：虚能量变化率（EV 近 N tick 平均）
- `pool_concentration` -> `pool.energy_concentration`：状态池能量聚集度（Herfindahl 指数）
- `pool_concentration_rate` -> `pool.energy_concentration`：状态池能量聚集度变化率（Herfindahl 指数 近 N tick 平均）
- `pool_total_energy` -> `pool.total_energy`：状态池总能量（ΣER+ΣEV）
- `pool_total_energy_rate` -> `pool.total_energy`：状态池总能量变化率（ΣER+ΣEV 近 N tick 平均）
- `map_total_ev` -> `memory_activation.total_ev`：记忆赋能池总虚能量（MAP ΣEV）
- `nt_delta` -> `emotion.nt.{channel}`：情绪递质变化量（NT，需填写 channel）
- `nt_rate` -> `emotion.nt.{channel}`：情绪递质变化率（NT，近 N tick 平均，需填写 channel）
- `recency_state` -> `item.recency_gain`：近因增益状态（Recency Gain）
- `structure_match_score` -> `retrieval.structure.best_match_score`：查存一体匹配分数（结构级，best_match_score）
- `structure_match_score_target` -> `retrieval.structure.match_scores`：查存一体匹配分数（结构级，按目标 group_id）
- `stimulus_match_score_target` -> `retrieval.stimulus.match_scores`：查存一体匹配分数（刺激级，按目标 structure_id）

### 9.2 动作已实现但默认未使用

- `focus`：引擎支持，默认 YAML 没直接使用；目前更多由 `action_trigger(kind=attention_focus)` 间接达成。
- `pool_energy`：已支持落到状态池，但默认规则没正式用。二期会很有用，尤其适合认知感受直接调参或能量注入实验。
- `delay`：已支持延时调度，但默认规则没正式启用。未来很适合做“短延迟后确认/后悔/反刍/反射动作”。
- `log`：已支持，但默认规则没使用。适合复杂 V2 原型中做审计钩子。

### 9.3 selector 已实现但默认未充分使用

- `specific_ref`：只在分支或动作定向里隐式用得较多，默认顶层条件用得还不够。
- `specific_item`：主要用于 branch 验证链，很少被当作一般规则条件使用。
- `has_packet_attribute / has_runtime_attribute / has_any_attribute` 的分裂口径还没被系统性利用。
- `selector.where` 这个能力其实很强，但默认规则只拿它做了少量 CP/输入门控。

### 9.4 历史模式已实现但默认未充分使用

- `prev_state`
- `delta`
- `avg_rate`
- `changed`
- `prev_gate`

尤其是 `prev_gate`，它正是你现在想表达的许多拟人规则的核心基础，例如：

- 先有压力，再下降 -> 缓解
- 先有期待，再有 ER 验证 -> 满足感
- 先有违和，再稳定下降 -> 正确感
- 先被强烈注意，再持续关注 -> 逐步拆解为更细粒度 SA

---

## 10. 二期最推荐补充的拟人先天规则

| rec_id | 规则名 | 主要读取条件 | 主要动作 | 目的 |
|---|---|---|---|---|
| REC-001 | 违和感通道对称化 | 对象级 CP/|CP| 变化率与变化量 | 给对象写入同一 signed 通道：正值=违和，负值=正确；显著时再注入独立认知节点。 | 完全对齐用户当前口径，减少正确感/违和感两套规则的重复。 |
| REC-002 | 期待/压力双通道 + 渐变验证 | 预测对象 EV、reward/punish 属性、验证锚点 ER 变化率 | 保持 expectation/pressure + verified/unverified 四支，但统一成可连续计分的模板。 | 一期已部分实现，二期应升级到 V2 Bn/C* 框架。 |
| REC-003 | 近因锚点增益 + 极短期抑制 | item.recency_gain + 最近激活历史 | 近期被激活对象先增益后回落，极短期重复时轻抑制。 | 实现‘想到老虎时把橘猫看成小老虎’的心理锚点。 |
| REC-004 | 后继优势内核 | STM 短窗 + successor kernel | 对短期主线附近的后继对象给虚能量增益，越远越弱，过远转惩罚。 | 保持语序连续，同时允许外界/情绪打断。 |
| REC-005 | 视焦点自动转移 | 视觉对象位置 + 注意力选择结果 | 当某视觉对象被注意力选中时，注入 move_visual_focus 行动驱动力。 | 实现类似人眼追踪说话者或显著目标。 |
| REC-006 | 持续注视导致分辨率展开 | 结构候选疲劳 + 注视时长 | 同一对象持续高注意时，大结构 SA 疲劳上升，小粒度 SA 竞争力上升。 | 实现从整体看字到拆偏旁部首笔画。 |
| REC-007 | 传感器底噪疲劳 | 100 tick 窗口出现频率 | 长期恒定刺激的外源性实能量缩放下降。 | 拟人化地忽略持续不变的触觉/背景音/背景亮度。 |
| REC-008 | 复杂度驱动注意力预算 | pool.complexity_score + NT.ADR + NT.FOC | 复杂环境或高 ADR 时增加预算并收窄聚焦；低复杂度时允许发散。 | 直接服务 0.1 秒 tick 的实时稳定性。 |
| REC-009 | 属性实例化阈值 | 属性通道数值 + family 饱和度 | 只有属性显著时才生成属性实例 SA 进入状态池。 | 避免属性噪声把状态池塞爆。 |
| REC-010 | 查询缓存固定增量上限 | R_state.delta_added_sa_count | 超过预算时做稀疏采样，不超过时全收。 | 固定复杂度且在稳定环境中自然变聪明。 |
| REC-011 | 多峰不确定感 | Bn 分数熵 + pool_effective_peak_count | 当多峰竞争且 grasp 不高时注入 uncertainty。 | 比仅凭高 residual 更拟人。 |
| REC-012 | 预测图景自洽感 | C* coherence / alignment | 综合预测场越自洽，越容易产生正确感、安心感、敢行动。 | 服务主动性与稳定输出。 |
| REC-013 | 失败预感 | pressure 通道上升 + pun 持续高 + recall 未命中 | 为回忆失败、行动无路可走时注入一种先天性的退缩/规避偏置。 | 更贴近真实代理。 |
| REC-014 | 教师监督淡出机制 | teacher alias hit ratio + 自主验证成功率 | 验证率稳定后逐步降低教师奖励/惩罚对规则的权重。 | 支持从被教导向自主学习迁移。 |
| REC-015 | 传感器-动作闭环惊觉 | 音频响度峰值/视觉新热点 + 当前视焦点 | 突发输入优先争夺注意，但要受已有高压力/高正确主线抑制。 | 兼顾专注与对突发事件的自然反应。 |

### 10.1 我最建议优先落地的第一批

如果二期实现资源有限，我建议优先顺序如下：

1. `REC-001 违和/正确 signed 通道`
2. `REC-002 期待/压力双通道 + 渐变验证`
3. `REC-003 近因锚点增益 + 极短期抑制`
4. `REC-004 后继优势内核`
5. `REC-008 复杂度驱动注意力预算`
6. `REC-009 属性实例化阈值`
7. `REC-010 查询缓存固定增量上限`
8. `REC-005 视焦点自动转移`

因为这 8 条会直接决定：

- 系统是否更像一个会持续“想下去”的体，而不是每拍散掉。
- 系统是否能在不额外上复杂专门模块的前提下，维持自然语序与自然跳题。
- 系统是否能在多模态接入后仍把复杂度稳定在可控范围内。
- 系统是否能让属性、感受、期待、压力真正成为可被认知和学习的对象。

---

## 11. 与 HDB-V2 草案的统一口径

这份先天规则底表需要和已经写好的 HDB-V2 主草案保持统一，否则后面大模型继续读时会出现两套说法。当前统一口径如下：

1. 状态池仍保留高级对象壳，但 SA 能量场是主体。
2. 记忆召回不只看一个短刺激片段，而是看完整状态池现状经固定预算读出的 `R_state`。
3. 一级召回结果是 `Bn`，二级分形/时空扩展结果形成 `C_i`，默认融合为综合预测包 `C*`。
4. 注意力负责：
   - 对当前状态池显著项做滤波和放大
   - 形成新的显意识小记忆
   - 但**不再**是唯一的记忆召回查询来源；召回来源以 `R_state` 为主
5. 后继优势不是独立的额外序列系统，而是对二级预测虚能量分配和短期记忆主线的偏置核。
6. 先天规则系统不直接替代 HDB-V2 计算主流程，而是读取主流程产出的结构化摘要和旁路信息，对感受、递质、行动、聚焦、属性实例化等做调制。

这意味着二期规则上下文最终至少要扩展出这几个新块：

- `query`：`R_state` 多头预算读出摘要
- `short_memory`：短期高分辨率原文窗口、主线片段、增量缓存
- `retrieval_v2`：`Bn` 级统计
- `prediction`：`C_i` / `C*` 一致度、峰数、主导方向
- `sensor.vision` / `sensor.audio`：视觉听觉感受器摘要
- `sa_competition`：多尺度 SA / 结晶 SA 竞争摘要
- `attribute_instance`：属性实例化显著性与通道值

---

## 12. 未来规则编辑器应该怎样利用这份表

如果以后要做你说的那种“一目了然的规则编辑，而不是想到什么就手工加什么”的系统，我建议前端/编辑器严格围绕这份底表做：

1. 左侧先按 `scope` 和 `family_id` 分类浏览条件。
2. 每个条件家族显示：
   - 中文名
   - 路径/公式
   - 支持的 mode
   - 支持的 compare op
   - 一期是否真实已接线
   - 默认是否已有规则在用
3. 动作面单独做“危险等级”分层：
   - 低风险：emit_script / log
   - 中风险：cfs_emit / emotion_update / action_trigger / focus
   - 高风险：pool_energy / pool_bind_attribute / query_budget_update / sensor_focus_move
4. 每条规则允许绑定到一个“认知目的模板”：
   - 违和/正确
   - 期待/压力
   - 聚焦/发散
   - 记忆/回忆
   - 传感器本能
   - 行动偏置
   - 安全护栏

这样后面扩规则时，就不是在 YAML 上随手长草，而是在一个有完整数据边界和动作边界的词典上长。

---

## 13. 这次没有采纳或暂未采纳的口径

为了给未来大模型阅读时保留决策痕迹，这里补一段“中间讨论过但目前没采纳或暂缓采纳”的口径。

### 13.1 不采纳“完全没有高级对象壳、状态池里只有全局唯一 SA 节点”

原因不是这个想法不优雅，而是：

1. 旁路归因、规则定向、验证锚点、时间感受回忆、动作目标这些都需要一个相对稳定的对象壳。
2. 没有对象壳，许多规则虽然还能“数学上算出”，但很难说清楚‘是谁导致了这个感受/期待/压力/动作’。
3. 二期可以把实时运行尽量做成 SA 主体，但旁路结构仍应保留轻量对象壳。

### 13.2 不采纳“用专门外部语序系统强行保序”

当前更推荐：

- 短期记忆主线
- 后继优势内核
- 二级预测分配偏置

来共同塑造连续输出。这样更符合 AP 闭环哲学，也更白箱。

### 13.3 不采纳“所有属性永远独立成为 SA 并无限常驻状态池”

当前更推荐属性 family + 数值通道 + 显著时实例化，因为这样：

1. 属性能被认知。
2. 属性能参与匹配召回。
3. 但不会把状态池复杂度拉爆。

---

## 14. 最终结论

从这次底层梳理可以得出几个很明确的结论：

1. **一期 IESM 已经远不是占位接口。**
   它已经有成熟的 declarative YAML 规则引擎、对象级/全局级/情绪级/检索级条件、分支、延时、属性绑定、行动驱动与递质更新能力。

2. **一期真正缺的不是“能不能写规则”，而是“规则到底该看哪些底层量、边界是否系统化”。**
   也正因此，这份底表是必要的。

3. **二期规则层最大的升级点不在动作，而在可追踪条件面的扩张。**
   特别是 `R_state`、短期主线、`Bn`、`C_i`、`C*`、后继优势、属性实例、视觉焦点、音频节律、查询缓存命中率这些新条件，才是真正决定二期拟人性的地方。

4. **用户当前提出的拟人方向，是可以和一期 IESM 顺滑接起来的。**
   尤其是：
   - 违和/正确对称通道
   - 期待/压力与验证/不验
   - 近因锚点增益
   - 后继优势
   - 视焦点自动转移
   - 传感器底噪疲劳
   - 固定增量缓存

5. **最值得立刻做的不是再加十几条零散规则，而是先把二期规则上下文字段标准化。**
   只有上下文标准化后，后续规则编辑器、实验对比、论文写作、白箱审计才会真正清晰。

---

## 15. 附录 A：一期 54 个指标预设总表

| preset | metric | mode | window | 分组 | 中文标签 |
|---|---|---|---:|---|---|
| `cam_concentration` | `cam.energy_concentration` | state | 0 | 注意力记忆体（CAM） | CAM 能量聚集度（Herfindahl 指数） |
| `cam_size` | `cam.size` | state | 0 | 注意力记忆体（CAM） | 当前注意记忆体大小（CAM 条目数） |
| `complexity_score` | `pool.complexity_score` | state | 0 | 全局指标（Global） | 繁/简综合复杂度（complexity_score，0~1） |
| `core_complexity_score` | `pool.core_complexity_score` | state | 0 | 全局指标（Global） | 核心繁/简复杂度（core_complexity_score，0~1） |
| `cp_abs_rate` | `item.cp_abs` | avg_rate | 4 | 对象能量（Item Energy） | 认知压大小变化率（|CP| 近 N tick 平均） |
| `cp_abs_state` | `item.cp_abs` | state | 0 | 对象能量（Item Energy） | 认知压大小状态（|CP|） |
| `cp_rate` | `item.cp_delta` | avg_rate | 4 | 对象能量（Item Energy） | 认知压变化率（CP 近 N tick 平均） |
| `cp_state` | `item.cp_delta` | state | 0 | 对象能量（Item Energy） | 认知压状态（CP 带符号，ER-EV） |
| `er_rate` | `item.er` | avg_rate | 4 | 对象能量（Item Energy） | 实能量变化率（ER 近 N tick 平均） |
| `er_state` | `item.er` | state | 0 | 对象能量（Item Energy） | 实能量状态（ER 当前值） |
| `ev_rate` | `item.ev` | avg_rate | 4 | 对象能量（Item Energy） | 虚能量变化率（EV 近 N tick 平均） |
| `ev_state` | `item.ev` | state | 0 | 对象能量（Item Energy） | 虚能量状态（EV 当前值） |
| `fatigue_state` | `item.fatigue` | state | 0 | 对象能量（Item Energy） | 疲劳度状态（Fatigue） |
| `got_cp` | `item.cp_delta` | delta | 0 | 对象能量（Item Energy） | 获得认知压（CP 变化量，可正可负） |
| `got_cp_abs` | `item.cp_abs` | delta | 0 | 对象能量（Item Energy） | 获得认知压大小（|CP| 变化量） |
| `got_er` | `item.er` | delta | 0 | 对象能量（Item Energy） | 获得实能量（ER 变化量） |
| `got_ev` | `item.ev` | delta | 0 | 对象能量（Item Energy） | 获得虚能量（EV 变化量） |
| `got_total_energy` | `item.total_energy` | delta | 0 | 对象能量（Item Energy） | 获得总能量（ER+EV 变化量） |
| `grasp_score` | `retrieval.stimulus.grasp_score` | state | 0 | 查存一体（Retrieval） | 把握感/置信度综合得分（grasp_score，0~1） |
| `map_item_count` | `memory_activation.item_count` | state | 0 | 记忆赋能池（MAP） | 记忆赋能池条目数（MAP item_count） |
| `map_total_ev` | `memory_activation.total_ev` | state | 0 | 记忆赋能池（MAP） | 记忆赋能池总虚能量（MAP ΣEV） |
| `nt_changed` | `emotion.nt.{channel}` | delta | 0 | 情绪递质（NT） | 情绪递质变化了（NT，需填写 channel） |
| `nt_delta` | `emotion.nt.{channel}` | delta | 0 | 情绪递质（NT） | 情绪递质变化量（NT，需填写 channel） |
| `nt_rate` | `emotion.nt.{channel}` | avg_rate | 4 | 情绪递质（NT） | 情绪递质变化率（NT，近 N tick 平均，需填写 channel） |
| `nt_state` | `emotion.nt.{channel}` | state | 0 | 情绪递质（NT） | 情绪递质状态（NT，需填写 channel） |
| `pool_concentration` | `pool.energy_concentration` | state | 0 | 状态池（StatePool / SP） | 状态池能量聚集度（Herfindahl 指数） |
| `pool_concentration_rate` | `pool.energy_concentration` | avg_rate | 4 | 状态池（StatePool / SP） | 状态池能量聚集度变化率（Herfindahl 指数 近 N tick 平均） |
| `pool_cp_abs_rate` | `pool.total_cp_abs` | avg_rate | 4 | 状态池（StatePool / SP） | 状态池认知压大小变化率（Σ|CP| 近 N tick 平均） |
| `pool_cp_abs_total` | `pool.total_cp_abs` | state | 0 | 状态池（StatePool / SP） | 状态池认知压大小总量（Σ|CP|） |
| `pool_cp_rate` | `pool.total_cp_delta` | avg_rate | 4 | 状态池（StatePool / SP） | 状态池认知压变化率（ΣCP 近 N tick 平均，带符号） |
| `pool_cp_total` | `pool.total_cp_delta` | state | 0 | 状态池（StatePool / SP） | 状态池认知压总量（ΣCP，带符号） |
| `pool_effective_peak_count` | `pool.effective_peak_count` | state | 0 | 状态池（StatePool / SP） | 状态池有效波峰数量（≈1/聚集度） |
| `pool_er_rate` | `pool.total_er` | avg_rate | 4 | 状态池（StatePool / SP） | 状态池实能量变化率（ΣER 近 N tick 平均） |
| `pool_er_total` | `pool.total_er` | state | 0 | 状态池（StatePool / SP） | 状态池实能量总量（ΣER） |
| `pool_ev_rate` | `pool.total_ev` | avg_rate | 4 | 状态池（StatePool / SP） | 状态池虚能量变化率（ΣEV 近 N tick 平均） |
| `pool_ev_total` | `pool.total_ev` | state | 0 | 状态池（StatePool / SP） | 状态池虚能量总量（ΣEV） |
| `pool_item_count` | `pool.item_count` | state | 0 | 状态池（StatePool / SP） | 状态池对象数量（SP item_count） |
| `pool_total_energy` | `pool.total_energy` | state | 0 | 状态池（StatePool / SP） | 状态池总能量（ΣER+ΣEV） |
| `pool_total_energy_got` | `pool.total_energy` | delta | 0 | 状态池（StatePool / SP） | 状态池获得总能量（ΣER+ΣEV 变化量） |
| `pool_total_energy_rate` | `pool.total_energy` | avg_rate | 4 | 状态池（StatePool / SP） | 状态池总能量变化率（ΣER+ΣEV 近 N tick 平均） |
| `punish_got` | `emotion.pun` | delta | 0 | 奖励/惩罚（Rwd/Pun） | 惩罚信号增加（PUN 变化量） |
| `punish_rate` | `emotion.pun` | avg_rate | 4 | 奖励/惩罚（Rwd/Pun） | 惩罚信号变化率（PUN 近 N tick 平均） |
| `punish_state` | `emotion.pun` | state | 0 | 奖励/惩罚（Rwd/Pun） | 惩罚信号状态（PUN 当前值） |
| `recency_state` | `item.recency_gain` | state | 0 | 对象能量（Item Energy） | 近因增益状态（Recency Gain） |
| `reward_got` | `emotion.rwd` | delta | 0 | 奖励/惩罚（Rwd/Pun） | 奖励信号增加（RWD 变化量） |
| `reward_rate` | `emotion.rwd` | avg_rate | 4 | 奖励/惩罚（Rwd/Pun） | 奖励信号变化率（RWD 近 N tick 平均） |
| `reward_state` | `emotion.rwd` | state | 0 | 奖励/惩罚（Rwd/Pun） | 奖励信号状态（RWD 当前值） |
| `stimulus_match_score` | `retrieval.stimulus.best_match_score` | state | 0 | 查存一体（Retrieval） | 查存一体匹配分数（刺激级，best_match_score） |
| `stimulus_match_score_target` | `retrieval.stimulus.match_scores` | state | 0 | 查存一体（Retrieval） | 查存一体匹配分数（刺激级，按目标 structure_id） |
| `stimulus_residual_ratio` | `stimulus.residual_ratio` | state | 0 | 刺激级过程（Stimulus Process） | 刺激级剩余能量比例（Residual Ratio） |
| `structure_match_score` | `retrieval.structure.best_match_score` | state | 0 | 查存一体（Retrieval） | 查存一体匹配分数（结构级，best_match_score） |
| `structure_match_score_target` | `retrieval.structure.match_scores` | state | 0 | 查存一体（Retrieval） | 查存一体匹配分数（结构级，按目标 group_id） |
| `total_energy_rate` | `item.total_energy` | avg_rate | 4 | 对象能量（Item Energy） | 总能量变化率（ER+EV 近 N tick 平均） |
| `total_energy_state` | `item.total_energy` | state | 0 | 对象能量（Item Energy） | 对象总能量状态（ER+EV 当前值） |

## 16. 附录 B：一期指标别名总表

| alias | normalized preset |
|---|---|
| `CAM能量聚集度` | `cam_concentration` |
| `CAM大小` | `cam_size` |
| `当前注意记忆体大小` | `cam_size` |
| `繁简综合复杂度` | `complexity_score` |
| `繁简综合得分` | `complexity_score` |
| `核心繁简复杂度` | `core_complexity_score` |
| `核心繁简得分` | `core_complexity_score` |
| `认知压大小变化率` | `cp_abs_rate` |
| `认知压大小状态` | `cp_abs_state` |
| `认知压的大小` | `cp_abs_state` |
| `认知压变化率` | `cp_rate` |
| `认知压状态` | `cp_state` |
| `实能量变化率` | `er_rate` |
| `实能量变化速率` | `er_rate` |
| `实能量状态` | `er_state` |
| `虚能量变化率` | `ev_rate` |
| `虚能量变化速率` | `ev_rate` |
| `虚能量状态` | `ev_state` |
| `疲劳度状态` | `fatigue_state` |
| `获得认知压` | `got_cp` |
| `获得认知压大小` | `got_cp_abs` |
| `认知压大小变化量` | `got_cp_abs` |
| `获得实能量` | `got_er` |
| `获得虚能量` | `got_ev` |
| `获得总能量` | `got_total_energy` |
| `把握感得分` | `grasp_score` |
| `把握感综合得分` | `grasp_score` |
| `置信度综合得分` | `grasp_score` |
| `记忆赋能池条目数` | `map_item_count` |
| `记忆赋能池总虚能量` | `map_total_ev` |
| `情绪递质__变化了` | `nt_changed` |
| `情绪递质变化了` | `nt_changed` |
| `情绪递质__变化量` | `nt_delta` |
| `情绪递质变化量` | `nt_delta` |
| `情绪递质__变化率` | `nt_rate` |
| `情绪递质变化率` | `nt_rate` |
| `情绪递质__状态` | `nt_state` |
| `情绪递质状态` | `nt_state` |
| `状态池能量聚集度` | `pool_concentration` |
| `状态池能量聚集度变化率` | `pool_concentration_rate` |
| `状态池认知压大小变化率` | `pool_cp_abs_rate` |
| `状态池认知压大小变化速率` | `pool_cp_abs_rate` |
| `状态池认知压大小总量` | `pool_cp_abs_total` |
| `状态池认知压变化率` | `pool_cp_rate` |
| `状态池认知压变化速率` | `pool_cp_rate` |
| `状态池认知压总量` | `pool_cp_total` |
| `有效波峰数量` | `pool_effective_peak_count` |
| `状态池有效波峰数量` | `pool_effective_peak_count` |
| `状态池实能量变化率` | `pool_er_rate` |
| `状态池实能量变化速率` | `pool_er_rate` |
| `状态池实能量总量` | `pool_er_total` |
| `状态池虚能量变化率` | `pool_ev_rate` |
| `状态池虚能量变化速率` | `pool_ev_rate` |
| `状态池虚能量总量` | `pool_ev_total` |
| `状态池对象数量` | `pool_item_count` |
| `状态池条目数` | `pool_item_count` |
| `状态池总能量` | `pool_total_energy` |
| `状态池总能量状态` | `pool_total_energy` |
| `状态池总能量变化量` | `pool_total_energy_got` |
| `状态池获得总能量` | `pool_total_energy_got` |
| `状态池总能量变化率` | `pool_total_energy_rate` |
| `状态池总能量变化速率` | `pool_total_energy_rate` |
| `惩罚信号增加` | `punish_got` |
| `惩罚信号变化率` | `punish_rate` |
| `惩罚信号变化速率` | `punish_rate` |
| `惩罚信号状态` | `punish_state` |
| `近因增益状态` | `recency_state` |
| `奖励信号增加` | `reward_got` |
| `奖励信号变化率` | `reward_rate` |
| `奖励信号变化速率` | `reward_rate` |
| `奖励信号状态` | `reward_state` |
| `查存一体过程匹配分数` | `stimulus_match_score` |
| `查存一体过程匹配分数（按目标）` | `stimulus_match_score_target` |
| `刺激级查存一体结束时的剩余能量比例` | `stimulus_residual_ratio` |
| `查存一体过程匹配分数（结构级）` | `structure_match_score` |
| `结构级查存一体匹配分数` | `structure_match_score` |
| `查存一体过程匹配分数（结构级按目标）` | `structure_match_score_target` |
| `结构级查存一体匹配分数（按目标）` | `structure_match_score_target` |
| `总能量变化率` | `total_energy_rate` |
| `总能量变化速率` | `total_energy_rate` |
| `对象总能量状态` | `total_energy_state` |
| `总能量状态` | `total_energy_state` |

## 17. 附录 C：一期默认规则使用面速查

- 顶层 selector mode 使用：`['contains_text', 'has_attribute', 'top_n']`
- 嵌套 branch selector mode 使用：`['specific_item']`
- 顶层 preset 使用：`['complexity_score', 'core_complexity_score', 'cp_abs_state', 'ev_state', 'fatigue_state', 'got_cp_abs', 'got_er', 'got_total_energy', 'grasp_score', 'map_item_count', 'nt_state', 'pool_effective_peak_count', 'punish_state', 'reward_state', 'stimulus_match_score', 'stimulus_residual_ratio', 'total_energy_state']`
- branch 内 preset 使用：`['er_rate', 'er_state']`
- 直接 metric 路径使用：`['item.exists']`
- 当前动作使用：`['action_trigger', 'branch', 'cfs_emit', 'emit_script', 'emotion_update', 'pool_bind_attribute']`
- 当前 compare op 已见：`['<=', '>', '>=', 'exists']`
- 当前 mode 已见：`['avg_rate', 'state']`
- 默认规则中已显式使用的 capture 变量：`['curiosity_residual', 'deja_vu_cp', 'expect_er_rate', 'expect_pred_ev', 'expect_teacher_er_rate', 'expect_teacher_total_energy', 'familiarity_grasp', 'fear_unknown_cp', 'pressure_actual_er', 'pressure_pred_ev', 'pressure_runtime_actual_er', 'pressure_runtime_pred_ev', 'pressure_teacher_actual_er', 'pressure_teacher_total_energy', 'reassurance_drop', 'simplicity_complexity', 'uncanny_cp', 'uncertainty_cp']`

这份速查表的意义在于：以后如果某个大模型要继续生成新规则，不需要再从源码扫一遍，直接先看这里，就知道当前“真实工程风格”是什么。
