# APV2.1 当前完整状态说明书（冷保存草稿）
Date: 2026-06-10
Workspace: `H:\AP原型实验第二期\APV2.1版本原型测试`

## 0. 写法约定

这份文档按“先定边界，再定机制，再定证据，再定下一步”的方式写。

它不是防御清单，而是当前实现的白盒说明：

- 它证明了什么；
- 它怎么做出来；
- 它为什么这样设计；
- 它现在还差什么。

## 1. 三条路线

### 1.1 AP-Core

AP-Core 是内生认知闭环本体。它负责：

- 状态池；
- 快/慢召回；
- 注意力竞争；
- 情绪慢量；
- 行动竞争；
- 反馈消化；
- 记忆写回。

### 1.2 GL

GL 是教学与实验生产线。它负责：

- 技能包；
- 教学协议；
- scaffold 退坡；
- teacher-off / cold-retest；
- 证据生成与整理。

### 1.3 桌宠 / SNS

桌宠 / SNS 是产品壳与展示入口。它负责：

- UI；
- 桌面交互；
- 演示；
- 权限与产品层封装。

## 2. 当前实现模块

### 2.1 核心运行

- `core/runtime/engine.py`
- `core/runtime/budget_controller.py`
- `core/state_pool/state_pool.py`
- `memory/store/memory_store.py`
- `core/attention/selector.py`
- `core/emotion/emotion_state.py`
- `core/emotion/emotion_modulator.py`
- `core/action/*`
- `core/innate/engine.py`
- `core/learning/innate_event_router.py`
- `education/intervention.py`
- `education/skill_protocol_v2.py`

### 2.2 记忆与检索

- `memory/retrieval/posting_index.py`
- `memory/retrieval/numeric_feature_index.py`
- `memory/retrieval/faiss_index.py`
- `memory/embedding/online_store.py`
- `memory/spacetime/transition_store.py`
- `memory/relations/relative_relation_store.py`

### 2.3 短时缓冲

- `memory/short_term/echo_buffer.py`
- `memory/short_term/focus_buffer.py`
- `memory/short_term/focus_successor_bias.py`
- `memory/short_term/memory_window.py`

## 3. 当前默认数值

### 3.1 状态池

- `real_decay = 0.9`
- `virtual_decay = 0.86`
- `attention_gain_decay = 0.9`
- `fatigue_decay = 0.82`
- `prune_threshold = 0.045`
- `query_limit = 8`
- `snapshot_limit = 24`
- `memory_snapshot_limit = 1024`
- `r_state_head_limit = 7`
- `r_state_items_per_head = 256`
- `maintenance_budget = 48`
- `recent_external_limit = 2048`
- `hot_anchor_limit = 2048`
- `prediction_validation_actual_limit = 256`
- `prediction_validation_update_limit = 128`
- `focus_boost = 0.3`
- `focus_fatigue_step = 0.18`
- `prediction_fatigue_enabled = True`
- `prediction_fatigue_min_mass = 0.18`
- `prediction_fatigue_ratio = 0.18`
- `prediction_fatigue_gain = 0.06`
- `prediction_fatigue_max_step = 0.18`
- `cstar_trace_top_labels = 8`
- `bootstrap_virtual_energy = 0.6`

### 3.2 记忆

- `recall_top_k = 5`
- `predict_top_k = 5`
- `prediction_energy_scale = 0.55`
- `max_snapshots_per_kind = 256`
- `candidate_limit = 256`
- `core_item_limit = 1024`
- `query_feature_limit = 1024`
- `numeric_dim = 64`
- `numeric_candidate_limit = 64`
- `numeric_top_k_per_channel = 24`
- `numeric_weight = 1.15`
- `relation_score_weight = 0.68`
- `relation_focus_score_weight = 0.92`
- `temporal_tick_seconds = 0.1`
- `temporal_long_half_life_ticks = 25_920_000`

### 3.3 注意力

- `focus_limit = 8`
- `pressure_gain = 0.6`
- `attention_gain_weight = 0.8`
- `fatigue_weight = 0.5`
- `continuation_bias = 0.35`
- successor bias `gain = 0.42`
- successor bias `max = 0.48`

### 3.4 短时

- `focus_history_limit = 12`
- `echo_history_limit = 128`
- `echo_max_age_ticks = 8`
- `memory_window_history_limit = 64`
- `memory_window_max_age_ticks = 48`
- `memory_window_recall_limit = 8`

### 3.5 情绪通道

8 通道基线：

- `DA = 0.12`
- `ADR = 0.05`
- `OXY = 0.12`
- `SER = 0.18`
- `END = 0.10`
- `COR = 0.06`
- `NOV = 0.08`
- `FOC = 0.10`

### 3.6 认知感受

当前默认认知感受主通道：

- `uncertainty`
- `evidence_gap`
- `quantity_grasp`
- `step_closure`
- `computation_pressure`
- `sensory_clarity`

## 4. AP 的核心机制

### 4.1 SA

SA 是状态原子。它是进入状态池的一切可观察材料的基本单位。

### 4.2 状态池

状态池不是单纯存储器，而是当前 tick 的认知场。

每个 entry 都有：

- `real_energy`
- `virtual_energy`
- `cognitive_pressure = real_energy - virtual_energy`
- `attention_gain`
- `fatigue`
- `last_seen_tick`
- `last_updated_tick`
- `provenance`
- `anchor_meta`
- `numeric_features`
- `reconstruction_payload`

### 4.3 B / Bn

- `B` 是候选中的历史近邻与现实经验压力；
- `Bn` 是被当前 tick 认知场重新判断后的可读近邻。

### 4.4 C / Cn / C*

- `C` 是后继预测；
- `Cn` 是当前 tick 的后继判断；
- `C*` 是同标签虚量合并后的预测审计口径。

### 4.5 认知感受

认知感受不是外界字段的直译，而是内生证据的投影：

- 预测残差；
- 抓握程度；
- 关闭感；
- 不确定；
- 压力；
- 顺畅；
- 任务完成感。

### 4.6 情绪慢量

情绪慢量是跨 tick 的调制层，不直接替代理解，但会影响：

- 注意力资源；
- 学习率；
- 行动阈值；
- 探索倾向。

### 4.7 行动反馈

行动反馈是闭环证据，不是装饰日志。

它会更新：

- 行动结果记忆；
- 参数记忆；
- 后继偏好；
- 未完成感；
- 未来驱动力。

## 5. 为什么 AP 的泛化不是关键词硬门

AP 的泛化来自多证据叠加，不来自关键词路由。

关键点是：

- 外界输入只是普通 SA；
- 召回靠状态竞争，不靠答案表；
- 后继来自多条历史；
- 在线学习写的是关系与偏好，不是隐藏标准答案；
- 教学层只提供材料、软 bias、反馈，不替代认知。

所以它更像“会从过程里长出语言和行动”，而不是“查表命中”。

## 6. 语言学习六阶段

1. `echo imitation`
2. `successor prediction`
3. `multi-reply aggregation`
4. `process-paradigm binding`
5. `keyword organization`
6. `grammar/style refinement`

当前门控原则：

- 如果没有清晰后继波峰，偏向模仿 / 回看；
- 如果有清晰后继波峰，偏向接着说 / 聚合。

这不是硬路由，而是过程门控。

## 7. 当前实现的完整流程

感知 -> 外源 / 教学 -> echo -> fast R_state recall -> attention + successor bias -> 短时缓冲 -> slow recall -> feelings -> expectation pressure -> emotion update -> action consequence eval -> planner -> SafetyGate -> control effects -> feedback -> memory write

## 8. 状态池细节

### 8.1 双能量与懒衰减

状态池采用双能量：

- `real_energy`
- `virtual_energy`

衰减是懒执行的，只在 entry 被触碰时才按 tick 差计算。

### 8.2 快读与视图

`read_r_state()` 才是 fast-system 正确入口。

它分 5 个头：

- `head_recent`
- `head_anchor`
- `head_prediction`
- `head_residual`
- `head_global`

### 8.3 读取粒度

- `head_recent`：最近外源证据；
- `head_anchor`：热锚点；
- `head_prediction`：预测虚量槽；
- `head_residual`：残差槽；
- `head_global`：局部合并视图。

### 8.4 记忆写入粒度

状态池写入时保留：

- SA label 粒度；
- 过程锚 metadata 粒度；
- numeric channel 粒度；
- reconstruction payload 粒度。

它会刻意保留低粒度过程元数据，避免把学习材料压扁成裸文本熟悉度。

## 9. 记忆召回

### 9.1 通道

召回依次叠加：

- posting
- vector
- numeric
- relation
- online embedding
- temporal applicability

### 9.2 候选构成

query features 会构造：

- labels
- displays
- bigrams
- sequence_bigrams
- relation_tokens
- focus_labels
- numeric_features

### 9.3 打分构成

总分来自：

- label overlap
- display overlap
- bigram overlap
- focus overlap
- state match
- energy overlap
- sequence overlap
- posting score
- vector score
- numeric score
- relation score
- learned score
- time match
- temporal applicability

### 9.4 B / C 的区别

- `recall()` 是 B 侧检索；
- `predict()` / successor link 是 C 侧判断；
- `C*` 只做同标签审计合并，不把重复当新知识。

## 10. 本地在线学习嵌入

在线嵌入不是学生侧 LLM，也不是隐藏 solver。

它是本地的、受限的、可审计的 token 关联表：

- `observe_positive_pair`
- `observe_negative_pair`
- `observe_transition_pair`

它输出：

- `learned_similarity`
- `learned_transition`
- `pair_evidence`

当前实现的要点是：

- 只有被 promote 的 token 才参与 learned similarity；
- 每 tick 更新有上限；
- 写入来自 snapshot / event pipeline，不来自答案注入。

## 11. 注意力机制

注意力 selector 的核心分数：

- `cognitive_pressure * pressure_gain`
- `attention_gain * attention_gain_weight`
- `virtual_energy * 0.25`
- `fatigue * fatigue_weight`
- `continuation_bias`
- successor bias
- innate bias
- action bias

它先做竞争，再截断到 `focus_limit = 8`。

### 11.1 当前新门控

- 没有清晰后继波峰 -> 偏模仿 / 回看；
- 有清晰后继波峰 -> 偏接着说 / 聚合。

## 12. 数据库现在是什么形式

当前是分层形态：

- 运行时内存索引：posting / vector / numeric / relation / transition / online；
- 持久层：`memory/persistence/*` 对应的 authoritative store；
- warm load：只加载热窗口，不全量灌内存。

它不是单一“大表”，而是“白盒快索引 + 权威持久层”的双层结构。

## 13. 情绪通道

8 通道都有基线、衰减和软上限。

对注意力、HDB、行动的映射分别如下：

- DA -> 奖励驱动 / 资源倍增；
- ADR -> 警觉 / 行动阈值变化；
- OXY -> 联结与传播；
- SER -> 稳定与阈值；
- END -> 奖励后的缓冲；
- COR -> 警戒 / 负荷；
- NOV -> 探索；
- FOC -> 焦点锁定。

## 14. 行动器与行动节点

### 14.1 注册的 actuator

当前注册 14 个：

- `actuator::attention_allocation`
- `actuator::visual_gaze_center`
- `actuator::visual_focus_scale`
- `actuator::auditory_band_center`
- `actuator::auditory_band_width`
- `actuator::memory_recall`
- `actuator::text_editor`
- `actuator::computer_pointer`
- `actuator::computer_keyboard`
- `actuator::llm_call`
- `actuator::tool_api`
- `actuator::timing`
- `actuator::protective_orientation`
- `actuator::legacy_internal`

### 14.2 注册的 action node

当前代码里有 39 个左右的节点，覆盖：

- attention
- gaze
- auditory
- memory
- text
- pointer
- keyboard
- llm
- tool
- wait
- protective
- legacy prediction

### 14.3 SafetyGate

外部动作会被复核，依据：

- pressure
- COR
- expectation anchor risk
- action control risk
- external confidence

## 15. 教学协议

### 15.1 边界

教学系统在 AP core 外部。

它可以给：

- state items；
- soft action biases；
- reward / punishment / correctness feedback。

它不能：

- 直接执行动作；
- 直接写 hidden answer table；
- 让 teacher-off 变成偷偷给答案。

### 15.2 标准退坡

`demonstrate -> strong_scaffold -> weak_scaffold -> feedback_only -> teacher_off -> cold_retest`

### 15.3 技能包

技能包的设计核心是“砖块化”：

- 语言砖；
- 顺序砖；
- 修订砖；
- 反馈砖；
- 组合砖。

### 15.4 幼儿阶段验收

当前语言学习与开放对话应按早期语言发展阶段去看：

- 模仿能成；
- 后继能成；
- 多答能叠加；
- 关键词可先于完整语法出现；
- 后续再做 grammar/style refinement。

## 16. 已完成证据

### 16.1 DPP1

当前可直接引用的正向表述是：

- 控制版 DPP：`300/300`
- live fresh：`286/300`

这更像近失，不是毕业。

### 16.2 可纳入论文的正向证据

- risky action confirmation
- mixed script name/object gap
- greeting then unknown word

### 16.3 还需硬化的模式

- overload complex small step
- recency impression emotion bias after conflict
- repetition confusion

## 17. 当前最优先下一步

### 理论硬化

- 后继波峰门控
- 主体身份峰
- 直接经验边界
- 关系印象衰减
- 低唤醒陪伴 vs 任务压力竞争
- 幼儿阶段验收

### 实验硬化

- `FeedbackOverride-1`
- `PersistenceReload-1`
- `NegativeFeedback-Ablation-1`
- `DPP-1 v0.3`

### 论文硬化

- `ParamSensitivity-1`
- `Representation-Ablation-1`
- 保留旧版优点并增强说服力

## 18. 结论

APV2.1 当前已经不是“单点回忆系统”，而是一条可以自洽运行的内生认知闭环：

- 经验进入状态池；
- 状态池分出快慢系统；
- 快慢系统共同驱动注意与行动；
- 行动结果再回写记忆与情绪；
- 教学层只提供砖块，不代替认知。

这也是它目前最重要的证明。

## 19. 实现附录: 模块、数据库与边界

### 19.1 模块总表

| 模块组 | 关键文件 | 当前职责 | 为什么需要它 |
|---|---|---|---|
| 运行总线 | `core/runtime/engine.py` | 把一次 tick 拆成感知、外源、快读、注意、慢读、情绪、行动、反馈、写回 | 让 AP 的“想-看-说-改-记”成为可审计链路 |
| 状态池 | `core/state_pool/state_pool.py` | 双能量池、R_state 五头、snapshot / memory_write 视图、预测校验、残差桶 | 这是 AP 内生认知场的中枢 |
| 记忆检索 | `memory/store/memory_store.py` | posting / vector / numeric / relation / online / temporal 的多通道召回 | 让泛化来自多证据叠加，而不是单句命中 |
| 在线嵌入 | `memory/embedding/online_store.py` | 本地 token 关联表、正负对、transition 对、promote 门槛 | 让学习是本地可审计的，不是隐藏 solver |
| 关系与时序 | `memory/relations/relative_relation_store.py` / `memory/spacetime/transition_store.py` | 文本顺序、视觉空间、音频顺序、状态后继链接 | 让“过程”成为记忆的一等对象 |
| 短时系统 | `memory/short_term/echo_buffer.py` / `focus_buffer.py` / `memory_window.py` / `focus_successor_bias.py` | 回声、焦点连续、工作记忆、后继偏置 | 让 AP 像人一样保留刚发生的东西 |
| 注意系统 | `core/attention/selector.py` | 关注点竞争、continuation、successor、action bias | 让选择基于内在证据而不是硬门 |
| 情绪与感受 | `channels/*`、`core/emotion/*` | 认知感受、任务感、时间感、节律感、期待压力、8 通道 NT | 让“感觉”成为慢调制，不成为答案表 |
| 行动系统 | `core/action/*` | registry、planner、consequence evaluator、control effects、SafetyGate、outcome memory | 让动作、后果和抑制都可审计 |
| 教学系统 | `education/*`、`GL_TaskBuilder/EDUCATION_PROTOCOL.md` | scaffold、teacher-off、cold retest、skill package | 让 GL 负责教与证据生产，AP 负责认知 |
| 持久层 | `memory/persistence/*` | PostgreSQL-first 权威存储与热窗口回读 | 让历史不靠 RAM 常驻 |

### 19.2 数据库表总表

| 表 | 作用 |
|---|---|
| `ap_memory_schema_version` | schema 版本登记 |
| `ap_runs` | 一次运行的元信息 |
| `ap_ticks` | tick 粒度运行日志 |
| `memory_snapshots` | 一条记忆快照的权威载体 |
| `memory_snapshot_items` | 快照中的逐项 SA |
| `memory_state_field_items` | Bn 主认知场的主表 |
| `memory_core_items` | legacy 兼容视图 |
| `memory_posting_tokens` | posting 倒排 token |
| `memory_vectors` | 向量表示 |
| `memory_numeric_features` | 64 维 numeric 特征 |
| `memory_relation_features` | 关系 token / 关系权重 |
| `memory_transitions` | 状态后继边 |
| `memory_learning_events` | 学习事件审计 |
| `memory_action_feedback_events` | 行动反馈审计 |
| `memory_asset_refs` | 多模态资产引用 |
| `memory_index_audit_runs` | 索引审计任务 |
| `memory_index_audit_rows` | 审计明细 |

### 19.3 持久层边界

- PostgreSQL-first 是权威真相层。
- runtime 的 posting / ANN / numeric / relation / transition / online 都是可重建视图。
- `memory_state_field_items` 是 Bn 主场，不是 legacy “core” 的附属注释。
- `load_recent_snapshots()` 只加载热窗口，不全量灌内存。
- `resident_hot_snapshots_per_kind=4096`，`warm_prefetch_limit=512`，这说明恢复时要的是“热的最近层”，不是数据库搬家。

## 20. 默认值附录

### 20.1 输入与资产

| 项 | 默认值 |
|---|---:|
| `TextSensorConfig.budget_limit` | 1024 |
| `TextSensorConfig.competition_limit` | 1024 |
| `TextSensorConfig.dynamic_phrase_min_observations` | 2 |
| `TextSensorConfig.dynamic_phrase_max_len` | 3 |
| `TextSensorConfig.dynamic_phrase_scan_budget` | 256 |
| `TextSensorConfig.dynamic_phrase_emit_budget` | 32 |
| `VisionSensorConfig.mode` | `native_numeric` |
| `VisionSensorConfig.max_objects` | 4 |
| `VisionSensorConfig.max_side` | 160 |
| `VisionSensorConfig.preview_side` | 96 |
| `VisionSensorConfig.fallback_to_legacy` | true |
| `AudioSensorConfig.mode` | `native_numeric` |
| `AudioSensorConfig.max_samples` | 32768 |
| `AudioSensorConfig.band_count` | 12 |
| `AudioSensorConfig.fallback_to_legacy` | true |
| `MultimodalAssetConfig.enabled` | false |
| `MultimodalAssetConfig.max_assets` | 256 |
| `MultimodalAssetConfig.persist_payloads` | false |
| `MultimodalAssetConfig.keep_hot_payloads` | true |
| `MultimodalAssetConfig.preview_retention_ticks` | 64 |
| `MultimodalAssetConfig.object_proxy_retention_ticks` | 256 |
| `MultimodalAssetConfig.raw_frame_retention_ticks` | 256 |
| `MultimodalAssetConfig.focus_tile_retention_ticks` | 512 |
| `MultimodalAssetConfig.focus_tile_max_count` | 4 |
| `MultimodalAssetConfig.focus_tile_padding_ratio` | 0.08 |
| `MultimodalAssetConfig.audio_focus_window_max_bytes` | 48000 |

### 20.2 状态池、记忆、注意力

| 项 | 默认值 |
|---|---:|
| `StatePool.real_decay` | 0.9 |
| `StatePool.virtual_decay` | 0.86 |
| `StatePool.attention_gain_decay` | 0.9 |
| `StatePool.fatigue_decay` | 0.82 |
| `StatePool.prune_threshold` | 0.045 |
| `StatePool.r_state_head_limit` | 7 |
| `StatePool.r_state_items_per_head` | 256 |
| `StatePool.maintenance_budget` | 48 |
| `StatePool.recent_external_limit` | 2048 |
| `StatePool.hot_anchor_limit` | 2048 |
| `StatePool.query_limit` | 8 |
| `StatePool.snapshot_limit` | 24 |
| `StatePool.memory_snapshot_limit` | 1024 |
| `StatePool.prediction_validation_actual_limit` | 256 |
| `StatePool.prediction_validation_update_limit` | 128 |
| `StatePool.focus_boost` | 0.3 |
| `StatePool.focus_fatigue_step` | 0.18 |
| `StatePool.bootstrap_virtual_energy` | 0.6 |
| `Memory.recall_top_k` | 5 |
| `Memory.predict_top_k` | 5 |
| `Memory.prediction_energy_scale` | 0.55 |
| `Memory.max_snapshots_per_kind` | 256 |
| `Memory.candidate_limit` | 256 |
| `Memory.core_item_limit` | 1024 |
| `Memory.query_feature_limit` | 1024 |
| `Memory.posting_label_token_limit` | 256 |
| `Memory.posting_display_token_limit` | 128 |
| `Memory.posting_bigram_token_limit` | 192 |
| `Memory.posting_sequence_token_limit` | 192 |
| `Memory.vector_token_limit` | 512 |
| `Memory.scoring_candidate_limit` | 96 |
| `Memory.learned_rerank_limit` | 16 |
| `Memory.numeric_dim` | 64 |
| `Memory.numeric_candidate_limit` | 64 |
| `Memory.numeric_top_k_per_channel` | 24 |
| `Memory.numeric_weight` | 1.15 |
| `Memory.relation_score_weight` | 0.68 |
| `Memory.relation_focus_score_weight` | 0.92 |
| `Memory.temporal_tick_seconds` | 0.1 |
| `Memory.temporal_long_half_life_ticks` | 25_920_000 |
| `Attention.focus_limit` | 8 |
| `Attention.pressure_gain` | 0.6 |
| `Attention.attention_gain_weight` | 0.8 |
| `Attention.fatigue_weight` | 0.5 |
| `Attention.continuation_bias` | 0.35 |
| `Attention.successor_bias_gain` | 0.42 |
| `Attention.successor_bias_max` | 0.48 |
| `Attention.successor_bias_top_k` | 12 |
| `Attention.successor_bias_context_limit` | 2048 |
| `Attention.successor_bias_max_successors_per_context` | 64 |
| `Attention.successor_bias_max_context_labels` | 8 |
| `Attention.successor_bias_max_order` | 3 |
| `Attention.successor_bias_per_tick_update_limit` | 16 |
| `Attention.successor_bias_entropy_floor` | 0.28 |

### 20.3 短时、感受、行动

| 项 | 默认值 |
|---|---:|
| `ShortTerm.focus_history_limit` | 12 |
| `ShortTerm.recency_decay` | 0.78 |
| `ShortTerm.synthetic_query_weight` | 1.1 |
| `ShortTerm.replay_decay` | 0.72 |
| `ShortTerm.replay_query_weight` | 0.82 |
| `ShortTerm.max_replay_items` | 8 |
| `ShortTerm.episode_break_overlap` | 0.22 |
| `ShortTerm.echo_history_limit` | 128 |
| `ShortTerm.echo_max_age_ticks` | 8 |
| `ShortTerm.echo_decay` | 0.68 |
| `ShortTerm.echo_sensory_gain` | 0.22 |
| `ShortTerm.echo_thought_gain` | 0.18 |
| `ShortTerm.echo_max_energy` | 0.28 |
| `ShortTerm.echo_max_items_per_tick` | 18 |
| `ShortTerm.memory_window_history_limit` | 64 |
| `ShortTerm.memory_window_max_age_ticks` | 48 |
| `ShortTerm.memory_window_recency_decay` | 0.86 |
| `ShortTerm.memory_window_fatigue_decay` | 0.70 |
| `ShortTerm.memory_window_fatigue_step` | 0.45 |
| `ShortTerm.memory_window_max_items_per_event` | 12 |
| `ShortTerm.memory_window_recall_limit` | 8 |
| `CognitiveFeeling.min_activation` | 0.12 |
| `TaskFeeling.min_activation` | 0.12 |
| `TaskFeeling.boredom_gain` | 1.0 |
| `TaskFeeling.fulfillment_gain` | 1.0 |
| `RuntimeLoad.min_activation` | 0.08 |
| `RuntimeLoad.target_load_ratio` | 1.0 |
| `RuntimeLoad.ideal_load_ratio` | 0.58 |
| `RuntimeLoad.state_item_soft_limit` | 1024 |
| `RuntimeLoad.r_state_item_soft_limit` | 1792 |
| `RuntimeLoad.attention_candidate_soft_limit` | 256 |
| `RuntimeLoad.pending_index_soft_limit` | 24 |
| `RuntimeLoad.family_overflow_soft_limit` | 8 |
| `RuntimeLoad.residual_mass_soft_limit` | 24.0 |
| `TimeFeeling.threshold` | 0.22 |
| `TimeFeeling.gain` | 0.95 |
| `TimeFeeling.min_confidence` | 0.24 |
| `TimeFeeling.default_radius_ticks` | 4.0 |
| `TimeFeeling.recall_gain` | 0.22 |
| `TimeFeeling.max_sources` | 6 |
| `Rhythm.window` | 12 |
| `Rhythm.min_hits` | 3 |
| `Rhythm.min_period` | 2 |
| `Rhythm.max_period` | 12 |
| `Rhythm.pulse_threshold` | 0.18 |
| `Rhythm.phase_threshold` | 0.14 |
| `ExpectationPressure.min_activation` | 0.1 |
| `ExpectationPressure.anchor_max_anchors` | 32 |
| `ExpectationPressure.anchor_decay` | 0.88 |
| `ExpectationPressure.anchor_min_level` | 0.03 |
| `ExpectationPressure.anchor_min_outcome_virtual` | 0.045 |
| `Action.selection_threshold` | 0.32 |
| `Action.max_selected_actions` | 4 |
| `Action.consequence_max_successor_rows` | 12 |
| `Action.consequence_max_evidence_per_action` | 8 |
| `Action.consequence_max_horizon` | 3 |
| `Action.consequence_branching` | 3 |
| `Action.consequence_path_decay` | 0.72 |
| `Action.outcome_memory_learning_rate` | 0.18 |
| `Action.outcome_memory_decay_per_tick` | 0.992 |
| `Action.outcome_memory_support_scale` | 6.0 |
| `Action.outcome_memory_max_drive_bias` | 0.75 |
| `OnlineEmbedding.dim` | 32 |
| `OnlineEmbedding.token_limit` | 2048 |
| `OnlineEmbedding.min_support_to_promote` | 2 |
| `OnlineEmbedding.per_tick_update_limit` | 8 |
| `OnlineEmbedding.scoring_token_limit` | 256 |
| `OnlineEmbedding.learned_weight` | 0.28 |
| `OnlineEmbedding.transition_learned_weight` | 0.18 |

### 20.4 情绪与观测

| 项 | 默认值 |
|---|---:|
| `Emotion.cfs_gain` | 1.0 |
| `Emotion.rwd_pun_gain` | 1.0 |
| `Tuner.ema_alpha` | 0.04 |
| `Tuner.min_support_ticks` | 12 |
| `Tuner.target_prediction_alignment` | 0.58 |
| `Tuner.max_normal_pressure` | 3.5 |
| `Tuner.target_action_success` | 0.52 |
| `Tuner.adjustment_rate` | 0.025 |
| `Tuner.rollback_threshold` | 0.18 |
| `Observability.default_trace_mode` | `summary` |
| `Observability.target_tick_ms` | 100.0 |
| `Observability.disable_gc_during_tick` | true |
| `Observability.trace_item_preview_limit` | 32 |
| `Observability.trace_r_state_item_preview_limit` | 8 |
| `Observability.trace_text_preview_chars` | 512 |
| `Observability.trace_matched_token_preview_limit` | 12 |

## 21. 召回、能量与学习的计算法

### 21.1 状态池的能量图景

每个 `PoolEntry` 不是单一分数，而是三件事：

- `real_energy`：当前真实进入场的证据力度。
- `virtual_energy`：当前被预测、被期待、被召回的力度。
- `cognitive_pressure = real_energy - virtual_energy`：真实与预期之间的张力。

快系统常见的两条公式是：

- `query_weight = real*1.0 + virtual*0.65 + attention_gain*0.8 - fatigue*0.3`
- `attention_score = cognitive_pressure*0.7 + attention_gain - fatigue*0.5 + virtual_energy*0.25`

这意味着：

- 真经验最重；
- 预测能量有用，但不是现实本身；
- 注意增长不是“更像答案”，而是“更值得继续看”。

### 21.2 召回通道

召回不是一条路，而是六条路叠加：

1. `posting`
2. `vector`
3. `numeric`
4. `relation`
5. `online embedding`
6. `temporal applicability`

### 21.3 召回算法

1. 先从 `snapshot` 或 `focus continuation` 构造 query items。
2. `query_features` 提取 label、display、bigram、sequence_bigrams、relation_tokens、focus_labels、numeric_features。
3. `PostingIndex` 先给出倒排候选。
4. `HashVectorIndex + FAISS/HNSW` 给出向量候选，若 ANN 不可用则只在 posting 候选上做向量重排。
5. `NumericFeatureIndex` 对 64 维 numeric 通道做候选与重排。
6. 合并三路候选后，再算：
   - label overlap
   - display overlap
   - bigram overlap
   - focus overlap
   - state_match
   - energy_overlap
   - sequence overlap
   - posting_score
   - vector_score
   - numeric_score
   - relation_score
   - learned_score
   - time_match
7. 最后乘 `temporal_applicability`。

核心分数可以直接读成：

```text
score_before_temporal =
  label_overlap*1.15
  + display_overlap*0.45
  + bigram_overlap*0.9
  + focus_overlap*0.7
  + state_match*0.55
  + energy_overlap*1.35
  + sequence_overlap*0.8
  + posting_score*0.35
  + vector_score*0.4
  + numeric_score*1.15
  + relation_score
  + learned_score*0.28
  + time_match
score = score_before_temporal * temporal_applicability
```

### 21.4 C / Cn / C* 的现在实现

- `apply_predictions()` 会把预测项写入 `virtual_energy` 和 `prediction_slot`。
- 同标签的虚量会在 `C*` 审计里合并成预算轨迹，不把重复当新知识。
- `PredictionEnergyUpdater` 会把 matched / missed / unexpected 转成：
  - `alignment_score`
  - `mismatch_ratio`
  - `missed_predicted_labels`
  - `unexpected_labels`
  - `energy_updates`
- `ResidualTracker` 只是侧桶，不是第二个状态池。

### 21.5 本地在线学习嵌入

当前在线嵌入是一个很小、很本地、可审计的 token 关联表：

- 维度 `32`
- token 上限 `2048`
- 每 tick 更新上限 `8`
- promoted token 才参与 learned similarity
- 只在 snapshot / event pipeline 里写入
- 不做学生侧 LLM，不做隐藏 solver，不做答案表

操作分三类：

- `observe_positive_pair(a,b)`：正向共现
- `observe_negative_pair(a,b)`：负向共现
- `observe_transition_pair(a,b)`：后继关系

它返回的不是“答案”，而是：

- `learned_similarity`
- `learned_transition`
- `pair_evidence`

这说明它现在的性质是“可解释关联记忆”，不是大模型语义库。

## 22. 行动器、行动节点与抑制

### 22.1 注册的 actuator

| actuator_id | 外部 | 半外部 | 默认/ tick | 阈值范围 | 冲突域 |
|---|---|---|---:|---|---|
| `actuator::attention_allocation` | 否 | 否 | 1 | 0.38-0.58 | attention_focus_width_and_anchor |
| `actuator::visual_gaze_center` | 否 | 是 | 1 | 0.48-0.72 | single_visual_center |
| `actuator::visual_focus_scale` | 否 | 是 | 1 | 0.58-0.78 | visual_sampling_scale |
| `actuator::auditory_band_center` | 否 | 是 | 1 | 0.50-0.70 | single_auditory_band_center |
| `actuator::auditory_band_width` | 否 | 是 | 1 | 0.48-0.68 | auditory_sampling_width |
| `actuator::memory_recall` | 否 | 否 | 1 | 0.50-0.78 | primary_recall_query |
| `actuator::text_editor` | 是 | 否 | 1 | 0.45-1.05 | single_text_buffer_edit |
| `actuator::computer_pointer` | 是 | 否 | 1 | 0.95-1.35 | single_os_pointer |
| `actuator::computer_keyboard` | 是 | 否 | 1 | 1.05-1.35 | single_os_keyboard |
| `actuator::llm_call` | 是 | 否 | 1 | 1.05-1.40 | single_llm_request |
| `actuator::tool_api` | 是 | 否 | 1 | 1.10-1.50 | tool_mutex |
| `actuator::timing` | 否 | 否 | 1 | 0.22-0.45 | wait_hold_pause |
| `actuator::protective_orientation` | 否 | 否 | 1 | 0.42-0.76 | protective_orientation |
| `actuator::legacy_internal` | 否 | 否 | 1 | 0.30-0.55 | legacy_internal_prediction |

### 22.2 全部 action node

按 actuator 分组，当前共有 39 个。

| action_id | actuator_id | base_threshold | fatigue_type | 备注 |
|---|---|---:|---|---|
| `action::focus_anchor` | `actuator::attention_allocation` | 0.45 | action_internal | 锚定当前焦点 |
| `action::continue_focus` | `actuator::attention_allocation` | 0.38 | action_internal | 续看 / 续说 |
| `action::diverge_attention` | `actuator::attention_allocation` | 0.52 | action_internal | 分散注意 |
| `action::inspect_residual` | `actuator::attention_allocation` | 0.50 | mismatch | 看残差 |
| `action::release_focus` | `actuator::attention_allocation` | 0.48 | action_internal | 放焦点 |
| `action::move_gaze_to` | `actuator::visual_gaze_center` | 0.68 | action_internal | 视觉定位 |
| `action::nudge_gaze` | `actuator::visual_gaze_center` | 0.48 | fast_sensory | 细调 gaze |
| `action::scan_visual_field` | `actuator::visual_gaze_center` | 0.62 | action_internal | 视觉扫描 |
| `action::hold_gaze` | `actuator::visual_gaze_center` | 0.40 | positive_validation | 保持注视 |
| `action::zoom_visual_focus` | `actuator::visual_focus_scale` | 0.72 | action_internal | 缩放视野 |
| `action::widen_visual_focus` | `actuator::visual_focus_scale` | 0.62 | action_internal | 放宽视野 |
| `action::slide_audio_band` | `actuator::auditory_band_center` | 0.60 | fast_sensory | 移动音带 |
| `action::lock_audio_band` | `actuator::auditory_band_center` | 0.50 | positive_validation | 锁定音带 |
| `action::narrow_audio_band` | `actuator::auditory_band_width` | 0.55 | action_internal | 收窄音带 |
| `action::widen_audio_band` | `actuator::auditory_band_width` | 0.50 | action_internal | 放宽音带 |
| `action::recall_recent_context` | `actuator::memory_recall` | 0.55 | action_internal | 近文回忆 |
| `action::replay_recent_context` | `actuator::memory_recall` | 0.55 | action_internal | legacy 到 recall_recent_context |
| `action::recall_by_timefelt` | `actuator::memory_recall` | 0.62 | rhythm | 按时间感召回 |
| `action::recall_by_expectation` | `actuator::memory_recall` | 0.65 | expectation | 按期待锚召回 |
| `action::replay_episode` | `actuator::memory_recall` | 0.76 | action_internal | 回放 episode |
| `action::text_reread` | `actuator::text_editor` | 0.45 | action_external | 只改内部草稿 |
| `action::text_insert` | `actuator::text_editor` | 0.78 | action_external | 内部插入 |
| `action::text_delete` | `actuator::text_editor` | 0.86 | action_external | 内部删除 |
| `action::text_replace` | `actuator::text_editor` | 0.92 | action_external | 内部替换 |
| `action::text_commit` | `actuator::text_editor` | 1.05 | action_external | 提交边界 |
| `action::pointer_move` | `actuator::computer_pointer` | 0.95 | action_external | 指针移动 |
| `action::pointer_click` | `actuator::computer_pointer` | 1.20 | action_external | 点击 |
| `action::pointer_drag` | `actuator::computer_pointer` | 1.32 | action_external | 拖拽 |
| `action::pointer_scroll` | `actuator::computer_pointer` | 1.05 | action_external | 滚动 |
| `action::keyboard_type` | `actuator::computer_keyboard` | 1.18 | action_external | 键入 |
| `action::keyboard_hotkey` | `actuator::computer_keyboard` | 1.30 | action_external | 热键 |
| `action::llm_think` | `actuator::llm_call` | 1.10 | action_external | 外部 LLM 思考 |
| `action::llm_critique` | `actuator::llm_call` | 1.25 | action_external | 外部 LLM 评审 |
| `action::llm_write_draft` | `actuator::llm_call` | 1.35 | action_external | 外部 LLM 草稿 |
| `action::tool_call` | `actuator::tool_api` | 1.25 | action_external | 工具调用 |
| `action::wait` | `actuator::timing` | 0.25 | action_internal | 等待 |
| `action::avoid` | `actuator::protective_orientation` | 0.58 | protective | 回避 |
| `action::withdraw` | `actuator::protective_orientation` | 0.62 | protective | 撤退 |
| `action::stabilize_prediction` | `actuator::legacy_internal` | 0.32 | action_internal | legacy 预测稳定 |

### 22.3 SafetyGate 现在怎么做

- 只审外部动作。
- 核心阈值：
  - `veto_pressure_threshold=0.64`
  - `veto_cor_threshold=0.68`
  - `review_pressure_threshold=0.42`
  - `review_cor_threshold=0.50`
  - `min_external_confidence=0.58`
- 风险来自：
  - 预测压力
  - COR
  - expectation pressure anchor risk
  - action control risk
  - 外部 confidence 不足
- 它的作用不是替代行动，而是把可执行外部动作变成可审计的门。

### 22.4 行动后果与结果记忆

- `ActionConsequenceEvaluator` 读取 fast/slow 的 Bn/Cn successor rows。
- `max_successor_rows=12`，`max_evidence_per_action=8`，`max_horizon=3`，`branching=3`，`path_decay=0.72`。
- `ActionControlEffectRouter` 只把胜出的动作变成短寿命控制 SA / slow query hints，不直接写入概念表征。
- `ActionOutcomeMemory` 记录 reward / punishment / correctness / confidence 的长期趋势，`support_scale=6.0`，`max_drive_bias=0.75`。

## 23. 教学协议与技能包

### 23.1 协议边界

教学系统在 AP core 外部。它可以给：

- state items
- soft action biases
- reward / punishment / correctness feedback

它不能：

- 直接执行 AP actions
- 直接写隐藏答案表
- 在 teacher-off 测试里偷偷注入 final answer
- 用 regex route / keyword hard gate / 整句宏代替学习

### 23.2 packet 形状

标准 packet 是 `education_intervention/v1`。它被规范化成：

- `state_items`
- `action_biases`
- `feedback`
- `notes`

其中：

- `state_items` 是普通 SA，不是命令。
- `action_biases` 是 soft drive delta，不是强制动作。
- `feedback` 是结果后证据，不应携带 hidden answer。

`SkillScaffoldProtocolV2Controller` 的阶段是：

```text
demonstrate
-> strong_scaffold
-> weak_scaffold
-> feedback_only
-> teacher_off
-> cold_retest
```

### 23.3 技能包为什么需要

技能包不是大答案，是可复用砖块：

- 语言砖
- 顺序砖
- 修订砖
- 反馈砖
- 组合砖

它的哲学是：

1. 先教可复用材料。
2. 再让 AP 自己竞争。
3. 再让 AP 自己修订。
4. 再把修订后的过程写回记忆。

这就是为什么 `teacher-off` 不该掉回“查表题”，而应该能继续依赖已学砖块。

### 23.4 六阶段语言学习如何落地

语言学习六阶段现在不是口号，而是 GL / DPP / Skill37 的课程线：

1. `echo imitation`
2. `successor prediction`
3. `multi-reply aggregation`
4. `process-paradigm binding`
5. `keyword organization`
6. `grammar/style refinement`

它对应的当前判断是：

- `278~286/300` 这种近失说明 process-paradigm 已在，但 keyword organization / identity peak / direct-experience boundary 还要继续硬化。
- 所以当前阶段更像 `imitation_successor_developmental_stage`，不是成人级通用对话。

## 24. 证据矩阵与下一步

### 24.1 当前已完成的证据

| 项目 | 结果 | 说明 |
|---|---|---|
| DPP controlled | `300/300` | 过程协议跑通，且学生侧边界干净 |
| Skill37 strict30 live | `30/30` | 小样本 strict live 可以过，但不能替代大样本 |
| Live Fresh300 | `286/300` | 近失，不是毕业 |
| Provider boundary | `blocked_no_provider` | 无 provider 时不会伪造 live pass |
| ParamSensitivity-1 | base `1.000` / core basin `0.9407` / extreme `0.8211` | 过程锚对参数有稳定盆地 |
| Representation-Ablation-1 | R1 `1.000` / R2 `0.5926` / R3 `1.000` / R4 `1.000` / R5 `0.5222` | 过程事件桥比表层 token 适配更稳 |

### 24.2 现在可以进论文的正向表述

- 过程锚点有稳定参数盆地。
- process-event bridge 能恢复 structured process SA 的效果。
- 关键不是表层关键词，而是 case-level 过程来源正确。
- 语言学习可以写成 infant-stage online dialogue learning，而不是“已经开放聊天毕业”。

### 24.3 仍然缺的证据

- 主体身份峰
- 直接经验边界
- 关系印象衰减
- 低唤醒陪伴 vs 任务压力竞争
- 幼儿阶段验收更稳的长样本

### 24.4 当前最优先实验

| 方向 | 实验 |
|---|---|
| 理论硬化 | 后继波峰门控、主体身份峰、直接经验边界、关系印象衰减、低唤醒陪伴与任务压力竞争、幼儿阶段验收 |
| 实验硬化 | `FeedbackOverride-1`、`PersistenceReload-1`、`NegativeFeedback-Ablation-1`、`DPP-1 v0.3` |
| 论文硬化 | `ParamSensitivity-1`、`Representation-Ablation-1`、保留旧版优点并增强说服力 |

## 25. 细节补录：当前实现的责任边界、粒度与设计理由

这一章把当前实现里最容易被写散的部分再压成一份可维护的总表。它不是重复前文，而是把前文已经成立的事实，整理成“模块负责什么、对象粒度是什么、默认值是什么、为什么这样设计”。

### 25.1 模块分工总表

| 模块 | 当前职责 | 设计理由 |
|---|---|---|
| `core/runtime/engine.py` | 组织每个 tick 的阶段链：`ingest -> apply_external -> fast_recall_initial -> time_and_fast_recall -> attention_and_rhythm -> slow_recall -> feelings -> expectation_pressure -> action_planning -> SafetyGate -> control_effects -> feedback -> memory_write` | 把一次经验做成可审计的闭环，而不是一段不可见的流水线 |
| `core/state_pool/state_pool.py` | 维护双能量状态池、快慢读视图、预测校验、残差桶、白箱能量流 | 让“状态”成为一等对象，而不是把输入字段直接当答案 |
| `memory/store/memory_store.py` | 负责 posting / vector / numeric / relation / online / temporal 的多通道召回与后继检索 | 把记忆做成多证据合流，而不是单一关键词命中 |
| `memory/short_term/*` | 维护 focus 断续、回声残留、工作记忆窗口、后继偏置 | 把“刚刚还在想什么”与“现在还要继续什么”分开 |
| `channels/*` | 产生认知感受、任务感受、时间感受、节律感受、运行负载感受、期待压力场 | 把内部过程转成可进入状态池的软材料 |
| `core/action/*` | 负责行动候选竞争、行动后果评估、参数记忆、结果记忆、安全门和控制效应 | 让行动先在内部竞争，再以可审计方式影响下一 tick |
| `memory/persistence/*` | 负责 PostgreSQL-first 的权威存储、热窗重载、环境检查、测试录写 | 把长期历史放在可重建的 durable 层，不压在运行时内存里 |
| `education/*` 与 `GL_TaskBuilder/*` | 负责教学协议、技能包、课程退坡、证据生成 | 把学习变成课程工程，而不是隐藏答案注入 |

### 25.2 状态池的对象粒度

状态池当前的基本对象粒度是 `SA label` 级。

这意味着：

- 一个 `PoolEntry` 代表一个可被状态竞争的 SA；
- `display_text` 只是同一 SA 的表层读法；
- `anchor_meta` 记录的是过程来源、过程角色、位置、读帧、反馈等低粒度过程信息；
- 状态池匹配时不是拿整句去硬对整句，而是拿 label / display / bigram / focus / energy / sequence / relation / learned / time 一起做证据合并。

当前白盒字段是：

`sa_label / display_text / family / source_type / real_energy / virtual_energy / cognitive_pressure / attention_gain / fatigue / last_seen_tick / last_updated_tick / provenance / anchor_meta / numeric_features / reconstruction_payload`

当前两个最重要的派生公式是：

```text
query_weight = real*1.0 + virtual*0.65 + attention_gain*0.8 - fatigue*0.3
attention_score = cognitive_pressure*0.7 + attention_gain - fatigue*0.5 + virtual_energy*0.25
```

它们的含义很直白：

- `real_energy` 更像“当前真的在场”；
- `virtual_energy` 更像“当前值得继续期待”；
- `cognitive_pressure` 是二者的张力；
- `attention_gain` 让某些条目更值得继续看；
- `fatigue` 让重复项逐渐让位。

### 25.3 读写视图与容量

| 视图 | 当前上限 | 作用 |
|---|---:|---|
| `query_view()` | `8` | 白箱观测视图，便于检查当前状态，不是主召回入口 |
| `attention_view()` | `8` | 白箱注意力视图，便于解释当前焦点，不是主召回入口 |
| `read_r_state()` | `7` 个头，每头 `256` 项 | fast-system 正门，给 Bn / Cn 用 |
| `snapshot()` | `24` | 白箱可读视图，用于观测和调试 |
| `snapshot_for_memory_write()` | `1024` | 长期记忆写回视图，保留完整 SA 级写回粒度 |

`read_r_state()` 当前五头分别是：

1. `head_recent`
2. `head_anchor`
3. `head_prediction`
4. `head_residual`
5. `head_global`

它们分别代表：

- `head_recent`：最近外源证据；
- `head_anchor`：热锚点；
- `head_prediction`：预测槽中的虚量；
- `head_residual`：未解残差；
- `head_global`：从多头和维护环里合并出来的全局近似。

### 25.4 召回算法的当前形状

当前召回不是单路命中，而是六路合流：

1. `posting`
2. `vector`
3. `numeric`
4. `relation`
5. `online embedding`
6. `temporal applicability`

查询特征当前包含：

- `labels`
- `displays`
- `bigrams`
- `sequence_bigrams`
- `relation_tokens`
- `focus_labels`
- `numeric_features`

总分的当前骨架是：

```text
score_before_temporal =
  label_overlap*1.15
  + display_overlap*0.45
  + bigram_overlap*0.9
  + focus_overlap*0.7
  + state_match*0.55
  + energy_overlap*1.35
  + sequence_overlap*0.8
  + posting_score*0.35
  + vector_score*0.4
  + numeric_score*1.15
  + relation_score
  + learned_score*0.28
  + time_match
score = score_before_temporal * temporal_applicability
```

这套分布的意义是：

- label/display 让表层可读；
- bigram/sequence 让过程顺序可读；
- focus 让当前注意史可读；
- energy/state_match 让真实认知强度可读；
- posting/vector/numeric/relation/learned 让不同证据源各自贡献；
- temporal_applicability 让旧经验还能在现在被重新点亮，但不会把时间感抹平。

### 25.5 后继波峰门控与语言学习

当前门控的核心不是“关键词命中”，而是“后继波峰是否清楚”。

- 没有清晰后继波峰时，系统偏向模仿、回看、重读、回声；
- 有清晰后继波峰时，系统偏向接着说、接着写、聚合多个后继片段。

`FocusSuccessorBias` 当前参数是：

- `context_limit = 2048`
- `max_successors_per_context = 64`
- `max_context_labels = 8`
- `max_order = 3`
- `top_k = 12`
- `per_tick_update_limit = 16`
- `gain = 0.42`
- `max_bias = 0.48`
- `entropy_floor = 0.28`

它学的不是答案表，而是“什么前后文经常接在一起、且有足够支持度”的小统计偏置。

这正好服务语言学习六阶段：

1. `echo imitation`
2. `successor prediction`
3. `multi-reply aggregation`
4. `process-paradigm binding`
5. `keyword organization`
6. `grammar/style refinement`

当前阶段之所以仍然像幼儿期，是因为过程锚已经有了，但关键词组织、主体身份峰和直接经验边界还在继续硬化中。

### 25.6 本地在线学习嵌入

在线嵌入当前是一个本地 token 关联表，不是学生侧 LLM，也不是隐藏 solver。

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `dim` | `32` | 低维本地关联空间 |
| `token_limit` | `2048` | 可保留 token 上限 |
| `min_support_to_promote` | `2` | 支持度达到后才算 promoted |
| `per_tick_update_limit` | `8` | 每 tick 最多更新 8 次 |
| `scoring_token_limit` | `256` | 参与打分的 token 上限 |
| `learned_weight` | `0.28` | learned similarity 权重 |
| `transition_learned_weight` | `0.18` | learned transition 权重 |

它支持三种学习事件：

- `observe_positive_pair(a, b)`：共现证据；
- `observe_negative_pair(a, b)`：对比分歧证据；
- `observe_transition_pair(a, b)`：后继证据。

它返回的也是证据读数，而不是答案：

- `learned_similarity`
- `learned_transition`
- `pair_evidence`

这让它更像“可解释关联记忆”，而不是把模型变成一个外置大词典。

### 25.7 记忆、时间与残差

当前短时系统有三层：

| 组件 | 上限 | 作用 |
|---|---:|---|
| `FocusBuffer` | 历史 `12` | 保存最近在想什么、怎么断续 |
| `ShortTermEchoBuffer` | 历史 `128` | 保存感官回声与思维残响 |
| `ShortTermMemoryWindow` | 历史 `64` | 保存可主动回看的工作记忆 |

回声寿命当前按模态分配为：

- `vision = 4`
- `audio = 24`
- `text = 10`
- `thought = 14`

`TimeFeelingChannel` 当前给出的时间感字段是：

- `dominant_delta_t`
- `confidence`
- `cluster_mass`
- `dominance`
- `source_count`
- `fatigue`
- `recall_gain`

`RhythmChannel` 当前给出的节律感字段是：

- `period_ticks`
- `regularity`
- `recurrence`
- `recovery_match`
- `groove`
- `phase_expectation`

`BAnchorExpectationVerifier` 让 B 记忆锚变成跨 tick 的期待/压力验证项。它的关键意义是：

- B 是被召回的记忆对象；
- B 不是状态池里的普通 SA；
- 期待与压力是从 B 的 successor 证据里长出来的，不是事后口头贴标签。

### 25.8 认知感受与情绪通道

当前直接认知感受通道是：

- `surprise`
- `coherence`
- `dissonance`
- `correctness`
- `grasp`
- `expectation`
- `pressure`

工厂派生的认知感受是：

- `uncertainty`
- `evidence_gap`
- `quantity_grasp`
- `step_closure`
- `computation_pressure`
- `sensory_clarity`

任务感受是：

- `boredom`
- `fulfillment`
- `task_available`
- `unfinished_strength`
- `recall_strength`
- `successor_clarity`
- `external_quiet`

运行负载感受是：

- `complexity`
- `simplicity`
- `load_ratio`

期待压力场是：

- `expectation_level`
- `pressure_level`
- `satisfaction_level`
- `expectation_gap`

8 通道 NT 基线仍然是：

| 通道 | 基线 |
|---|---:|
| `DA` | `0.12` |
| `ADR` | `0.05` |
| `OXY` | `0.12` |
| `SER` | `0.18` |
| `END` | `0.10` |
| `COR` | `0.06` |
| `NOV` | `0.08` |
| `FOC` | `0.10` |

情绪调制器的作用是把这些软感受映射成注意力、学习与行动的调制参数，而不是替代 AP 自己的判断。

### 25.9 行动系统的当前组织

当前注册 actuator 共 `14` 个，action node 共 `39` 个。

安全门当前只审外部动作，阈值是：

- `veto_pressure_threshold = 0.64`
- `veto_cor_threshold = 0.68`
- `review_pressure_threshold = 0.42`
- `review_cor_threshold = 0.50`
- `min_external_confidence = 0.58`

行动的长时学习分成两层：

| 记忆 | 当前参数 | 学什么 |
|---|---|---|
| `ActionOutcomeMemory` | `learning_rate = 0.18`，`decay_per_tick = 0.992`，`support_scale = 6.0`，`max_drive_bias = 0.75` | 动作结果好不好、趋向是什么 |
| `ActionParameterMemory` | `learning_rate = 0.20`，`decay_per_tick = 0.994`，`support_scale = 3.0`，`max_records_per_action = 24`，`max_drive_bias = 0.22` | 这个动作在什么参数样式下更顺手 |

这两个记忆合起来，构成了“动作本身”和“动作参数样式”的分离学习。

### 25.10 数据库是什么形式

权威存储当前是 `PostgreSQL-first`，schema 版本是 `apv21_postgres_memory_schema/v2`。

核心表包括：

- `ap_memory_schema_version`
- `ap_runs`
- `ap_ticks`
- `memory_snapshots`
- `memory_snapshot_items`
- `memory_state_field_items`
- `memory_core_items`
- `memory_posting_tokens`
- `memory_vectors`
- `memory_numeric_features`
- `memory_relation_features`
- `memory_transitions`
- `memory_learning_events`
- `memory_action_feedback_events`
- `memory_asset_refs`
- `memory_index_audit_runs`
- `memory_index_audit_rows`

这里的层次关系是：

- `memory_snapshots` 是权威历史；
- `memory_state_field_items` 是 Bn 主认知视图；
- `memory_core_items` 只是兼容旧工具的外部锚视图；
- posting / vector / numeric / relation / transition / online 都是可重建运行时视图。

热重载只拿热窗，不把全历史灌进内存。

### 25.11 近期证据现在说明了什么

当前最稳的证据组是：

- `DPP controlled = 300/300`
- `Skill37 strict30 live = 30/30`
- `Live Fresh300 = 286/300`
- `Provider boundary = blocked_no_provider`
- `ParamSensitivity-1 = base 1.000 / core basin 0.9407 / extreme 0.8211`
- `Representation-Ablation-1 = R1 1.000 / R2 0.5926 / R3 1.000 / R4 1.000 / R5 0.5222`

它们共同说明了三件事：

1. 过程锚点已经是稳定对象；
2. 过程事件桥能恢复结构化过程效果；
3. 目前语言学习仍更像幼儿期的在线对话学习，而不是成人级开放聊天毕业。

### 25.12 现在最值得继续做的三类工作

| 方向 | 重点 |
|---|---|
| 理论硬化 | 后继波峰门控、主体身份峰、直接经验边界、关系印象衰减、低唤醒陪伴与任务压力竞争、幼儿阶段验收 |
| 实验硬化 | `FeedbackOverride-1`、`PersistenceReload-1`、`NegativeFeedback-Ablation-1`、`DPP-1 v0.3` |
| 论文硬化 | `ParamSensitivity-1`、`Representation-Ablation-1`、保留旧版优点并增强说服力 |

这份说明书后续的更新方式也会保持同样的节奏：

1. 设计；
2. 审查完善；
3. 通过落地；
4. 严谨验收测试；
5. 最终汇总报告。

## 26. 本轮新增的底层补强钉子

这一轮不再重复旧架构，而是把最该先补的三根钉子写死：

### 26.1 短期叙事槽

- 目标不是再加一个缓存，而是把每个 tick 的完整注意焦点包显式成一个叙事槽。
- 槽内顺序只做软偏置。
- 槽与槽之间的相对顺序更强，但仍然是偏置。
- 槽的目标容量先按 32 个多通道 SA 设计，再看是否需要更细分层。
- 这个槽每 tick 都应汇成一个内源性的短期刺激包，进入状态池时以虚能量为主，作为唯一的叙事性内感受槽来维持思维连续性。
- 这个刺激包要把槽内对象、槽位系数、相对顺序和连续性一起折进 SA；顺序通道本身也应是 SA，而不是附属注释。
- 它的作用不是盖住外界输入，而是像人类工作记忆那样，让“我刚刚在想什么”持续有一个可读、可接、可回看的内部场。

### 26.2 残差式 B 召回

- 每轮只取一个最强 B。
- 本轮成功匹配的 SA 在下一轮降低有效权重。
- 未匹配的 SA 在下一轮自然更容易冒头。
- 这样更适合连续动作、连续说话和多行动器并行。

### 26.3 峰型 C 后继

- `Δt=1` 要有明显主峰。
- `Δt=2` 起快速下跌。
- 更远处保留低尾巴。
- 节拍场景再叠加 Rhythm 周期门控。

### 26.4 这三根钉子对应的验证

- `SuccessorPeakGate-1`
- `ResidualBRecall-1`
- `SlotHistoryReload-1`

它们会和 `FeedbackOverride-1`、`PersistenceReload-1`、`NegativeFeedback-Ablation-1`、`DPP-1 v0.3` 组成下一轮的核心补强组。
