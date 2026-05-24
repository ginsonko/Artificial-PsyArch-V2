# AP 二期工程原型

> 当前状态：**持续迭代中的实验原型仓，不是成品，不是通用产品，也不是“已经定型的最终理论实现”**。  
> 这个仓库主要用于：  
> 1. 验证 AP 二期认知闭环中已经落地的部分；  
> 2. 给外部审阅者提供一个**可复查、可运行、可对照论文与实验报告**的工程入口；  
> 3. 持续暴露工程问题、理论边界和后续调优方向。  
>
> 如果你是第一次审阅这个项目，**请不要直接从局部模块（例如先天规则、某个调参器、某份旧报告）外推整个架构结论**。  
> 推荐先看：
>
> 1. [docs/APV2_对Claude审计与常见误解的纠偏说明_20260525.md](./docs/APV2_对Claude审计与常见误解的纠偏说明_20260525.md)
> 2. [evidence/README_证据索引_20260525.md](./evidence/README_证据索引_20260525.md)
> 3. [docs/V2_阶段性实验评估报告_20260520.md](./docs/V2_阶段性实验评估报告_20260520.md)
> 4. [docs/V2_多模态教学与海豚训练综合实验报告_2026-05-23.md](./docs/V2_多模态教学与海豚训练综合实验报告_2026-05-23.md)
>
> 这四份材料会先告诉你：
>
> - AP 二期当前到底在证明什么；
> - 哪些批评是成立的；
> - 哪些“纯规则系统 / 没有任何学习或直觉能力 / 只是 YAML if-then”的判断属于范围偷换或证据不足；
> - 应该怎样公平地做复现实验，而不是拿不合理输入或错版本材料直接下结论。

这是 `AP` 第二期原型的本地工程仓。当前目标不是一次性把论文里的全部强版本都做完，而是按已经确认过的二期设计草案，逐段把真正可跑、可测、可观测、可恢复、可继续扩展的工程底盘做扎实。

到现在，这个仓已经从“文本传感器 + 最小演示页”的阶段，推进到一版更接近项目级底盘的形态：

1. 文本主链最小闭环已经稳定跑通  
   `文本输入 -> SA 竞争 -> 状态池 -> R_state -> Bn -> C_i -> C* -> rules -> 执行器审计 -> A_focus -> 短期记忆 -> 长期记忆`
2. `MemoryStoreV2` 已升级为混合召回版  
   `posting + bigram + 向量嵌入 + FAISS ANN + 时空邻域`
3. 多模态主链已经统一  
   文本、图片、音频都能进入同一条 tick 主链；截图感知也已经接入
4. runtime 导出 / 导入 / checkpoint / continue 已可用
5. 观测台已经从一页 JSON 预览，升级为带图表、指标卡、规则编辑器、调参编辑器的控制台
6. 执行器已经不再只是“概念 sandbox”  
   已经具备安全白名单动作、手动动作 API、截图预览、dry-run/live 双模式
7. 规则层已经从硬编码 if 升级为“声明式规则集 + 调参档”持久化结构
8. run 级增量 rollup / tick list cache 已接入，前端不再靠反复全量扫描 chunk 文件重建
9. 自动化回归已经扩大到 `103` 项测试，并全部通过
10. 规则编辑器与调参编辑器已经具备结构化增删条件 / 效果 / 档位 / 调参项的能力，不再主要依赖手写 JSON 文本框
11. 观测台已进一步补强为结构化实验控制台：
    - Tick 表格不再只有 `Bn`
    - 审计区不再只有规则名和动作名
    - 已新增近期 run 对比表
    - 已新增 run 批量摘要预取，减少最近 run 列表刷新时的 N+1 请求
    - 已新增 tick 内情绪 / 规则 / 动作连续趋势图
12. 默认配置中的浏览器 smoke test 临时规则 / 临时调参档已清理，避免测试条目污染正式默认配置
13. 观测台自动刷新已做轻量化：
    - 高频自动刷新默认走 `runtime summary`
    - 只有显式查看 `Runtime JSON` 或点击按钮时才拉全量 `runtime export`
    - 手动选中的旧 tick 不会再被自动刷新强行拉回最新 tick
    - 自动刷新已增加重入保护，避免长时间运行时多轮刷新互相叠加
    - 已选 tick 的 `tick + sidecar` 结果会被前端缓存复用，减少重复请求
14. 观测台结构化白箱继续补强：
    - 新增 `Sidecar 结构`
    - 新增 `Runtime 结构`
    - 新增 `Live 结构`
    - 新增缓存命中 / 自动刷新状态小面板
    - 新增规则与调参“启用可见 / 停用可见”批量操作
15. 观测台长跑可观测性继续补强：
    - 新增 `运行新鲜度` 提示
    - 运行态下会提示是否可能停在同一 tick
    - 完成态不再把“未继续增长”误显示成疑似卡住
16. `Manifest / 索引` 页签继续结构化：
    - 新增 `Run 元数据`
    - 新增 `索引专题`
    - 新增更细的时空索引参数展示
17. 规则编辑器与调参编辑器继续提升可操作性：
    - 规则新增 `family 过滤` 与 `启用状态过滤`
    - 调参新增 `target 过滤` 与 `启用状态过滤`
    - 新增当前筛选结果统计提示，便于批量启停前确认范围
18. 长跑观测控制继续增强：
    - 自动刷新间隔可直接在前端调整
    - 新增 `刷新诊断` 提示，显示最近成功时间、连续失败次数、最近错误
19. `Sidecar 结构` 继续补细：
    - 新增一级召回细表
    - 新增预测分支细表
    - 更适合直接核对 `bn_list / c_i_list`，少翻 JSON
20. 编辑器工作流继续补顺：
    - 新增或复制规则 / 调参档后会自动滚动到新卡片
    - 新卡片会短暂高亮，减少长列表里“新增了但没看到”的感觉
21. 刷新链路与规则试算可信度继续补强：
    - 前端刷新入口统一收口到 `triggerRefresh()`，减少长跑时未处理 Promise 异常的风险
    - `规则试算` 不再覆盖最近真实 `rules_result`
    - 审计区现在明确区分：
      - `最近真实执行`
      - `当前草稿试算`
    - `规则试算` 会直接试前端当前草稿，不必先保存才能看结果
    - 可一键 `回到真实审计`
22. 观测台编辑器与审计联动继续补强：
    - 规则编辑器新增 `审计过滤`
      - 全部
      - 只看真实命中
      - 只看草稿新增
      - 只看草稿缺失
      - 只看有 warning
    - 调参编辑器新增 `审计过滤`
      - 全部
      - 只看真实命中
      - 只看草稿新增命中
      - 只看草稿缺失命中
      - 只看有 warning
    - 规则卡片可直接显示：
      - 真实命中
      - 草稿命中 / 草稿新增 / 草稿缺失
      - 驱动力变化
      - warning 数
    - 调参档卡片可直接显示：
      - 真实命中
      - 草稿命中 / 草稿新增命中 / 草稿缺失命中
      - 调参变化
      - warning 数
    - 卡片说明会直接给出：
      - 真实驱动 / 草稿驱动
      - 真实调参 / 草稿调参
      - warning code 摘要
23. 编辑器草稿脏检查继续校准：
    - 前端保存态 / 草稿态比较前会先做 canonical 规范化
    - `规则试算` 不再把未修改的编辑器误标成“未保存改动”
    - `导入 / 保存 / 刷新 / 重置` 后的状态线语义更稳定
24. 观测台继续减少高频 JSON 依赖：
    - `Manifest / 索引` 页签新增 `配置快照 / 运行轨迹 / Run 备注`
    - `Runtime 结构` 页签新增 `锚点热点 / 近期外源输入`
    - 审计对照区新增 `对照过滤`
      - 默认 `只看变化项`
      - 可切换为 `显示全部`
    - 对照区会同步显示当前过滤后仍保留下来的规则 / 驱动 / 调参档 / 调参目标数量
25. 观测台继续向“长期实验控制台”靠拢：
    - `Sidecar 结构` 新增 `采样 / 疲劳` 结构化细表
    - `Sidecar 结构` 新增 `竞争 / 壳细节` 结构化细表
    - `Live 结构` 新增 `近期 focus 热点`
    - `Live 结构` 新增 `Run 状态摘要`
26. 审计对照与编辑器联动继续补强：
    - 对照表中的规则项可直接 `定位规则卡`
    - 对照表中的调参档项可直接 `定位调参卡`
    - 点击后会自动设置搜索条件并滚动到对应卡片
    - 更适合 `看到差异 -> 直接修改 -> 再试算` 的循环
27. 规则 / 调参提醒工作流继续结构化：
    - `规则提醒` 与 `调参提醒` 不再只是文本提示
    - 新增提醒等级统计卡：`ERROR / WARNING / INFO`
    - 新增提醒过滤：
      - `等级过滤`
      - `Code 过滤`
      - `Path / 文本过滤`
    - 新增 `清空过滤`
    - 新增提醒表格视图，直接显示：
      - 等级
      - code
      - path
      - message
      - `定位规则卡 / 定位调参卡`
28. Tick 研究卡继续上翻高频白箱信息：
    - Tick 卡片页新增 `Tick 研究摘要`
    - Tick 卡片页新增 `短期链路观察`
    - 直接上翻：
      - `rules_preview.rules_fired`
      - `emotion_channels`
      - `tuner_preview.matched_profiles`
      - `tuner_preview.adjustments`
      - `competition_summary`
      - `bn_preview` 头部
      - `c_star_preview` 头部
      - `rule_log_preview`
      - `short_term_preview`
      - `r_state_heads`
      - `bn_candidate_source_histogram`
29. 高频 JSON 渲染继续减负：
    - `Tick JSON / Sidecar JSON / Runtime JSON / Manifest JSON / Live JSON`
      改为按需渲染
    - 不再在每次自动刷新时默认对所有 JSON 面板执行全量 pretty-print
    - 更适合长时间挂机运行
30. 规则 / 调参编辑器继续向“研究控制台”推进：
    - 新增 `模板新建` 区：
      - `规则模板`
      - `调参模板`
      - 一键按模板新增而不是只给空白对象
    - 新增 `规则参考面板`
      - 规则数 / 条件数 / 效果数
      - 常用 family
      - 可用 metric
      - 可用 effect type
      - 情绪通道 / 公式种类
    - 新增 `调参参考面板`
      - 档位数 / 条件数 / 调参项数
      - 推荐模板
      - 常用 metric
      - 可调 target
      - 面向长跑实验的调参建议
31. 编辑器渲染链继续收紧：
    - warning 诊断台从状态摘要里拆出独立渲染函数
    - 输入草稿时不再顺手重建整块 warning 表
    - 保留 warning 过滤 / 定位 / 白箱建议，但减少长时间编辑时的不必要重绘
32. 编辑器大列表工作流继续补强：
    - 表单输入改为“单卡同步 + 去抖汇总”
    - 状态线显式区分：
      - `表单已同步`
      - `输入待同步`
    - 长时间编辑时更容易判断当前改动是否已经进草稿
33. 过滤链继续轻量化：
    - rule 搜索 / family / 启用状态 / 审计过滤
    - tuner 搜索 / target / 启用状态 / 审计过滤
    - warning 过滤变更
    - 以上优先走显隐更新，不再默认整表重建
34. 编辑器新增快捷骨架：
    - 条件快捷插入：
      - `高耗时保护`
      - `锚点主线`
      - `已有一级召回`
      - `期待占优`
      - `违和感抬升`
    - 效果快捷插入：
      - `违和感地板`
      - `注入违和感 SA`
      - `继续聚焦驱动`
      - `记录规则日志`
    - 调参快捷插入：
      - `聚焦增益`
      - `采样预算`
      - `后继优势`
      - `锚点偏置`
      - `违和感增益`
35. `Runtime 结构` 页签继续上翻：
    - `状态池摘要`
    - `向量 / Posting / 时空`
    - `短期记忆主线`
    - `执行器摘要`
    - 以 digest 卡形式提供第一眼可扫视的运行态摘要
36. `Sidecar 结构 / Live 结构 / 长期研究专题` 继续上翻为 digest 控制台：
    - `Sidecar 结构` 新增
      - `感受器采样摘要`
      - `召回与预测主线`
      - `focus / exact 摘要`
      - `规则与动作摘要`
    - `Live 结构` 新增
      - `运行新鲜度`
      - `近期主线摘要`
      - `Run 混合摘要`
      - `最新 Run 摘要`
    - `长期研究专题` 新增
      - `逻辑耗时判断`
      - `规模与容量判断`
      - `规则与来源热点`
      - `风险与提醒摘要`
    - 整体目标是让用户先抓住“当前主线 / 热点 / 风险”，再按需下钻表格，而不是先读长表或 JSON

## 当前核心能力

### 1. 文本闭环

已经具备：

- 文本固定预算采样
- 文本疲劳抑制
- phrase SA 与基础 SA 共存
- `Bn` 混合召回
- `C_i` / `C*` 预测链
- `A_focus`
- 短期记忆
- exact memory / focus chain memory 回写
- 规则层注入
- 执行器审计输出

### 2. HDB-V2 当前实现口径

当前 `MemoryStoreV2` 已不再是纯 overlap demo，而是带白箱解释的混合召回层：

- `label_posting`
- `unit_posting`
- `bigram_posting`
- `recent_window`
- `HashEmbeddingV2`
- `VectorIndexV2 (FAISS HNSW / fallback flat search)`
- `SpacetimeIndexV2`
- `score_breakdown`
- `candidate_sources`
- `vector_tokens`
- `memory_index_summary`

当前不是最终论文级数据库，但已经具备专业原型应有的骨架：

- 支持增量写入
- 支持 ANN 检索
- 支持时空近邻扩展
- 支持 sidecar 白箱回放

### 3. 多模态统一编排

已经具备：

- 文本输入
- 图片 bytes 输入
- 音频 wav bytes 输入
- 截图感知输入
- 同一 tick sidecar 中统一可见：
  - `sensor_packet`
  - `image_packet`
  - `audio_packet`
  - `bn_list`
  - `c_i_list`
  - `c_star`
  - `rules_result`
  - `sandbox_result`

当前仍属于 V1 级感受器实现：

- 视觉仍是 patch 级
- 听觉仍是 window 级
- 连续视频流已升级为 lazy realtime source
- webcam / microphone 已具备真实设备流入口与 unavailable 降级路径
- 双耳定位 / 更高频连续截图循环仍未完成

### 4. 连续运行与 checkpoint

已经具备：

- `export-runtime`
- `import-runtime`
- `continue-from-checkpoint`
- 跨 run 的 `runtime_tick_index`
- run 内本地 `tick_index`

当前系统明确区分：

- `tick_index`：当前 run 内回放 tick
- `runtime_tick_index`：跨 run 累积的认知 tick

这保证：

- 观测台回放保持简单
- 近因增益、连续性、后继优势不因 run 切换而断裂

### 5. 执行器 / sandbox

当前已经具备：

- 安全白名单动作筛选
- dry-run / live 两种执行模式
- 手动动作 API
- 最近动作事件审计
- 截图预览 API
- 截图感知 run

当前支持的动作包括：

- `move_mouse`
- `click`
- `double_click`
- `scroll`
- `type_text`
- `press_key`
- `move_gaze`
- `continue_focus`
- `inspect_residual`
- `noop`

默认配置仍然是安全边界优先：

- `executor_enabled = false`
- `executor_dry_run = true`

也就是说，当前仓默认不会擅自真实操作电脑，但这条链已经做通了。

### 6. 观测台 V2

当前观测台已经具备：

- Run 列表
- Tick 列表
- Run 总览卡片
- 运行趋势图表
- 情绪 / 规则 / 动作连续趋势图
- Tick 卡片 / Tick 表格 / JSON 切换
- Sidecar 结构卡片 / 表格
- Runtime 结构卡片 / 表格
- Manifest / 索引 / 审计卡片 + JSON 切换
- Live 结构卡片 + JSON 切换
- 运行控制入口
- 文本闭环启动
- 多模态样例启动
- 截图感知启动
- 截图预览
- 手动动作触发
- 先天规则编辑器
- 自适应调参器编辑器
- 规则试算入口
- 近期 run 对比表
- 最近 run 批量摘要预取接口
  - `/api/runs/overview-batch`
- 规则 / 调参编辑器结构化概览卡
  - 启用数
  - 总是触发 / 总是命中数
  - 空效果 / 空调参项数
  - 热门 family / target
- 自动刷新与缓存状态卡
  - 自动刷新运行 / 暂停状态
  - tick 缓存数量
  - run 摘要缓存数量
  - tick / run 缓存命中率
- 规则 / 调参批量启停
  - 启用可见规则
  - 停用可见规则
  - 启用可见调参档
  - 停用可见调参档
- Tick 结构化细表：
  - `state_top`
  - `C*`
  - `Bn + score_breakdown`
  - `C_i`
  - `short_term_preview`
  - `R_state heads`
  - `candidate source histogram`
  - `state_pool_summary / sidecar 摘要`
- Tick 研究摘要卡：
  - `规则命中`
  - `情绪通道`
  - `调参命中`
  - `调参输出`
  - `竞争摘要`
  - `一级召回头部`
  - `综合预测头部`
  - `规则日志`
- 短期链路观察卡：
  - `focus_memory`
  - `focus_text`
  - `bn_refs_tail`
  - `r_state_heads`
  - `候选来源分布`
- 审计 / 规则结构化细表：
  - 条件命中详情
  - 效果摘要
  - 规则日志
  - 动作驱动力
  - 调参命中与调参结果
  - 指标快照
- Sidecar 结构化细表：
  - 输入源 / 感受器预算 / 外源写入 / 预测写入
  - focus_memory / exact_memory
  - spacetime / modality / vector token 口径
  - c_star / competition / pool_result 摘要
  - 热锚缓存
  - 最近执行事件
  - 状态池旁路摘要
- Runtime 结构化细表：
  - runtime 状态池摘要
  - 向量 / posting / 时空索引摘要
  - 短期记忆 tail
  - 执行器状态
- Manifest / 索引结构化细表：
  - Run 元数据
  - 索引专题摘要
  - 更细的时空索引参数项
- Live 结构化细表：
  - live 状态 / 活动 run / 最新 run / 最新 tick
  - 最近 tick ring
  - 已知 run 列表
  - live 元信息
  - 运行新鲜度 / 停滞提示

设计原则是：

- 在线轻观测
- 离线重审计
- 前端尽量消费后端增量汇总缓存，不重走一期那种容易拉爆内存的路线
- 尽量把高价值白箱信息上翻为表格和图卡，而不是要求操作者反复阅读 JSON
- 长时间挂机运行时，默认自动刷新应尽量走轻摘要，不把完整 runtime 当成高频心跳包
- 若只是结构化观测，优先停留在卡片 / 表格页；JSON 面板保留给定点审计

## 目录结构

- `core/`
  - `runtime_v2.py`
  - `state_pool_v2.py`
  - `sa_registry_v2.py`
- `memory/`
  - `memory_store_v2.py`
  - `embedding_v2.py`
  - `vector_index_v2.py`
  - `spacetime_index_v2.py`
  - `short_term_memory_v2.py`
- `sensors/`
  - `text_sensor_v2.py`
  - `vision_sensor_v1.py`
  - `hearing_sensor_v1.py`
- `iesm/`
  - `rules_engine_v2.py`
- `observatory_v2/`
  - `app.py`
  - `web.py`
  - `run_rollup.py`
  - `computer_executor.py`
  - `agent_sandbox.py`
  - `web_static/index.html`
- `docs/`
  - 各阶段验收记录
  - 操作入口总表

## 常用启动入口

### 双击入口

- [启动观测台.bat](./启动观测台.bat)
- [启动单次实验.bat](./启动单次实验.bat)
- [启动多模态样例实验.bat](./启动多模态样例实验.bat)
- [启动批量多模态样例实验.bat](./启动批量多模态样例实验.bat)
- [启动截图感知实验.bat](./启动截图感知实验.bat)
- [启动视频流样例实验.bat](./启动视频流样例实验.bat)
- [启动自主循环样例.bat](./启动自主循环样例.bat)
- [导出当前Runtime检查点.bat](./导出当前Runtime检查点.bat)
- [安装环境.bat](./安装环境.bat)
- [环境自检.bat](./环境自检.bat)
- [从Checkpoint继续文本运行.bat](./从Checkpoint继续文本运行.bat)

### 命令行入口

启动观测台：

```bash
python -m observatory_v2 serve --host 127.0.0.1 --port 8766 --no-browser
```

环境安装与可复制运行说明：

- [docs/环境安装与可复制运行说明.md](./docs/环境安装与可复制运行说明.md)
- [docs/视频流与自主循环入口说明.md](./docs/视频流与自主循环入口说明.md)

文本闭环：

```bash
python -m observatory_v2 run-text --text "今天 天气 不错" --text "今天 天气 有点 冷" --text "算了 不说了"
```

多模态样例：

```bash
python -m observatory_v2 run-dataset --dataset config\sample_dataset_multimodal.json --label "多模态样例运行"
```

批量文本实验：

```bash
python -m observatory_v2 run-dataset --dataset config\sample_dataset_text.json --label "批量文本实验"
```

批量多模态实验：

```bash
python -m observatory_v2 run-dataset --dataset config\sample_dataset_multimodal.json --label "批量多模态实验"
```

工程化数据集运行：

```bash
python -m observatory_v2 run-dataset --dataset config\sample_dataset_pipeline.json --label "数据集管线实验"
```

AP-agent 数据集级耦合样例：

```bash
python -m observatory_v2 run-dataset --dataset config\sample_dataset_ap_agent_pipeline.json --label "AP-agent 耦合样例"
```

持续自主 Session 数据集样例：

```bash
python -m observatory_v2 run-dataset --dataset config\sample_dataset_autonomous_session_pipeline.json --label "持续自主 Session 样例"
```

截图感知：

```bash
python -m observatory_v2 run-screen --ticks 1 --text "截图感知样例"
```

导出 runtime：

```bash
python -m observatory_v2 export-runtime --out outputs\runtime_checkpoint.json
```

从 checkpoint 继续：

```bash
python -m observatory_v2 continue-from-checkpoint --in outputs\runtime_checkpoint.json --text "继续 看看" --text "再 想 一下"
```

真实设备流入口：

```bash
python -m observatory_v2 run-webcam-stream --max-frames 8 --text-prefix "观察摄像头"
python -m observatory_v2 run-microphone-stream --max-windows 8 --tick-window-ms 50 --text-prefix "监听环境"
```

## 观测重点

### 在线接口

- `/api/live`
- `/api/runs`
- `/api/runs/latest`
- `/api/runs/<run_id>/manifest`
- `/api/runs/<run_id>/overview`
- `/api/runs/<run_id>/rollup`
- `/api/runs/<run_id>/ticks`
- `/api/runs/<run_id>/ticks/<tick_index>`
- `/api/runs/<run_id>/ticks/<tick_index>/sidecar`
- `/api/runtime/export`
- `/api/executor/status`
- `/api/executor/screen-preview`
- `/api/executor/manual-action`
- `/api/rules`
- `/api/rules/simulate`
- `/api/tuner`

### sidecar 白箱点

- `competition_packet`
- `bn_list`
- `score_breakdown`
- `candidate_sources`
- `c_i_list`
- `c_star`
- `rules_result`
- `sandbox_result`
- `memory_index_summary`
- `state_pool_sidecar`
- `short_term_snapshot`

### 规则与调参

当前 `rules_engine_v2` 已支持：

- `config/innate_rules_v2.json`
- `config/auto_tuner_v2.json`
- 规则条件读取
- 规则效果执行
- 规则试算
- 前端结构化编辑
- 调参档读取与保存
- 条件行 / 效果行 / 调参项的结构化增删
- 复杂公式 JSON 兜底编辑
- 保存时的结构化校验与 warning 回传
- 重复 `rule_id` / `profile_id` 自动改名保护
- 非法条件 / 非法公式 / 非法调参项的 warning 提示
- 导入前基础 schema 预检

## 自动化验证

回归命令：

```bash
python -m unittest discover -s tests -v
```

当前通过数：

- `83` 项测试全部通过

覆盖重点包括：

1. 文本感受器预算与疲劳
2. 状态池 sidecar 上界
3. phrase SA / mixed recall / `C*`
4. 向量召回与时空邻域
5. runtime export/import/forget
6. checkpoint 后继续
7. 多模态统一 run
8. API 路径
9. 跨 run 全局 `runtime_tick_index`
10. 执行器状态与手动动作 API

## 当前仍未完成的部分

虽然已经比之前成熟很多，但这版仍然不是最终冲顶版本。当前主要缺口有：

1. 真正的大规模向量数据库落地  
   当前是内嵌式 FAISS + Python 索引骨架，不是完整外部库部署版
2. 视觉/听觉长流  
   已支持音频文件 lazy stream、视频文件 lazy stream、webcam、microphone 入口；双耳定位与更高频连续循环仍未完成
3. 奖励/惩罚闭环  
   当前已具备自动反馈、教师反馈、动作反馈和长期 bias/tuner 塑形；但真实长期环境任务管理与更强外界奖惩来源仍待补强
4. 更完整的观测专题页  
   长期统计与规则/来源专题已补上，但仍缺更深的记忆连续性专题、遗忘专题与多 run 关联分析
5. 冷归档 / 遗忘系统增强  
   当前已支持策略化 forget（`latest_only` / `score_prune`）、dry-run 预演、kind 保护、现实度阈值、总能量阈值与数量上限；完整冷归档/分层迁移仍是后续增强方向。  
   另外，最近一次真实 forget 摘要与最近一次 dry-run 预演摘要现在都会持久化到 `outputs/live/service_runtime_state.json`，观测台重启、runtime checkpoint 导入后仍可恢复。  
   `manifest / live / checkpoint / service state` 相关 JSON 当前统一走原子替换写入，降低长时间运行时的半截写坏风险。

## 当前建议

如果继续推进，最顺的顺序还是：

1. 把 `MemoryStoreV2` 继续抬成更正式的向量/ANN/时空库
2. 把连续截图 / 视频 / 音频流闭环做实
3. 把执行器与奖励/惩罚反馈接起来
4. 把观测台扩到长期实验级
5. 再做规模实验与长期跑数

## 备注

当前仓默认仍是：

- 本地优先
- 验证优先
- 观测优先
- 安全边界优先

这个仓目前仍不是 Git 仓。

## 2026-05-19 深夜继续补强（三）

这一轮继续聚焦“长期研究工作台”完成度，而不是新增主链哲学分支。

### 本轮新增能力

1. 观测台新增跨 run 长期趋势区：
   - `mean_logic_ms`
   - `memory_count_last`
   - `state_pool_size_last`
   - `bn_peak`
   - `tick_count`
2. 观测台新增研究导出能力：
   - 导出当前 `runtime checkpoint` 文件
   - 导入既有 `runtime checkpoint` 文件
   - 导出当前 run 的 Markdown 报告
   - 导出近期 run 对比 Markdown 报告
3. 规则编辑器与调参编辑器继续增强：
   - 未保存状态提示
   - 重置改动
   - 导入 JSON 草稿
   - 导出 JSON 草稿
   - 刷新前覆盖确认
4. 修补了一个高风险编辑器问题：
   - 搜索过滤现在只影响“显示”
   - 不再因为搜索状态下保存而丢失未命中的规则或调参档
5. 观测台新增配置元数据摘要表：
   - `repo_root`
   - `outputs_root`
   - `executor_enabled / dry_run`
   - `tick_list_limit`
   - `run_chunk_size`
   - `memory_vector_dim`
   - `memory_ann_engine`

### 本轮验证

1. 前端内嵌脚本重新通过语法检查
2. 重新执行：

```bash
python -m unittest discover -s tests -v
```

结果仍为：

- `26` 项测试全部通过

3. 新增 `/api/config` 自动化检查，保证观测台依赖的配置元数据接口不会静默失效
4. 本地服务已重新启动并确认：

```text
http://127.0.0.1:8766
```

5. 浏览器侧已确认新增控件存在并可见：
   - `导出 Runtime 文件`
   - `导入 Runtime 文件`
   - `跨 Run 长期趋势`
   - `重置改动`
   - `导出 JSON`
   - `导出当前 Run 报告`

## 2026-05-19 深夜继续补强（四）

这一轮继续把项目往“长期研究工作台”方向补强，重点不是再加主链概念，而是把配置编辑安全性、研究统计与报告质量做得更专业。

### 本轮新增能力

1. 规则编辑器 / 调参编辑器新增后端结构化校验：
   - `schema_id` 不匹配提醒
   - 重复 `rule_id` / `profile_id` 自动改名
   - 空 `display_name` 自动补全
   - 条件、效果、公式、调参项的类型与字段检查
   - 非法值自动兜底并返回 warning
2. 前端新增提醒展示：
   - 规则提醒区
   - 调参提醒区
   - 保存后提醒计数
   - 条件数 / 效果数 / 调参项数摘要
3. 观测台新增“长期研究专题”区：
   - 近期平均逻辑耗时
   - 近期峰值逻辑耗时
   - 记忆增长趋势
   - 状态池增长趋势
   - 最常触发规则
   - 最常候选来源
   - 规则累计分布 Top
   - 候选来源累计分布 Top
4. Run 报告导出增强：
   - `rules histogram`
   - `candidate source histogram`
   - `emotion tail`
   - `executor snapshot`
5. 近期 Run 对比报告增强：
   - 增加平均值与峰值摘要
   - 自动给出最佳平均耗时 run
   - 自动给出最大记忆规模 run

### 本轮验证

1. 新增 `tests/test_rules_engine_v2.py`
2. 回归命令：

```bash
python -m unittest discover -s tests -v
```

结果：

- `28` 项测试全部通过

3. 前端内嵌脚本重新通过 Node 语法检查
4. 浏览器只读验收已确认以下页面结构真实存在：
   - `长期研究专题`
   - `长期分布 Top`
   - `rulesWarningsBox`
   - `tunerWarningsBox`
   - `runCompareChart`
   - `导出近期对比报告`

## 2026-05-19 深夜继续补强（五）

这一轮继续专注于“规则/调参编辑工作流”的专业化，不改主链哲学，只补编辑体验、校验链路和观测便利性。

### 本轮新增能力

1. 新增“不落盘校验”接口：
   - `POST /api/rules/validate`
   - `POST /api/tuner/validate`
2. 规则编辑器新增：
   - `校验规则` 按钮
   - 校验后直接回显 warning / stats
3. 调参编辑器新增：
   - `校验调参器` 按钮
   - 校验后直接回显 warning / stats
4. 规则效果中的公式编辑从 `formula JSON textarea` 升级为结构化公式编辑：
   - `kind`
   - `metric`
   - `metrics(list)`
   - `value`
   - `base`
   - `factor`
   - `min`
   - `max`
5. 前端在不保存的情况下也能先做草稿体检，更适合长时间调规则时反复检查

### 本轮验证

1. 再次执行：

```bash
python -m unittest discover -s tests -v
```

结果仍为：

- `28` 项测试全部通过

2. 前端内嵌脚本再次通过 Node 语法检查
3. 浏览器只读验收已确认以下控件真实存在：
   - `校验规则`
   - `校验调参器`
   - 结构化公式编辑器中的 `formula-kind`
   - 结构化公式编辑器中的 `formula-metric`

## 2026-05-19 深夜继续补强（八）

这一轮主要做的是两件事：

1. 把刚落地的“真实执行 vs 草稿试算”差异视图和编辑器折叠工作流做成可验收、可持续维护的能力
2. 把前端“结构还在但整页脚本没跑起来”这类隐性风险纳入自动化护栏

### 本轮新增改进

1. 审计区的 `真实执行 vs 草稿试算` 差异区完成真实验收：
   - 规则数对比
   - 动作驱动数对比
   - 调参命中数对比
   - 新增 / 缺失规则摘要
   - 驱动力变化摘要
2. 规则编辑器折叠工作流完成真实验收：
   - 单卡片折叠 / 展开
   - `折叠可见规则`
   - `展开可见规则`
3. 调参编辑器折叠工作流完成真实验收：
   - 单卡片折叠 / 展开
   - `折叠可见档位`
   - `展开可见档位`
4. 新增前端静态护栏：
   - 直接提取 `observatory_v2/web_static/index.html` 内联脚本
   - 用 Node 编译检查语法
   - 防止再出现“结构测试仍通过，但整页脚本因语法错误提前中断”的情况

### 本轮发现并确认的点

1. 浏览器真实验收中曾发现一类隐性问题：
   - 页面 DOM 骨架仍然存在
   - 但前端脚本如果在加载早期语法中断，刷新链、run 列表、tick 列表、规则审计都会像“没初始化”一样失效
2. 当前盘面代码已重新确认：
   - 内联脚本可编译
   - 页面可正常初始化
   - 最新 run / tick / 审计差异区都能真实出数

### 本轮验证

1. 自动化回归：

```bash
python -m unittest discover -s tests -v
```

结果：

- `32` 项测试全部通过

2. 浏览器真实验收确认：
   - `规则试算` 可切到草稿试算视图
   - `回到真实审计` 可切回最近真实执行视图
   - 差异区在“无明显差异”时也会给出结构化摘要
   - 调参区 `折叠可见档位 / 展开可见档位` 可真实生效

## 2026-05-19 观测台继续补强（十一）

这一轮继续做的是“把观测台从能看推进到更适合长时间研究控制”，重点仍然是前端结构化和长跑稳定性。

### 本轮新增

1. 规则提醒 / 调参提醒诊断台新增 `建议` 列：
   - 针对常见 warning code 直接给出可操作修法
   - 例如重复 ID、空条件、空效果、非法公式、未知 target 等都会给出更直白的修改建议
2. `Live 结构` 新增更完整的元信息：
   - `latest_run_label`
   - `latest_run_path`
   - 更方便长期挂机后快速定位最近一次 run 的真实落盘位置
3. 前端主渲染链改成更偏按需：
   - 主视图只重绘当前激活页签
   - 右侧视图只重绘当前激活页签
   - JSON 页继续保持按需 pretty-print
   - 自动刷新时进一步减少隐藏区块的无意义重建

### 本轮验证

1. 自动化回归：

```bash
python -m unittest tests.test_observatory_frontend_phase17 -v
python -m unittest discover -s tests -v
```

结果：

1. 前端专项 `2` 项通过
2. 全量 `32` 项通过

2. 本地 / HTTP 前端脚本编译复验：

1. `observatory_v2/web_static/index.html` 内联脚本可通过 Node 编译
2. `http://127.0.0.1:8766/` 返回的实际 HTML 内联脚本也可通过 Node 编译

3. 真页验收：

1. `Live 结构` 页签真实显示：
   - `latest_run_label`
   - `latest_run_path`
2. 浏览器插件层对输入类自动化仍有限制，`locator.fill(...)` 会受虚拟剪贴板限制，因此“真页中直接改规则 ID 再点校验”未作为项目前端缺陷处理
3. 对 warning 诊断台的本轮验收采用：
   - 后端 `POST /api/rules/validate`
   - 后端 `POST /api/tuner/validate`
   - 前端静态结构护栏
   - HTTP 实际页面脚本编译
   的组合方式完成

### 本轮边界与收获

1. 本轮曾真实发现一个环境边界：
   - 本地常驻 `8766` 观测台进程在某一阶段仍提供旧服务逻辑
   - 表现为 `/` 已能返回新版静态页，但 `POST /api/rules/validate` 与 `POST /api/tuner/validate` 仍像旧版本
2. 最终通过平滑重启本地观测台进程解决，说明后续每次做观测台真页验收时，应明确区分：
   - 磁盘文件是否已更新
   - 本地服务是否已经加载到当前版本
## 2026-05-19 观测台继续补强（十二）

这一轮继续沿着“减少高频 JSON 依赖，强化结构化编辑器”的方向推进，重点不再是加新面板，而是把规则编辑器和调参编辑器真正做得更像长期可用的研究操作台。

### 本轮新增

1. 规则编辑器新增聚焦工作流：
   - `仅看当前规则`
   - `取消规则聚焦`
   - 每张规则卡也新增 `聚焦此卡 / 取消聚焦`
   - 适合长列表里围绕单条规则做连续修改、试算、回看
2. 调参编辑器新增聚焦工作流：
   - `仅看当前档位`
   - `取消档位聚焦`
   - 每张调参卡也新增 `聚焦此卡 / 取消聚焦`
3. 行级复制能力补齐：
   - 条件行支持复制
   - 规则效果行支持复制
   - 调参项支持复制
   - 适合做相近规则/相近调参的快速变体实验
4. 规则效果中的文本元信息不再只靠紧凑 JSON textarea：
   - `display_text`
   - `reason`
   - `message`
   已在真页中结构化展开为独立输入框，更适合直接阅读和编辑
5. 编辑器增强层新增类型提示：
   - `effect type` 会给出用途提示
   - `tuner target` 会给出用途提示
   - 更方便在不翻文档的情况下直接判断字段意图

### 本轮验证

1. 自动化：

```bash
python -m unittest tests.test_observatory_frontend_editor_focus -v
python -m unittest discover -s tests -v
```

结果：
- 新增前端专项 `1` 项通过
- 全量 `33` 项测试通过

2. HTTP 验证：
   - `http://127.0.0.1:8766/?ts=editorfocus`
   - 已确认源码中存在：
     - `focusVisibleSingleRuleBtn`
     - `clearRulePinBtn`
     - `focusVisibleSingleTunerBtn`
     - `clearTunerPinBtn`
     - `duplicate-condition`
     - `duplicate-effect`
     - `duplicate-adjustment`
     - `pin-rule-card`
     - `pin-tuner-card`
     - `effect-display-text / effect-reason / effect-message`

3. 真页验收：
   - 页面标题正常：`AP 二期观测台 V2`
   - 新增四个聚焦按钮真实可见
   - 真页中实际检测到：
     - `pinRuleButtons = 6`
     - `pinTunerButtons = 2`
     - `duplicateConditionButtons = 10`
     - `duplicateEffectButtons = 12`
     - `duplicateAdjustmentButtons = 6`
     - `effectDisplayInputs = 12`
     - `effectReasonInputs = 12`
     - `effectMessageInputs = 12`

### 本轮意义

这轮的价值不在“又多了几个按钮”，而在于把编辑器从“能改结构化数据”再往前推进了一步，变成：

- 更适合围绕单张卡长时间打磨
- 更适合复制小变体做快速实验
- 更适合直接理解规则/调参的意图
- 更少要求操作者手动读紧凑 JSON

## 2026-05-19 观测台继续补强（十三）

这一轮继续沿着“减少编辑噪音、让结构化编辑真正像研究工作台”的方向推进，不再满足于只是给字段加提示，而是让字段本身跟随当前语义自动收敛。

### 本轮新增

1. 规则效果编辑器新增 `effect type` 驱动字段显隐：
   - `set_emotion_floor` 主要强调 `channel + formula`
   - `inject_sa` 主要强调 `sa_label + when_channel + threshold + display_meta + formula`
   - `add_action_drive` 主要强调 `action_id + reason/display_meta + formula`
   - `append_rule_log` 主要强调 `display_meta`，并把 `formula` 弱化为说明性字段
2. 公式编辑器新增 `formula kind` 渐进显示：
   - `constant` 只强调 `value`
   - `metric` 只强调 `metric`
   - `mul` 强调 `metric + factor`
   - `affine` 强调 `metric + base + factor`
   - `max_metric` 强调 `metrics(list)`
3. 公式提示不再只有静态摘要：
   - 现在会同时显示 `当前公式摘要`
   - 以及当前 `kind` 的中文说明
4. effect / formula 原始模板补齐稳定语义锚点：
   - `data-effect-field=*`
   - `data-formula-key=*`
   - 便于后续增强、测试和审计联动继续深化

### 本轮验证

1. 自动化：

```bash
python -m unittest tests.test_observatory_frontend_editor_focus -v
python -m unittest discover -s tests -v
```

结果：
- 前端专项 `2` 项通过
- 全量 `34` 项测试通过

2. HTTP 验证：
   - `http://127.0.0.1:8766/?ts=phase17e`
   - 已确认源码中存在：
     - `effectFieldMode`
     - `formulaFieldMode`
     - `applyEffectRowMode`
     - `applyFormulaPanelMode`
     - `data-effect-field="channel"`
     - `data-formula-key="kind"`
     - `mode-hidden / mode-muted / field-inline-note`

3. 真页验收：
   - `http://127.0.0.1:8766/?ts=phase17e-browser`
   - 已真实切换并确认：
     - `set_emotion_floor` 会隐藏无关注入/动作字段
     - `inject_sa` 会突出 `sa_label + when_channel + threshold`
     - `append_rule_log` 会突出日志文本并弱化公式
     - `constant / affine / max_metric` 会真实切换公式参数可见性

### 本轮意义

这轮真正解决的是编辑器“看起来结构化了，但还是很吵”的问题。

现在规则编辑和公式编辑都更接近：

- 当前语义只看当前需要的字段
- 不相关字段自动退场
- 不必靠记忆去推断这个类型到底应该改什么
- 更适合长时间做规则实验、调参实验和论文截图级展示

## 2026-05-19 观测台继续补强（十四）

这一轮继续沿着“把编辑器做成真正的研究控制台”推进，不再只是补字段级体验，而是补全局结构感和进入编辑态的闭环。

### 本轮新增

1. 规则参考面板升级为结构概览面板：
   - 新增 `family` 分布
   - 新增 `effect type` 分布
   - 新增 `formula kind` 分布
   - 新增情绪 / 门槛通道使用分布
   - 新增主 family / 主 effect 摘要
2. 调参参考面板升级为结构概览面板：
   - 新增 `target` 分布
   - 新增条件 `metric` 分布
   - 新增主 target / 空调参档数量摘要
3. “定位卡片”升级为“定位并聚焦”：
   - 从 jump / 搜索路径进入规则卡时，会直接设置 `pinnedRuleId`
   - 从 jump / 搜索路径进入调参卡时，会直接设置 `pinnedTunerId`
   - 状态文案同步改为 `已定位并聚焦...`
4. 参考面板新增轻量条形分布视图：
   - 不引入额外图表依赖
   - 直接用轻量结构表达当前规则集 / 调参器的整体构成

### 本轮验证

1. 自动化：

```bash
python -m unittest tests.test_observatory_frontend_editor_focus -v
python -m unittest tests.test_observatory_frontend_phase17 -v
python -m unittest discover -s tests -v
```

结果：
- 全量 `34` 项测试通过

2. HTTP 验证：
   - `http://127.0.0.1:8766/?ts=phase17f`
   - 已确认源码中存在：
     - `renderReferenceBarChart`
     - `topEntriesFromHist`
     - `editor-reference-bar`
     - `state.pinnedRuleId = cleanId`
     - `state.pinnedTunerId = cleanId`

3. 真页验收：
   - `http://127.0.0.1:8766/?ts=phase17f-browser`
   - 已确认：
     - 规则参考面板出现多组结构分布条
     - 调参参考面板出现结构分布条
     - `rule::residual_dissonance` 经搜索后点击 `仅看当前规则` 可收敛为 `visibleRuleCards = 1`
     - `rulesFilterHint` 会显示 `聚焦=rule::residual_dissonance`

### 本轮意义

这轮开始让规则编辑器和调参编辑器不只是“能改”，而是：

- 一打开就能看出当前这套系统主要在做什么
- 从提醒 / 搜索 / 对照进入后能直接只看那一张卡
- 更适合长时间做大规模规则实验与调参实验

## 2026-05-19 持续自主 Session 与底层验收补充

这一轮不是再加新花样，而是把已经补进主链的底层能力做了一次更严格的工程验收和小范围收口。

### 本轮补齐

1. 持续自主 session 的白箱状态语义已经完整上翻：
   - `session_goal`
   - `session_context`
   - `session_health`
   - `recover_hint`
2. 观测台右侧 `Live 结构` 已确认真实显示：
   - `session_phase`
   - `session_health`
   - `session_recover_hint`
   - `session_focus`
   - `session_actions`
3. `autonomousSessionStatusBox` 的展示口径已统一：
   - 统一为 `ticks=X/Y`
   - 不再出现一处写 `tick=4`、另一处写 `ticks=4/4` 的细小分裂

### 本轮验证

1. 自动化回归：

```bash
python -m unittest tests.test_observatory_frontend_phase17 -v
python -m unittest discover -s tests -v
```

结果：
- 前端定向测试通过
- 全量 `103/103` 通过

2. 真页验收：

- 本地启动：

```bash
python -m observatory_v2 serve --host 127.0.0.1 --port 8766 --no-browser
```

- 真实执行：

```bash
python -m observatory_v2 run-autonomous-session --server-url http://127.0.0.1:8766 --max-ticks 4 --wait --text-hint "health browser smoke"
```

- 已确认：
  - `autonomousSessionStatusBox` 真实显示  
    `session: completed / phase=completed / health=completed / ticks=4/4 / 已完成，无需恢复`
  - `Live 结构 -> Run 状态摘要` 真实显示：
    - `session_phase = completed`
    - `session_health = completed / target_completed`
    - `session_recover_hint = 已完成，无需恢复`
    - `session_focus = health / browser / smoke`
    - `session_actions = continue_focus`

### 当前判断

到这一轮为止，下面这些底层已经不是“概念壳”了：

1. 持续自主 session 生命周期
2. session checkpoint / recover
3. session 运行态健康摘要与恢复提示
4. 外部教师 provider 协议骨架
5. FAISS ANN + posting + 时空索引的混合记忆层
6. 统一 realtime source abstraction

但下面这些仍属于下一层能力，而不是本轮已彻底完成：

1. 真正联网的外部 LLM provider 生产化接入
2. 更成熟的长期 goal manager / planner
3. 连续视频流 / 麦克风流在长期真实设备环境下的更重压测
4. 双耳定位与更完整具身场景

### 环境经验

- 如果真页表现和磁盘代码不一致，优先检查本地 `8766` 常驻观测台进程是否已经重启到当前版本。
- `index.html` 是前端静态资源，不适合用 `python -m py_compile` 验证；前端脚本的自动化校验仍以 `tests.test_observatory_frontend_phase17` 为准。
