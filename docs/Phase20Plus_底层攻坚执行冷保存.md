# AP 二期 Phase20+ 底层攻坚执行冷保存

版本：v1  
日期：2026-05-19  
用途：作为本轮“剩余底层问题全部解决”工作的统一执行锚点、续跑锚点、验收锚点。

---

## 1. 本轮之前已经完成的底座

截至本文件落地时，仓库已经不再处于“底层断链”状态，而是已经具备以下稳定基础：

1. AP 主链已经稳定跑通  
   文本 / 图像 / 音频 / 截图输入都可以进入统一 tick 主链，形成：
   - 感受器输入
   - 状态池更新
   - 记忆召回
   - 预测包
   - 规则层
   - 动作驱动
   - 教师审阅
   - 奖惩回写
   - 短期记忆 / 长期记忆

2. 外部教师已经不再只是概念壳  
   当前已有：
   - `core/external_teacher_gateway_v1.py`
   - `TeacherLayerV1` 外部教师接入口
   - `teacher_feedback` 注入状态池
   - `teacher_feedback` 写入长期记忆
   - `teacher_provenance` 审计链路

3. 自主循环已经不是纯演示  
   当前已有：
   - 截图感知
   - 动作沙箱
   - 自动反馈
   - 教师审阅
   - 记忆 / 调参塑形
   - 长时间运行的基本保护阈值

4. 视频流入口已经走到工程化中间态  
   当前已有：
   - 视频文件入口
   - `OpenCV + tempfile` 解码
   - 抽帧配置
   - 帧级 sidecar 元数据

5. 当前测试基线已通过  
   最新已知通过：
   - `python -m unittest tests.test_teacher_layer_v1 -v`
   - `python -m unittest tests.test_autonomous_loop_phase19 -v`
   - `python -m unittest tests.test_stream_adapter_phase18 -v`
   - `python -m unittest tests.test_schema_tools -v`
   - `python -m unittest discover -s tests -v`
   - 全量 `62 / 62 OK`

这意味着：

> 现在不是“从零搭底座”，而是“在已经稳定的主链上，继续把正式 provider、持续 session、统一 realtime source abstraction 做完整”。

---

## 2. 本轮必须彻底解决的剩余问题

本轮要按顺序完成并验收这三大块：

### 2.1 正式 external teacher provider protocol

当前问题：

- 只有 `stub_file`
- 没有正式 provider registry
- 没有标准 request / response schema
- 没有正式的 `http_json` provider
- 未来真实外部教师无法无缝接入

本轮完成标准：

1. 外部教师网关升级为 provider abstraction
2. 至少支持：
   - `off`
   - `stub_file`
   - `http_json`
3. request / response 都有正式 schema
4. provider 输入被严格约束，只能看到：
   - 候选动作
   - 焦点预览
   - 记忆预览 id
   - 状态摘要
   - 少量运行上下文
5. provider 输出被严格约束，只能回：
   - `allow / warn / block`
   - `reward / punishment`
   - `risk_tags`
   - `warning_code`
   - `explanation`
6. provider 失败时，fail-open / fail-closed 行为可测、可审计

### 2.2 持续 autonomous session

当前问题：

- `start_autonomous_run(...)` 仍是预生成固定 tick 的 batch run
- 还不是成熟的长期 session 生命周期
- 没有正式的：
  - start
  - pause
  - resume
  - stop
  - status
- 还不能很好承载长期自主运行和中断恢复

本轮完成标准：

1. 新增连续 session 生命周期
2. session 支持：
   - `start`
   - `pause`
   - `resume`
   - `stop`
   - `status`
3. session 在主链内逐 tick 生成，而不是预先造完整 item 列表
4. session 状态可持久化到 run 目录的 live/system 侧
5. 暂停、恢复、停止都有明确事件和审计记录
6. session 仍然坚持：
   - AP 先产出主链
   - 教师只做审阅/奖惩/安全门控
   - 不让外部教师替代 AP 主认知

### 2.3 统一 realtime source abstraction

当前问题：

- 音频文件、图片序列、视频文件、截图、自主循环入口还各走各路
- 没有统一 source adapter
- 没有统一 source status / cursor / close 语义
- webcam / microphone / live screen / future realtime stream 不容易接
- 视频 decoder 仍偏“入口专用逻辑”

本轮完成标准：

1. 建立统一 source abstraction
2. 统一支持以下 source kind：
   - `audio_file`
   - `image_sequence`
   - `video_file`
   - `screen_capture`
   - `webcam`（允许 graceful unavailable）
   - `microphone`（允许 graceful unavailable）
3. 每个 source 都有统一的：
   - `next_item()`
   - `status()`
   - `close()`
   - cursor / index / total / realtime 标记
4. 视频 decoder 进入统一 source 路径，不再只是单独的 split helper
5. 现有 `start_audio_stream_run / start_image_stream_run / start_video_stream_run / autonomous session`
   尽量都走同一层 source adapter

---

## 3. 本轮执行顺序

严格按下面顺序推进，不跳步：

1. 落本冷保存文件
2. 补 provider protocol 与 schema
3. 跑 provider 定向测试
4. 抽取统一单 tick 处理路径，给 batch run 和 session 共用
5. 落持续 autonomous session 生命周期
6. 跑 session 定向测试
7. 落统一 realtime source abstraction
8. 跑 source / video / audio / autonomous 回归
9. 更新文档与入口说明
10. 跑全量测试
11. 如有粗糙处继续补硬，直到主链、测试、文档三者一致

---

## 4. 本轮设计约束

### 4.1 AP 必须始终是主体

外部教师只能：

- 审阅候选动作
- 警告
- 阻断
- 稀疏奖惩
- 给解释

外部教师不能：

- 直接生成 AP 的主认知内容
- 直接替代 AP 的预测链
- 直接替代 AP 的动作产生逻辑

### 4.2 不重造平行主链

新增的 provider / session / source 都必须尽量挂在现有统一主链上，不新造一套旁路认知逻辑。

### 4.3 先白箱，再功能

新增结构必须优先具备：

- schema
- 审计字段
- 运行状态
- 错误状态
- 事件记录

而不是先追求“能跑一下”。

### 4.4 允许设备能力缺失，但不允许接口塌陷

例如：

- 没有摄像头时，`webcam` source 可以返回 unavailable
- 没有麦克风时，`microphone` source 可以返回 unavailable

但接口本身要统一、可测、可审计。

---

## 5. 本轮验收标准

只有同时满足以下条件，才可以宣称这轮底层攻坚“完成”：

### A. Provider protocol 验收

1. `stub_file` 仍然可用
2. `http_json` 可用
3. request / response schema 校验通过
4. fail-open / fail-closed 行为经过测试
5. 审计链中能看到：
   - provider mode
   - provider reviewer
   - request digest
   - warning code
   - risk tags

### B. Session 生命周期验收

1. session 可以启动
2. session 可以暂停
3. session 可以恢复
4. session 可以停止
5. session status 可查询
6. pause / resume / stop 都有日志与状态文件
7. session 逐 tick 追加写 summary / sidecar / metrics

### C. Realtime source abstraction 验收

1. `audio_file` / `image_sequence` / `video_file` 走统一 source
2. `screen_capture` 可作为统一 source 使用
3. `webcam` / `microphone` 在缺设备环境下不崩溃
4. source metadata 统一且可审计
5. 视频 decoder metadata 更正式

### D. 总体验收

1. 定向测试全部通过
2. 全量测试全部通过
3. 文档与实现口径一致
4. 不破坏现有批量 run、视频流 run、自主 run、观测台现有入口

---

## 6. 预计代码热点

本轮最可能大改的文件：

- `core/external_teacher_gateway_v1.py`
- `core/teacher_layer_v1.py`
- `observatory_v2/app.py`
- `observatory_v2/web.py`
- `observatory_v2/__main__.py`
- `observatory_v2/config.py`
- `config/runtime_config.json`
- `schemas/app_config.schema.json`
- `sensors/stream_adapter_v1.py`
- `tests/test_teacher_layer_v1.py`
- `tests/test_autonomous_loop_phase19.py`
- `tests/test_stream_adapter_phase18.py`
- `tests/test_web_api.py`
- 以及新增的 protocol / session / source 相关 schema 或测试文件

---

## 7. 本轮完成后的目标状态

本轮如果彻底完成，项目状态应从：

> “主链底座稳定，但 provider / session / realtime source 还偏原型级”

推进到：

> “AP 主导的自主系统底层已经具备正式外部教师协议、连续自主 session 生命周期、统一 realtime source abstraction，能够作为更长期真实实验与更强执行器闭环的专业底座。”

---

## 8. 一句话执行口径

> 本轮不再做表层体验补丁，而是把剩余底层短板按顺序彻底补齐：先正规化 external teacher provider protocol，再把 autonomous run 推成持续 session，再把 audio / image / video / screen / future live device 收敛到同一层 realtime source abstraction，最后以定向测试和全量回归作为唯一完成标准。

---

## 9. 2026-05-19 晚间续做进度补记

本冷保存文档落地后，已先完成其中 external teacher provider 正式化这一块，且已经验收通过。

本次已实际完成：

1. `external_teacher_fail_open` 从 `llm_gate_fail_open` 中独立拆出
2. `http_json` provider 增加 transport 审计：
   - `attempt_count`
   - `duration_ms`
   - `status_code`
   - `transport_result`
   - `transport_error_kind`
3. `http_json` provider 增加：
   - `autonomous_external_teacher_max_retries`
   - `autonomous_external_teacher_retry_backoff_ms`
4. CLI / Web API / runtime / observatory teacher 摘要已全部对齐新口径
5. 新增重试成功、独立 fail-closed、tick meta 透传等测试
6. 全量回归已通过：`104 / 104 OK`

详细验收记录见：

- [Phase24_ExternalTeacherProvider正式化验收记录_20260519.md](H:\AP原型实验第二期\docs\Phase24_ExternalTeacherProvider正式化验收记录_20260519.md)
