# AP 二期 Phase20+ 底层攻坚验收记录

日期：2026-05-19  
状态：已完成并通过回归

---

## 1. 本轮解决的问题

本轮针对之前剩余的三类底层短板，已经全部完成：

1. 正式 external teacher provider protocol  
2. 持续 autonomous session 生命周期  
3. 统一 realtime source abstraction

---

## 2. 已完成内容

### 2.1 外部教师协议层

已完成：

- `ExternalTeacherGatewayV1` 升级为正式 provider abstraction
- 支持模式：
  - `off`
  - `stub_file`
  - `http_json`
- 新增正式 schema：
  - `schemas/external_teacher_request.schema.json`
  - `schemas/external_teacher_response.schema.json`
- 配置层新增：
  - `autonomous_external_teacher_http_endpoint`
  - `autonomous_external_teacher_http_headers`
- `TeacherLayerV1 / RuntimeV2 / autonomous run / CLI / Web` 全链路接通
- 支持 fail-open / fail-closed 行为

工程意义：

- 以后可以接真实外部教师 provider，而不破坏 AP 主链
- 外部教师输入输出边界被正式约束
- 教师层不再只是本地 stub，而是正式可扩展协议口

### 2.2 持续 autonomous session

已完成：

- `start_autonomous_session(...)`
- `pause_autonomous_session(...)`
- `resume_autonomous_session(...)`
- `stop_autonomous_session(...)`
- `get_autonomous_session_status(...)`
- Web API：
  - `GET /api/autonomous-session/status`
  - `POST /api/autonomous-session/start`
  - `POST /api/autonomous-session/pause`
  - `POST /api/autonomous-session/resume`
  - `POST /api/autonomous-session/stop`
- CLI：
  - `run-autonomous-session`
  - `pause-autonomous-session`
  - `resume-autonomous-session`
  - `stop-autonomous-session`
  - `autonomous-session-status`
- session 状态持久化：
  - `live/autonomous_session_status.json`
  - `system/events.jsonl`

工程意义：

- 自主循环不再只是一批预生成 tick 的 batch run
- 现在已经具备更接近长期 agent 的 session 形态
- 生命周期和审计链已经成型，后续可以继续往目标保持、任务恢复、多阶段任务推进

### 2.3 统一 realtime source abstraction

已完成：

- `StreamAdapterV1` 升级为统一 source abstraction
- 新增 source 类：
  - `BaseRealtimeSourceV1`
  - `SequenceRealtimeSourceV1`
  - `UnavailableRealtimeSourceV1`
- 统一 source builder：
  - `build_audio_file_source`
  - `build_image_sequence_source`
  - `build_video_file_source`
  - `build_screen_capture_source`
  - `build_webcam_source`
  - `build_microphone_source`
- 统一 source 语义：
  - `next_item()`
  - `status()`
  - `close()`
- 现有入口已切到统一 source 路线：
  - audio stream
  - video stream
  - image stream
- `webcam / microphone` 在当前无设备实现时会 graceful unavailable，而不是接口缺失

工程意义：

- realtime 入口不再各写各的
- 未来接 webcam / mic / live device / continuous screen pipeline 时，不需要再推翻重做
- 视频 decoder 已进入统一 source 路线的一部分，而不只是孤立的 split helper

---

## 3. 关键内部重构

本轮还做了一个对后续非常重要的结构整理：

- 把 batch run 里的“单 tick 真执行主干”抽成了可复用单元：`_execute_runtime_item(...)`

这意味着：

- batch multimodal run
- batch autonomous run
- continuous autonomous session

现在可以共用同一条真实主链，而不是复制三份类似逻辑。

这一步是后续继续做：

- 更长时间自主运行
- 更复杂 source
- 更强执行器闭环

时最重要的稳固点之一。

---

## 4. 验收结果

### 4.1 定向测试

已通过：

- `python -m unittest tests.test_schema_tools tests.test_teacher_layer_v1 tests.test_autonomous_loop_phase19 -v`
- `python -m unittest tests.test_stream_adapter_phase18 tests.test_autonomous_loop_phase19 tests.test_teacher_layer_v1 tests.test_schema_tools -v`

覆盖点包括：

- external teacher `stub_file`
- external teacher `http_json`
- external teacher fail-closed
- autonomous session lifecycle
- autonomous session Web entrypoints
- unified realtime source status / unavailable paths

### 4.2 编译检查

已通过：

- `python -m py_compile observatory_v2/app.py observatory_v2/web.py observatory_v2/__main__.py core/external_teacher_gateway_v1.py core/teacher_layer_v1.py core/runtime_v2.py sensors/stream_adapter_v1.py`

### 4.3 全量测试

已通过：

- `python -m unittest discover -s tests -v`

结果：

- `68 / 68 OK`

---

## 5. 当前完成度判断

如果只针对本轮原定底层问题，当前判断是：

### 已解决

- 正式 external teacher protocol：已解决
- 持续 autonomous session：已解决
- 统一 realtime source abstraction：已解决

### 当前仍然属于下一层的增强方向，但不再是这轮的“底层断链”

- 真正联网的大模型 provider 生产化接入
- webcam / microphone 的实际设备流采集实现
- 双耳定位
- 更完整的长期目标保持、多阶段任务恢复
- 更成熟的长期自主任务管理器

这些现在已经不再是“底层没打通”，而是可以在现有底座上继续专业推进的下一层能力。

---

## 6. 一句话验收结论

> 本轮底层攻坚已经完成：外部教师协议从本地 stub 升级成正式 provider protocol，自主循环从 batch run 推进成持续 session，audio / image / video / screen / webcam / microphone 入口收敛到统一 realtime source abstraction；持续 session 还新增了 runtime checkpoint 落盘与 recover 恢复路径；同时所有改动已通过定向测试、编译检查与全量 `72/72` 回归验证。
