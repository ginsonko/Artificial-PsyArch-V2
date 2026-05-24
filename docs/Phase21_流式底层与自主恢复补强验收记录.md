# Phase21 流式底层与自主恢复补强验收记录

## 本轮目标

在不改变 AP 二期既定哲学的前提下，继续优先补底层：

1. 把连续媒体输入从“先整段展开再运行”推进到“按 tick 懒加载消费”
2. 把 `webcam / microphone` 从占位接口推进到真实设备流入口
3. 把持续自主 session 从 `start / pause / resume / stop` 推进到“可恢复”
4. 保持：
   - AP 主认知链不被外部教师替代
   - 设备不存在时优雅降级
   - 白箱日志、manifest、live snapshot 一起承接新能力

## 本轮完成

### 1. realtime source 统一升级为 lazy 消费

文件：

- [sensors/stream_adapter_v1.py](<H:\AP原型实验第二期\sensors\stream_adapter_v1.py>)
- [observatory_v2/app.py](<H:\AP原型实验第二期\observatory_v2\app.py>)

新增/升级：

1. `AudioWavRealtimeSourceV1`
2. `OpenCVVideoRealtimeSourceV1`
3. `WebcamRealtimeSourceV1`
4. `MicrophoneRealtimeSourceV1`

关键变化：

- 音频文件不再默认先全部切成列表再交给 run
- 视频文件不再默认先全部抽帧展开到内存
- app 层新增 `start_realtime_source_run(...)`
- `audio / image / video` 已开始统一走按 tick 消费主线

这意味着：

- 长流输入的内存占用更稳定
- 更贴近“固定 tick 预算 + 长期运行”的工程目标
- 未来接更长视频、更长音频、更高频 screen loop 时，不必推翻现有结构

### 2. webcam / microphone 真实入口补齐

文件：

- [sensors/stream_adapter_v1.py](<H:\AP原型实验第二期\sensors\stream_adapter_v1.py>)
- [observatory_v2/app.py](<H:\AP原型实验第二期\observatory_v2\app.py>)
- [observatory_v2/web.py](<H:\AP原型实验第二期\observatory_v2\web.py>)
- [observatory_v2/__main__.py](<H:\AP原型实验第二期\observatory_v2\__main__.py>)

新增入口：

- CLI:
  - `run-webcam-stream`
  - `run-microphone-stream`
- Web API:
  - `POST /api/runs/webcam-stream/start`
  - `POST /api/runs/microphone-stream/start`

实现口径：

- webcam: `opencv VideoCapture`
- microphone: `sounddevice.rec`
- 无设备、无驱动、无依赖时：
  - source 侧可返回 `unavailable`
  - app / web 层会明确报错
  - 不会静默崩溃

### 3. 持续 autonomous session 新增 checkpoint + recover

文件：

- [observatory_v2/app.py](<H:\AP原型实验第二期\observatory_v2\app.py>)
- [observatory_v2/web.py](<H:\AP原型实验第二期\observatory_v2\web.py>)
- [observatory_v2/__main__.py](<H:\AP原型实验第二期\observatory_v2\__main__.py>)

新增能力：

1. 每个自主 tick 后自动写：
   - `live/autonomous_runtime_checkpoint.json`
2. session 状态新增：
   - `recoverable`
3. app 层新增：
   - `recover_autonomous_session(...)`
4. CLI 新增：
   - `recover-autonomous-session`
5. Web API 新增：
   - `POST /api/autonomous-session/recover`

当前恢复语义：

- 从最近落盘的 runtime checkpoint 载入
- 从已有 `tick_done` 继续
- 保持原 session 的 `run_id / session_id`
- 恢复后继续写同一个 run 的 chunks / live / system 轨迹

### 4. 观测与状态承接补强

文件：

- [observatory_v2/app.py](<H:\AP原型实验第二期\observatory_v2\app.py>)
- [README.md](<H:\AP原型实验第二期\README.md>)
- [docs/视频流与自主循环入口说明.md](<H:\AP原型实验第二期\docs\视频流与自主循环入口说明.md>)
- [requirements-optional.txt](<H:\AP原型实验第二期\requirements-optional.txt>)

补强点：

1. `live snapshot` 新增 `active_stream_source`
2. manifest 在流式运行中持续刷新 `source_meta`
3. 文档明确说明：
   - lazy realtime source
   - webcam / microphone 已接入
   - 真实设备流依赖是可选依赖

## 本轮验收

### 定向验收

执行：

```bash
python -m py_compile sensors\stream_adapter_v1.py observatory_v2\app.py observatory_v2\web.py observatory_v2\__main__.py
python -m unittest tests.test_stream_adapter_phase18 tests.test_autonomous_loop_phase19 tests.test_web_api -v
```

结果：

- 16/16 通过

### 全量回归

执行：

```bash
python -m unittest discover -s tests -v
```

结果：

- 72/72 通过

### 真页验收

已实际启动：

```bash
python -m observatory_v2 serve --host 127.0.0.1 --port 8766 --no-browser
python scripts/external_teacher_stub_server.py --host 127.0.0.1 --port 8877
```

并实际打开观测台页面检查：

- 页面可正常加载
- 持续自主 session 面板可见
- `teacher / source / session` 状态区可承接底层数据

## 当前判断

到这一轮为止，下面这些底层点已经明显不再只是“原型口子”：

1. 连续媒体输入的运行模型
2. 持续自主 session 生命周期
3. 外部教师协议
4. 真实设备流入口

但仍未完全到“最终部署版”的点还有：

1. microphone / webcam 依赖真实设备环境，当前测试主要验证：
   - 有设备可走
   - 无设备优雅失败
2. 双耳定位仍未实现
3. 更成熟的长期任务管理：
   - 多阶段目标保持
   - 更丰富的 session planner
   - 更强的 session 恢复策略
4. 前端还没把 webcam / microphone / recover 做成显式按钮

## 结论

本轮更重要的价值，不是“又加了几个按钮”，而是把二期底层真正往长期运行系统推进了一步：

- 流式输入更稳定
- 真实设备入口已接通
- 自主 session 有了可恢复语义
- 观测、文档、测试一起承接住了这些变化

这使得当前项目更接近“专业研究底座”，而不再只是“可演示的多模态样例壳子”。
