# AP 二期 Phase 0 / Phase 1 最小底座操作说明

本文档对应当前已经落地的最小工程底座。

当前范围：

1. 配置加载
2. 轻量 schema 校验
3. run manifest
4. JSONL 追加写日志
5. 最小 live API
6. 最小 tick 回放 API
7. 中文 Web 观测页
8. BAT 启动入口
9. 文本感受器最小实现
10. 状态池最小内核
11. 最小 `R_state` 多头读出

当前明确还没有做：

1. 真实文本闭环
2. 真实 HDB-V2 召回主链
3. 真实 `R_state -> Bn -> C_i -> C*`
4. 真实 IESM 规则
5. 视觉 / 听觉
6. Agent sandbox

## 一、如何启动观测台

直接双击：

- [启动观测台.bat](<H:\AP原型实验第二期\启动观测台.bat>)

启动后会自动运行：

```bash
python -m observatory_v2 serve
```

默认地址：

- `http://127.0.0.1:8766`

## 二、如何执行一次文本最小闭环运行

直接双击：

- [启动单次实验.bat](<H:\AP原型实验第二期\启动单次实验.bat>)

它会执行：

```bash
python -m observatory_v2 run-text --text "今天 天气 不错" --text "今天 天气 不错" --text "我 想 出门"
```

如果只想执行 Phase 1 演示运行，也可以手动执行：

```bash
python -m observatory_v2 run-demo --ticks 12 --interval-ms 120
```

## 三、输出目录

当前所有运行产物写入：

- `outputs/runs/<run_id>/`

典型结构：

```text
outputs/
  runs/
    phase1_demo_.../
      manifest.json
      live/
        latest.json
      chunks/
        ticks_000000_000999.summary.jsonl
        ticks_000000_000999.metrics.jsonl
      system/
        events.jsonl
```

## 四、API 一览

### 1. 健康检查

- `GET /api/health`

### 2. 配置查看

- `GET /api/config`

### 3. Live 预览

- `GET /api/live`

### 4. Runs 列表

- `GET /api/runs`

### 5. 最新 Run

- `GET /api/runs/latest`

### 6. 单个 Manifest

- `GET /api/runs/<run_id>/manifest`

### 7. 单 tick 回放

- `GET /api/runs/<run_id>/ticks/<tick_index>`

### 8. 启动最小演示运行

- `POST /api/runs/demo/start`

请求体示例：

```json
{
  "tick_count": 12,
  "tick_interval_ms": 120,
  "label": "页面触发的最小演示运行"
}
```

### 9. 启动文本最小闭环运行

- `POST /api/runs/text/start`

请求体示例：

```json
{
  "label": "页面触发的 Phase2 文本最小闭环",
  "tick_interval_ms": 0,
  "texts": [
    "今天 天气 不错",
    "今天 天气 不错",
    "我 想 出门"
  ]
}
```

### 10. 读取单 tick sidecar

- `GET /api/runs/<run_id>/ticks/<tick_index>/sidecar`

## 五、如何运行测试

在根目录执行：

```bash
python -m unittest discover -s tests -v
```

## 六、这一步的验收标准

当前阶段最小验收标准是：

1. 观测台可以启动
2. 页面可以打开
3. 页面可以启动一次文本最小闭环 run
4. `outputs/runs/<run_id>/manifest.json` 能生成
5. `summary.jsonl` / `metrics.jsonl` / `sensor.jsonl` / `sidecar.jsonl` 能生成
6. `GET /api/live` 有值
7. `GET /api/runs/<run_id>/ticks/<tick>` 能回放
8. `GET /api/runs/<run_id>/ticks/<tick>/sidecar` 能返回文本感受器和 `R_state` 摘要

只要这些成立，就说明二期已经把“后续不迷路的实验底座”立住了。
