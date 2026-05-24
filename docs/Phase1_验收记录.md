# AP 二期 Phase 1 验收记录

日期：2026-05-18

## 本阶段目标

本阶段对应执行手册中的：

1. Phase 0：项目基线冻结
2. Phase 1：实验底座与最小观测底座

当前目标不是实现真实认知闭环，而是先把后续工程不迷路的底盘立住。

## 已落地内容

1. `observatory_v2/` 最小服务包
2. `config/runtime_config.json`
3. `schemas/app_config.schema.json`
4. `schemas/run_manifest.schema.json`
5. `schemas/tick_summary.schema.json`
6. `schemas/tick_metrics.schema.json`
7. `outputs/runs/<run_id>/` 目录协议
8. `manifest.json`
9. `*.summary.jsonl`
10. `*.metrics.jsonl`
11. `live/latest.json`
12. `system/events.jsonl`
13. 最小中文 HTML 观测页
14. 中文 BAT 启动入口
15. 批量演示脚本与重建 live 快照脚本

## 已验证内容

### 自动化测试

执行：

```bash
python -m unittest discover -s tests -v
```

结果：

1. schema 校验测试通过
2. storage / run 写盘测试通过
3. Web API 测试通过
4. live ring 重启恢复测试通过

### 命令行运行验证

执行：

```bash
python -m observatory_v2 run-demo --ticks 4 --interval-ms 10 --label "本地验收最小演示运行"
```

验证结果：

1. 成功生成 `run_id`
2. `manifest.json` 状态为 `completed`
3. `tick_done = tick_planned`
4. `summary.jsonl` 存在
5. `metrics.jsonl` 存在

### HTTP 验证

验证：

1. `GET /api/live`
2. `GET /api/runs/latest`

结果：

1. 可返回最新 run 信息
2. 可返回 live 快照

## 当前限制

当前仍然只是 Phase 1 底座，不代表真实 AP V2 已实现。

尚未接入：

1. 真实文本感受器
2. 真实状态池 V2
3. 真实 `R_state`
4. 真实 HDB-V2 混合召回
5. 真实 `A_focus`
6. 真实 IESM

## 下一步建议

最推荐的下一步是继续按执行手册进入：

1. Phase 2：文本输入最小闭环
2. Phase 3：状态池 V2 最小内核

不要现在跳去做视觉、听觉、Agent 或复杂前端。
