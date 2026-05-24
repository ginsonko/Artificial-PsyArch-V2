# AP 二期 Phase 2 文本最小闭环验收记录

日期：2026-05-18

## 本阶段目标

对应执行手册中的：

1. 阶段 2：文本输入最小闭环

当前目标不是实现完整 HDB-V2，而是先把：

`文本输入 -> 文本感受器 -> 状态池最小核 -> 最小 R_state -> A_focus 摘要 -> 观测与日志`

这条真链打通。

## 本阶段已落地内容

1. [text_sensor_v2.py](<H:\AP原型实验第二期\sensors\text_sensor_v2.py>)
2. [state_pool_v2.py](<H:\AP原型实验第二期\core\state_pool_v2.py>)
3. `text_input_envelope.schema.json`
4. `text_sensor_packet.schema.json`
5. `POST /api/runs/text/start`
6. `GET /api/runs/<run_id>/ticks/<tick_index>/sidecar`
7. 文本 run 写入：
   - `inputs/inputs.jsonl`
   - `chunks/*.sensor.jsonl`
   - `chunks/*.sidecar.jsonl`

## 已验证内容

### 1. 自动化测试

执行：

```bash
python -m unittest discover -s tests -v
```

结果：

1. `11` 项测试全部通过
2. 新增文本 Phase 2 测试通过

### 2. 文本闭环命令行验收

执行：

```bash
python -m observatory_v2 run-text --text "今天 天气 不错" --text "今天 天气 不错" --text "我 想 出门" --label "Phase2 本地验收文本闭环"
```

结果：

1. run 成功完成
2. `run_kind = phase2_text_min_loop`
3. `tick_done = tick_planned = 3`
4. `summary.jsonl`、`metrics.jsonl`、`sensor.jsonl`、`sidecar.jsonl` 全部生成

### 3. 白箱结果核查

从 `summary.jsonl` 与 `sidecar.jsonl` 可看到：

1. 文本被切成 SA 单元
2. 文本预算稳定
3. 状态池在连续 tick 中累积与衰减
4. `R_state` 以多头摘要形式输出
5. `A_focus` 已能从更新后的状态池读出

## 当前阶段结论

Phase 2 已经成立，且满足执行手册中最重要的几条要求：

1. 文本预算稳定
2. 不同输入形成不同刺激包
3. 观测台可看到感受器摘要
4. 状态池与 `R_state` 真正进入运行链，而不是假字段

## 当前仍存在的限制

1. 第二次重复输入的疲劳门槛当前设定较保守，示例 run 中第 2 tick 尚未明显压低能量
2. `A_focus` 目前更接近“状态池峰值摘要”，还没有后继优势和语言链连续性
3. 还没有接入 Bn / `C_i` / `C*`
4. 还没有真实记忆召回

## 下一步建议

最推荐的下一步是：

1. 微调 Phase 2/3 交界处的状态池读出偏置
2. 进入 Phase 3：状态池 V2 最小内核正式强化
3. 之后再进入 Phase 5：HDB-V2 最小主链
