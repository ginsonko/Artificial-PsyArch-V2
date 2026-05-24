# Phase24 ExternalTeacher Provider 正式化验收记录
日期：2026-05-19

## 本轮目标

把 `external teacher` 从“能通的原型挂件”推进到更正式、可审计、可配置、可回归的底层模块，重点解决：

1. `external teacher` 的 fail-open / fail-closed 语义不再借用 `llm_gate_fail_open`
2. `http_json` provider 具备正式的 transport 审计字段
3. `http_json` provider 具备有限重试和回退时间配置
4. CLI / Web API / runtime / observatory 摘要口径统一
5. 保持 AP 为主链主体，teacher 仍只做审阅、门控、奖励/惩罚、审计

## 已完成改动

### 1. 配置与 schema

已新增配置项：

- `autonomous_external_teacher_fail_open`
- `autonomous_external_teacher_max_retries`
- `autonomous_external_teacher_retry_backoff_ms`

已同步更新：

- `config/runtime_config.json`
- `observatory_v2/config.py`
- `schemas/app_config.schema.json`

### 2. ExternalTeacherGatewayV1 正式化

已在 [core/external_teacher_gateway_v1.py](H:\AP原型实验第二期\core\external_teacher_gateway_v1.py) 中完成：

- `stub_file` 与 `http_json` 的统一 transport 审计输出
- `http_json` provider 的有限重试
- `duration_ms / status_code / attempt_count / transport_result / transport_error_kind`
- 本地 transport 审计优先，不再依赖远端 provider 自报字段
- `off` / `unsupported mode` 也返回结构化 transport 审计

当前标准审计字段为：

- `provider`
- `attempt_count`
- `duration_ms`
- `status_code`
- `success`
- `transport_result`
- `transport_error_kind`

### 3. TeacherLayerV1 语义拆分

已在 [core/teacher_layer_v1.py](H:\AP原型实验第二期\core\teacher_layer_v1.py) 中完成：

- `external_teacher_fail_open` 独立于 `llm_gate_fail_open`
- provider 不可用时：
  - `fail_open = true`：仅保留 warning / 审计，不强制阻断
  - `fail_open = false`：仅对 risky action 触发 `external_teacher_unavailable_fail_closed`
- `external_teacher_review` 现包含：
  - `provider`
  - `reviewer`
  - `fail_open`
  - `fail_closed`
  - `transport_audit`

### 4. runtime / CLI / Web API / observatory 对齐

已在以下入口完成参数透传：

- [core/runtime_v2.py](H:\AP原型实验第二期\core\runtime_v2.py)
- [observatory_v2/app.py](H:\AP原型实验第二期\observatory_v2\app.py)
- [observatory_v2/web.py](H:\AP原型实验第二期\observatory_v2\web.py)
- [observatory_v2/__main__.py](H:\AP原型实验第二期\observatory_v2\__main__.py)

新增支持：

- `external_teacher_fail_open`
- `external_teacher_max_retries`
- `external_teacher_retry_backoff_ms`

观测台 runtime teacher 摘要也已更新为新口径：

- external mode
- external fail_open
- external retry

位置：

- [observatory_v2/web_static/index.html](H:\AP原型实验第二期\observatory_v2\web_static\index.html)

## 测试与回归

### 定向测试

已通过：

```powershell
py -3 -m unittest tests.test_teacher_layer_v1 -v
py -3 -m unittest tests.test_schema_tools -v
py -3 -m unittest tests.test_observatory_frontend_phase17 -v
py -3 -m unittest tests.test_external_teacher_autonomous_e2e -v
py -3 -m unittest tests.test_autonomous_loop_phase19 tests.test_external_teacher_autonomous_e2e tests.test_teacher_layer_v1 -v
```

新增覆盖点包括：

- `http_json` warn 流程
- `http_json` 首次失败、重试一次后成功
- external teacher 独立 fail-closed
- autonomous tick meta 中的新配置透传

### 全量回归

已通过：

```powershell
py -3 -m unittest discover -s tests -v
```

结果：

- `104 / 104 OK`

## 真实验收结论

### 1. 持续自主 session 真跑仍正常

已验证：

```powershell
python -m observatory_v2 run-autonomous-session --server-url http://127.0.0.1:8766 --max-ticks 2 --wait --text-hint "cli session wait"
```

结果：

- session 正常 completed
- `autonomous_tick_meta` 中已包含：
  - `external_teacher_fail_open`
  - `external_teacher_max_retries`
  - `external_teacher_retry_backoff_ms`

### 2. external teacher e2e 正常

已验证：

```powershell
py -3 -m unittest tests.test_external_teacher_autonomous_e2e -v
```

结果：

- 自主 run 中 teacher review 正常进入 sidecar
- teacher feedback 正常进入 AP 奖惩闭环

## 当前结论

这轮之后，`external teacher` 已经不再是“借用 llm gate 容错语义的临时层”，而是：

1. 有独立策略口径
2. 有正式 transport 审计
3. 有有限重试
4. 有完整入口透传
5. 有回归覆盖

也就是说，这一块底层现在已经达到了“项目级研究底座”的完成度，而不是单纯能演示的原型状态。

## 后续最值得继续补的底层点

在当前 external teacher provider 正式化完成后，剩余更高价值的底层增强点主要还有：

1. 真实外部 LLM provider 对接
   - 当前 `http_json` 协议已就位
   - 下一步更像是 provider 接入问题，而不是底层协议问题

2. 连续视频 / 麦克风 / 双耳定位继续向设备级长期流推进
   - 当前已具备统一 source abstraction
   - 但更深的设备级连续能力仍可继续增强

3. 长期自主任务管理
   - 现在 session 生命周期已经稳定
   - 下一步可继续增强更长期的目标保持、多阶段任务恢复与策略层
