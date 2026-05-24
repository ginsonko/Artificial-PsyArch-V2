# Phase23 持续自主 Session 长期语义与真页验收记录

日期：2026-05-19

## 本轮目标

这轮不是再扩新模块，而是确认前一轮补进底层的持续自主 session 语义层，已经真正同时满足：

1. 后端状态文件完整
2. manifest / live snapshot 可承接
3. 前端真实页面可读
4. 全量回归稳定

也就是把“代码里已经有 goal/context/health/recover_hint”推进到“操作者在真实观测台里真的能看见，而且口径一致”。

## 本轮完成

### 1. session 运行态语义已完整进入主链

当前持续自主 session 状态已包含：

- `session_goal`
  - `phase_label`
  - `phase_index`
  - `phase_status`
  - `ticks_completed`
  - `remaining_tick_budget`
  - `completion_ratio`
  - `focus_preview`
  - `selected_action_names`
  - `recover_hint`
- `session_context`
  - `last_tick_id`
  - `last_input_preview`
  - `last_focus_preview`
  - `last_bn_ids`
  - `last_selected_action_names`
  - `last_selected_action_statuses`
  - `last_teacher_mode`
  - `last_external_teacher_mode`
- `session_health`
  - `health_status`
  - `health_reason`
  - `recover_hint`
  - `idle_ticks`
  - `capture_failures`
  - `action_errors`
  - `last_logic_ms`
  - `last_sleep_ms`
  - `last_screen_capture_ok`
  - `last_focus_preview`
  - `last_input_preview`
  - `last_selected_action_names`
  - `last_selected_action_statuses`
  - `last_bn_ids`
  - `last_tick_generated_at_ms`
  - `last_checkpoint_at_ms`
  - `last_checkpoint_tick_done`

这些信息已经不仅存在于：

- `live/autonomous_session_status.json`

也已经进入：

- manifest overlay
- `/api/live`
- `/api/autonomous-session/status`
- 前端 `Live 结构`

### 2. 前端 session 摘要口径已统一

本轮顺手修了一个虽小但不够专业的细节：

- 一处显示 `tick=4`
- 另一处显示 `ticks=4/4`

现在统一为：

- `ticks=X/Y`

并抽成公共前端格式化函数，避免两个渲染路径再次分叉。

### 3. Live 结构细表已确认真实显示 session 细节

`Live 结构 -> Run 状态摘要` 当前真实显示：

- `session_phase`
- `session_health`
- `session_recover_hint`
- `session_focus`
- `session_actions`

这意味着操作者现在不用只看大段 JSON，也能直接在结构化表格里判断：

- 当前处于哪个阶段
- 当前健康状态如何
- 是否可恢复 / 为什么可恢复
- 最近聚焦主线是什么
- 最近动作主线是什么

## 本轮验证

### 1. 自动化回归

执行：

```bash
python -m unittest tests.test_observatory_frontend_phase17 -v
python -m unittest discover -s tests -v
```

结果：

- 前端定向测试通过
- 全量 `103/103` 通过

### 2. 真实页面验收

启动：

```bash
python -m observatory_v2 serve --host 127.0.0.1 --port 8766 --no-browser
```

真实运行：

```bash
python -m observatory_v2 run-autonomous-session --server-url http://127.0.0.1:8766 --max-ticks 4 --wait --text-hint "health browser smoke"
```

实际验收结果：

1. 顶部持续自主 session 摘要真实显示：

```text
session: completed / phase=completed / health=completed / ticks=4/4 / 已完成，无需恢复
```

2. `Live 结构 -> Run 状态摘要` 真实显示：

```text
session_phase         completed
session_health        completed / target_completed
session_recover_hint  已完成，无需恢复
session_focus         health / browser / smoke
session_actions       continue_focus
```

3. `/api/live` 返回中也已包含对应 `autonomous_session` 对象与完整字段。

## 本轮意义

这一轮的价值不在于“又加了一个功能按钮”，而在于把持续自主 session 从：

- 可启动
- 可暂停
- 可恢复

进一步推进到了：

- 可解释
- 可诊断
- 可恢复理由可见
- 长期挂机时更容易排查

这对后面继续做：

- 更成熟的 goal manager
- 更长期的自主 agent 验证
- 更强的执行闭环审计

都很关键。

## 当前边界

到本轮为止，以下内容可以算“已做实”：

1. 持续自主 session 生命周期
2. session checkpoint / recover
3. goal/context/health/recover_hint 白箱状态层
4. Live 结构的真实可观测承接
5. 全量回归稳定

但以下仍属于下一层能力，而不是本轮已彻底完成：

1. 真正联网的外部 LLM provider 生产化接入
2. 更成熟的长期 goal manager / planner
3. 连续视频流 / 麦克风流在真实设备环境下的更重压测
4. 双耳定位与更完整具身智能场景

## 环境经验

1. 如果真页表现和磁盘代码不一致，优先检查本地 `8766` 常驻观测台进程是否已经重启到当前版本。
2. `index.html` 是静态前端资源，不适合用 `python -m py_compile`；这类资源继续以：
   - `tests.test_observatory_frontend_phase17`
   - 真实浏览器验收
   作为主要验证手段。
