# Phase20Plus 持续自主 Session 竞态修复验收记录（2026-05-19）

## 本轮目标

把持续自主 session 里最危险、最容易导致“看起来能跑，但长期一压测就卡死/误判”的三个底层问题彻底修掉：

1. `get_manifest()` 误把当前进程仍在运行中的 live session 读成 `interrupted`
2. `resume_autonomous_session()` 在 pause/resume 交错时错误 early-return，导致 `pause_event` 没被清掉
3. `while pause_event.is_set()` 内部重复累计 `paused` transition，导致 `paused_count` 虚高

## 根因结论

### 1. live session 被读路径误判为 interrupted

之前 `observatory_v2/app.py::get_manifest()` 读取到 `live/autonomous_session_status.json` 后，
无条件走 `_normalize_bootstrap_autonomous_session_status(...)`。

这本来是给“进程重启后，磁盘残留 active session 状态文件”做 bootstrap 修复的。
但如果当前 app 自己其实仍持有 live thread，这个 normalize 就会把真实的 `running/paused`
误改成 `interrupted`。

### 2. resume 竞态导致 pause_event 残留

之前如果：

- 已经发了 `pause_request`
- 主循环还没来得及正式进入 `paused`
- 这时立刻又发 `resume`

可能看到的表面状态是：

- `status == running`
- `paused == false`

于是 `resume_autonomous_session()` 直接 early-return，误以为“已经恢复了”。
但实际上 `pause_event` 还没清掉，后面主循环还是会掉进 pause loop，造成 session 长期卡住。

### 3. paused_count 被长暂停放大

之前 `while pause_event.is_set()` 每 50ms 都会再次：

- 设 `status = paused`
- `_mark_autonomous_session_transition(..., "paused")`

所以长暂停一会儿，`paused_count` 就会疯狂增长。
这不是“进入 paused 的次数”，而是“停在 paused 里的循环次数”。

## 已完成修复

### A. manifest overlay 改成区分 live / stale 两条路径

文件：

- `observatory_v2/app.py`

新增：

- `_read_session_status_for_manifest_overlay(...)`

策略：

- 如果 `run_id == self._active_run_id`
- 且当前 `self._active_thread.is_alive()`
- 且 `self._autonomous_session_status["run_id"] == run_id`

则直接用内存中的 live session 状态做 overlay，
只补默认字段，不再走 bootstrap interrupted normalization。

只有非当前 live session 才允许走 `_normalize_bootstrap_autonomous_session_status(...)`。

### B. resume 必须真正清掉 pause_event

文件：

- `observatory_v2/app.py`

修复点：

- 只有在：
  - `status == running`
  - `paused == false`
  - `pause_event.is_set() == false`

时，才允许视为“已经恢复，无需动作”。

否则即使外表仍是 `running`，只要 `pause_event` 还在，就必须：

- `pause_event.clear()`
- 写 `status = running`
- 记一次 `resumed` transition
- 落盘

### C. paused transition 只在进入 paused 那一刻记录一次

文件：

- `observatory_v2/app.py`

修复点：

在 pause loop 内先判断是否已经：

- `status == paused`
- `paused == true`

如果已经是 paused，就不再重复：

- `_mark_autonomous_session_transition(..., "paused")`
- `_persist_autonomous_session_status()`

这样 `paused_count` 语义恢复为“进入 paused 的次数”。

## 新增回归测试

### 1. resume 竞态回归

文件：

- `tests/test_autonomous_loop_phase19.py`

新增测试：

- `test_autonomous_session_resume_clears_pause_request_even_if_status_still_running`

验证：

- 即使 `status` 仍是 `running`
- 只要 `pause_event` 还 set
- `resume_autonomous_session()` 也必须清掉 pause_event 并累计 `resume_count`

### 2. live manifest 不误报 interrupted

文件：

- `tests/test_storage_and_runs.py`

新增测试：

- `test_get_manifest_keeps_live_autonomous_session_status_without_bootstrap_interrupting_it`

验证：

- 当前 app 仍持有 live thread 时
- `get_manifest(run_id)` 返回的状态仍应是 `paused/running`
- 不能被 bootstrap normalize 成 `interrupted`

## 验收结果

### 定向测试

通过：

- `tests.test_autonomous_loop_phase19.AutonomousLoopPhase19Tests.test_autonomous_session_resume_clears_pause_request_even_if_status_still_running`
- `tests.test_storage_and_runs.StorageAndRunTests.test_get_manifest_keeps_live_autonomous_session_status_without_bootstrap_interrupting_it`
- `tests.test_batch_runner_v2.BatchRunnerV2Tests.test_dataset_runner_supports_autonomous_session_pipeline_and_hooks`

### 高频复现压测

对最容易抖动的用例：

- `test_dataset_runner_supports_autonomous_session_pipeline_and_hooks`

进行了 12 次连续循环复现，全部通过。

### 相关回归

通过：

- `tests.test_autonomous_loop_phase19`
- `tests.test_storage_and_runs`
- `tests.test_batch_runner_v2`
- `tests.test_external_teacher_autonomous_e2e`
- `tests.test_ap_agent_coupling_phase22`
- `tests.test_web_api`
- `tests.test_runtime_export_and_forget_phase8_14`

### 全量回归

通过：

- `py -3 -m unittest discover -s tests -v`

结果：

- `101 / 101 OK`

## 当前底层状态更新

到这一轮为止，之前那几个最关键的底层风险点里：

1. 持续自主 session 的 pause/resume/get_manifest 时序竞态  
   - 已修复并验收通过

2. external teacher provider protocol  
   - 当前已不是只有本地 `stub_file`
   - 已支持 schema 化 `http_json` 协议与对应回归

3. realtime source abstraction  
   - 当前已经统一收敛到 `BaseRealtimeSourceV1 / StreamAdapterV1`
   - 覆盖 image / audio / video / screen / webcam / microphone

4. 向量 / ANN / 时空层  
   - 当前已经是 layered bundle + vector ANN + spacetime index 的工程形态
   - 不是“还没接上”的状态

## 仍可继续补强但不再属于断链问题

以下更像“继续追求更强工程规格”的方向，而不是当前主链还有断点：

1. 把部分视频解码链进一步从 `OpenCV + tempfile` 路线继续抽象清洁
2. 把观测台长期统计专题继续做深
3. 把持续自主 session 再往更长期任务管理、阶段恢复、目标保持推进

## 本轮结论

本轮不是表面补丁，而是把持续自主 session 最核心的状态机竞态真正修实了。

当前这套底层已经不再是“看起来能跑、长一点就抖”的状态，
而是已经通过定向竞态回归、高频复现压测、相关链路回归和全量测试验证的状态。
