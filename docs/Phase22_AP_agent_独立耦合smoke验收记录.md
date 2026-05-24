# AP 二期 Phase22 AP-agent 独立耦合 smoke 验收记录

日期：2026-05-19

## 目标

这次不是做一个“让外部 agent 接管 AP”的 demo，而是验证下面这条严格边界已经在当前工程里成立：

1. AP 自己完成感知、状态池演化、记忆召回、预测、规则命中与动作驱动力产生
2. teacher 层只做审阅、轻微改权、阻断、奖惩与审计
3. executor / sandbox 只做执行桥，不替代 AP 的动作生成
4. 奖惩与执行结果会再回到 AP 内部，塑形后续 action bias / tuner bias

## 本轮 smoke 内容

使用文本链做一个最小但可重复的耦合验证：

- tick0: `今天 天气 不错`
- tick1: `今天 天气 不错`
- tick2: `我 想 出门`

配置口径：

- `executor_enabled = true`
- `executor_dry_run = true`
- `autonomous_external_teacher_enabled = false`

这意味着：

- action 真的会走 executor 链
- 但不会真实操作电脑
- external teacher 不参与主导，只保留本地 teacher 审阅层

## 验收要点

### 1. AP 先产生动作驱动力

在 tick1，`rules_result.action_drives` 已出现：

- `action::continue_focus`

说明动作候选首先由 AP 自己的：

- `Bn`
- `C_i`
- `C*`
- `rules`

这条主链得出，而不是 teacher 或 executor 凭空生成。

### 2. teacher 只做审阅与轻度改权

在同一个 tick 中，`teacher_review.scored_action_drives` 继续包含：

- `action::continue_focus`

但它只是对 drive 做了轻微调制，并补了：

- `teacher_reward`
- `teacher_penalty`
- `teacher_notes`
- `llm_gate`

这说明 teacher 层是“基于 AP 候选做处理”，不是替代 AP 出候选。

### 3. executor 只消费已审阅动作

在 `sandbox_result.selected_actions` 中，仍然是：

- `action::continue_focus`

并且当前是 `dry_run` 执行。

这说明 executor 只是：

- 接收已审阅候选
- 选择并执行
- 回传执行结果

没有把自己的行动逻辑倒灌成主认知链。

### 4. 奖惩回流后，会塑形下一 tick 的动作 bias

在 tick1 已能看到 teacher feedback：

- `reward > 0`

并且在 tick2，对同一个动作：

- `learned_bias > 0`

说明上一 tick 的反馈已经进入 AP 内部学习层，开始影响下一轮 drive。

这正是我们要的哲学：

- 外层可以教
- 但学到的东西回流到 AP 自己
- 不是永远靠外层代算

## 新增自动化测试

文件：

- [tests/test_ap_agent_coupling_phase22.py](<H:\AP原型实验第二期\tests\test_ap_agent_coupling_phase22.py>)

测试名：

- `test_ap_remains_primary_and_teacher_executor_only_gate_and_feedback`

它验证了：

1. rules 层先产出 `action::continue_focus`
2. teacher review 继续处理这个 action，而不是另起 action
3. sandbox 执行的是同一个 action
4. teacher feedback 为正
5. 下一 tick 的 `learned_bias` 已大于 0

## 结论

当前仓已经具备一个可以独立成立的 AP-agent 耦合底层：

- AP 是主体
- teacher 是审阅/奖惩层
- executor 是执行桥
- 学习闭环回到 AP 内部

这说明“把 AP 做成心智核心，外部 agent/LLM 只做辅助层”的方向，当前工程底座已经不是概念口径，而是有可跑 smoke、可回归测试、可白箱审计的实际链路。
