# Phase17D 观测台编辑器聚焦与结构化补强验收记录
日期：2026-05-19

## 1. 本轮目标

在不改变 AP 二期核心哲学、不引入额外前端框架、也不把编辑器重新做成“另一个复杂系统”的前提下，继续补强观测台的规则编辑器与调参编辑器，使它们更适合：

1. 长列表下围绕单张卡持续打磨
2. 快速复制小变体做实验
3. 直接阅读和编辑规则效果文本元信息
4. 减少仍然残留的“像在改 JSON”感

## 2. 本轮实际完成

### 2.1 卡片聚焦工作流

新增控制：

1. 规则区：
   - `仅看当前规则`
   - `取消规则聚焦`
   - 卡片级 `聚焦此卡 / 取消聚焦`
2. 调参区：
   - `仅看当前档位`
   - `取消档位聚焦`
   - 卡片级 `聚焦此卡 / 取消聚焦`

落地方式：

- 新增 `state.pinnedRuleId`
- 新增 `state.pinnedTunerId`
- 在 `applyRulesFilters()` / `applyTunerFilters()` 中把“筛选命中”再叠加一层“是否正被聚焦”
- 聚焦卡片时同时打 `pinned` 样式，并沿用现有 `pendingScrollRuleId / pendingScrollTunerId` 滚动定位链路

目标效果：

1. 当规则很多时，可以先用搜索 / family / target / audit 过滤收敛，再一键只看当前卡
2. 在试算、复制、增删条件时，不会被长列表干扰
3. 聚焦状态是显式可见、可取消、可回退的

### 2.2 行级复制能力

本轮补齐：

1. `duplicate-condition`
2. `duplicate-effect`
3. `duplicate-adjustment`

覆盖范围：

- 规则条件
- 规则效果
- 调参条件
- 调参项

实现方式：

- 基于当前卡实时同步后的 `state.rulesPayload / state.tunerPayload`
- 用深拷贝在原行后插入一份副本
- 立即重渲染编辑器

目标效果：

1. 快速构建相近阈值 / 相近公式 / 相近 target 的实验变体
2. 减少手工重复录入
3. 保持现有结构化 schema，不引入新的草稿格式

### 2.3 规则效果文本元信息结构化展开

原先残留问题：

- `display_text / reason / message` 仍绑定在一个紧凑 JSON textarea 中
- 虽然机器可解析，但人工编辑体验差
- 很像在“继续硬改 JSON”

本轮处理：

1. 保持底层 schema 不变
2. 通过编辑器增强层，把该 textarea 结构化展开为三个独立输入：
   - `effect-display-text`
   - `effect-reason`
   - `effect-message`
3. 收集时兼容两种来源：
   - 若仍存在原 textarea，则继续可解析
   - 若已被增强层替换，则改读三个结构化字段

目标效果：

1. 人工可以直接看懂和修改文本元信息
2. 兼容原有保存格式
3. 不改变后端接口和规则 schema

### 2.4 类型提示增强

新增提示：

1. `effectTypeHint(type)`
2. `tunerTargetHint(target)`

作用：

- 在编辑器中直接提示 effect type / tuner target 的用途
- 降低操作者对外部文档的依赖

## 3. 关键实现点

本轮关键代码位置：

- `observatory_v2/web_static/index.html`

新增或调整的关键逻辑：

1. 编辑器状态：
   - `pinnedRuleId`
   - `pinnedTunerId`
2. 筛选聚焦：
   - `pinSingleVisibleRuleCard()`
   - `clearPinnedRuleCard()`
   - `pinSingleVisibleTunerCard()`
   - `clearPinnedTunerCard()`
3. DOM 增强层：
   - `ensureActionStack()`
   - `insertStackButton()`
   - `enhanceRuleEffectRow()`
   - `enhanceConditionRow()`
   - `enhanceAdjustmentRow()`
   - `attachCardPinButtons()`
   - `enhanceRulesEditorDom()`
   - `enhanceTunerEditorDom()`
4. 结构化提示：
   - `effectTypeHint()`
   - `tunerTargetHint()`

## 4. 自动化验证

### 4.1 新增专项测试

新增文件：

- `tests/test_observatory_frontend_editor_focus.py`

覆盖点：

1. 聚焦按钮锚点存在
2. 聚焦函数存在
3. 行级复制锚点存在
4. 结构化效果文本字段锚点存在
5. pin 状态字段存在

### 4.2 自动化回归

执行：

```bash
python -m unittest tests.test_observatory_frontend_editor_focus -v
python -m unittest discover -s tests -v
```

结果：

1. 新增前端专项 1 项通过
2. 全量 33 项通过

## 5. HTTP 验证

验证地址：

- `http://127.0.0.1:8766/?ts=editorfocus`

源码确认命中：

1. `focusVisibleSingleRuleBtn`
2. `clearRulePinBtn`
3. `focusVisibleSingleTunerBtn`
4. `clearTunerPinBtn`
5. `duplicate-condition`
6. `duplicate-effect`
7. `duplicate-adjustment`
8. `pin-rule-card`
9. `pin-tuner-card`
10. `effect-display-text`
11. `effect-reason`
12. `effect-message`

## 6. 真页验收

使用新开页 + query 参数方式对真实页面验收，确认：

1. 页面标题正常：
   - `AP 二期观测台 V2`
2. 四个聚焦按钮真实存在：
   - `仅看当前规则`
   - `取消规则聚焦`
   - `仅看当前档位`
   - `取消档位聚焦`
3. 增强后卡片内真实出现：
   - 规则卡聚焦按钮
   - 调参卡聚焦按钮
   - 条件复制按钮
   - 效果复制按钮
   - 调参项复制按钮
   - 结构化效果文本输入框

真页计数结果：

1. `ruleCards = 6`
2. `tunerCards = 2`
3. `pinRuleButtons = 6`
4. `pinTunerButtons = 2`
5. `duplicateConditionButtons = 10`
6. `duplicateEffectButtons = 12`
7. `duplicateAdjustmentButtons = 6`
8. `effectDisplayInputs = 12`
9. `effectReasonInputs = 12`
10. `effectMessageInputs = 12`

## 7. 本轮意义

本轮的意义不是简单“补几个按钮”，而是把观测台编辑器继续从：

`结构化配置页`

推进到更接近：

`研究者长时间停留、聚焦、试错、复制变体的操作台`

具体价值：

1. 长列表编辑压力明显下降
2. 规则与调参小变体实验更顺手
3. 文本元信息更适合直接阅读与维护
4. 在不改后端 schema 的前提下，继续减少对紧凑 JSON 的依赖

## 8. 后续最值得继续补强的点

1. 让 `effect type` 进一步驱动字段显隐：
   - 例如 `append_rule_log` 时弱化 `channel / threshold`
   - 例如 `inject_sa` 时突出 `sa_label`
2. 给聚焦工作流增加“只保留此卡在顶部”的模式
3. 为结构化效果文本字段增加更明确的中文说明
4. 若后续还要继续降 JSON 感，可考虑把公式编辑也做成按 kind 逐步显隐，而不是通铺字段
