# Phase17E 观测台字段显隐与公式渐进编辑验收记录
日期：2026-05-19

## 1. 本轮目标

在前一轮已经完成：

1. 编辑器聚焦工作流
2. 行级复制
3. 规则效果文本元信息结构化展开

的基础上，继续把观测台编辑器从“结构化但仍偏通铺”推进到更接近研究级操作台的状态。

本轮目标非常明确：

1. 让 `effect type` 真正驱动字段显隐，而不是只给一条提示
2. 让 `formula kind` 真正驱动参数显隐，而不是所有参数同时铺开
3. 在不修改后端 schema、不破坏现有保存格式、不引入新前端框架的前提下，明显降低编辑噪音

## 2. 本轮实际完成

### 2.1 effect type 驱动字段显隐

新增：

1. `effectFieldMode(type)`
2. `applyEffectRowMode(row)`
3. `setCellMode(cell, mode, noteText)`

当前支持的 effect type：

1. `set_emotion_floor`
2. `inject_sa`
3. `add_action_drive`
4. `append_rule_log`

各类型现在不再展示同样的一整套字段，而是按实际语义做收敛：

#### `set_emotion_floor`

主要强调：

1. `channel`
2. `formula`

默认隐藏：

1. `when_channel`
2. `threshold`
3. `action_id`
4. `sa_label`
5. `display_meta`

因为后端真实语义就是：

- 读 `channel`
- 读 `formula`
- 抬高对应情绪通道下限

其它字段不会被该类型实际使用。

#### `inject_sa`

主要强调：

1. `sa_label`
2. `when_channel`
3. `threshold`
4. `display_meta`
5. `formula`

默认隐藏：

1. `channel`
2. `action_id`

并新增解释：

1. `sa_label` 是要注入状态池的 SA 标签
2. `when_channel + threshold` 是情绪门槛组合
3. `display_text` 会进入观测和显示
4. `formula` 输出的是注入能量

#### `add_action_drive`

主要强调：

1. `action_id`
2. `display_meta`
3. `formula`

默认隐藏：

1. `channel`
2. `when_channel`
3. `threshold`
4. `sa_label`

这里特意保留了 `display_meta` 可见，因为：

- `reason` 会被写入动作审计
- 对研究者回看“为什么出现这个驱动力”非常重要

#### `append_rule_log`

主要强调：

1. `display_meta`

默认隐藏：

1. `channel`
2. `when_channel`
3. `threshold`
4. `action_id`
5. `sa_label`

同时：

- `formula` 不再隐藏，而是改为 `muted`
- 并明确提示“该类型不读取公式，保留为空即可”

这样做的原因是：

1. 不让操作者误以为这个字段是必填
2. 但又保留结构连续性，避免界面跳得太碎

### 2.2 formula kind 驱动参数显隐

新增：

1. `formulaKindHint(kind)`
2. `formulaFieldMode(kind)`
3. `applyFormulaPanelMode(panel)`

当前支持的公式类型：

1. `constant`
2. `metric`
3. `mul`
4. `affine`
5. `max_metric`

各类型的字段策略如下：

#### `constant`

主要显示：

1. `value`
2. `min`
3. `max`

隐藏：

1. `metric`
2. `metrics`
3. `base`
4. `factor`

#### `metric`

主要显示：

1. `metric`
2. `min`
3. `max`

隐藏：

1. `metrics`
2. `value`
3. `base`
4. `factor`

#### `mul`

主要显示：

1. `metric`
2. `factor`
3. `min`
4. `max`

隐藏：

1. `metrics`
2. `value`
3. `base`

#### `affine`

主要显示：

1. `metric`
2. `base`
3. `factor`
4. `min`
5. `max`

隐藏：

1. `metrics`
2. `value`

#### `max_metric`

主要显示：

1. `metrics`
2. `min`
3. `max`

隐藏：

1. `metric`
2. `value`
3. `base`
4. `factor`

### 2.3 公式说明栏同步变成动态内容

原先公式下方只有静态摘要。

现在改成：

1. `当前公式摘要：...`
2. `说明：...`

例如：

- `current formula summary: max(state.top_energy)`
- `说明：从多个指标中取最大值，适合“谁更强就跟谁”`

这使得编辑器从“知道字段名”提升到“直接理解当前公式思路”。

### 2.4 原始模板锚点也做了同步补强

虽然本轮依旧主要走“增强层”路线，但为了：

1. 让测试更稳定
2. 让 DOM 语义更明确
3. 降低后续增强时的猜测成本

本轮也把关键锚点回填进了原始模板：

#### effect 行锚点

新增：

1. `data-effect-field="channel"`
2. `data-effect-field="when_channel"`
3. `data-effect-field="threshold"`
4. `data-effect-field="action_id"`
5. `data-effect-field="sa_label"`
6. `data-effect-field="display_meta"`
7. `data-effect-field="formula"`

#### formula 行锚点

新增：

1. `data-formula-key="kind"`
2. `data-formula-key="metric"`
3. `data-formula-key="metrics"`
4. `data-formula-key="value"`
5. `data-formula-key="base"`
6. `data-formula-key="factor"`
7. `data-formula-key="min"`
8. `data-formula-key="max"`

## 3. 为什么这么设计

### 3.1 目标不是“隐藏越多越好”

本轮不是在追求激进删字段，而是在追求：

1. 当前任务相关字段更突出
2. 明显无关字段先退出视野
3. 对少量“保留但不主推”的字段，用 `muted` 而不是生硬移除

所以本轮采用的是：

1. `hidden`
2. `muted`
3. `inline note`

三层语义，而不是单一的“全显 / 全藏”。

### 3.2 目标是减少研究噪音，而不是改 schema

这次没有改：

1. 后端规则 schema
2. 保存格式
3. 校验器语义
4. 规则引擎执行逻辑

这样做的好处是：

1. 风险很低
2. 回归面很小
3. 当前所有 API、保存文件、旧草稿都继续兼容
4. 改善的是研究者体验，而不是引入新的结构债

### 3.3 为什么保留 `append_rule_log.formula` 但弱化

后端对 `append_rule_log` 的真实逻辑是不读取 `formula`。

如果完全删除：

1. 界面会突兀跳动
2. 和其它 effect 类型的布局一致性变差
3. 复制 effect 变体时，操作者更难看出“这个字段是没用，而不是丢了”

所以本轮把它做成：

1. 仍可见
2. 明显弱化
3. 附带“该类型不读取公式”的明确提示

这是更稳妥的研究界面策略。

## 4. 自动化验证

### 4.1 新增前端专项断言

更新：

- `tests/test_observatory_frontend_editor_focus.py`

新增覆盖：

1. `effectFieldMode`
2. `formulaFieldMode`
3. `formulaKindHint`
4. `applyEffectRowMode`
5. `applyFormulaPanelMode`
6. `data-effect-field=*`
7. `data-formula-key=*`
8. `mode-hidden / mode-muted / field-inline-note`

并额外检查：

1. 四种 effect type 的语义分支都在
2. 五种 formula kind 的语义分支都在
3. 公式提示文本会拼接摘要和说明

### 4.2 自动化回归结果

执行：

```bash
python -m unittest tests.test_observatory_frontend_editor_focus -v
python -m unittest discover -s tests -v
```

结果：

1. 前端专项 `2` 项通过
2. 全量 `34` 项测试通过

## 5. HTTP 验证

验证地址：

- `http://127.0.0.1:8766/?ts=phase17e`

源码命中确认：

1. `effectFieldMode`
2. `formulaFieldMode`
3. `applyEffectRowMode`
4. `applyFormulaPanelMode`
5. `data-effect-field="channel"`
6. `data-formula-key="kind"`
7. `mode-hidden`
8. `field-inline-note`

说明：

这一步确认的不是磁盘代码，而是本地服务真正吐出的页面源码，避免“文件改了但服务还是旧版”的误判。

## 6. 真页验收

验收地址：

- `http://127.0.0.1:8766/?ts=phase17e-browser`

### 6.1 页面基础状态

已确认：

1. 页面标题正常：`AP 二期观测台 V2`
2. 规则卡 `6` 张
3. 调参卡 `2` 张
4. 规则效果结构化文本输入框仍在
5. 内联说明真实出现在页面中

### 6.2 effect type 驱动显隐

真实切换后确认：

#### `set_emotion_floor`

状态：

1. `channel` 可见
2. `formula` 可见
3. `when_channel / threshold / action_id / sa_label / display_meta` 隐藏
4. 提示文本正确切换为“抬高情绪通道下限...”

#### `append_rule_log`

状态：

1. `display_meta` 可见
2. `formula` 变为 `muted`
3. `channel / action_id` 等无关字段隐藏
4. 提示文本正确切换为“只记录规则命中日志...”

#### `inject_sa`

状态：

1. `sa_label / when_channel / threshold / display_meta` 可见
2. `channel / action_id` 隐藏
3. 字段说明与感受显化语义一致

### 6.3 formula kind 渐进显示

真实切换后确认：

#### `constant`

状态：

1. 只强调 `value`
2. `metric / metrics / base / factor` 隐藏

#### `affine`

状态：

1. `metric / base / factor` 可见
2. `value / metrics` 隐藏

#### `max_metric`

状态：

1. `metrics` 可见
2. `metric / value / base / factor` 隐藏
3. 公式提示会同步更新为“从多个指标中取最大值...”

## 7. 本轮意义

本轮的真正价值是：

把规则编辑器继续从：

`结构化配置页`

推进到更接近：

`研究者长期停留、低噪音、高可解释的规则工作台`

更具体一点说，本轮解决的是下面这种真实痛点：

1. 打开一条规则效果，总看到很多当前类型根本无关的字段
2. 明明只是改一个公式，却要在一堆不相关的参数里找真正有意义的项
3. 新人或隔天回看时，只看到字段名，不知道这组参数到底为什么这样配

现在这些痛点都明显下降了。

## 8. 后续最值得继续补强的点

### 8.1 把类型驱动显隐进一步扩展到调参编辑器

当前调参器已经有 target hint，但还没有像 effect / formula 这样更强的字段导向。

可继续考虑：

1. 对不同 target 家族给出更细的操作建议
2. 对危险 target 增加醒目提示

### 8.2 让审计 diff 直接驱动卡片聚焦与折叠

现在已经能定位对应卡片。

下一步可做：

1. 定位后自动聚焦
2. 自动折叠其它卡
3. 把“看到差异 -> 改卡片 -> 试算”链条再压短一步

### 8.3 进一步减少 JSON 心智负担

仍可继续推进：

1. effect type 分布图
2. formula kind 分布图
3. rule family / tuner target 结构占比
4. warning code 热点统计

这样研究者在大规则集下能更快做全局判断，而不是只靠逐卡阅读。
