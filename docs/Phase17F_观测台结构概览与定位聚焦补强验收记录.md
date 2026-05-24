# Phase17F 观测台结构概览与定位聚焦补强验收记录
日期：2026-05-19

## 1. 本轮目标

在前几轮已经完成：

1. 编辑器聚焦工作流
2. 行级复制
3. effect type / formula kind 渐进显隐

之后，本轮继续沿着“研究操作台化”的方向补两块高价值能力：

1. 让参考面板不再只是几行静态 chip，而是能直接看出规则集 / 调参器的整体结构
2. 让从提醒、对照、搜索进入编辑器时，更自然地收敛到“只看这一张卡”

## 2. 本轮实际完成

### 2.1 规则参考面板升级为结构概览面板

新增：

1. `topEntriesFromHist(hist, limit)`
2. `renderReferenceBarChart(rows, options)`

并扩展了 `summarizeRuleDraft(payload)` 的统计输出，新增：

1. `familyHist`
2. `effectTypeHist`
3. `formulaKindHist`
4. `channelHist`

现在 [observatory_v2/web_static/index.html](H:\AP原型实验第二期\observatory_v2\web_static\index.html) 中的规则参考面板除了原有：

1. 规则数
2. 条件数
3. 效果数

之外，还会新增展示：

1. 当前规则集结构概览
   - 主 family
   - 主 effect
2. family 分布
3. effect type 分布
4. formula kind 分布
5. 情绪 / 门槛通道使用分布

这意味着研究者一打开规则编辑区，不必先下钻到单条规则，就能大致知道：

- 当前这套先天规则主要偏哪一类
- 主要在做情绪调制、注入 SA，还是动作驱动
- 公式主要是常量、线性、仿射还是 max 聚合

### 2.2 调参参考面板升级为结构概览面板

扩展了 `summarizeTunerDraft(payload)`，新增：

1. `targetHist`
2. `metricHist`

现在调参参考面板会额外显示：

1. 当前调参器结构概览
   - 主 target
   - 空调参档数量
2. target 分布
3. 条件 metric 分布

这样做以后，研究者可以直接判断：

1. 当前调参器主要在调注意力、采样预算、后继优势，还是锚点偏置
2. 调参规则的触发条件主要盯着逻辑耗时、状态池、情绪还是召回量

### 2.3 “定位卡片”升级为“定位并聚焦”

原先 `jumpToEditorCard(kind, targetId)` 主要做：

1. 搜索框写入
2. 过滤条件收敛
3. 滚动到对应卡片

本轮进一步补强为：

1. 规则跳转时自动设置 `state.pinnedRuleId = cleanId`
2. 调参跳转时自动设置 `state.pinnedTunerId = cleanId`
3. 状态文案同步改为：
   - `已定位并聚焦规则卡`
   - `已定位并聚焦调参档`

这带来的直接效果是：

从提醒区、差异区或搜索路径进入编辑器时，不再只是“找到它”，而是直接进入：

`只看这一张卡`

更适合长时间围绕单卡做：

1. 对照
2. 修改
3. 试算
4. 回看

### 2.4 轻量图形样式补齐

新增样式：

1. `.editor-reference-chart`
2. `.editor-reference-bar`
3. `.editor-reference-caption`

这些不是装饰性样式，而是为了在不引入额外图表库的前提下，提供稳定、轻量、无额外依赖的结构分布视图。

## 3. 为什么这样设计

### 3.1 规则/调参编辑器不该只有“字段编辑”

如果一个编辑器只负责改字段，而不负责给出“整体结构感”，那研究者会一直在：

1. 单卡片局部视角
2. 与全局结构脱节
3. 容易忘记当前规则集的主导风格和失衡点

所以这次把参考面板从“说明板”推进成了“轻量结构总览板”。

### 3.2 不引入重图表库，优先轻量与稳定

本轮没有上新的图表依赖，而是复用前端已有样式体系，用极轻量的条形比例图完成：

1. family / type / kind / target / metric 分布表达
2. 全局结构一眼可读
3. 不增加 bundle 和刷新成本

这符合本项目当前阶段对前端的要求：

1. 要清晰
2. 要直观
3. 要低风险
4. 不能为了“更炫”而引入新的不稳定因素

### 3.3 “定位”如果不“聚焦”，仍然很容易丢

只做搜索并滚动到一张卡，在长列表里其实还不够。

因为研究者下一步通常还要：

1. 人眼重新确认是哪张
2. 自己再做聚焦
3. 手动排除其它干扰项

所以本轮直接把“跳到卡片”补成“跳到并聚焦卡片”，让流程更短，也更像专业工具。

## 4. 自动化验证

更新：

- [tests/test_observatory_frontend_editor_focus.py](H:\AP原型实验第二期\tests\test_observatory_frontend_editor_focus.py)

新增检查：

1. `renderReferenceBarChart`
2. `topEntriesFromHist`
3. `editor-reference-bar / editor-reference-chart / editor-reference-caption`
4. `state.pinnedRuleId = cleanId`
5. `state.pinnedTunerId = cleanId`
6. `已定位并聚焦规则卡`
7. `已定位并聚焦调参档`
8. `topFamilyBars / topEffectTypeBars / topFormulaBars / topChannelBars`
9. `topTargetBars / topMetricBars`

执行：

```bash
python -m unittest tests.test_observatory_frontend_editor_focus -v
python -m unittest tests.test_observatory_frontend_phase17 -v
python -m unittest discover -s tests -v
```

结果：

1. 前端专项通过
2. Phase17 编译与结构检查通过
3. 全量 `34` 项测试通过

## 5. HTTP 验证

验证地址：

- `http://127.0.0.1:8766/?ts=phase17f`

源码中已确认存在：

1. `renderReferenceBarChart`
2. `topEntriesFromHist`
3. `editor-reference-bar`
4. `state.pinnedRuleId = cleanId`
5. `state.pinnedTunerId = cleanId`
6. `已定位并聚焦规则卡`
7. `已定位并聚焦调参档`

## 6. 真页验收

验收地址：

- `http://127.0.0.1:8766/?ts=phase17f-browser`

### 6.1 参考面板真实可见

真页确认：

1. 规则参考面板中出现了结构分布条
2. 调参参考面板中出现了结构分布条

实测计数：

1. `ruleReferenceBars = 17`
2. `tunerReferenceBars = 4`

说明这不是只在源码里写了函数，而是页面里真实渲染出了多组条形分布。

### 6.2 聚焦闭环真实可用

真页中通过：

1. 在规则搜索框写入 `rule::residual_dissonance`
2. 点击 `仅看当前规则`

实测结果：

1. `pinnedRuleId = rule::residual_dissonance`
2. `visibleRuleCards = 1`
3. `rulesFilterHint` 中出现 `聚焦=rule::residual_dissonance`

说明“搜索 -> 聚焦 -> 收敛到单卡”的工作流已真实可用。

## 7. 本轮意义

本轮的意义不是单纯“多了几根条形图”，而是：

1. 编辑器第一次具备了真正的全局结构感
2. 从提醒 / 搜索 / 对照进入编辑的路径更短了
3. 规则和调参不再只是“字段维护”，而开始变成“结构化研究与调试”

如果说前几轮主要在解决“别让我看纯 JSON”，那这一轮主要在解决：

`给我一个足够像研究控制台的整体感`

## 8. 后续最值得继续补强的点

1. 在参考面板上继续加入：
   - warning code 热点
   - action drive 热点
   - condition metric 与 effect type 的交叉热点
2. 从审计差异表点击后，除了聚焦卡片外，还可以：
   - 自动折叠其它卡
   - 自动切到对应 effect / adjustment 子区块
3. 继续把右侧白箱区中仍偏文本型的信息，上翻成更直观的结构卡或比例图
