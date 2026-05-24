# V2 视觉结构偏好通道与 OCR 偏置修复记录

## 1. 这轮解决的问题

此前 V2 视觉链路已经具备：

- 原图分辨率稀疏采样
- 边缘优先候选点生成
- 视焦点附近偏置
- 注意力聚焦驱动下一 tick 动态提高采样率

但在 OCR-like 探针里，仍存在明显结构性偏置：

- `3` 更容易被压成 `8`
- `1024 raw` 和 `1536 raw` 并不总是优于 `512 raw`
- 说明仅有边缘和局部亮度/颜色还不够，需要更贴近人类形状识别的局部结构通道

## 2. 本轮新增的结构偏好通道

这轮在 `sensors/vision_sensor_v1.py` 的 `_local_shape_metrics(...)` 和 `_to_memory_feature_sample(...)` 中补入了以下结构信息：

### 2.1 基础结构通道

- `endpoint_likeness`
- `corner_likeness`
- `opening_likeness`
- `closure_likeness`
- `arc_balance`

### 2.2 新增的人类偏好型结构通道

- `straight_likeness`
- `curvilinear_likeness`
- `angularity`
- `roundness`
- `local_symmetry`

### 2.3 进一步提升判别力的通道

- `opening_dir_x`
- `opening_dir_y`
- `opening_direction_strength`
- `structure_discriminability`

其中：

- `opening_direction_strength` 用于区分“开口但朝哪边开”
- `structure_discriminability` 用于提升对类别区分更有价值的局部结构样本优先级

## 3. 记忆编码策略调整

本轮不再把所有结构量都细粒度离散成独立大段编码，而是改成更稳的组合式编码：

- 保留颜色、边缘、笔画、相对坐标、绝对坐标
- 结构部分改成：
  - `shape family`：直线/曲线/尖锐/圆润中的主导家族
  - `shape strength bin`
  - `opening tag`：闭合 / 左开 / 右开 / 上开 / 下开 / 不明确
  - `symmetry bin`
  - `discriminability bin`

这样做的目的不是减少信息，而是避免视觉记忆特征被切得过碎，导致同类局部结构难以合并召回。

## 4. 优先级逻辑调整

这轮不仅改了记忆编码，也改了进入焦点和记忆写入的排序逻辑：

- `structure_priority` 仍保留，但不再单纯奖励“闭合 + 圆润 + 对称”
- `symmetry` 会经过 `symmetry_support` 再参与结构优先级
- `focus_priority` 和 `raw_priority` 中新增了 `structure_discriminability`

因此当前的倾向是：

- 不是谁更圆谁就更容易进记忆
- 而是谁更能区分当前对象，谁更容易进焦点高优先区

## 5. 当前实验结论

对照输出目录：

- `outputs/vision_ocr_bias_study/small_run_shape_v2`
- `outputs/vision_ocr_bias_study/small_run_shape_v3`
- `outputs/vision_ocr_bias_study/small_run_shape_v4`

### 5.1 结果走势

- `v2`：首次加入结构通道后，`1024 raw / 2 tick` 可以把 `three` 拉回来，但会伤到 `eight`
- `v3`：加入直线/曲线/尖锐/圆润/对称后，长观察更像“仔细看”有效，但高 raw 下又更偏向 `eight`
- `v4`：加入开口方向与辨识度后，高 raw 下明显回稳，`1536 raw` 不再像 `v3` 那样崩掉

### 5.2 当前最值得记住的点

1. 视觉结构通道是有效的，但不能只奖励“闭合”和“圆润”
2. 对于 `3/8` 这种局部共享笔画很多的任务，开口方向和判别力优先级是必要的
3. 更高 raw 预算并不自动更好，必须配合更稳的视觉记忆编码与结构筛选
4. 多 tick 连续观察仍然非常重要，`4 tick` 普遍显著优于 `2 tick`

## 6. 当前最佳解释

这轮结果支持一个重要判断：

- AP 风格的视觉识别，不是靠单 tick 一眼锁死
- 而是靠“高 raw 现状 + 少量高判别力焦点结构写入记忆 + 连续多 tick 观察”来逐步稳定

这和人类的体验更接近：

- 惊鸿一瞥时容易把相近形状看混
- 盯着看几拍后会越来越稳

## 7. 下一步建议

下一步最值得继续推进的是：

1. 在前端把 `opening tag / shape family / discriminability` 做成可视化图层
2. 针对更复杂字符集做更大规模测试，而不是只盯 `3/8`
3. 把“注意力聚焦 -> 局部 raw 提升 -> 结构判别力上升”的链条做成可直接观测的实验面板
4. 后续扩展到中文手写、英文字符、图标轮廓识别时，继续坚持“不上传统 OCR 外挂”的路线

## 8. attention 闭环补充验证

在本轮中，还额外确认了一个重要实现事实：

- 早期 OCR 探针虽然已经开启了 attention boost 参数
- 但并没有真正执行 `continue_focus / inspect_residual` 等视觉动作
- 所以“仔细看”机制当时并未正式参与识别

随后补做了两步：

1. 在 OCR probe tick 里接入 `runtime.apply_selected_actions(...)`
2. 在规则层补入视觉场景下的 `continue_focus` 驱动：
   - `modal.has_image == 1`
   - `bn.count > 0`
   - `emotion.expectation_minus_pressure >= 0`

这样之后，链条变成：

- 第 1 个形成一级召回的视觉 tick
- 触发 `continue_focus`
- 下一 tick 视觉采样预算和聚焦强度提升
- 若后续违和感升高，则转入 `inspect_residual`

### 8.1 attention on/off 对照结果

对照目录：

- `outputs/vision_ocr_bias_study/attention_compare_on_v5`
- `outputs/vision_ocr_bias_study/attention_compare_off_v5`

最关键的现象是：

- 在 `1024 raw` 条件下，`attention off` 时，`three` 在 `2 / 4 / 6 tick` 几乎一直被 `eight` 吸走
- 打开 `attention on` 后，`three` 的召回明显回稳，尤其在持续观察里不再持续塌缩成 `eight`

这说明当前可以把贡献拆成两层：

1. **结构通道层贡献**  
   证明 AP 原生视觉链条本身已经具备基础的图像文字区分能力。

2. **仔细看闭环贡献**  
   证明“动作 -> 下一 tick 感知增强 -> 连续观察更稳”的机制可以进一步提升视觉辨识效果。

### 8.2 当前最稳妥的结论

现阶段最严谨的说法是：

- AP V2 已经不是“只能靠单 tick 稀疏采样蒙对”
- 而是可以通过
  - 结构化视觉特征
  - 少量高判别力焦点记忆
  - 连续多 tick 观察
  - 注意力驱动的感知增强
  
  来形成更稳定的视觉识别过程

这对后续扩到更复杂的 OCR-like、图标识别、局部目标跟踪，都有直接意义。
