# V2 动态视觉联动 OCR 实验报告

- 生成时间：2026-05-23T02:08:20
- 目标：严谨区分三件事。
  1. 动态视觉是否已经具备“对外界运动本身进行区分”的基础能力。
  2. 这条动态链是否已经接入注意力 / 视线重定向，因此可作为未来先天动作或后天学习的输入接口。
  3. 动态视觉是否已经稳定提升复杂场景下的 OCR-like 识别闭环。

## 一、总评
- 已证明：动态视觉链已经成立。系统不仅能产生运动相关内部量，还能在较强运动条件下把外界运动提升为对象级摘要。
- 已证明：动态视觉链已经接入视线/注意力接口。在低预算单目标运动实验中，动态开启后 gaze 路径会随目标迁移，而被动条件下 gaze 基本保持不动。
- 已证明：动态视觉会显著改变低预算单目标运动场景下的 gaze 路径和内部处理链，因此联动接口是真实存在的。
- 仅初步成立：动态链已经能显著提高动态轨迹数量、动态显著性与视线重定向，但这还不足以证明它已经稳定增强复杂 clutter 场景 OCR。
- 尚未证明：仅靠当前动态视觉联动，就能稳定压低复杂场景中的 surprise/dissonance 并把 grasp/correctness 拉起，从而形成稳定的最终识别闭环。

## 二、实验一：纯动态辨别能力
- 协议：固定 gaze，不开启自动回看，避免把“自己眼睛在动”误当成“外界物体在动”。
- 条件：
  - `static_repeat_fixed_gaze`：同一矩形连续重复。
  - `big_shift_motion_fixed_gaze`：同一矩形横向大幅位移。
  - `big_approach_motion_fixed_gaze`：同一矩形持续放大，模拟接近。
- `static_repeat_fixed_gaze`：peak_dynamic_object_count_after_first=0，peak_dynamic_objectness_after_first=0.5835，peak_motion_speed_after_first=0.171，avg_surprise_after_first=0.7994
- `big_shift_motion_fixed_gaze`：peak_dynamic_object_count_after_first=2，peak_dynamic_objectness_after_first=0.654，peak_motion_speed_after_first=0.2177，avg_surprise_after_first=0.9679
- `big_approach_motion_fixed_gaze`：peak_dynamic_object_count_after_first=2，peak_dynamic_objectness_after_first=0.6434，peak_motion_speed_after_first=0.2178，avg_surprise_after_first=0.9532

结论：
- 静态重复条件下，`dynamic_object_count` 在后续 tick 中保持 0。
- 大幅横移与大幅接近条件下，`dynamic_object_count` 在后续 tick 中上升到 2，说明系统不是只积累局部边缘差分，而是已经能把持续运动提升成对象级动态摘要。
- 运动条件的 `surprise` 也维持更高，说明系统把持续运动当作持续的新异输入，而不是完全习惯掉。
- 这一步已经符合你的长期目标方向：运动本身可以成为可区分、可调用、可继续接动作或学习的内部量。

## 三、实验二：低预算单目标运动 OCR 联动
- 简单运动训练底座：raw=512 / memory=24 / focus=12 / epoch=2
- 复杂压力训练底座：raw=512 / memory=24 / focus=12 / epoch=6
- 测试：只有目标 `8` 在画布上移动，无 distractor、无 clutter。
- `moving_passive_no_dynamic`：first_success_tick=2，success_count=1，mean_elapsed_ms=32.9688，gaze_start={'x': 0.5, 'y': 0.5}，gaze_end={'x': 0.5, 'y': 0.5}
- `moving_dynamic_auto`：first_success_tick=3，success_count=1，mean_elapsed_ms=48.6216，gaze_start={'x': 0.5645, 'y': 0.168}，gaze_end={'x': 0.2547, 'y': 0.4355}
- `moving_dynamic_full`：first_success_tick=3，success_count=1，mean_elapsed_ms=63.2059，gaze_start={'x': 0.5645, 'y': 0.168}，gaze_end={'x': 0.2547, 'y': 0.4355}

结论：
- 这组简单运动实验没有全部成功，因此当前仍不能宣称“动态联动已经稳定增强单目标 OCR”；它更多证明了动态链会改变 gaze 与内部处理路径。
- `moving_passive_no_dynamic` 的 gaze 基本保持在中心；而 `moving_dynamic_auto` / `moving_dynamic_full` 的 gaze 会随目标移动。
- 这证明动态视觉并不是旁路日志，而是真的接到了 runtime 的视线/注意力接口。
- 即便简单运动实验成功，它也未必显示“动态条件比被动条件更早成功”；因此这部分主要用于证明联动链成立，而不是直接证明识别优势。

## 四、实验三：复杂 clutter 压力场景 OCR 联动
- 测试：目标 `8` 在复杂背景中移动，同时放入固定 distractor `3`。
- `moving_passive_no_dynamic`：first_success_tick=None，success_count=0，mean_dynamic_track_count=9.3333，mean_dynamic_object_count=0.0，mean_surprise=0.9317，mean_dissonance=0.9648
- `moving_dynamic_auto`：first_success_tick=None，success_count=0，mean_dynamic_track_count=19.6667，mean_dynamic_object_count=0.0，mean_surprise=0.9322，mean_dissonance=0.9651
- `moving_dynamic_full`：first_success_tick=None，success_count=0，mean_dynamic_track_count=20.1667，mean_dynamic_object_count=0.1667，mean_surprise=0.9316，mean_dissonance=0.9647

结论：
- 三个条件在这组压力测试中全部 `success_count=0`。
- 打开动态链后，`dynamic_track_count` 和 `dynamic_salience` 确实上升了，说明系统更强地“看见了运动”。
- 但最终 `state_best_text` 仍长期偏向 distractor `three`，同时 `correctness` / `grasp` 没有稳定抬升。
- 因此目前最诚实的结论是：动态视觉已经显著改变了内部处理路径，但还没有稳定完成‘把动态对象推成最终识别波峰’这一步。

## 五、认知感受与情绪通道
- 在运动与 clutter 场景中，`surprise` 与 `dissonance` 普遍维持高位，说明系统确实把这些变化当作新异和错配来处理。
- 但 `correctness` 与 `grasp` 没有相应稳定抬升，说明‘注意到了变化’和‘已经认清了对象’目前还是两步。
- 从你的哲学来说，这个结果很有价值：它表明动态视觉已经能制造后续学习/动作所需的“被注意、被惊到、值得进一步处理”的状态，但还没完全闭环成稳定把握。

## 六、对你的目标意味着什么
- 你希望未来系统能根据接近、远离、横移等运动情况，触发先天规则或后天学习。就这一步而言，接口已经开始成形。
- 当前可以把 `dynamic_objectness`、`motion_speed`、`motion_surprise`、`motion_coherence`、`dynamic_object_count` 这些量，看作未来规则系统或奖惩学习的直接输入候选。
- 特别是“接近”条件能够在固定 gaze 下产生对象级动态摘要，这对以后做趋避、警觉、主动回看都很关键。

## 七、当前边界
- 现在能证明动态视觉已经打通，但不能证明它已经成熟到稳定提升复杂 OCR。
- 复杂场景中的瓶颈不是“看不见运动”，而是“看见了运动，但还没有把运动对象稳定并入最终识别闭环”。
- 因而这阶段最合理的表述应当是：动态视觉联动已经具备工程与理论价值，但在复杂识别上的优势仍属初步迹象。

## 八、下一步建议
- 优先增强：让动态对象摘要更稳定地进入最终 `Bn / C* / state top` 决策主干。
- 再做联动：把接近、横移、遮挡重现这些模式接到先天动作阈值调制或奖励/惩罚学习。
- 最后再验收：用更长时程、多次重复、带空 tick 稳定段的协议，验证动态对象能否被习得为可靠的行动线索。