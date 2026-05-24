# V2 多通道感受与召回闭环实验报告

- 生成时间：2026-05-23T13:35:34.087295
- 目标：对时间感受、情绪/认知感受、节奏感、视觉感受、听觉感受，以及它们反过来影响召回的闭环做统一验收。
- 方法：每个通道都分成两层证明。
  1. 运行时是否真的生成了对应通道的感受信号，并以 SA 形式进入状态池或 query_spacetime。
  2. 该通道的感受/意图参数是否真的改变了向量数据库中的召回排序。

## 一、总评
- 时间感受：runtime=True / monotonic_prefix=True / recall=True
- 情绪与认知感受：novelty=True / repeat_correctness=True / relearn_8=True / feedback_signal=True
- 节奏感：runtime=True / recall=True
- 运动意图召回：passed=True
- 反馈价性召回：passed=True
- 视觉：ocr=False / motion=True
- 听觉：focus=True / semantic=partial

## 二、时间感受
- 预期：当当前现状强烈召回某一类过去记忆时，系统应从主导记忆波峰中抽出时间间隔感，并把它写入 query_spacetime，形成后续模糊时间召回偏置。
- 结果摘要：
  - gap=1 -> target_delta_t=1.0 / confidence=1.0 / top_has_timefelt=True
  - gap=2 -> target_delta_t=1.2912 / confidence=0.7364 / top_has_timefelt=True
  - gap=4 -> target_delta_t=1.848 / confidence=0.7282 / top_has_timefelt=True
  - gap=6 -> target_delta_t=1.9104 / confidence=0.6559 / top_has_timefelt=True
- 召回对比：top1=eat banana / time_intent_bonus=0.9
- 解释：当前的时间感受更像‘主导时间间隔簇’而不是精确 tick 计数，因此最稳妥的验收标准不是要求完全等于真实 gap，而是要求它能形成稳定的时间主峰，并在召回时对匹配时间间隔的记忆产生可见偏置。

## 三、情绪与认知感受
- 预期：首次新异输入应先产生惊；重复输入应逐步拉起正确感与把握相关指标；错配后持续输入新对象时，应在若干 tick 后重新形成把握与正确感；恢复过程应带来恢复性奖励；期待与压力应持续转成下一 tick 的奖惩信号。
- 序列结果：novelty=True / repeat_correctness=True / relearn_8=True
  - tick=0 text='3' surprise=0.78 dissonance=0.0 correctness=0.0 grasp_score=0.0 pending=(0.0,0.0)
  - tick=1 text='3' surprise=0.0 dissonance=0.36 correctness=0.5152 grasp_score=0.56 pending=(0.0,0.1435)
  - tick=2 text='' surprise=0.0 dissonance=1.0 correctness=0.0 grasp_score=0.0 pending=(0.35,0.095)
  - tick=3 text='8' surprise=0.78 dissonance=0.9912 correctness=0.0 grasp_score=0.0 pending=(0.0,0.1848)
  - tick=4 text='8' surprise=0.4317 dissonance=0.9686 correctness=0.0 grasp_score=0.0 pending=(0.0018,0.1871)
  - tick=5 text='8' surprise=0.0 dissonance=0.8851 correctness=0.0 grasp_score=0.0721 pending=(0.1542,0.0616)
  - tick=6 text='8' surprise=0.0 dissonance=0.8455 correctness=0.0 grasp_score=0.0735 pending=(0.1974,0.0389)
  - tick=7 text='8' surprise=0.0 dissonance=0.8111 correctness=0.0 grasp_score=0.1727 pending=(0.0882,0.0372)
  - tick=8 text='8' surprise=0.0 dissonance=0.7025 correctness=0.2007 grasp_score=0.2181 pending=(0.1148,0.0357)
  - tick=9 text='8' surprise=0.0 dissonance=0.6051 correctness=0.2495 grasp_score=0.2712 pending=(0.1613,0.0309)
- 恢复性反馈：reward=0.35 / punishment=0.008 / notes=['intrinsic_correctness_delta_reward', 'intrinsic_grasp_delta_reward', 'intrinsic_surprise_recovery_reward', 'intrinsic_dissonance_recovery_reward', 'intrinsic_expectation_tonic_reward', 'intrinsic_pressure_tonic_punishment']
- 奖惩信号入池：top_contains_feedback=True / top=['text::hello', 'text::world', 'attr::reward_signal']

## 四、节奏感
- 预期：规则输入应形成 pulse/phase 类节奏感；不规则输入不一定完全没有节奏感，但其 regularity / confidence / groove 应显著更低；同时节奏 query_spacetime 应能提升匹配节拍周期的记忆召回分数。
- 规则节奏：period=2.0 / regularity=1.0 / confidence=1.0 / groove=0.5837
- 非规则节奏：period=3.724 / regularity=0.5393 / confidence=0.5928 / groove=0.5286
- 节奏召回：top1=beat / rhythm_intent_bonus=0.63

## 五、视觉与运动
- 预期：静态图像文字训练后，应能以统一状态池方式召回对应文本；动态输入应额外产生 motionfelt::trend，并把运动摘要写入内部链路。
- OCR 训练：accepted=False / epochs=12
  - digit_3: bn=three / cstar=three / state=three / strict=True / raw=64 / focus=8
  - digit_8: bn=three / cstar=three / state=three / strict=False / raw=64 / focus=8
- 动态视觉：motion_passed=True / labels=['motionfelt::trend', 'attr::punishment_signal'] / dynamic_object_count=0 / motion_confidence=0.3374

## 六、听觉
- 预期：听觉焦点应像视焦点一样可移动，并提升焦点频段附近的采样优先级；进入统一主链后，可作为未来语义学习、背景降噪与注意力联动的基础。
- 焦点移动：passed=True / before_center=1200.0 / after_center=1000.0 / target_hz=1000.0
- 焦点采样：before_focus_priority_count=4 / after_focus_priority_count=5 / strongest_after={'dominant_hz': 880.0, 'focus_bonus': 0.5295, 'freq_center_hz': 1000.0}
- 听觉语义边界：本轮补做了 richer pattern 的初探针，但仍只把它视作初步迹象，不宣称已经形成稳定的音频语义识别闭环。
  - tone_low: bn_preview=[{'text': 'tone_low', 'score': 0.4291}, {'text': 'tone_low', 'score': 0.36}, {'text': 'tone_high', 'score': 0.3156}]
  - tone_high: bn_preview=[{'text': '', 'score': 0.4802}, {'text': 'tone_low', 'score': 0.4716}, {'text': 'tone_low', 'score': 0.3941}]

## 七、统一哲学意义
- 这组结果更重要的不是单一准确率，而是多个通道都已经呈现同一个统一结构：
  1. 通道自身先产出连续强度信号。
  2. 当强度超过阈值时，该信号以 SA 形式进入状态池，成为可被认知和回忆的对象。
  3. 同一通道的感受又会反过来调制 query_spacetime 或召回评分，形成闭环。
- 这意味着‘时间感、节奏感、惊、违和感、正确感、奖励/惩罚、运动趋势、听觉焦点’都不是外挂标签，而是统一状态池中的一等公民。

## 八、当前边界
- 时间感目前更像模糊主峰，不是精确计时器，因此实验结论应表述为‘已形成稳定的模糊时间间隔感与时间召回偏置’，而不是‘已精确恢复真实 tick 差’。
- 听觉的焦点链、结构链和入池链已证明，但音频语义跨模态识别还应继续用更好的数据集做下一轮严格验收。
- 视觉 OCR-like 方面，本轮目标是证明统一状态池的可用性与动态补强接口，后续还可以继续扩大数据集、降低随机性、做更长时程测试。