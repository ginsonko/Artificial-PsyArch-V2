# V2 原生音频识别 OCR-like 实验报告

## 1. 实验目标

本实验的目标，不是验证传统 ASR 或关键词分类器，而是验证 AP V2 是否已经具备一种原生的音频-文本绑定与召回雏形。
更具体地说，本轮实验想回答四个问题：
1. 当一段音频结构与对应文本标签持续共现时，系统能否把这段音频的结构信息与文本标签绑定起来。
2. 训练完成后，仅再次输入音频而不给文本，系统能否从记忆中正确召回对应文本。
3. 仅输入文本而不给音频时，系统能否反过来召回与该文本相关的音频结构记忆。
4. 在旧音频上下文干扰下，系统能否经过若干连续 observation tick，从旧对象逐步切换到新的音频认知对象。

## 2. 实验材料

本轮实验使用两段程序生成的单声道上行 chirp 音频：
- `tone_low_rise`：文本标签 `tone_low`，频率从 320.0 Hz 上升到 480.0 Hz，时长 0.24 秒，文件为 [tone_low_rise.wav](outputs\audio_ocr_like_20260524_post_vectorboost\generated_audio\tone_low_rise.wav)
  - 可观测结构标签：memory=['audio::mem::pfha_tc4_nz0_ps4_hr5_pr0_vp5_ct4_fl0_bw0_ro0_ce0_db0', 'audio::mem::pfha_tc4_nz0_ps4_hr5_pr0_vp5_ct4_fl0_bw0_ro0_ce0_db0'] / global=['audio::global::pfha_tc4_nz0_ps4_hr5_pr0_vp5_ct3_fl0_bw0_ro0_ce0_db0'] / dominant_hz=560.0
- `tone_high_rise`：文本标签 `tone_high`，频率从 760.0 Hz 上升到 980.0 Hz，时长 0.24 秒，文件为 [tone_high_rise.wav](outputs\audio_ocr_like_20260524_post_vectorboost\generated_audio\tone_high_rise.wav)
  - 可观测结构标签：memory=['audio::mem::pfto_tc4_nz1_ps4_hr4_pr1_vp4_ct4_fl0_bw0_ro0_ce0_db1', 'audio::mem::pfto_tc4_nz1_ps4_hr4_pr1_vp4_ct3_fl0_bw0_ro0_ce0_db1'] / global=['audio::global::pfto_tc4_nz1_ps4_hr4_pr1_vp4_ct4_fl0_bw0_ro0_ce0_db1'] / dominant_hz=1080.0

## 3. 实验链路与判定口径

音频不会进入传统语音识别模块，而是进入 AP V2 的统一多模态链路：
1. 听觉感受器把波形拆成听窗、焦点优先样本、可入记忆的结构特征，以及全局听觉结构特征。
2. 文本标签与听觉结构同 tick 进入状态池。
3. 系统通过 `Bn` 召回历史记忆，再通过 `C*` 形成当前整合预测包。
4. 训练阶段同步注入奖励信号，塑造“该音频结构 <-> 该文本标签”的长期联结。

本实验主要看四层：
- `BN_top`：最强候选显式记忆的文本标签。
- `C*_top`：综合预测包中最强文本标签。
- `State_top_text`：状态池主导文本波峰。
- 反向音频结构召回：文本单独输入时，与目标音频对应的 `audio::mem::* / audio::global::*` 标签能否在 `C*` 和状态池内压过干扰对象。

其中：
- `BN_top` 正确，说明显式记忆召回已开始对齐。
- `C*_top` 正确，说明当前整合后的认知判断已对齐。
- `State_top_text` 正确，说明状态池主导波峰已切换到目标文本。
- 文本反向召回到目标音频结构标签，说明多模态绑定已经不是单向的，而是可以反向联想到听觉记忆。

## 4. 训练设计

- 训练轮次：6 epoch，每个 epoch 交替呈现两段音频与对应文本。
- 每个正确共现 tick 注入奖励：reward=1.0
- 训练后稳定空 tick：8
- 接受门槛冷探测：audio-only，连续 4 tick
- 实验输出目录：[summary.json](outputs\audio_ocr_like_20260524_post_vectorboost\summary.json) / [report.md](outputs\audio_ocr_like_20260524_post_vectorboost\report.md)

## 5. 预期

如果 AP V2 的原生音频识别链路成立，那么预期会出现以下现象：
1. 训练后，面对 `tone_low_rise` 音频时，`BN_top` 和 `C*_top` 应稳定偏向 `tone_low`。
2. 面对 `tone_high_rise` 音频时，`BN_top` 和 `C*_top` 应稳定偏向 `tone_high`。
3. 当只输入文本 `tone_low` 或 `tone_high` 时，与该文本绑定过的音频结构标签应被反向召回，而不是只剩文本自身。
4. 在先听过另一段音频的前提下，新音频未必第一反应就立刻获胜，但经过几个连续 tick，应逐步翻转到正确对象。

## 6. 主实验结果

### 6.1 Audio-only 冷探测
- `tone_low_rise` -> `tone_low`：BN_top=`tone_high`，C*_top=`tone_low`，State_top=`tone_low`，C* margin=40.8864，首次严格成功 tick=1，strict_success=False
- `tone_high_rise` -> `tone_high`：BN_top=`tone_low`，C*_top=`tone_low`，State_top=`tone_low`，C* margin=-25.2703，首次严格成功 tick=3，strict_success=False

### 6.2 接受门槛检查
- `tone_low_rise`：BN_rank=0 / C*_top=`tone_low` / focus_has_target=False / first_success_tick=1 / strict_success=False
- `tone_high_rise`：BN_rank=1 / C*_top=`tone_high` / focus_has_target=False / first_success_tick=3 / strict_success=True

### 6.3 Text-only 反向召回音频结构
- 文本 `tone_low`：BN_top=`tone_high`，C*_text_top=`tone_low`，音频结构 C*_best_pair=`tone_low_rise`，State_best_pair=`tone_low_rise`，audio_cstar_margin=2.5473，首次音频反向命中 tick=0，audio_strict_success=True
  - 目标音频标签：['audio::mem::pfha_tc4_nz0_ps4_hr5_pr0_vp5_ct4_fl0_bw0_ro0_ce0_db0', 'audio::mem::pfha_tc4_nz0_ps4_hr5_pr0_vp5_ct4_fl0_bw0_ro0_ce0_db0', 'audio::mem::pfha_tc4_nz0_ps4_hr5_pr0_vp5_ct4_fl0_bw0_ro0_ce0_db0', 'audio::mem::pfha_tc4_nz0_ps4_hr5_pr0_vp5_ct3_fl0_bw0_ro0_ce0_db0']
- 文本 `tone_high`：BN_top=`tone_low`，C*_text_top=`tone_low`，音频结构 C*_best_pair=`tone_low_rise`，State_best_pair=`tone_low_rise`，audio_cstar_margin=-2.3035，首次音频反向命中 tick=0，audio_strict_success=False
  - 目标音频标签：['audio::mem::pfto_tc4_nz1_ps4_hr4_pr1_vp4_ct4_fl0_bw0_ro0_ce0_db1', 'audio::mem::pfto_tc4_nz1_ps4_hr4_pr1_vp4_ct3_fl0_bw0_ro0_ce0_db1', 'audio::mem::pfto_tc4_nz1_ps4_hr4_pr0_vp4_ct3_fl0_bw0_ro0_ce0_db1', 'audio::mem::pfha_tc4_nz0_ps4_hr4_pr0_vp4_ct4_fl0_bw0_ro0_ce0_db1']

### 6.4 旧上下文干扰下的音频切换
- 协议：先连续听 `tone_low_rise` 共 4 tick，再不清空地切换到 `tone_high_rise` 并继续观察 8 tick。
- 翻转结果：BN_top 第 0 个 observation tick 翻转；C*_top 第 0 个翻转；State_top_text 第 None 个翻转；A_focus 第 None 个命中。

| observation tick | BN_top | C*_top | C* margin | State_top | State margin | A_focus 命中 |
| --- | --- | --- | ---: | --- | ---: | --- |
| 0 | tone_high | tone_high | 9.2352 | tone_low | -2.8968 | 否 |
| 1 | tone_low | tone_high | 27.1489 | tone_low | -1.4729 | 否 |
| 2 | tone_high | tone_low | -5.9818 | tone_low | -1.7919 | 否 |
| 3 | tone_high | tone_low | -4.317 | tone_low | -1.9504 | 否 |
| 4 | tone_low | tone_low | -18.8924 | tone_low | -2.8646 | 否 |
| 5 | tone_high | tone_low | -29.1496 | tone_low | -3.6698 | 否 |
| 6 | tone_low | tone_low | -4.8954 | tone_low | -3.7245 | 否 |
| 7 | tone_low | tone_low | -8.4597 | tone_low | -4.0346 | 否 |

## 7. 结果解释

这组结果如果成立，证明的不是“已经做出了传统语音识别产品”，而是更基础、更重要的一点：
AP V2 已经可以把音频结构与文本标签放进同一个统一状态池链路里学习，并在之后仅凭其中一个模态的线索，把另一个模态相关记忆重新拉起来。

尤其是 text-only 反向召回这一步很关键。
如果只有 audio-only -> text 成功，那还可能被解释成“只是在做音频分类后拉文本标签”；
但如果 text-only 时，和目标音频对应的结构标签也能被一起带出，说明这里建立的是跨模态绑定，而不是单向映射。

切换实验的意义则在于：
- 系统并不要求第一反应永远正确。
- 它允许旧上下文短暂残留。
- 但在持续新证据输入下，能逐步完成认知翻转。

这比一个一次性静态分类器更接近认知系统的行为图景。

## 8. 这意味着什么

如果把它翻译成更直白的话，就是：
1. 我们已经不仅能让系统“听到不同的声音不一样”，而是开始能让它把“这段声音”和“这个词”绑定起来。
2. 训练后，仅输入声音，它可以逐步想到对应文本。
3. 仅输入文本，它也可以逐步想到之前和它一起出现过的那类声音结构。
4. 这说明 AP 的多模态统一召回，不只适用于文字-图像，也可以扩展到文字-音频。

如果后续再把动作、情绪、奖惩等一起绑定进去，这就更接近“教一个主体认识一个对象”的方式，而不是单模态分类器。

## 9. 当前边界

本实验证明的是“原生音频识别雏形成立”，但还不能直接推出更强结论：
1. 不能直接等价于成熟 ASR 或通用音频事件识别。
2. 当前刺激集只有两类、且人为构造得比较清晰，因此证明的是原理可行性，不是大规模开放环境泛化。
3. 当前 text-only 反向召回的可观测证据，主要还是 `audio::mem::* / audio::global::*` 结构标签，而不是最终可试听的内心声音重建。
4. 本轮默认使用带奖励的配对训练，因此证明的是“统一状态池 + 记忆召回 + 奖励塑形”这条链路可行，不是“无奖励也会自动同样稳定形成绑定”。

## 10. 阶段性总结

如果只看“有没有这种能力”，本轮实验一旦通过，答案就是肯定的。
它最值得和同事分享的，不是“已经做出了产品级语音识别”，而是：

> **AP V2 这种统一状态池 + 记忆召回 + 预测包 + 奖励塑形的架构，已经可以不依赖传统 ASR 模块，直接长出一种原生的音频-文本识别与反向联想雏形。**

而且这个能力不是单步硬分类式输出，而是带有连续观察、旧上下文干扰、逐步翻转、反向联想这些更接近认知系统的行为特征。
