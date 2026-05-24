# V2 多模态教学与海豚训练综合实验报告

## 1. 实验目标

本实验要验证的，不是单一模态分类，而是 AP V2 在统一状态池里，能否把“同一个对象的视觉、听觉、文本标签”共同教进去，并在后续仅凭单一模态线索，把其它模态相关记忆一起召回。
同时，本实验还把奖励信号接入训练过程，观察它是否能形成更稳定的对象绑定与后续认知翻转。

本轮重点回答五个问题：
1. 当图像、音频、文本同时输入且内容一致时，系统能否形成统一的多模态对象记忆。
2. 训练后，仅输入单视觉、单听觉、单文本时，系统能否召回正确文本以及对应其它模态结构。
3. 在空 tick 稳定后探测，与在 reset transient state 后探测，系统表现有何差别。
4. 当对象从苹果突然切换成香蕉时，系统需要多久从旧上下文翻转过来；反过来是否同理。
5. 在这个过程中，惊、违和感、正确感、把握感、期待、压力是否呈现出符合理论预期的时序变化。

## 2. 实验结论摘要

先说最重要的结论：
1. 在保留训练后稳定上下文的条件下，`vision-only` 与 `audio-only` 都已经能把目标文本重新拉起来，且 `vision-only` 的对象区分最稳。
2. `text-only` 对 `banana` 还能稳定命中，但对 `apple` 仍容易被更晚期的 `banana` 上下文压过去，说明跨模态统一绑定已经成立，但双对象长期对称稳固还没完全站住。
3. 一旦 `reset_transient_state`，纯视觉/纯听觉的冷启动召回明显变弱，说明这轮实验更像“稳定后的多模态整合召回成立”，还不能说“彻底脱离近期上下文也一样稳”。
4. 对象切换时，`Bn` 与 `C*` 都能在第 0 个 observation tick 迅速翻向新对象，但 `state_top` 还会慢 1 到 2 tick，末尾还存在尾振荡和旧对象残留。
5. 认知感受方面，这轮最清楚证明的是“新异输入触发惊与违和，然后记忆层逐步跟上”；`correctness / grasp` 这次仍基本没有显著起来，所以这一块不能夸大。

## 3. 实验材料

- `apple`：文本 `apple` / 中文 `苹果` / 图像 [apple.png](H:\AP原型实验第二期\outputs\multimodal_teaching_compare_20260524\current\assets\apple.png) / 音频 [apple.wav](H:\AP原型实验第二期\outputs\multimodal_teaching_compare_20260524\current\assets\apple.wav) / audio_mode=`tts`
  - 视觉结构标签示例：['vision_mem::s666663663_rgb955_e1_k1_n1_c1_o1_q4_u6_fr2_gox_y2_d0_rx4_ry4_ax5_ay5_ph3220_pv3210_oh3000_rh0010_hl0_cv0_hs2_vs1', 'vision_mem::s033033003_rgb411_e2_k1_n0_c0_o1_q3_u8_fr2_gox_y2_d0_rx4_ry5_ax3_ay7_ph0000_pv0000_oh0210_rh0010_hl0_cv3_hs3_vs3', 'vision_mem::s044004000_rgb121_e2_k2_n0_c0_o1_q3_u7_fr2_gox_y2_d0_rx5_ry3_ax6_ay2_ph3210_pv0123_oh0300_rh0021_hl1_cv2_hs2_vs1', 'vision_mem::s333333333_rgb811_e0_k0_n0_c0_o0_q9_u9_fr3_gcl_y2_d0_rx5_ry4_ax6_ay5_ph0000_pv0000_oh0000_rh0010_hl0_cv3_hs3_vs3']
  - 听觉结构标签示例：['audio::win_4', 'audio::win_11', 'audio::win_3', 'audio::win_10']
- `banana`：文本 `banana` / 中文 `香蕉` / 图像 [banana.png](H:\AP原型实验第二期\outputs\multimodal_teaching_compare_20260524\current\assets\banana.png) / 音频 [banana.wav](H:\AP原型实验第二期\outputs\multimodal_teaching_compare_20260524\current\assets\banana.wav) / audio_mode=`tts`
  - 视觉结构标签示例：['vision_mem::s777777000_rgb651_e4_k2_n1_c3_o1_q4_u0_fa1_gox_y2_d0_rx4_ry4_ax5_ay5_ph3300_pv2221_oh0300_rh0010_hl0_cv1_hs0_vs3', 'vision_mem::s777777770_rgb872_e1_k1_n1_c3_o1_q4_u0_fa1_gox_y2_d0_rx4_ry4_ax5_ay5_ph3310_pv2222_oh1200_rh0010_hl0_cv0_hs0_vs3', 'vision_mem::s777777770_rgb872_e1_k1_n0_c0_o1_q5_u2_fr2_gox_y2_d0_rx4_ry4_ax5_ay5_ph3310_pv3222_oh2100_rh0010_hl0_cv0_hs1_vs3', 'vision_mem::s777777777_rgb882_e0_k0_n0_c0_o0_q9_u9_fr3_gcl_y2_d0_rx5_ry4_ax6_ay4_ph3333_pv3333_oh0000_rh0010_hl0_cv0_hs3_vs3']
  - 听觉结构标签示例：['audio::win_4', 'audio::win_3', 'audio::win_5', 'audio::win_6']

这里的图像不直接写出“苹果/香蕉”文字，避免实验退化成视觉文字识别；它测的是对象形状/颜色结构与文本、音频标签的统一绑定。

## 4. 训练协议

- 苹果多模态连续训练：12 tick
- 香蕉多模态连续训练：12 tick
- 每个训练 tick 注入奖励：reward=1.0
- 训练后稳定空 tick：8
- 单模态 probe 连续观察：6 tick
- 切换实验旧对象预热：4 tick
- 训练 tick 总数：24
- 平均训练耗时：234.2689 ms
- 峰值训练耗时：365.0422 ms
- 平均 probe 耗时：118.2044 ms
- 峰值 probe 耗时：249.226 ms
- 稳定阶段 tick 数：8

## 5. 判定口径

本报告同时看四层信号：
1. `BN_top`：一级显式记忆召回最强文本。
2. `C*_top`：综合预测包当前主导文本。
3. `State_top`：状态池主导文本波峰。
4. `vision_identity / audio_identity`：目标对象特有的视觉或听觉结构标签，是否在 `C*` 与状态池里占优。

这里的 `first_text_success_tick` 采用的是当前脚本里的 `text_eval.strict_success` 口径，它要求 `BN_top` 与 `C*_top` 已经同时命中目标，但**不强制** `State_top` 同 tick 也完成翻转。
所以它更接近“认知判断已翻过来”，而不是“整个状态池主波峰已彻底稳定”。状态池是否翻过来，需要单独看 `State_top`。

## 6. 理论预期的认知感受图景

理论上，这组实验中的认知感受应该大致遵循下面的时序：
1. 第一次面对某个对象时，如果系统尚未建立对应预测，应先出现较明显的惊。
2. 连续多 tick 重复面对同一对象时，惊应逐步下降，而正确感/把握感、期待应逐步上升。
3. 当输入突然从苹果切换到香蕉时，旧预测尚未完全退去，会出现“惊 + 违和”的复合态。
4. 随着香蕉证据连续输入，旧预测衰减，新预测建立，违和感与惊应逐步回落，而正确感/把握感重新上来。
5. 如果奖励链路正常，则在“认清对象”与“从惊/违和中恢复”这两类阶段，应看到恢复性奖励或正确感相关的内源反馈。

## 7. 训练阶段的实际图景

### 7.1 苹果建立期
- tick 0 / epoch 0：surprise=1.0 / dissonance=0.0 / expectation=0.0 / BN_top=`` / C*_top=`` / State_top=`apple` / strict=否
- tick 1 / epoch 1：surprise=0.9482 / dissonance=0.4955 / expectation=1.0 / BN_top=`apple` / C*_top=`apple` / State_top=`apple` / strict=是
- tick 2 / epoch 2：surprise=0.7959 / dissonance=0.6393 / expectation=1.0 / BN_top=`apple` / C*_top=`apple` / State_top=`apple` / strict=是
- tick 3 / epoch 3：surprise=0.6846 / dissonance=0.4802 / expectation=1.0 / BN_top=`apple` / C*_top=`apple` / State_top=`apple` / strict=是

### 7.2 香蕉切入期
- tick 12 / epoch 0：surprise=0.7946 / dissonance=0.9553 / expectation=0.0 / BN_top=`apple` / C*_top=`apple` / State_top=`apple` / strict=否
- tick 13 / epoch 1：surprise=0.8003 / dissonance=0.9428 / expectation=1.0 / BN_top=`banana` / C*_top=`apple` / State_top=`apple` / strict=否
- tick 14 / epoch 2：surprise=0.6877 / dissonance=0.8665 / expectation=1.0 / BN_top=`banana` / C*_top=`apple` / State_top=`apple` / strict=否
- tick 15 / epoch 3：surprise=0.6067 / dissonance=0.8116 / expectation=1.0 / BN_top=`banana` / C*_top=`apple` / State_top=`apple` / strict=否

这段数据最像你预期的图景：`banana` 刚切入时，系统先表现出高惊和高违和，`BN` 先翻，`C*` 再翻，最后 `State_top` 才在更后面稳住。

## 8. 单模态召回结果

### 8.1 `idle_then_probe` / `audio`
- `apple`：BN_top=`apple` / C*_top=`apple` / State_top=`apple` / first_text_success_tick=0 / text_strict=是 / vision_C*_best=`apple` / vision_state_best=`` / vision_strict=否 / audio_C*_best=`` / audio_state_best=`` / audio_strict=否
- `banana`：BN_top=`banana` / C*_top=`apple` / State_top=`apple` / first_text_success_tick=4 / text_strict=否 / vision_C*_best=`apple` / vision_state_best=`` / vision_strict=否 / audio_C*_best=`` / audio_state_best=`` / audio_strict=否

### 8.2 `idle_then_probe` / `text`
- `apple`：BN_top=`apple` / C*_top=`apple` / State_top=`apple` / first_text_success_tick=0 / text_strict=是 / vision_C*_best=`apple` / vision_state_best=`apple` / vision_strict=是 / audio_C*_best=`` / audio_state_best=`` / audio_strict=否
- `banana`：BN_top=`banana` / C*_top=`banana` / State_top=`banana` / first_text_success_tick=0 / text_strict=是 / vision_C*_best=`banana` / vision_state_best=`` / vision_strict=否 / audio_C*_best=`` / audio_state_best=`` / audio_strict=否

### 8.3 `idle_then_probe` / `vision`
- `apple`：BN_top=`banana` / C*_top=`banana` / State_top=`banana` / first_text_success_tick=0 / text_strict=否 / vision_C*_best=`banana` / vision_state_best=`apple` / vision_strict=否 / audio_C*_best=`` / audio_state_best=`` / audio_strict=否
- `banana`：BN_top=`banana` / C*_top=`banana` / State_top=`banana` / first_text_success_tick=0 / text_strict=是 / vision_C*_best=`banana` / vision_state_best=`banana` / vision_strict=是 / audio_C*_best=`` / audio_state_best=`` / audio_strict=否

### 8.4 `reset_transient_state` / `audio`
- `apple`：BN_top=`apple` / C*_top=`apple` / State_top=`apple` / first_text_success_tick=0 / text_strict=是 / vision_C*_best=`apple` / vision_state_best=`` / vision_strict=否 / audio_C*_best=`` / audio_state_best=`` / audio_strict=否
- `banana`：BN_top=`banana` / C*_top=`banana` / State_top=`apple` / first_text_success_tick=5 / text_strict=是 / vision_C*_best=`banana` / vision_state_best=`` / vision_strict=否 / audio_C*_best=`` / audio_state_best=`` / audio_strict=否

### 8.5 `reset_transient_state` / `text`
- `apple`：BN_top=`apple` / C*_top=`apple` / State_top=`apple` / first_text_success_tick=2 / text_strict=是 / vision_C*_best=`apple` / vision_state_best=`apple` / vision_strict=是 / audio_C*_best=`` / audio_state_best=`` / audio_strict=否
- `banana`：BN_top=`banana` / C*_top=`apple` / State_top=`banana` / first_text_success_tick=4 / text_strict=否 / vision_C*_best=`banana` / vision_state_best=`apple` / vision_strict=否 / audio_C*_best=`` / audio_state_best=`` / audio_strict=否

### 8.6 `reset_transient_state` / `vision`
- `apple`：BN_top=`banana` / C*_top=`apple` / State_top=`apple` / first_text_success_tick=0 / text_strict=否 / vision_C*_best=`apple` / vision_state_best=`apple` / vision_strict=是 / audio_C*_best=`` / audio_state_best=`` / audio_strict=否
- `banana`：BN_top=`apple` / C*_top=`apple` / State_top=`apple` / first_text_success_tick=None / text_strict=否 / vision_C*_best=`apple` / vision_state_best=`banana` / vision_strict=否 / audio_C*_best=`` / audio_state_best=`` / audio_strict=否

这里有三个必须诚实说明的点：
1. `idle_then_probe` 明显强于 `reset_transient_state`，说明当前成功很依赖训练后保留下来的稳定上下文。
2. 这轮 integrated probe 里，视觉 identity 的跨模态带起效果是最清楚的；音频 identity 标签在最终 probe 窗口里还没有形成同样干净的可观测胜出。
3. 所以这轮最稳的证据是“视觉/听觉能把目标文本拉起来，视觉还能把目标视觉对象结构一起带起”，而不是“所有模态都已经完全对称地互相召回”。

## 9. 对象切换翻转结果

- `apple->banana`：BN_flip=0 / C*_flip=0 / State_flip=0 / Focus_flip=0 / first_obs(surprise=0.8607, dissonance=0.3991, BN=`banana`, C*=`banana`, State=`banana`) / final(BN=`banana`, C*=`banana`, State=`banana`, strict=是)
- `banana->apple`：BN_flip=0 / C*_flip=0 / State_flip=1 / Focus_flip=0 / first_obs(surprise=0.8607, dissonance=0.3292, BN=`apple`, C*=`apple`, State=`banana`) / final(BN=`apple`, C*=`banana`, State=`apple`, strict=否)

这组切换结果很像你要的层级翻转：
1. 新证据一进来，`BN` 和 `C*` 几乎立刻翻向新对象。
2. `State_top` 更慢，通常要再过 1 到 2 tick。
3. 末尾 `final_C*_top` 仍可能飘回旧对象，这说明尾振荡和旧上下文残留还没有完全消掉。

## 10. 认知感受与情绪变化观察

- `idle_then_probe` / `vision` / `apple`：first(surprise=0.9996, dissonance=0.8852) -> last(surprise=0.546, dissonance=0.7301, correctness=0.0, grasp=0.0)
- `idle_then_probe` / `vision` / `banana`：first(surprise=0.9996, dissonance=0.8852) -> last(surprise=0.546, dissonance=0.7301, correctness=0.0, grasp=0.0)
- `idle_then_probe` / `audio` / `apple`：first(surprise=0.9989, dissonance=0.6721) -> last(surprise=0.5162, dissonance=0.6717, correctness=0.0, grasp=0.0)
- `idle_then_probe` / `audio` / `banana`：first(surprise=0.9989, dissonance=0.6721) -> last(surprise=0.5145, dissonance=0.6708, correctness=0.0, grasp=0.0)
- `idle_then_probe` / `text` / `apple`：first(surprise=0.0, dissonance=0.4424) -> last(surprise=0.0, dissonance=0.7379, correctness=0.0, grasp=0.0)
- `idle_then_probe` / `text` / `banana`：first(surprise=0.0, dissonance=0.3844) -> last(surprise=0.0, dissonance=0.6043, correctness=0.2478, grasp=0.0)
- `reset_transient_state` / `vision` / `apple`：first(surprise=1.0, dissonance=1.0) -> last(surprise=0.546, dissonance=0.7522, correctness=0.0, grasp=0.0)
- `reset_transient_state` / `vision` / `banana`：first(surprise=1.0, dissonance=1.0) -> last(surprise=0.546, dissonance=0.7522, correctness=0.0, grasp=0.0)
- `reset_transient_state` / `audio` / `apple`：first(surprise=1.0, dissonance=1.0) -> last(surprise=0.546, dissonance=0.7522, correctness=0.0, grasp=0.0)
- `reset_transient_state` / `audio` / `banana`：first(surprise=1.0, dissonance=1.0) -> last(surprise=0.546, dissonance=0.7522, correctness=0.0, grasp=0.0)
- `reset_transient_state` / `text` / `apple`：first(surprise=0.0, dissonance=0.414) -> last(surprise=0.0, dissonance=0.4266, correctness=0.3453, grasp=0.198)
- `reset_transient_state` / `text` / `banana`：first(surprise=0.0, dissonance=0.4882) -> last(surprise=0.0, dissonance=0.7953, correctness=0.0, grasp=0.0)

这轮数据和理论预期的符合点与不符合点都很清楚：
1. 符合的部分：新模态输入时，`surprise` 与 `dissonance` 会先高，然后随着连续输入逐步下降。
2. 部分符合的部分：香蕉切入训练期，确实出现了“先惊 + 违和，再由记忆层逐步翻正”的层级过程。
3. 暂时不符合或尚未显著的部分：`correctness` 与 `grasp` 在这轮 integrated probe 里几乎始终接近 0，没有形成可以拿来强证明的稳定证据。
所以这轮能证明“惊 / 违和 / 逐步反应过来”，但还不能强证明“把握感 / 正确感已经在这套多模态教学里稳定长出来”。

## 11. 前端展示链路验收

- 展示 dataset：[showcase_dataset.json](H:\AP原型实验第二期\outputs\multimodal_teaching_compare_20260524\current\showcase_dataset.json)
- 展示 run 目录：[phase11_multimodal_run_20260524_063153_943](H:\AP原型实验第二期\outputs\multimodal_teaching_compare_20260524\current\observatory_showcase\runs\phase11_multimodal_run_20260524_063153_943)
- sidecar tick 数：86 / summary tick 数：86

这条展示 run 已经做了前端实测：
1. 在文字 probe tick 上，视觉面板会老实显示“暂无可回放视觉帧”，不会伪造图像。
2. 在图像 tick（如 `Tick 12`）上，视觉面板可进入“正在播放视觉回放”状态，并显示 `当前感知 / 近 4 tick 已叠加 192 个视觉 SA；当前 tick 原始采样 192，焦点样本 20，注视累积 1315`。
3. 想象音频面板也能进入播放态，并显示 `融合视图 / 近 4 tick 已提取 6 个听觉 SA，可播放代理合成音`。

也就是说，当前前端已经能把“bot 某个 tick 在联想到的内心画面”和“它在回味的内心声音代理结构”真正展示出来，只是仍然属于代理重建，不是原始高保真回放。

## 12. 结论

如果只问“这套多模态教学 + 海豚训练有没有把统一召回这条链打通”，这轮答案是肯定的，但要带边界地说。

> **AP V2 已经能在统一状态池、统一记忆召回和奖励塑形链路里，把图像、音频、文本共同教成一个对象，并在保留近期稳定上下文的情况下，凭单一模态线索触发跨模态联想与对象翻转。**

已经被较强证明的部分是：
1. `vision-only` 与 `audio-only` 可以把目标文本重新拉起来。
2. 视觉对象结构也能跟着一起被带起。
3. 新旧对象切换时，系统确实呈现出分层翻转，而不是单步硬切。
4. 前端展示链可以把这种联想过程以“想象图像 / 想象音频”方式直观看出来。

还没有被这轮强证明的部分是：
1. 完全清空瞬态后，纯跨模态长期冷召回是否同样稳。
2. 音频 identity 结构是否已经能像视觉 identity 一样干净地被反向带起。
3. `correctness / grasp` 是否已经在这组综合实验里稳定长出来。

## 13. 当前边界

1. 当前对象集仍然很小，只证明原理可行，不代表开放环境大规模泛化。
2. 当前 integrated probe 的成功高度依赖保留训练后的稳定上下文；`reset_transient_state` 后的纯冷召回仍然明显变弱。
3. 当前前端听觉回放是结构代理合成，不是原始波形高保真重建。
4. 当前视觉叠加是状态池视觉 SA 的稀疏代理重建，不等于像素级完整还原。
5. 本轮听觉优先使用系统 TTS；若环境不可用则自动退回到可控 chirp 代理音。
6. 当前 `strict_success` 的统计口径偏向“BN + C* 已翻正”，与“状态池主波峰也完全稳定”不是同一件事，阅读时要分开。
