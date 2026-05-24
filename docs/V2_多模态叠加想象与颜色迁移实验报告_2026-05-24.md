# V2 多模态叠加想象与颜色迁移实验报告

## 1. 实验目标

这次只回答两个很具体的问题：
1. 在已经学会 `apple` 和 `banana` 之后，如果直接输入 `apple banana` 或 `banana apple`，系统是否会同时拉起两个对象，而不是只剩一个对象。
2. 如果输入一个“黄色苹果”，系统会不会以苹果为主，同时因为颜色相近而带起一点香蕉联想。

为了避免上一段短时上下文残留直接污染结果，本轮正式 probe 在导入长期记忆后，先 `reset_transient_state`，只保留 memory store，再开始测试。这样更接近“学会之后重新看题”的口径。

## 2. 实验设置

- 苹果多模态训练：12 tick
- 香蕉多模态训练：12 tick
- 每个训练 tick 外部奖励：1.0
- 训练后稳定空 tick：8
- 平均训练耗时：406.6336 ms
- probe 平均耗时：145.0657 ms
- 稳定阶段 tick 数：8

报告里同时看 6 组量：
1. `text C*`：综合预测包里，`apple / banana` 两个文本对象的能量。
2. `text state`：状态池顶部里，`apple / banana` 两个文本对象的能量。
3. `vision C*`：视觉身份标签映射到 `apple / banana` 后的能量。
4. `audio C*`：听觉身份标签映射到 `apple / banana` 后的能量。
5. `secondary_ratio`：次强对象能量 / 最强对象能量。
6. 情绪：重点看 `surprise / dissonance`。

文中说“明显双活化”时，只是一个便于阅读的口径：`secondary_ratio >= 0.25`。原始能量值全部保留。

## 3. 基线校准

先做单概念文本 probe，确认系统不是连单个苹果/香蕉都拉不起来。

- `apple`：text 主导=`apple` / text 次级=`` ratio=0.0 / vision 主导=`apple`
- `banana`：text 主导=`apple` / text 次级=`banana` ratio=0.8372 / vision 主导=`banana`

这里要实话实说：`apple` 的单概念基线是干净的，但 `banana` 的 text C* 仍然带着明显的 `apple` 残留。
不过 `banana` 的视觉 recall 仍然是以香蕉为主，所以后面的双概念实验至少不是建立在“视觉层连单对象都认不出”的坏底座上。

## 4. 实验 A：双概念文本叠加想象

### 4.1 `apple banana`

- tick 0: apple: textC*=77.3934, textState=2.1695, visionC*=0.0, audioC*=0.0 | banana: textC*=99.2677, textState=2.5, visionC*=0.0, audioC*=0.0 / surprise=0.0 / dissonance=0.6242
- tick 1: apple: textC*=192.246, textState=4.4526, visionC*=21.1692, audioC*=0.0 | banana: textC*=8.8113, textState=3.33, visionC*=0.0, audioC*=0.0 / surprise=1.0 / dissonance=0.419
- tick 2: apple: textC*=107.8911, textState=6.5073, visionC*=14.5197, audioC*=0.0 | banana: textC*=53.5476, textState=4.7415, visionC*=0.0, audioC*=0.0 / surprise=0.8482 / dissonance=0.4353
- tick 3: apple: textC*=48.3653, textState=8.3566, visionC*=22.9997, audioC*=0.0 | banana: textC*=29.0379, textState=6.168, visionC*=0.0, audioC*=0.0 / surprise=0.714 / dissonance=0.5971
- tick 4: apple: textC*=105.1853, textState=10.0209, visionC*=15.5881, audioC*=0.0 | banana: textC*=38.5552, textState=7.101, visionC*=0.0, audioC*=0.0 / surprise=0.7098 / dissonance=0.4111
- tick 5: apple: textC*=25.6189, textState=11.5188, visionC*=18.731, audioC*=0.0 | banana: textC*=12.2747, textState=8.1096, visionC*=5.7545, audioC*=0.0 / surprise=0.6577 / dissonance=0.8386

结论：`apple banana` 最终在 text C* 上是 `apple` 主导，`banana` 次级，secondary_ratio=0.4791；state text 上同样保留明显双对象，secondary_ratio=0.704。
更重要的是，这一顺序下视觉 C* 也已经进入双活化：主导=`apple`，次级=`banana`，secondary_ratio=0.3072。
所以如果直接看当前前端的能量叠加视图，这一段更像“苹果轮廓更亮、香蕉轮廓更暗，但两者同时在场”的叠加态。
音频侧仍然没有形成对应的双活化，最终 audio C* secondary_ratio=0.0。

### 4.2 `banana apple`

- tick 0: apple: textC*=78.6489, textState=2.3511, visionC*=0.0, audioC*=0.0 | banana: textC*=87.3185, textState=2.5, visionC*=0.0, audioC*=0.0 / surprise=0.0 / dissonance=0.7878
- tick 1: apple: textC*=180.9091, textState=4.616, visionC*=24.7965, audioC*=0.0 | banana: textC*=19.1668, textState=3.4089, visionC*=0.0, audioC*=0.0 / surprise=1.0 / dissonance=0.4262
- tick 2: apple: textC*=53.3135, textState=6.5744, visionC*=10.6257, audioC*=0.0 | banana: textC*=56.317, textState=5.568, visionC*=0.0, audioC*=0.0 / surprise=0.7303 / dissonance=0.8376
- tick 3: apple: textC*=15.4273, textState=7.1285, visionC*=0.0, audioC*=0.0 | banana: textC*=109.4345, textState=7.5112, visionC*=1.3952, audioC*=0.0 / surprise=0.7859 / dissonance=0.4462
- tick 4: apple: textC*=59.526, textState=8.9157, visionC*=11.8292, audioC*=0.0 | banana: textC*=28.7442, textState=8.4844, visionC*=0.0, audioC*=0.0 / surprise=0.6971 / dissonance=0.4133
- tick 5: apple: textC*=47.2215, textState=10.5241, visionC*=17.9231, audioC*=0.0 | banana: textC*=11.2094, textState=8.9921, visionC*=3.0562, audioC*=0.0 / surprise=0.6531 / dissonance=0.5041

结论：`banana apple` 这一顺序里，state text 仍然是强双活化，secondary_ratio=0.8544；但 text C* 本身只到 secondary_ratio=0.2374，刚好低于这份报告里“明显双活化”的阅读阈值。
视觉 C* 里两个对象也都还有能量，但次级只到 secondary_ratio=0.1705，比 `apple banana` 更弱。
音频侧双活化仍然偏弱：最终 audio C* 主导=``，次级 ratio=0.0。

### 4.3 对实验 A 的判断

1. “同时想着两样东西”这件事已经成立，但目前最稳的是文本层和视觉层，听觉层还不够稳。
2. 顺序会强烈影响主导对象，说明当前系统不仅会叠加，还会保留顺序偏置。
3. 从这轮正式数据看，`apple banana` 反而比 `banana apple` 更容易出现你想要的“双轮廓叠加”。

## 5. 实验 B：黄色苹果的颜色迁移

黄色苹果 probe 使用的是“苹果轮廓 + 香蕉色相”的图像，只输入视觉，不输入文本和音频。

- tick 0: apple: textC*=54.8199, textState=1.5, visionC*=32.3891, audioC*=1.5402 | banana: textC*=0.0, textState=0.0, visionC*=0.0, audioC*=0.0 / surprise=1.0 / dissonance=1.0
- tick 1: apple: textC*=20.005, textState=2.85, visionC*=10.2572, audioC*=0.6512 | banana: textC*=0.0, textState=0.0, visionC*=0.0, audioC*=0.0 / surprise=0.9482 / dissonance=0.9749
- tick 2: apple: textC*=13.654, textState=3.1827, visionC*=0.0, audioC*=0.0 | banana: textC*=33.1581, textState=1.5, visionC*=6.0224, audioC*=0.0 / surprise=0.7942 / dissonance=0.8897
- tick 3: apple: textC*=34.3026, textState=4.3644, visionC*=22.7162, audioC*=0.0 | banana: textC*=13.9184, textState=1.9586, visionC*=2.8443, audioC*=0.0 / surprise=0.6833 / dissonance=0.8283
- tick 4: apple: textC*=27.0965, textState=5.428, visionC*=16.7269, audioC*=0.0 | banana: textC*=8.466, textState=2.2314, visionC*=1.6326, audioC*=0.0 / surprise=0.6035 / dissonance=0.7841
- tick 5: apple: textC*=6.1704, textState=4.8852, visionC*=1.7484, audioC*=0.0 | banana: textC*=30.8356, textState=3.5083, visionC*=13.7791, audioC*=0.0 / surprise=0.546 / dissonance=0.7522

- 首 tick 低层视觉标签重合：apple overlap=1（contour=1），banana overlap=0（contour=0）
- 最终 recall：text C* 主导=`banana` / 次级=`apple` / secondary_ratio=0.2001
- 最终视觉 recall：vision C* 主导=`banana` / 次级=`apple` / secondary_ratio=0.1269

这组结果很清楚：
1. 首 tick 的低层轮廓重合 actually 只对苹果成立，说明轮廓通道本身并没有把黄色苹果看成香蕉。
2. 但随着连续几个 tick 的整合，最终高层 recall 反而翻成了香蕉主导。
3. 这说明问题不在“轮廓没提出来”，而在后续 recall / competition 过程中，黄色带来的颜色相似性、已有香蕉记忆优势，压过了苹果轮廓锚点。

也就是说，如果你的预期是“黄色苹果应该以苹果为主，只稍微想到一点香蕉”，那当前版本没有做到，甚至会在后续整合里翻错到香蕉主导。

## 6. 对你关心问题的直接回答

### 6.1 轮廓重建能不能两个轮廓都有

能，但不是所有顺序都一样强。
`apple banana` 这组里，视觉 C* 已经出现两个对象同时有正能量，而且次级占比过了 0.25，所以按现在前端的能量叠加逻辑，应该会出现“双轮廓同场、主次明暗不同”的效果。
`banana apple` 也不是完全没有第二对象，但视觉次级更弱，更像隐约叠上了一层较淡的轮廓。

### 6.2 黄色苹果会不会苹果为主、香蕉略微被带起

这次正式实验里，并不是“苹果主、香蕉副”，而是最后会逐步翻成“香蕉主、苹果副”。
更准确地说，是“首 tick 的轮廓判断偏苹果，但后续 recall 整合把结果推向了香蕉”。

## 7. Showcase 回看路径

- 输出目录：[overlay_color_probe](H:\AP原型实验第二期\outputs\multimodal_overlay_color_probe\20260524_overlay_color_v1)
- 展示 dataset：[showcase_dataset.json](H:\AP原型实验第二期\outputs\multimodal_overlay_color_probe\20260524_overlay_color_v1\showcase_dataset.json)
- 展示 run 目录：[phase11_multimodal_run_20260524_132837_792](H:\AP原型实验第二期\outputs\multimodal_overlay_color_probe\20260524_overlay_color_v1\observatory_showcase\runs\phase11_multimodal_run_20260524_132837_792)
- sidecar / summary tick 数：78 / 78

建议重点回看这些 tick 段：
- `train::apple`：tick 0 -> 9
- `idle_after_train::apple`：tick 10 -> 13
- `train::banana`：tick 14 -> 23
- `idle_after_train::banana`：tick 24 -> 27
- `probe::text::apple`：tick 28 -> 33
- `idle::apple`：tick 34 -> 37
- `probe::text::banana`：tick 38 -> 43
- `idle::banana`：tick 44 -> 47
- `probe::text::apple_banana`：tick 48 -> 53
- `idle::apple_banana`：tick 54 -> 57
- `probe::text::banana_apple`：tick 58 -> 63
- `idle::banana_apple`：tick 64 -> 67
- `probe::vision::yellow_apple`：tick 68 -> 73
- `idle::yellow_apple`：tick 74 -> 77

如果你在前端里看“内心展示”，最值得盯的段落是：
1. `probe::text::apple_banana`：看是不是主要苹果、但仍有香蕉痕迹。
2. `probe::text::banana_apple`：看是不是更明显地出现双轮廓叠加。
3. `probe::vision::yellow_apple`：看是不是苹果主导，同时几乎没有稳定香蕉副本。

## 8. 当前结论

> 这次实验最明确的结论是：AP V2 已经具备“多概念同时激活”的雏形，而且在 `apple banana` 这类顺序下，视觉层确实能进入双轮廓叠加态；但颜色迁移目前不是“弱副联想”，而是会把黄色苹果逐步推成香蕉主导，说明颜色/记忆偏置在后续整合里压过了轮廓锚点。

如果后面你希望“黄色苹果 -> 以苹果为主、稍微想到香蕉”更稳定成立，比较自然的改进方向不是硬调规则，而是让轮廓通道在多 tick 整合中的稳定权重更高，同时把颜色/材质通道保留为次级相似来源，而不是让它在后续竞争里反客为主。
