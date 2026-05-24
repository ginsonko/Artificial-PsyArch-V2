# V2 视觉 OCR-like 预实验报告

## 1. 训练设定
- 训练轮次：12 epoch（每个 epoch 交替呈现两张手写风格数字图像）
- 训练采样：raw=256 / memory=16 / focus=8
- 奖励信号：每个正确图像-文本共现 tick 注入 reward=1.0
- 稳定空 tick：10

## 2. 主实验结论
- `digit_3` -> 目标文本 `three`：BN_top=`three`，C*_top=`three`，margin=115.1551，strict_success=True
- `digit_8` -> 目标文本 `eight`：BN_top=`eight`，C*_top=`eight`，margin=46.4996，strict_success=True

## 3. 接受门槛检查（训练后 1024 raw / 4 tick 冷探测）
- `digit_3`: BN_rank=1 / C*_top=`three` / focus_has_target=False / strict_success=True
- `digit_8`: BN_rank=1 / C*_top=`eight` / focus_has_target=False / strict_success=True

## 4. 采样率-准确率扫描
- raw=256 时，首次达到 strict_accuracy=1.0 所需 observation_ticks=1；best_strict_accuracy=1.0

## 5. 解释边界
- 这个实验若成功，证明的是 AP 视觉稀疏采样特征可以和文本标签建立可召回联结，属于 OCR-like 的初步关联识别。
- 它还不能直接等价于通用 OCR，更不能证明跨字体、跨噪声、跨布局的完整泛化能力。
- 奖励信号已纳入训练流程，但本轮默认没有做无奖励对照，因此证明的是“带奖励的配对训练可行”，不是“奖励是唯一原因”。
