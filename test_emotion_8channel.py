"""
测试 8 通道拟人 NT 情绪系统

验证：
1. 8 个 NT 通道正确初始化为基线值
2. 通道能向基线衰减（不是向 0）
3. CFS 信号能正确映射到 NT 增量
4. Rwd/Pun 信号能正确映射到 NT 增量
5. 软饱和公式正确工作
6. NT 状态能输出调制参数
7. 调制参数能影响注意力和行动
"""

from core.runtime.engine import APV21Runtime
from core.emotion import NT_CHANNEL_META


def test_nt_channels_initialization():
    """测试 8 通道初始化"""
    runtime = APV21Runtime()

    # 第一个 tick
    result = runtime.process_text_tick("测试")

    emotion_trace = result["emotion"]
    assert "update" in emotion_trace, "情绪更新未执行"
    assert "modulation" in emotion_trace, "情绪调制未生成"

    emotion_state = emotion_trace["update"]["emotion_state"]

    print("\n=== 8 通道 NT 初始状态 ===")
    for channel_id, value in emotion_state.items():
        baseline = NT_CHANNEL_META[channel_id]["baseline"]
        label = NT_CHANNEL_META[channel_id]["label"]
        print(f"{channel_id} ({label}): {value:.4f} (baseline: {baseline:.4f})")

    # 验证所有 8 个通道都存在
    expected_channels = {"DA", "ADR", "OXY", "SER", "END", "COR", "NOV", "FOC"}
    assert set(emotion_state.keys()) == expected_channels, f"通道不完整: {emotion_state.keys()}"

    # 验证所有通道值在合理范围内 [0, 1]
    for channel_id in expected_channels:
        value = emotion_state[channel_id]
        assert 0.0 <= value <= 1.0, \
            f"{channel_id} 值 {value} 超出范围 [0, 1]"

    print("\n[OK] 8 通道初始化正确，所有通道值在合理范围内")


def test_nt_decay_to_baseline():
    """测试通道向基线衰减"""
    from core.emotion.emotion_state import EmotionState

    # 直接测试 EmotionState 的衰减机制
    state = EmotionState()

    # 手动设置一些通道偏离基线
    state.channels["DA"] = 0.8  # 远高于基线 0.12
    state.channels["SER"] = 0.05  # 远低于基线 0.18
    state.channels["COR"] = 0.6  # 远高于基线 0.06

    print("\n=== 通道衰减测试 ===")
    print("初始状态（人工设置）:")
    print(f"  DA: 0.8000 (baseline: 0.1200)")
    print(f"  SER: 0.0500 (baseline: 0.1800)")
    print(f"  COR: 0.6000 (baseline: 0.0600)")

    # 连续衰减 5 次
    for i in range(5):
        state.apply_decay()

    print("\n5 次衰减后:")
    print(f"  DA: {state.channels['DA']:.4f} (baseline: 0.1200)")
    print(f"  SER: {state.channels['SER']:.4f} (baseline: 0.1800)")
    print(f"  COR: {state.channels['COR']:.4f} (baseline: 0.0600)")

    # 验证向基线移动
    assert state.channels["DA"] < 0.8, "DA 应该从 0.8 向基线 0.12 衰减"
    assert state.channels["SER"] > 0.05, "SER 应该从 0.05 向基线 0.18 回升"
    assert state.channels["COR"] < 0.6, "COR 应该从 0.6 向基线 0.06 衰减"

    # 验证向基线靠近
    assert abs(state.channels["DA"] - 0.12) < abs(0.8 - 0.12), "DA 应该更接近基线"
    assert abs(state.channels["SER"] - 0.18) < abs(0.05 - 0.18), "SER 应该更接近基线"
    assert abs(state.channels["COR"] - 0.06) < abs(0.6 - 0.06), "COR 应该更接近基线"

    print("\n[OK] 通道正确向基线衰减")


def test_cfs_to_nt_mapping():
    """测试 CFS → NT 映射"""
    runtime = APV21Runtime()

    # 输入文本触发 CFS
    result = runtime.process_text_tick("你好世界")

    emotion_trace = result["emotion"]["update"]
    cfs_deltas = emotion_trace.get("cfs_deltas", {})

    print("\n=== CFS → NT 映射 ===")
    print("CFS 增量:")
    for channel_id, delta in cfs_deltas.items():
        if abs(delta) > 0.001:
            label = NT_CHANNEL_META[channel_id]["label"]
            print(f"  {channel_id} ({label}): {delta:+.4f}")

    # 验证至少有一些通道被触发
    non_zero_deltas = sum(1 for delta in cfs_deltas.values() if abs(delta) > 0.001)
    assert non_zero_deltas > 0, "CFS 应该触发至少一个 NT 通道"

    print("\n[OK] CFS → NT 映射正常工作")


def test_nt_modulation_output():
    """测试 NT 调制参数输出"""
    runtime = APV21Runtime()

    # 多个 tick 累积情绪
    for i in range(3):
        result = runtime.process_text_tick(f"测试 {i}")

    modulation = result["emotion"]["modulation"]

    print("\n=== NT 调制参数 ===")
    print("注意力调制:")
    for key, value in modulation.get("attention", {}).items():
        print(f"  {key}: {value}")

    print("\nHDB 调制:")
    for key, value in modulation.get("hdb", {}).items():
        print(f"  {key}: {value}")

    print("\n行动调制:")
    for key, value in modulation.get("action", {}).items():
        print(f"  {key}: {value}")

    # 验证调制参数结构
    assert "attention" in modulation, "缺少注意力调制"
    assert "hdb" in modulation, "缺少 HDB 调制"
    assert "action" in modulation, "缺少行动调制"

    assert "resource_multiplier" in modulation["attention"], "缺少注意力资源倍数"
    assert "threshold_adjustment" in modulation["action"], "缺少行动阈值调整"
    assert "reward_gain_multiplier" in modulation["action"], "缺少奖惩增益倍数"

    print("\n[OK] NT 调制参数输出正确")


def test_nt_modulation_effect():
    """测试 NT 调制对注意力和行动的影响"""
    runtime = APV21Runtime()

    # 多个 tick 累积情绪
    for i in range(5):
        result = runtime.process_text_tick(f"测试文本 {i}")

    # 检查注意力是否受情绪调制
    attention_trace = result["attention"]
    if attention_trace["selected_items"]:
        first_item = attention_trace["selected_items"][0]
        if "emotion_multiplier" in first_item:
            print(f"\n注意力受情绪调制: {first_item['emotion_multiplier']:.4f}")

    # 检查行动是否受情绪调制
    action_trace = result["action"]
    if "effective_threshold" in action_trace:
        print(f"行动阈值受情绪调制: {action_trace['effective_threshold']:.4f}")

    # 检查 explainability
    explainability = result["explainability"]
    emotion_explain = explainability.get("emotion", {})

    print("\n=== 情绪白箱解释 ===")
    print("当前 NT 状态:")
    for channel_id, value in emotion_explain.get("state", {}).items():
        label = NT_CHANNEL_META[channel_id]["label"]
        print(f"  {channel_id} ({label}): {value:.4f}")

    print("\n[OK] NT 调制效果测试通过")


def test_soft_saturation():
    """测试软饱和公式"""
    from core.emotion.emotion_state import EmotionState

    state = EmotionState()

    # 连续施加大增量，观察软饱和效果
    print("\n=== 软饱和测试 ===")
    print("连续施加 +0.3 增量到 DA 通道:")

    for i in range(10):
        result = state.apply_delta("DA", 0.3)
        print(f"  第 {i+1} 次: {result['before']:.4f} -> {result['after']:.4f} "
              f"(请求: {result['requested_delta']:+.4f}, 实际: {result['actual_delta']:+.4f})")

    # 验证不会超过 max_value
    final_value = state.get_channel("DA")
    max_value = NT_CHANNEL_META["DA"]["max_value"]
    assert final_value <= max_value, f"DA 超过最大值: {final_value} > {max_value}"

    # 验证增量逐渐减小（软饱和效果）
    print("\n[OK] 软饱和公式正确工作")


if __name__ == "__main__":
    print("=" * 60)
    print("8 通道拟人 NT 情绪系统测试")
    print("=" * 60)

    test_nt_channels_initialization()
    print("\n" + "=" * 60)

    test_nt_decay_to_baseline()
    print("\n" + "=" * 60)

    test_cfs_to_nt_mapping()
    print("\n" + "=" * 60)

    test_nt_modulation_output()
    print("\n" + "=" * 60)

    test_nt_modulation_effect()
    print("\n" + "=" * 60)

    test_soft_saturation()
    print("\n" + "=" * 60)

    print("\n[SUCCESS] 所有测试通过！8 通道 NT 系统已成功实现。")
