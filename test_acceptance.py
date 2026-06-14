"""
简单验收测试：验证 8 通道 NT 系统没有破坏基本功能
"""

import sys
sys.path.insert(0, "H:\\AP原型实验第二期\\APV2.1版本原型测试")

from core.runtime.engine import APV21Runtime


def test_basic_text_processing():
    """测试基本文本处理"""
    print("\n=== 基本文本处理测试 ===")
    runtime = APV21Runtime()

    # 处理几个 tick
    texts = ["你好", "世界", "测试", "系统"]

    for i, text in enumerate(texts):
        result = runtime.process_text_tick(text)

        # 验证基本结构
        assert "state_pool" in result, f"Tick {i}: 缺少 state_pool"
        assert "attention" in result, f"Tick {i}: 缺少 attention"
        assert "action" in result, f"Tick {i}: 缺少 action"
        assert "emotion" in result, f"Tick {i}: 缺少 emotion"
        assert "explainability" in result, f"Tick {i}: 缺少 explainability"

        # 验证情绪结构
        emotion = result["emotion"]
        assert "update" in emotion, f"Tick {i}: 缺少 emotion.update"
        assert "modulation" in emotion, f"Tick {i}: 缺少 emotion.modulation"

        # 验证情绪状态有 8 个通道
        emotion_state = emotion["update"]["emotion_state"]
        expected_channels = {"DA", "ADR", "OXY", "SER", "END", "COR", "NOV", "FOC"}
        assert set(emotion_state.keys()) == expected_channels, \
            f"Tick {i}: 情绪通道不完整"

        # 验证调制参数结构
        modulation = emotion["modulation"]
        assert "attention" in modulation, f"Tick {i}: 缺少 attention 调制"
        assert "hdb" in modulation, f"Tick {i}: 缺少 hdb 调制"
        assert "action" in modulation, f"Tick {i}: 缺少 action 调制"

        print(f"Tick {i} ({text}): OK")

    print("\n[OK] 基本文本处理正常")


def test_empty_tick_continuation():
    """测试空 tick 延续"""
    print("\n=== 空 tick 延续测试 ===")
    runtime = APV21Runtime()

    # 先输入一些文本
    runtime.process_text_tick("初始化")
    runtime.process_text_tick("测试文本")

    # 然后空 tick
    for i in range(3):
        result = runtime.process_text_tick("")

        # 验证系统仍然运行
        assert "state_pool" in result, f"空 tick {i}: 系统未运行"
        assert "emotion" in result, f"空 tick {i}: 情绪系统未运行"

        # 验证情绪状态仍然存在
        emotion_state = result["emotion"]["update"]["emotion_state"]
        assert len(emotion_state) == 8, f"空 tick {i}: 情绪通道丢失"

        print(f"空 tick {i}: OK")

    print("\n[OK] 空 tick 延续正常")


def test_emotion_modulation_integration():
    """测试情绪调制集成"""
    print("\n=== 情绪调制集成测试 ===")
    runtime = APV21Runtime()

    # 多个 tick 累积情绪
    for i in range(5):
        result = runtime.process_text_tick(f"测试 {i}")

    # 检查调制参数是否传递到各个系统
    explainability = result["explainability"]

    # 检查情绪白箱解释
    emotion_explain = explainability.get("emotion", {})
    assert "state" in emotion_explain, "缺少情绪状态解释"
    assert "cfs_deltas" in emotion_explain, "缺少 CFS 增量解释"
    assert "modulation" in emotion_explain, "缺少调制参数解释"

    # 检查调制参数结构
    modulation = emotion_explain["modulation"]
    assert "attention" in modulation, "缺少注意力调制"
    assert "hdb" in modulation, "缺少 HDB 调制"
    assert "action" in modulation, "缺少行动调制"

    print("情绪状态:", list(emotion_explain["state"].keys()))
    print("调制参数:", list(modulation.keys()))

    print("\n[OK] 情绪调制集成正常")


if __name__ == "__main__":
    print("=" * 60)
    print("8 通道 NT 系统验收测试")
    print("=" * 60)

    try:
        test_basic_text_processing()
        print("\n" + "=" * 60)

        test_empty_tick_continuation()
        print("\n" + "=" * 60)

        test_emotion_modulation_integration()
        print("\n" + "=" * 60)

        print("\n[SUCCESS] 所有验收测试通过！")
        print("8 通道 NT 系统已成功集成，未破坏现有功能。")

    except Exception as e:
        print(f"\n[FAILED] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
