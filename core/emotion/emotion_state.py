"""
情绪状态表示 - 8 通道拟人 NT 系统

基于 PA 原型的 8 通道神经递质设计：
- DA (Dopamine): 奖励驱动
- ADR (Adrenaline): 警觉度
- OXY (Oxytocin): 联结感
- SER (Serotonin): 稳定感
- END (Endorphin): 痛感缓解
- COR (Cortisol): 警戒度
- NOV (Novelty): 新奇探索
- FOC (Focus): 焦点锁定

每个通道有：
- baseline: 基线值（通道的"性格基线"）
- decay_ratio: 衰减比例（向基线回归的速度）
- soft_cap_k: 软饱和参数（防止通道打满）
"""

from __future__ import annotations
import math


# 8 通道元数据（来自 PA 原型）
NT_CHANNEL_META = {
    "DA": {
        "label": "多巴胺(奖励驱动)",
        "baseline": 0.12,
        "decay_ratio": 0.91,
        "soft_cap_k": 0.35,
        "max_value": 1.0,
    },
    "ADR": {
        "label": "肾上腺素(警觉)",
        "baseline": 0.05,
        "decay_ratio": 0.85,
        "soft_cap_k": 0.35,
        "max_value": 1.0,
    },
    "OXY": {
        "label": "催产素(联结)",
        "baseline": 0.12,
        "decay_ratio": 0.93,
        "soft_cap_k": 0.35,
        "max_value": 1.0,
    },
    "SER": {
        "label": "血清素(稳定)",
        "baseline": 0.18,
        "decay_ratio": 0.94,
        "soft_cap_k": 0.35,
        "max_value": 1.0,
    },
    "END": {
        "label": "内啡肽(痛感缓解)",
        "baseline": 0.10,
        "decay_ratio": 0.91,
        "soft_cap_k": 0.35,
        "max_value": 1.0,
    },
    "COR": {
        "label": "皮质醇(警戒)",
        "baseline": 0.06,
        "decay_ratio": 0.86,
        "soft_cap_k": 0.35,
        "max_value": 1.0,
    },
    "NOV": {
        "label": "新奇探索",
        "baseline": 0.08,
        "decay_ratio": 0.89,
        "soft_cap_k": 0.33,
        "max_value": 1.0,
    },
    "FOC": {
        "label": "焦点锁定",
        "baseline": 0.10,
        "decay_ratio": 0.92,
        "soft_cap_k": 0.34,
        "max_value": 1.0,
    },
}


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class EmotionState:
    """8 通道 NT 情绪状态"""

    def __init__(self) -> None:
        # 初始化所有通道为基线值
        self.channels: dict[str, float] = {
            channel_id: meta["baseline"]
            for channel_id, meta in NT_CHANNEL_META.items()
        }

    def apply_decay(self) -> dict:
        """
        衰减：所有通道向基线回归

        公式：decayed = baseline + (current - baseline) * decay_ratio

        这不是向 0 衰减，而是向"性格基线"回归
        """
        decay_trace = {}
        for channel_id, current_value in self.channels.items():
            meta = NT_CHANNEL_META[channel_id]
            baseline = meta["baseline"]
            decay_ratio = meta["decay_ratio"]

            decayed = baseline + (current_value - baseline) * decay_ratio
            delta = decayed - current_value

            self.channels[channel_id] = _clamp(decayed, 0.0, meta["max_value"])

            decay_trace[channel_id] = {
                "before": _round4(current_value),
                "after": _round4(self.channels[channel_id]),
                "delta": _round4(delta),
                "baseline": baseline,
                "decay_ratio": decay_ratio,
            }

        return decay_trace

    def apply_delta(self, channel_id: str, delta: float) -> dict:
        """
        应用增量到指定通道

        - 负增量：直接减少（允许低于基线）
        - 正增量：使用软饱和公式防止打满

        软饱和公式：after = max - gap * exp(-delta / k)
        其中 gap = max - current
        """
        if channel_id not in self.channels:
            return {"error": f"Unknown channel: {channel_id}"}

        meta = NT_CHANNEL_META[channel_id]
        current = self.channels[channel_id]
        max_value = meta["max_value"]
        soft_cap_k = meta["soft_cap_k"]

        if delta < 0:
            # 负增量：直接减少
            new_value = _clamp(current + delta, 0.0, max_value)
            actual_delta = new_value - current
        else:
            # 正增量：软饱和
            gap = max_value - current
            if gap < 0.0001:
                # 已经接近上限，几乎不增长
                new_value = current
                actual_delta = 0.0
            else:
                new_value = max_value - gap * math.exp(-delta / soft_cap_k)
                new_value = _clamp(new_value, 0.0, max_value)
                actual_delta = new_value - current

        self.channels[channel_id] = new_value

        return {
            "channel": channel_id,
            "before": _round4(current),
            "after": _round4(new_value),
            "requested_delta": _round4(delta),
            "actual_delta": _round4(actual_delta),
            "soft_capped": delta > 0,
        }

    def get_state(self) -> dict[str, float]:
        """获取当前所有通道的状态"""
        return {
            channel_id: _round4(value)
            for channel_id, value in self.channels.items()
        }

    def get_channel(self, channel_id: str) -> float:
        """获取单个通道的当前值"""
        return self.channels.get(channel_id, 0.0)
