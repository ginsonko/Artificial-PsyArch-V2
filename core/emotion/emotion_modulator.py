"""
情绪调制器 - 8 通道 NT 系统的更新与调制逻辑

主要功能：
1. 根据 CFS (认知感受信号) 更新 NT 通道
2. 根据 Rwd/Pun (奖惩信号) 更新 NT 通道
3. 输出调制参数影响注意力、学习、行动系统
"""

from __future__ import annotations
from .emotion_state import EmotionState, NT_CHANNEL_META


def _round4(value: float) -> float:
    return round(float(value), 4)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


class EmotionModulator:
    """8 通道 NT 情绪调制器"""

    def __init__(
        self,
        *,
        cfs_gain: float = 1.0,
        rwd_pun_gain: float = 1.0,
    ) -> None:
        self.state = EmotionState()
        self.cfs_gain = max(0.0, float(cfs_gain))
        self.rwd_pun_gain = max(0.0, float(rwd_pun_gain))

    def update(
        self,
        *,
        cognitive_feelings: dict,
        reward: float = 0.0,
        punishment: float = 0.0,
        innate_deltas: dict | None = None,
    ) -> dict:
        """
        主更新流程：
        1. 衰减（向基线回归）
        2. CFS 信号映射到 NT 增量
        3. Rwd/Pun 信号映射到 NT 增量
        4. 应用所有增量
        """
        trace = {}

        # Step 1: 衰减
        decay_trace = self.state.apply_decay()
        trace["decay"] = decay_trace

        # Step 2: CFS → NT 映射
        cfs_deltas = self._compute_deltas_from_cfs(cognitive_feelings)
        trace["cfs_deltas"] = cfs_deltas

        # Step 3: Rwd/Pun → NT 映射
        rwd_pun_deltas = self._compute_deltas_from_rwd_pun(reward, punishment)
        trace["rwd_pun_deltas"] = rwd_pun_deltas
        innate_emotion_deltas = {
            str(channel): _round4(float(value or 0.0))
            for channel, value in dict(innate_deltas or {}).items()
            if str(channel) in NT_CHANNEL_META
        }
        trace["innate_deltas"] = innate_emotion_deltas

        # Step 4: 应用所有增量
        apply_trace = []
        all_deltas = {}

        # 合并 CFS 和 Rwd/Pun 增量
        for channel_id in NT_CHANNEL_META.keys():
            total_delta = (
                cfs_deltas.get(channel_id, 0.0) + rwd_pun_deltas.get(channel_id, 0.0)
                + innate_emotion_deltas.get(channel_id, 0.0)
            )
            all_deltas[channel_id] = total_delta

            if abs(total_delta) > 0.0001:
                apply_result = self.state.apply_delta(channel_id, total_delta)
                apply_trace.append(apply_result)

        trace["apply"] = apply_trace
        trace["emotion_state"] = self.state.get_state()

        return trace

    def get_modulation(self) -> dict:
        """
        输出调制参数

        基于当前 NT 状态，生成影响其他系统的调制参数：
        - attention_modulation: 影响注意力选择
        - hdb_modulation: 影响记忆学习
        - action_modulation: 影响行动规划
        """
        da = self.state.get_channel("DA")
        adr = self.state.get_channel("ADR")
        oxy = self.state.get_channel("OXY")
        ser = self.state.get_channel("SER")
        end = self.state.get_channel("END")
        cor = self.state.get_channel("COR")
        nov = self.state.get_channel("NOV")
        foc = self.state.get_channel("FOC")

        # 注意力调制
        attention_modulation = {
            # 注意力资源倍数：DA 和 FOC 提升，COR 降低
            "resource_multiplier": _round4(
                1.0 + da * 0.35 + foc * 0.28 - cor * 0.18
            ),
            # Top-N 调整：FOC 提升时减少 top-N（更聚焦），NOV 提升时增加 top-N（更分散）
            "top_n_adjustment": int((nov * 0.5 - foc * 0.4) * 10),
            # 阈值调整：SER 降低阈值（更容易选中），COR 提升阈值（更谨慎）
            "threshold_adjustment": _round4(cor * 0.12 - ser * 0.08),
        }

        # HDB 学习调制
        hdb_modulation = {
            # 学习率倍数：DA 和 NOV 提升学习率
            "learning_rate_multiplier": _round4(1.0 + da * 0.25 + nov * 0.20),
            # 传播强度：OXY 提升关联传播
            "propagation_multiplier": _round4(1.0 + oxy * 0.30),
            # 遗忘率：COR 降低遗忘（警戒时记得更牢）
            "forgetting_multiplier": _round4(1.0 - cor * 0.15),
        }

        # 行动调制
        action_modulation = {
            # 行动阈值调整：ADR 和 COR 降低阈值（更容易行动），SER 提升阈值（更稳定）
            "threshold_adjustment": _round4(
                -adr * 0.15 - cor * 0.12 + ser * 0.10
            ),
            # 奖惩增益倍数：DA 和 END 提升奖励敏感度
            "reward_gain_multiplier": _round4(1.0 + da * 0.28 + end * 0.18),
            # 探索偏置：NOV 提升探索倾向
            "exploration_bias": _round4(nov * 0.45),
        }

        return {
            "attention": attention_modulation,
            "hdb": hdb_modulation,
            "action": action_modulation,
        }

    def _compute_deltas_from_cfs(self, cognitive_feelings: dict) -> dict[str, float]:
        """
        CFS → NT 映射

        基于 PA 原型的映射规则：
        - 惊异 (surprise) → DA↑, ADR↑, NOV↑
        - 违和感 (dissonance) → COR↑, ADR↑, SER↓
        - 压力 (pressure) → COR↑, ADR↑
        - 期待 (expectation) → DA↑, FOC↑
        - 正确感 (correctness) → SER↑, DA↑
        - 把握感 (grasp) → SER↑, OXY↑, FOC↑
        - 流畅感 (fluency) → SER↑, DA↑
        """
        channels = cognitive_feelings.get("channels", {}) or {}

        surprise = float(channels.get("surprise", 0.0) or 0.0)
        dissonance = float(channels.get("dissonance", 0.0) or 0.0)
        pressure = float(channels.get("pressure", 0.0) or 0.0)
        expectation = float(channels.get("expectation", 0.0) or 0.0)
        correctness = float(channels.get("correctness", 0.0) or 0.0)
        grasp = float(channels.get("grasp", 0.0) or 0.0)
        fluency = float(channels.get("fluency", 0.0) or 0.0)
        boredom = float(channels.get("boredom", 0.0) or 0.0)
        fulfillment = float(channels.get("fulfillment", 0.0) or 0.0)

        deltas = {}

        # DA: 惊异、期待、正确感、流畅感
        deltas["DA"] = (
            surprise * 0.18 + expectation * 0.15 + correctness * 0.12 + fluency * 0.14
            + boredom * 0.06 + fulfillment * 0.10
        ) * self.cfs_gain

        # ADR: 惊异、违和感、压力
        deltas["ADR"] = (
            surprise * 0.16 + dissonance * 0.20 + pressure * 0.22
        ) * self.cfs_gain

        # OXY: 把握感
        deltas["OXY"] = (grasp * 0.18 + fulfillment * 0.04) * self.cfs_gain

        # SER: 正确感、把握感、流畅感，违和感降低
        deltas["SER"] = (
            correctness * 0.16 + grasp * 0.20 + fluency * 0.15 + fulfillment * 0.14
            - dissonance * 0.12 - boredom * 0.05
        ) * self.cfs_gain

        # END: 暂无直接 CFS 映射（主要由 Rwd/Pun 驱动）
        deltas["END"] = 0.0

        # COR: 违和感、压力
        deltas["COR"] = (dissonance * 0.22 + pressure * 0.24) * self.cfs_gain

        # NOV: 惊异
        deltas["NOV"] = (surprise * 0.20 + boredom * 0.18 - fulfillment * 0.04) * self.cfs_gain

        # FOC: 期待、把握感
        deltas["FOC"] = (expectation * 0.16 + grasp * 0.18 + fulfillment * 0.12 - boredom * 0.04) * self.cfs_gain

        return {k: _round4(v) for k, v in deltas.items()}

    def _compute_deltas_from_rwd_pun(
        self, reward: float, punishment: float
    ) -> dict[str, float]:
        """
        Rwd/Pun → NT 映射

        基于 PA 原型的映射规则：
        - 奖励 (reward) → DA↑, END↑, SER↑
        - 惩罚 (punishment) → COR↑, ADR↑, SER↓, DA↓
        """
        reward = max(0.0, float(reward))
        punishment = max(0.0, float(punishment))

        deltas = {}

        # DA: 奖励提升，惩罚降低
        deltas["DA"] = (reward * 0.28 - punishment * 0.18) * self.rwd_pun_gain

        # ADR: 惩罚提升
        deltas["ADR"] = (punishment * 0.22) * self.rwd_pun_gain

        # OXY: 奖励轻微提升
        deltas["OXY"] = (reward * 0.12) * self.rwd_pun_gain

        # SER: 奖励提升，惩罚降低
        deltas["SER"] = (reward * 0.20 - punishment * 0.16) * self.rwd_pun_gain

        # END: 奖励提升（痛感缓解）
        deltas["END"] = (reward * 0.24) * self.rwd_pun_gain

        # COR: 惩罚提升
        deltas["COR"] = (punishment * 0.26) * self.rwd_pun_gain

        # NOV: 暂无直接 Rwd/Pun 映射
        deltas["NOV"] = 0.0

        # FOC: 暂无直接 Rwd/Pun 映射
        deltas["FOC"] = 0.0

        return {k: _round4(v) for k, v in deltas.items()}

    def get_explainability(self) -> dict:
        """
        白箱解释：当前 NT 状态和调制参数的详细说明
        """
        modulation = self.get_modulation()
        state = self.state.get_state()

        return {
            "state": state,
            "state_labels": {
                channel_id: NT_CHANNEL_META[channel_id]["label"]
                for channel_id in state.keys()
            },
            "modulation": modulation,
            "modulation_explanation": {
                "attention": {
                    "resource_multiplier": f"DA({state['DA']:.2f})*0.35 + FOC({state['FOC']:.2f})*0.28 - COR({state['COR']:.2f})*0.18",
                    "top_n_adjustment": f"NOV({state['NOV']:.2f})*0.5 - FOC({state['FOC']:.2f})*0.4",
                    "threshold_adjustment": f"COR({state['COR']:.2f})*0.12 - SER({state['SER']:.2f})*0.08",
                },
                "hdb": {
                    "learning_rate_multiplier": f"DA({state['DA']:.2f})*0.25 + NOV({state['NOV']:.2f})*0.20",
                    "propagation_multiplier": f"OXY({state['OXY']:.2f})*0.30",
                    "forgetting_multiplier": f"1.0 - COR({state['COR']:.2f})*0.15",
                },
                "action": {
                    "threshold_adjustment": f"-ADR({state['ADR']:.2f})*0.15 - COR({state['COR']:.2f})*0.12 + SER({state['SER']:.2f})*0.10",
                    "reward_gain_multiplier": f"DA({state['DA']:.2f})*0.28 + END({state['END']:.2f})*0.18",
                    "exploration_bias": f"NOV({state['NOV']:.2f})*0.45",
                },
            },
        }
