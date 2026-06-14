from __future__ import annotations

from experiments.apv2_emotion_modulation_long_run_dynamics_1 import BASELINES, _probe


def test_emotion_modulation_long_run_dynamics_passes_and_respects_baselines() -> None:
    payload = _probe()
    assert payload["passed"], payload["checks"]

    trajectory = payload["trajectory"]
    assert len(trajectory) == 3

    reward, stress, silence = trajectory

    assert reward["channels"]["DA"] > BASELINES["DA"]
    assert reward["learning_rate_multiplier"] > 1.0

    assert stress["channels"]["COR"] > BASELINES["COR"]
    assert stress["channels"]["DA"] < reward["channels"]["DA"]
    assert stress["action_threshold_adjustment"] < 0.0

    assert abs(silence["channels"]["DA"] - BASELINES["DA"]) < abs(reward["channels"]["DA"] - BASELINES["DA"])
    assert silence["channels"]["COR"] < stress["channels"]["COR"]
    assert reward["attention_resource_multiplier"] >= 1.0
