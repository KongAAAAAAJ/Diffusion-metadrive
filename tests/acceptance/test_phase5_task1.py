from __future__ import annotations

from evaluation.reward_terms import compute_step_reward


def test_reward_contract():
    config = {
        "delta_s_max": 5.0,
        "d_norm": 10.0,
        "d_safe": 8.0,
        "w_progress": 1.0,
        "w_formation": 0.5,
        "w_safety": 0.3,
        "w_collision": 10.0,
        "w_road": 5.0,
        "w_comfort": 0.1,
    }

    base_info = {
        "progress": 2.0,
        "formation_error": 1.0,
        "min_gap": 10.0,
        "jerk": 0.1,
        "delta_steering": 0.05,
        "crash": False,
        "out_of_road": False,
    }

    reward = compute_step_reward(base_info, config)
    assert isinstance(reward, float)
    assert reward > 0.0

    crash_reward = compute_step_reward({**base_info, "crash": True}, config)
    assert crash_reward <= -10.0

    low_error = compute_step_reward({**base_info, "formation_error": 1.0}, config)
    high_error = compute_step_reward({**base_info, "formation_error": 5.0}, config)
    assert low_error > high_error

    overridden = compute_step_reward(base_info, {**config, "w_progress": 2.0})
    assert overridden != reward

    missing_key_reward = compute_step_reward({"progress": 1.0}, config)
    assert isinstance(missing_key_reward, float)

