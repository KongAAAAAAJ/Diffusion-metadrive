from expert_dataset import run_expert


def test_parse_args_accepts_expert_idm_overrides():
    args = run_expert.parse_args(
        [
            "--expert-idm-distance-wanted", "7.5",
            "--expert-idm-time-wanted", "1.2",
            "--expert-idm-enable-lane-change", "0",
            "--expert-idm-lane-change-freq", "17",
            "--expert-idm-heading-pid-kp", "2.8",
            "--expert-idm-lateral-pid-kd", "0.12",
        ]
    )

    assert args.expert_idm_distance_wanted == 7.5
    assert args.expert_idm_time_wanted == 1.2
    assert args.expert_idm_enable_lane_change == 0
    assert args.expert_idm_lane_change_freq == 17
    assert args.expert_idm_heading_pid_kp == 2.8
    assert args.expert_idm_lateral_pid_kd == 0.12
