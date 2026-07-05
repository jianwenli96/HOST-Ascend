X2_NORM = {
    "libero": {
        "action_keys": [
            ("relative_right_position", 3),
            ("relative_right_rotation", 3),
            ("follow_right_gripper",    1),
        ],
        # q01 / q99 from libero_128_norm_stats.json; norm_delta = q99 - q01 per dim
        "norm_min": [
            -0.7046250000751599,
            -0.80100000008544,
            -0.9375000001,
            -0.11467779149968735,
            -0.16395000004372,
            -0.2240490058320433,
            -1.0000000001,
        ],
        "norm_delta": [
            1.6417500001751198,
            1.668750000178,
            1.87462500019996,
            0.2464309345988557,
            0.35670000009511997,
            0.5593995055394168,
            1.9996000001999599,
        ],
    },
    "default": {
        "action_keys": [
            ("follow_left_position",  3),
            ("follow_left_rotation",  3),
            ("follow_left_gripper",   1),
            ("follow_right_position", 3),
            ("follow_right_rotation", 3),
            ("follow_right_gripper",  1),
        ],
        "norm_min":   [-0.1, -0.5, -0.5, -1.5, -1.5, -2.0, -0.5, -0.1, -0.5, -0.5, -1.5, -1.5, -2.0, -0.5],
        "norm_delta": [ 0.6,  1.0,  1.0,  3.0,  3.0,  3.5,  6.5,  0.6,  1.0,  1.0,  3.0,  3.0,  3.5,  6.5],
    },
    "AgiBotWorld-Beta-v2": {
        "action_keys": [
            ("relative_left_position",  3),
            ("relative_left_rotation",  3),
            ("follow_left_gripper",     1),
            ("relative_right_position", 3),
            ("relative_right_rotation", 3),
            ("follow_right_gripper",    1),
        ],
        # Stage 2: pos Q99.9=0.013m, rot Q99.9=0.054rad(3.1deg); filter: pos<0.05m, rot<10deg
        # Gripper from abs_constant.py agibotworld_beta: left min=34.6222/delta=86.1921, right min=34.6222/delta=85.7635
        "norm_min":   [-0.05, -0.05, -0.05, -0.20, -0.20, -0.20, 34.6222, -0.05, -0.05, -0.05, -0.20, -0.20, -0.20, 34.6222],
        "norm_delta": [ 0.10,  0.10,  0.10,  0.40,  0.40,  0.40, 86.1921,  0.10,  0.10,  0.10,  0.40,  0.40,  0.40, 85.7635],
    },
    "bridge_data_v2_img1_v2": {
        "action_keys": [
            ("relative_right_position", 3),
            ("relative_right_rotation", 3),
            ("follow_right_gripper",    1),
        ],
        # Gripper from abs_constant.py bridge_data_v2: min=0.0692, delta=0.9426
        "norm_min":   [-0.10, -0.10, -0.10, -0.50, -0.50, -0.50, 0.0692],
        "norm_delta": [ 0.20,  0.20,  0.20,  1.00,  1.00,  1.00, 0.9426],
    },
    "bridge_data_v2_img3_v2": {
        "action_keys": [
            ("relative_right_position", 3),
            ("relative_right_rotation", 3),
            ("follow_right_gripper",    1),
        ],
        # Gripper from abs_constant.py bridge_data_v2: min=0.0692, delta=0.9426
        "norm_min":   [-0.10, -0.10, -0.10, -0.50, -0.50, -0.50, 0.0692],
        "norm_delta": [ 0.20,  0.20,  0.20,  1.00,  1.00,  1.00, 0.9426],
    },
    "bridge_data_v2_img4_v2": {
        "action_keys": [
            ("relative_right_position", 3),
            ("relative_right_rotation", 3),
            ("follow_right_gripper",    1),
        ],
        # Gripper from abs_constant.py bridge_data_v2: min=0.0692, delta=0.9426
        "norm_min":   [-0.10, -0.10, -0.10, -0.50, -0.50, -0.50, 0.0692],
        "norm_delta": [ 0.20,  0.20,  0.20,  1.00,  1.00,  1.00, 0.9426],
    },
    "droid_success_only_v2": {
        "action_keys": [
            ("relative_right_position", 3),
            ("relative_right_rotation", 3),
            ("follow_right_gripper",    1),
        ],
        # Stage 2: pos Q99.9=0.052m, rot Q99.9=5.8deg; Max: pos=0.161m, rot=18.8deg; filter: pos<0.20m, rot<20deg
        # Gripper from abs_constant.py droid: min=0.0, delta=0.9912
        "norm_min":   [-0.20, -0.20, -0.20, -0.35, -0.35, -0.35, 0.0],
        "norm_delta": [ 0.40,  0.40,  0.40,  0.70,  0.70,  0.70, 0.9912],
    },
    "fractal20220817_data_v2": {
        "action_keys": [
            ("relative_right_position", 3),
            ("relative_right_rotation", 3),
            ("follow_right_gripper",    1),
        ],
        # Stage 2: pos Q99.9=0.124m, rot Q99.9=17.6deg(0.306rad); filter: pos<0.30m, rot<25deg
        # Gripper from abs_constant.py fractal: min=0.0, delta=1.0
        "norm_min":   [-0.30, -0.30, -0.30, -0.50, -0.50, -0.50, 0.0],
        "norm_delta": [ 0.60,  0.60,  0.60,  1.00,  1.00,  1.00, 1.0],
    },
    "Robochallenge_x2": {
        "action_keys": [
            ("relative_left_position",  3),
            ("relative_left_rotation",  3),
            ("follow_left_gripper",     1),
            ("relative_right_position", 3),
            ("relative_right_rotation", 3),
            ("follow_right_gripper",    1),
        ],
        # Stage 2: pos Q99.9=0.027m, rot Q99.9=7.0deg; mixed single/dual arm dataset
        # filter: pos<0.05m, rot<12deg; norm aligned to filter thresholds
        # Gripper from abs_constant.py x2_normal: min=-0.5, delta=6.5
        "norm_min":   [-0.05, -0.05, -0.05, -0.20, -0.20, -0.20, -0.5, -0.05, -0.05, -0.05, -0.20, -0.20, -0.20, -0.5],
        "norm_delta": [ 0.10,  0.10,  0.10,  0.40,  0.40,  0.40,  6.5,  0.10,  0.10,  0.10,  0.40,  0.40,  0.40,  6.5],
    },
    "robocoin_v2": {
        "action_keys": [
            ("relative_left_position",  3),
            ("relative_left_rotation",  3),
            ("follow_left_gripper",     1),
            ("relative_right_position", 3),
            ("relative_right_rotation", 3),
            ("follow_right_gripper",    1),
        ],
        # Stage 2: pos Q99=0.022m, rot Q99=4.5deg; extreme outliers in both pos/rot
        # filter: pos<0.05m, rot<10deg; norm aligned to filter thresholds
        # Gripper from abs_constant.py x2_normal: min=-0.5, delta=6.5
        "norm_min":   [-0.05, -0.05, -0.05, -0.20, -0.20, -0.20, -0.5, -0.05, -0.05, -0.05, -0.20, -0.20, -0.20, -0.5],
        "norm_delta": [ 0.10,  0.10,  0.10,  0.40,  0.40,  0.40,  6.5,  0.10,  0.10,  0.10,  0.40,  0.40,  0.40,  6.5],
    },
    "robomind_v2.0_v2": {
        "action_keys": [
            ("relative_left_position",  3),
            ("relative_left_rotation",  3),
            ("follow_left_gripper",     1),
            ("relative_right_position", 3),
            ("relative_right_rotation", 3),
            ("follow_right_gripper",    1),
        ],
        # Stage 2: pos Q99.9=0.064m, rot Q99.9=7.6deg; extreme outliers in both pos/rot
        # filter: pos<0.10m, rot<12deg; norm aligned to filter thresholds
        # Gripper from abs_constant.py x2_normal: min=-0.5, delta=6.5
        "norm_min":   [-0.10, -0.10, -0.10, -0.20, -0.20, -0.20, -0.5, -0.10, -0.10, -0.10, -0.20, -0.20, -0.20, -0.5],
        "norm_delta": [ 0.20,  0.20,  0.20,  0.40,  0.40,  0.40,  6.5,  0.20,  0.20,  0.20,  0.40,  0.40,  0.40,  6.5],
    },
    "robomindv2": {
        "action_keys": [
            ("relative_left_position",  3),
            ("relative_left_rotation",  3),
            ("follow_left_gripper",     1),
            ("relative_right_position", 3),
            ("relative_right_rotation", 3),
            ("follow_right_gripper",    1),
        ],
        # Stage 2: pos Q99.9=0.064m, rot Q99.9=7.6deg; extreme outliers in both pos/rot
        # filter: pos<0.10m, rot<12deg; norm aligned to filter thresholds
        # Gripper from abs_constant.py x2_normal: min=-0.5, delta=6.5
        "norm_min":   [-0.10, -0.10, -0.10, -0.20, -0.20, -0.20, -0.5, -0.10, -0.10, -0.10, -0.20, -0.20, -0.20, -0.5],
        "norm_delta": [ 0.20,  0.20,  0.20,  0.40,  0.40,  0.40,  6.5,  0.20,  0.20,  0.20,  0.40,  0.40,  0.40,  6.5],
    },
    "10042": {
        "action_keys": [
            ("relative_left_position",  3),
            ("relative_left_rotation",  3),
            ("follow_left_gripper",     1),
            ("relative_right_position", 3),
            ("relative_right_rotation", 3),
            ("follow_right_gripper",    1),
        ],
        # filter: pos<0.08m, rot<15deg(0.2618rad); norm aligned to filter thresholds
        # Gripper from abs_constant.py x2_normal: min=-0.5, delta=6.5
        "norm_min":   [-0.08, -0.08, -0.08, -0.30, -0.30, -0.30, -0.5, -0.08, -0.08, -0.08, -0.30, -0.30, -0.30, -0.5],
        "norm_delta": [ 0.16,  0.16,  0.16,  0.60,  0.60,  0.60,  6.5,  0.16,  0.16,  0.16,  0.60,  0.60,  0.60,  6.5],
    },
    "10053": {
        "action_keys": [
            ("relative_left_position",  3),
            ("relative_left_rotation",  3),
            ("follow_left_gripper",     1),
            ("relative_right_position", 3),
            ("relative_right_rotation", 3),
            ("follow_right_gripper",    1),
        ],
        # filter: pos<0.08m, rot<15deg(0.2618rad); norm aligned to filter thresholds
        # Gripper from abs_constant.py x2_normal: min=-0.5, delta=6.5
        "norm_min":   [-0.08, -0.08, -0.08, -0.30, -0.30, -0.30, -0.5, -0.08, -0.08, -0.08, -0.30, -0.30, -0.30, -0.5],
        "norm_delta": [ 0.16,  0.16,  0.16,  0.60,  0.60,  0.60,  6.5,  0.16,  0.16,  0.16,  0.60,  0.60,  0.60,  6.5],
    },
    "10058": {
        "action_keys": [
            ("relative_left_position",  3),
            ("relative_left_rotation",  3),
            ("follow_left_gripper",     1),
            ("relative_right_position", 3),
            ("relative_right_rotation", 3),
            ("follow_right_gripper",    1),
        ],
        # filter: pos<0.08m, rot<15deg(0.2618rad); norm aligned to filter thresholds
        # Gripper from abs_constant.py x2_normal: min=-0.5, delta=6.5
        "norm_min":   [-0.08, -0.08, -0.08, -0.30, -0.30, -0.30, -0.5, -0.08, -0.08, -0.08, -0.30, -0.30, -0.30, -0.5],
        "norm_delta": [ 0.16,  0.16,  0.16,  0.60,  0.60,  0.60,  6.5,  0.16,  0.16,  0.16,  0.60,  0.60,  0.60,  6.5],
    },
}

# Per-site toggles for CustomDataset logging.warning (train/datasets.py). True = emit warning.
DATA_WARNING_FLAGS = {
    "missing_action_key": False,
    "no_active_action_fields": False,
    "action_json_not_found": False,
    # No instruction.txt (or dataset alt); sample skipped via SilentFilterError (set True to log each skip)
    "missing_instruction_file": False,
    # load_task_video_frames fail_reason=... detail + caller "task tokens failed" (set False to silence)
    "load_task_video_frames_detail": True,
}
