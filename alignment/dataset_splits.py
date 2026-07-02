# coding=utf-8
"""List of subsets."""

DATASETS = {
    'pouring': {'train': 70, 'val': 14, 'test': 32},
    'baseball_pitch': {'train': 103, 'val': 63},
    'baseball_swing': {'train': 113, 'val': 57},
    'bench_press': {'train': 113, 'val': 57},
    'bowling': {'train': 134, 'val': 84},
    'clean_and_jerk': {'train': 40, 'val': 42},
    'golf_swing': {'train': 87, 'val': 77},
    'jumping_jacks': {'train': 56, 'val': 56},
    'pushups': {'train': 102, 'val': 105},
    'pullups': {'train': 98, 'val': 100},
    'situp': {'train': 50, 'val': 50},
    'squats': {'train': 114, 'val': 115},
    'tennis_forehand': {'train': 79, 'val': 74},
    'tennis_serve': {'train': 115, 'val': 68},
}


DATASET_TO_NUM_CLASSES = {
    'pouring': 5,
    'baseball_pitch': 4,
    'baseball_swing': 3,
    'bench_press': 2,
    'bowling': 3,
    'clean_and_jerk': 6,
    'golf_swing': 3,
    'jumping_jacks': 4,
    'pushups': 2,
    'pullups': 2,
    'situp': 2,
    'squats': 4,
    'tennis_forehand': 3,
    'tennis_serve': 4,
}
