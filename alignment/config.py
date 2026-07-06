# coding=utf-8
"""Configuration of an experiment."""

from easydict import EasyDict as edict

# Get the configuration dictionary whose keys can be accessed with dot.
CONFIG = edict()

# ******************************************************************************
# Experiment params
# ******************************************************************************

# Directory for the experiment logs.
CONFIG.LOGDIR = '/tmp/alignment_logs/'
# Size of images/frames (used by image loading/resize pipeline).
CONFIG.IMAGE_SIZE = 224

# ******************************************************************************
# Training params
# ******************************************************************************

# Number of training steps.
CONFIG.TRAIN = edict()
# Number of frames to use while training.
CONFIG.TRAIN.NUM_FRAMES = 24
CONFIG.TRAIN.NUM_ALIGN_FRAMES = 24
CONFIG.TRAIN.MAX_BATCH_FRAMES = 96
CONFIG.TRAIN.CHUNK_PROBS = [0.1, 0.2, 0.7] # 1x, 2x, 4x multipliers

# ******************************************************************************
# Eval params
# ******************************************************************************
CONFIG.EVAL = edict()
# Number of samples in each batch.
# Set to 1 when using 4x multiplier to ensure all samples are processed
CONFIG.EVAL.BATCH_SIZE = 1
# Number of frames to use while evaluating. Only used to see loss in eval mode.
CONFIG.EVAL.NUM_FRAMES = 24
# Chunk probabilities for evaluation [1x, 2x, 4x multipliers]
CONFIG.EVAL.CHUNK_PROBS = [0.0, 0.0, 1.0]  # Force 4x multiplier (96 frames total)

# Ref embedding cache (eval only)
CONFIG.EVAL.REF_CACHE_MAXSIZE = 48        # max cached sample-level ref embeddings
CONFIG.EVAL.REF_CACHE_MAIN_ONLY = False     # use main-only Qwen input when all refs are cached

# If True (eval + distributed), each rank gets a contiguous index range of the
# dataset (JSON order) instead of PyTorch DistributedSampler's strided indices.
CONFIG.EVAL.CONTIGUOUS_DISTRIBUTED_SHARD = False

# ******************************************************************************
# Model params
# ******************************************************************************
CONFIG.MODEL = edict()

CONFIG.MODEL.BASE_MODEL = edict()
CONFIG.MODEL.BASE_MODEL.NETWORK = 'Qwen3-VL-Embedding-8B'

# Embedder params (LinearEmbedder, applied on top of the Qwen backbone).
CONFIG.MODEL.CONV_EMBEDDER_MODEL = edict()
CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE = 128

# ******************************************************************************
# Alignment params
# ******************************************************************************
CONFIG.ALIGNMENT = edict()
CONFIG.ALIGNMENT.SOFTMAX_TEMPERATURE = 0.1 # 0.1 for non-normalization
CONFIG.ALIGNMENT.NORMALIZE_INDICES = True
CONFIG.ALIGNMENT.VARIANCE_LAMBDA = 0.001
# Hinge on |pred_time - true_time| for the regression main term only. Same units as steps/time
# after NORMALIZE_INDICES (if True, use margin in ~[0,1], e.g. 0.03). If NORMALIZE_INDICES=False,
# use a margin in raw frame-index units or set 0.0 for legacy squared error.
CONFIG.ALIGNMENT.TCC_REGRESSION_MARGIN = 0
CONFIG.ALIGNMENT.FORWARD_VARIANCE_LAMBDA = 0.001
CONFIG.ALIGNMENT.SIMILARITY_TYPE = 'l2'  # l2, cosine
CONFIG.ALIGNMENT.NORMALIZE_EMBEDDINGS = True
CONFIG.ALIGNMENT.SMOOTH_DTW_GAMMA_S = 1.0   # soft-min temperature (smaller = harder path)
CONFIG.ALIGNMENT.SMOOTH_DTW_GAMMA_F = 0.1   # column-norm temperature (0 = skip)
# False: forward-only (original); True: forward-backward (argmax → hard DTW path as γ_s→0)
CONFIG.ALIGNMENT.SMOOTH_DTW_BIDIRECTIONAL = True
# "loop": O(T²) sequential loop (safe for T1≠T2); "anti_diag": O(T) parallel (requires T1==T2)
CONFIG.ALIGNMENT.SMOOTH_DTW_METHOD = "anti_diag"
# D2TW path-cost loss: minimise sdtw[-1,-1] (the cumulative DTW alignment cost) directly.
CONFIG.ALIGNMENT.USE_D2TW_LOSS = True        # whether to add soft-DTW path cost as a loss term
CONFIG.ALIGNMENT.D2TW_LOSS_LAMBDA = 0.3       # weight (matches the 0.1 coefficient in the original smoothDTW.py)
# Hinge on normalized path cost: mean(ReLU(cost - m)) per direction (linear, like original D2TW cost).
# Same units as post-normalization cost. 0 = no hinge (full cost as before).
CONFIG.ALIGNMENT.D2TW_COST_MARGIN = 0
# ******************************************************************************
# Special Token IDs for Qwen3-VL
# ******************************************************************************
CONFIG.SPECIAL_TOKENS = edict()
CONFIG.SPECIAL_TOKENS.CLS_TOKEN_ID = 151664  # <|file_sep|> - 用于标记每个 group 的结束
CONFIG.SPECIAL_TOKENS.ALIGN_END_TOKEN_ID = 151662  # <|fim_pad|> - 用于标记 align video 的结束

# ******************************************************************************
# Data params
# ******************************************************************************
CONFIG.DATA = edict()
CONFIG.DATA.RANDOM_OFFSET = 1
CONFIG.DATA.NUM_STEPS = 3  # number of frames that will be embedded jointly,
# Camera order for JSON training: {CAM_MAPPING_DIR}/{dataset_name}_cam_mapping.json (task_path -> [view names]).
# For 3-camera setups, set NUM_STEPS to 6 so NUM_STEPS // num_views stays integral.
CONFIG.DATA.CAM_MAPPING_DIR = '/open_data/cgy/cam_mapping'
CONFIG.DATA.USE_CAM_MAPPING = True
CONFIG.DATA.FRAME_STRIDE = 10  # stride between context frames

# Dataset weights for sampling. Key: dataset name (from json filename), Value: weight.
CONFIG.DATA.DATASET_WEIGHTS = {
    "10042": 1.0,
}

# Dataset names that use segmented path format: "path:segment_id:start-end"
# Only these will be parsed by _parse_video_path; others are treated as plain path (video_dir = path_str).
CONFIG.DATA.SEGMENTED_PATH_DATASETS = ('AgiBotWorld','AgiBotWorld-Beta-v2','robocoin_v2')

# Path transformations for task_paths.json loading
# Format: {dataset_name: {old_prefix: new_prefix, ...}, ...}
# The dataset_name is extracted from the JSON filename (e.g., AgiBotWorld_video_paths.json -> 'AgiBotWorld')
# TODO(open-source): every prefix below is an internal-cluster path (including the
# "10042" entry, which is load-bearing for the verified alignment training run —
# do not remove without re-verifying end-to-end). See OPEN_SOURCE_PATH_TODOS.md.
CONFIG.DATA.TASK_PATHS_TRANSFORMS = {
    "10042": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
}

# ******************************************************************************
# Augmentation params
# ******************************************************************************
CONFIG.AUGMENTATION = edict()

# --- Part 1: Currently Used (in datasets.py) ---
CONFIG.AUGMENTATION.RANDOM_FLIP = True

# Switches for ColorJitter (PyTorch)
CONFIG.AUGMENTATION.BRIGHTNESS = True
CONFIG.AUGMENTATION.CONTRAST = True

# Magnitudes used by ColorJitter
CONFIG.AUGMENTATION.BRIGHTNESS_MAX_DELTA = 32.0 / 255
CONFIG.AUGMENTATION.CONTRAST_LOWER = 0.5
CONFIG.AUGMENTATION.CONTRAST_UPPER = 1.5

# ******************************************************************************
# Logging params
# ******************************************************************************
CONFIG.LOGGING = edict()
# Number of steps between summary logging.
CONFIG.LOGGING.REPORT_INTERVAL = 10
# Keys to exclude from WandB logging
CONFIG.LOGGING.EXCLUDE_KEYS = ['per_sample_loss', 'alignment_indices']
# Threshold for saving high loss samples
CONFIG.LOGGING.HIGH_LOSS_THRESHOLD = 0.3
# Threshold for saving low loss samples
CONFIG.LOGGING.LOW_LOSS_THRESHOLD = 0.2
# Max number of low loss samples to save
CONFIG.LOGGING.MAX_LOW_LOSS_SAMPLES = 30
# Step to start saving high loss samples
CONFIG.LOGGING.DEBUG_STEP_START = 0
# Interval steps to save the full batch of samples
CONFIG.LOGGING.SAVE_BATCH_INTERVAL = 30

# ******************************************************************************
# Checkpointing params
# ******************************************************************************
CONFIG.CHECKPOINT = edict()
# Number of steps between consecutive checkpoints.
CONFIG.CHECKPOINT.SAVE_INTERVAL = 500

# ******************************************************************************
# DeepSpeed params
# ******************************************************************************
CONFIG.DS_CONFIG = edict()
CONFIG.DS_CONFIG.train_micro_batch_size_per_gpu = 4
CONFIG.DS_CONFIG.gradient_clipping = 3.0

# Optimizer
CONFIG.DS_CONFIG.optimizer = edict()
CONFIG.DS_CONFIG.optimizer.type = 'AdamW'
CONFIG.DS_CONFIG.optimizer.params = edict()
CONFIG.DS_CONFIG.optimizer.params.lr = 0.00001
CONFIG.DS_CONFIG.optimizer.params.betas = [0.9, 0.999]
CONFIG.DS_CONFIG.optimizer.params.eps = 1e-8
CONFIG.DS_CONFIG.optimizer.params.weight_decay = 0.00001

# Scheduler
CONFIG.DS_CONFIG.scheduler = edict()
CONFIG.DS_CONFIG.scheduler.type = 'WarmupCosineLR'
CONFIG.DS_CONFIG.scheduler.params = edict()
CONFIG.DS_CONFIG.scheduler.params.total_num_steps = 10000
CONFIG.DS_CONFIG.scheduler.params.warmup_num_steps = 100
CONFIG.DS_CONFIG.scheduler.params.warmup_min_ratio = 0.0
CONFIG.DS_CONFIG.scheduler.params.cos_min_ratio = 0.3

# ******************************************************************************
# Joint state tokenization params
# ******************************************************************************
CONFIG.JOINTS = edict()
CONFIG.JOINTS.USE_JOINTS = True
CONFIG.JOINTS.NUM_BINS = 192
CONFIG.JOINTS.JOINT_ACTION_MAPPING_DIR = '/open_data/cgy/joint_action_mapping'

# ******************************************************************************
# Sync params from DS_CONFIG to other configs to avoid redundancy
# ******************************************************************************
CONFIG.TRAIN.BATCH_SIZE = CONFIG.DS_CONFIG.train_micro_batch_size_per_gpu
CONFIG.TRAIN.MAX_ITERS = CONFIG.DS_CONFIG.scheduler.params.total_num_steps


# ******************************************************************************
# Eval-mode config overrides
# ******************************************************************************

def apply_eval_overrides():
    """Apply eval-mode config overrides.

    Called once at the start of evaluate_v2.py, before any FLAGS-based overrides.
    FLAGS-based overrides take priority over anything set here.
    Add or remove entries freely — this is the single source of truth for
    eval-specific config defaults.
    """
    # --- Augmentation: disable all stochastic transforms ---
    CONFIG.AUGMENTATION.RANDOM_FLIP = False
    CONFIG.AUGMENTATION.BRIGHTNESS = False
    CONFIG.AUGMENTATION.CONTRAST = False
    CONFIG.ALIGNMENT.D2TW_LOSS_LAMBDA = 0
    CONFIG.EVAL.CONTIGUOUS_DISTRIBUTED_SHARD = True
    CONFIG.EVAL.REF_CACHE_MAIN_ONLY = True

    # --- Logging: report every step and save every batch during eval ---
    CONFIG.LOGGING.DEBUG_STEP_START = 0
    CONFIG.LOGGING.SAVE_BATCH_INTERVAL = 1

    CONFIG.TRAIN.CHUNK_PROBS = [0, 0, 1.0]
