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
# Debug mode
CONFIG.DEBUG = False
# Dataset for training alignment.
CONFIG.DATASETS = [
    'pouring',
]

# Path to tfrecords.
CONFIG.PATH_TO_TFRECORDS = '/tmp/%s_tfrecords/'
# Algorithm used for training: alignment, sal, alignment_sal_tcn,
# classification, tcn . (alignment is called tcc in paper)
CONFIG.TRAINING_ALGO = 'alignment'
# Size of images/frames.
CONFIG.IMAGE_SIZE = 224  # For ResNet50

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
CONFIG.TRAIN.DUSTBIN_COEFF = 0.7
CONFIG.TRAIN.DUSTBIN_LOSS_WEIGHT = 0.5
CONFIG.TRAIN.GLOBAL_DUSTBIN_LOSS_WEIGHT = 0
CONFIG.TRAIN.GLOBAL_DUSTBIN_LOSS_TOLERANCE = 0.1
CONFIG.TRAIN.CUT_RANGE = [0, 0]
CONFIG.TRAIN.VISUALIZE_INTERVAL = 20
CONFIG.TRAIN.USE_PRETRAINED = True 

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

CONFIG.EVAL.VAL_ITERS = 20
# If True (eval + distributed), each rank gets a contiguous index range of the
# dataset (JSON order) instead of PyTorch DistributedSampler's strided indices.
CONFIG.EVAL.CONTIGUOUS_DISTRIBUTED_SHARD = False
# A task evaluates the embeddings or the trained model.
CONFIG.EVAL.TASKS = [
    'algo_loss',
    'classification',
    'kendalls_tau',
    'event_completion',
    'few_shot_classification'
]

# DTW alignment settings (will be set after CONFIG.ALIGNMENT is defined)
CONFIG.EVAL.USE_DTW = True
CONFIG.EVAL.DTW_WINDOW = None

CONFIG.EVAL.FRAMES_PER_BATCH = 25
CONFIG.EVAL.KENDALLS_TAU_STRIDE = 5  # 5 for Pouring, 2 for PennAction
CONFIG.EVAL.KENDALLS_TAU_DISTANCE = 'sqeuclidean'  # cosine, sqeuclidean
CONFIG.EVAL.CLASSIFICATION_FRACTIONS = [0.1, 0.5, 1.0]
CONFIG.EVAL.FEW_SHOT_NUM_LABELED = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
CONFIG.EVAL.FEW_SHOT_NUM_EPISODES = 50

# ******************************************************************************
# Model params
# ******************************************************************************
CONFIG.MODEL = edict()

CONFIG.MODEL.EMBEDDER_TYPE = 'conv'

CONFIG.MODEL.BASE_MODEL = edict()
# Resnet50, VGGM
CONFIG.MODEL.BASE_MODEL.NETWORK = 'Resnet50_pretrained'
# conv4_block3_out, conv4 (respective layers in networks)
CONFIG.MODEL.BASE_MODEL.LAYER = 'conv4_block3_out'

# Select which layers to train.
# 'frozen' : Weights are fixed and batch_norm stats are also fixed.
# 'train_all': Everything is trained and batch norm stats are updated.
# 'only_bn': Only tune batch_norm variables and update batch norm stats.
CONFIG.MODEL.TRAIN_BASE = 'only_bn'
CONFIG.MODEL.TRAIN_EMBEDDING = True

CONFIG.MODEL.RESNET_PRETRAINED_WEIGHTS = '/tmp/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5'

# VGG_M-esque model
CONFIG.MODEL.VGGM = edict()
CONFIG.MODEL.VGGM.USE_BN = True

CONFIG.MODEL.CONV_EMBEDDER_MODEL = edict()
# List of conv layers defined as (channels, kernel_size, activate).
CONFIG.MODEL.CONV_EMBEDDER_MODEL.CONV_LAYERS = [
    (256, 3, True),
    (256, 3, True),
]
CONFIG.MODEL.CONV_EMBEDDER_MODEL.FLATTEN_METHOD = 'max_pool'
# List of fc layers defined as (channels, activate).
CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_LAYERS = [
    (256, True),
    (256, True),
]
CONFIG.MODEL.CONV_EMBEDDER_MODEL.CAPACITY_SCALAR = 2
CONFIG.MODEL.CONV_EMBEDDER_MODEL.EMBEDDING_SIZE = 128
CONFIG.MODEL.CONV_EMBEDDER_MODEL.L2_NORMALIZE = False
CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_RATE = 0.0
CONFIG.MODEL.CONV_EMBEDDER_MODEL.BASE_DROPOUT_SPATIAL = False
CONFIG.MODEL.CONV_EMBEDDER_MODEL.FC_DROPOUT_RATE = 0.1
CONFIG.MODEL.CONV_EMBEDDER_MODEL.USE_BN = True

# Conv followed by GRU Embedder
CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL = edict()
# List of conv layers defined as (channels, kernel_size, activate).
CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.CONV_LAYERS = [(512, 3, True),
                                                   (512, 3, True)]
# List of fc layers defined as (channels, activate).
CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.GRU_LAYERS = [
    128,
]
CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.DROPOUT_RATE = 0.0
CONFIG.MODEL.CONVGRU_EMBEDDER_MODEL.USE_BN = True

# ******************************************************************************
# Alignment params
# ******************************************************************************
CONFIG.ALIGNMENT = edict()
CONFIG.ALIGNMENT.CYCLE_LENGTH = 2
CONFIG.ALIGNMENT.LABEL_SMOOTHING = 0.1
CONFIG.ALIGNMENT.SOFTMAX_TEMPERATURE = 0.1 # 0.1 for non-normalization
CONFIG.ALIGNMENT.LOSS_TYPE = 'regression_mse_var'
CONFIG.ALIGNMENT.NORMALIZE_INDICES = True
CONFIG.ALIGNMENT.VARIANCE_LAMBDA = 0.001
# Hinge on |pred_time - true_time| for regression_mse_var main term only. Same units as steps/time
# after NORMALIZE_INDICES (if True, use margin in ~[0,1], e.g. 0.03). If NORMALIZE_INDICES=False,
# use a margin in raw frame-index units or set 0.0 for legacy squared error.
CONFIG.ALIGNMENT.TCC_REGRESSION_MARGIN = 0
CONFIG.ALIGNMENT.FORWARD_VARIANCE_LAMBDA = 0.001
CONFIG.ALIGNMENT.FRACTION = 1.0
CONFIG.ALIGNMENT.HUBER_DELTA = 0.1
CONFIG.ALIGNMENT.SIMILARITY_TYPE = 'l2'  # l2, cosine
CONFIG.ALIGNMENT.NORMALIZE_EMBEDDINGS = True
CONFIG.ALIGNMENT.STOCHASTIC_MATCHING = False
CONFIG.ALIGNMENT.CAUSAL_LAMBDA = 0
CONFIG.ALIGNMENT.CAUSAL_MARGIN = 0
CONFIG.ALIGNMENT.USE_ATTN_GATE = False
CONFIG.ALIGNMENT.GATE_HIDDEN_DIM = 256
CONFIG.ALIGNMENT.GATE_HEADS = 8
CONFIG.ALIGNMENT.GATE_LAYERS = 4
# DTW alignment settings
CONFIG.ALIGNMENT.USE_DTW = True
CONFIG.ALIGNMENT.DTW_WINDOW = None
CONFIG.ALIGNMENT.DTW_GUIDANCE_LAMBDA = 0 # 15
CONFIG.ALIGNMENT.DTW_ALIGNMENT_STRATEGY = "middle"
# SmoothDTW-based matching probability (replaces plain F.softmax for beta)
# Set USE_SMOOTH_DTW=True to enable; keep False to use original softmax.
CONFIG.ALIGNMENT.USE_SMOOTH_DTW = True
CONFIG.ALIGNMENT.SMOOTH_DTW_GAMMA_S = 1.0   # soft-min temperature (smaller = harder path)
CONFIG.ALIGNMENT.SMOOTH_DTW_GAMMA_F = 0.1   # column-norm temperature (0 = skip)
CONFIG.ALIGNMENT.SMOOTH_DTW_SOFTNING = "dtw_prob"  # 'dtw_prob' or 'dtw_minGamma'
# False: forward-only (original); True: forward-backward (argmax → hard DTW path as γ_s→0)
CONFIG.ALIGNMENT.SMOOTH_DTW_BIDIRECTIONAL = True
# "loop": O(T²) sequential loop (safe for T1≠T2); "anti_diag": O(T) parallel (requires T1==T2)
CONFIG.ALIGNMENT.SMOOTH_DTW_METHOD = "anti_diag"
# When True AND USE_SMOOTH_DTW=True: skip the separate numpy hard-DTW call entirely.
# Instead, argmax(beta[b, i, :]) from the already-computed student beta matrix is used as
# the DTW regression target. Empirically produces ~90% index agreement with hard DTW at any
# gamma_s; the remaining 10% is at genuinely ambiguous path positions where both choices are
# valid DTW paths. Requires USE_SMOOTH_DTW=True; has no effect when USE_SMOOTH_DTW=False.
CONFIG.ALIGNMENT.SMOOTH_DTW_AS_DTW_INDICES = True
# D2TW path-cost loss: minimise sdtw[-1,-1] (the cumulative DTW alignment cost) directly.
# Requires USE_SMOOTH_DTW=True (reuses the same DP table, zero extra compute).
CONFIG.ALIGNMENT.USE_D2TW_LOSS = True        # whether to add soft-DTW path cost as a loss term
CONFIG.ALIGNMENT.D2TW_LOSS_LAMBDA = 0.3       # weight (matches the 0.1 coefficient in the original smoothDTW.py)
# Divide terminal soft-DTW cost by (T_src + T_tgt) for this direction (matches DP grid axes).
# Stabilizes scale across chunk lengths; retune D2TW_LOSS_LAMBDA when toggling. False = legacy raw cost.
CONFIG.ALIGNMENT.D2TW_NORMALIZE_LENGTH_SUM = True
# Hinge on normalized path cost: mean(ReLU(cost - m)) per direction (linear, like original D2TW cost).
# Same units as post-normalization cost. 0 = no hinge (full cost as before).
CONFIG.ALIGNMENT.D2TW_COST_MARGIN = 0
# Persist scaled pre-softmax similarity sim_12 (Main->Ref / Ref->Main) for debugging.
# Only global rank 0 writes; filenames include step + timestamp. See deterministic_alignment._maybe_save_raw_sim12.
CONFIG.ALIGNMENT.SAVE_RAW_SIM12 = False
CONFIG.ALIGNMENT.SAVE_RAW_SIM12_DIR = ""  # e.g. under run log dir: "saved_sim12"
CONFIG.ALIGNMENT.SAVE_RAW_SIM12_EVERY = 1000  # save when global_step % EVERY == 0; use 1 for every step
CONFIG.ALIGNMENT.SAVE_RAW_SIM12_MAX_BATCH = None  # None = full batch; int = first N samples only
# ******************************************************************************
# Shuffle and Learn params
# ******************************************************************************
CONFIG.SAL = edict()
CONFIG.SAL.DROPOUT_RATE = 0.0
# List of fc layers defined as (channels, activate).
CONFIG.SAL.FC_LAYERS = [(128, True), (64, True), (2, False)]
CONFIG.SAL.SHUFFLE_FRACTION = 0.75
# Number of triplets to sample from each video in batch.
CONFIG.SAL.NUM_SAMPLES = 8
CONFIG.SAL.LABEL_SMOOTHING = 0.0

# ******************************************************************************
# Alignment and Shuffle and Learn and TCN params
# ******************************************************************************
CONFIG.ALIGNMENT_SAL_TCN = edict()
# The weight for the tcn loss is (1 - alignment_loss_weight - sal_loss_weight)
CONFIG.ALIGNMENT_SAL_TCN.ALIGNMENT_LOSS_WEIGHT = 0.33
CONFIG.ALIGNMENT_SAL_TCN.SAL_LOSS_WEIGHT = 0.33

# ******************************************************************************
# Classification/Supervised Learning of Per-frame Classes params
# ******************************************************************************
CONFIG.CLASSIFICATION = edict()
CONFIG.CLASSIFICATION.LABEL_SMOOTHING = 0.0
CONFIG.CLASSIFICATION.DROPOUT_RATE = 0.0

# ******************************************************************************
# Time Contrastive Network params
# ******************************************************************************
CONFIG.TCN = edict()
CONFIG.TCN.POSITIVE_WINDOW = 5
CONFIG.TCN.REG_LAMBDA = 0.002

# ******************************************************************************
# Optimizer params
# ******************************************************************************
CONFIG.OPTIMIZER = edict()
# Supported optimizers are: AdamOptimizer, MomentumOptimizer
CONFIG.OPTIMIZER.TYPE = 'AdamOptimizer'

CONFIG.OPTIMIZER.LR = edict()
# Learning rate decay strategy.
# Currently Supported strategies: fixed, exp_decay, manual, warmup_cosine
# CONFIG.OPTIMIZER.LR.DECAY_TYPE = 'warmup_cosine'
# CONFIG.OPTIMIZER.LR.EXP_DECAY_RATE = 0.97
# CONFIG.OPTIMIZER.LR.EXP_DECAY_STEPS = 1000
# CONFIG.OPTIMIZER.LR.MANUAL_LR_STEP_BOUNDARIES = [5000, 10000]
# CONFIG.OPTIMIZER.LR.MANUAL_LR_DECAY_RATE = 0.1
# CONFIG.OPTIMIZER.LR.NUM_WARMUP_STEPS = 0
# CONFIG.OPTIMIZER.LR.WARMUP_FRACTION = 0.1
# CONFIG.OPTIMIZER.LR.END_LR = 0.000003

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
CONFIG.DATA.SHUFFLE_QUEUE_SIZE = 0
CONFIG.DATA.NUM_PREFETCH_BATCHES = 1
CONFIG.DATA.RANDOM_OFFSET = 1
CONFIG.DATA.STRIDE = 16
CONFIG.DATA.SAMPLING_STRATEGY = 'offset_uniform'  # offset_uniform, stride
CONFIG.DATA.NUM_STEPS = 3  # number of frames that will be embedded jointly,
# Multi-view interleaving: datasets listed here split each NUM_STEPS chunk evenly
# across NUM_VIEWS camera views using path-string substitution (no extra file I/O).
CONFIG.DATA.MULTI_VIEW_DATASETS = {'AgiBotWorld-Beta-v2', 'Robochallenge_x2', 'droid_success_only_v2', 'robomind', 'robomindv2', 'robomind_v2.0_v2', 'robocoin_v2', '10042', '10058', '10053', '10042_part_1', '10042_part_2', '10042_part_3', '10042_part_4', '10053_part_1', '10053_part_2', '10053_part_3', '10053_part_4', '10058_part_1', '10058_part_2', '10058_part_3', '10058_part_4'}
CONFIG.DATA.NUM_VIEWS = 2
# Camera order for JSON training: {CAM_MAPPING_DIR}/{dataset_name}_cam_mapping.json (task_path -> [view names]).
# For 3-camera setups, set NUM_STEPS to 6 so NUM_STEPS // num_views stays integral.
CONFIG.DATA.CAM_MAPPING_DIR = '/open_data/cgy/cam_mapping'
CONFIG.DATA.USE_CAM_MAPPING = True
CONFIG.DATA.FRAME_STRIDE = 10  # stride between context frames
# Intra-chunk attention type for multi-view token sequences in Qwen forward pass.
# 'causal': standard lower-triangular mask within each chunk (each token only sees past tokens).
# 'full':   all tokens within a chunk (and the prefix) attend to each other bidirectionally;
#           cross-chunk backward attention is still blocked in both modes.
CONFIG.DATA.CHUNK_INTRA_ATTENTION = 'full'  # 'causal' | 'full'
# Set this to False if your TFRecords don't have per-frame labels.
CONFIG.DATA.FRAME_LABELS = True
CONFIG.DATA.PER_DATASET_FRACTION = 1.0  # Use 0 to use only one sample.
CONFIG.DATA.PER_CLASS = False
# stride of frames while embedding a video during evaluation.
CONFIG.DATA.SAMPLE_ALL_STRIDE = 1

# Dataset weights for sampling. Key: dataset name (from json filename), Value: weight.
CONFIG.DATA.DATASET_WEIGHTS = {
    'berkeley_autolab_ur5': 2.0,
    'cmu_play_fusion': 1.0,
    'libero': 1.0,
    'bridgev2': 1.0,
    'bridge_data_v2_img1_v2': 1.0,
    'bridge_data_v2_img3_v2': 1.0,
    'bridge_data_v2_img4_v2': 1.0,
    'Robochallenge_x2': 1.0,
    'droid': 0.2,
    'maniskill': 1.0,
    'calvin': 1.0,
    'fmb': 1.0,
    'rt1': 0.5,
    'language_table': 0.1,
    'SSv2':0.5,
    'Egodex': 0.1,
    'AgiBotWorld': 0.05,
    'AgiBotWorld-Beta-v2': 0.05,
    'robomindv2':0.3,
    'robomind_v2.0_v2':0.1,
    'robomind':0.5,
    'robocoin_v2':1.0,
    "10042": 0.3,
    "10058": 0.3,
    "10053": 0.3,
}

# Dataset names that use segmented path format: "path:segment_id:start-end"
# Only these will be parsed by _parse_video_path; others are treated as plain path (video_dir = path_str).
CONFIG.DATA.SEGMENTED_PATH_DATASETS = ('AgiBotWorld','AgiBotWorld-Beta-v2','robocoin_v2')

# Path transformations for task_paths.json loading
# Format: {dataset_name: {old_prefix: new_prefix, ...}, ...}
# The dataset_name is extracted from the JSON filename (e.g., AgiBotWorld_video_paths.json -> 'AgiBotWorld')
CONFIG.DATA.TASK_PATHS_TRANSFORMS = {
    'AgiBotWorld-Beta-v2': {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    'bridge_data_v2_img1_v2': {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    'bridge_data_v2_img3_v2': {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    'bridge_data_v2_img4_v2': {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    'droid_success_only_v2': {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    'fractal20220817_data_v2': {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    'Robochallenge_x2': {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    'robocoin_v2': {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    'robomind_v2.0_v2': {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    'robomindv2': {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    'robomind': {
        '/open_data/Open_Action_datasets_as_mp4': '/open_data/cgy/anns'
    },
    "10042": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10042_part_1": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10042_part_2": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10042_part_3": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10042_part_4": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10058": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10058_part_1": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10058_part_2": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10058_part_3": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10058_part_4": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10053": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10053_part_1": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10053_part_2": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10053_part_3": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    "10053_part_4": {
        '/x2robot_data/zhengwei': '/open_data/cgy/anns/real_data_new'
    },
    # 'InternDataA1': {
    #     '/x2robot_v2/share/xzn/datasets/x2robot': '/open_data/cgy/anns'
    # },
    # Add more datasets here as needed:
    # 'SomeOtherDataset': {
    #     '/old/path': '/new/path'
    # }
    #     'AgiBotWorld': {
    #     '/open_data': '/open_data/cgy/anns'
    # },
}

# ******************************************************************************
# Augmentation params
# ******************************************************************************
CONFIG.AUGMENTATION = edict()

# --- Part 1: Currently Used (in datasets.py) ---
CONFIG.AUGMENTATION.RANDOM_FLIP = True
CONFIG.AUGMENTATION.RANDOM_CROP = False
CONFIG.AUGMENTATION.REVERSE_PROB = 0.5
CONFIG.AUGMENTATION.CROP_MIN_SCALE = 0.8
CONFIG.AUGMENTATION.CROP_MAX_SCALE = 1.0

# Switches for ColorJitter (PyTorch)
CONFIG.AUGMENTATION.BRIGHTNESS = True
CONFIG.AUGMENTATION.CONTRAST = True
CONFIG.AUGMENTATION.HUE = False
CONFIG.AUGMENTATION.SATURATION = False

# Magnitudes used by ColorJitter
CONFIG.AUGMENTATION.BRIGHTNESS_MAX_DELTA = 32.0 / 255
CONFIG.AUGMENTATION.HUE_MAX_DELTA = 0.2
CONFIG.AUGMENTATION.CONTRAST_LOWER = 0.5
CONFIG.AUGMENTATION.CONTRAST_UPPER = 1.5
CONFIG.AUGMENTATION.SATURATION_LOWER = 0.5
CONFIG.AUGMENTATION.SATURATION_UPPER = 1.5

# --- Part 2: Legacy / Not Used (Placeholders from older versions) ---
# Note: Currently all important params are in Part 1.

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
CONFIG.JOINTS.JOINT_TOKEN_START_ID = 151669   # first unused ID in Qwen3 vocab (vocab_size=151936)

# ******************************************************************************
# Sync params from DS_CONFIG to other configs to avoid redundancy
# ******************************************************************************
CONFIG.TRAIN.BATCH_SIZE = CONFIG.DS_CONFIG.train_micro_batch_size_per_gpu
CONFIG.OPTIMIZER.CLIP_GRAD_NORM = CONFIG.DS_CONFIG.gradient_clipping
CONFIG.OPTIMIZER.LR.INITIAL_LR = CONFIG.DS_CONFIG.optimizer.params.lr
CONFIG.MODEL.L2_REG_WEIGHT = CONFIG.DS_CONFIG.optimizer.params.weight_decay
CONFIG.TRAIN.MAX_ITERS = CONFIG.DS_CONFIG.scheduler.params.total_num_steps

# ******************************************************************************
# Sync DTW params from ALIGNMENT to EVAL for backward compatibility
# ******************************************************************************
CONFIG.EVAL.USE_DTW = CONFIG.ALIGNMENT.USE_DTW
CONFIG.EVAL.DTW_WINDOW = CONFIG.ALIGNMENT.DTW_WINDOW


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
    CONFIG.AUGMENTATION.REVERSE_PROB = 0.0
    CONFIG.AUGMENTATION.RANDOM_FLIP = False
    CONFIG.AUGMENTATION.RANDOM_CROP = False
    CONFIG.AUGMENTATION.BRIGHTNESS = False
    CONFIG.AUGMENTATION.CONTRAST = False
    CONFIG.AUGMENTATION.HUE = False
    CONFIG.AUGMENTATION.SATURATION = False
    CONFIG.ALIGNMENT.D2TW_LOSS_LAMBDA = 0
    CONFIG.EVAL.CONTIGUOUS_DISTRIBUTED_SHARD = True
    CONFIG.EVAL.REF_CACHE_MAIN_ONLY = True

    # --- Logging: report every step and save every batch during eval ---
    CONFIG.LOGGING.DEBUG_STEP_START = 0
    CONFIG.LOGGING.SAVE_BATCH_INTERVAL = 1

    CONFIG.TRAIN.CHUNK_PROBS = [0, 0, 1.0]
