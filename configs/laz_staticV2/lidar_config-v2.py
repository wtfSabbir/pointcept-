_base_ = []

# ==============================================================================
# 1. ENGINE
# ==============================================================================
train = dict(type="DefaultTrainer")
test  = dict(type="SemSegTester", verbose=True)

# ==============================================================================
# 2. EXPERIMENT SETTINGS
# ==============================================================================
weight      = None
resume      = False
evaluate    = True
test_only   = False
seed        = 4880
save_path   = "/mnt/d/PointCloudsFiles/Pointcept/Refinement_PTv3"
enable_wandb = False

use_mlflow = True

num_worker  = 8
batch_size  = 2
batch_size_val  = 1
batch_size_test = 1
epoch       = 500
eval_epoch  = 50

sync_bn                  = False
enable_amp               = False
empty_cache              = True
empty_cache_per_epoch    = True
amp_dtype                = "float16"
find_unused_parameters   = False
mix_prob                 = 0
clip_grad                = 1.0
gradient_accumulation_steps = 1

param_dicts = [dict(keyword="backbone", lr_scale=0.1)]

# ==============================================================================
# 3. DATASET
# ==============================================================================
# v2: feat_keys=("coord", "color") where color is now (N, 2): intensity + v1_pred_normalized
# in_channels = 3 (XYZ) + 2 (intensity + v1_pred) = 5

data = dict(
    num_classes  = 10,
    ignore_index = 255,
    names = ["Unclassified", "Ground", "Pole", "Sign", "Bollard",
             "Trunk", "Vegetation", "Building", "Fence", "Gate"],

    train = dict(
        type      = "LidarDatasetV2",
        data_root = "/mnt/d/PointCloudsFiles/Pointcept/Refinement_PTv3/DataV2",
        split     = "train",
        loop      = 10,
        transform = [
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=True, p=0.5),
            dict(type="RandomScale",  scale=[0.9, 1.1]),
            dict(type="RandomFlip",   p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="GridSample", grid_size=0.05, hash_type="fnv",
                 mode="train", return_grid_coord=True),
            dict(type="SphereCrop", point_max=120000, mode="random"),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "grid_coord", "segment"),
                 feat_keys=("coord", "color")),  # XYZ + intensity + v1_pred = 5ch
        ],
        test_mode = False,
    ),

    val = dict(
        type      = "LidarDatasetV2",
        data_root = "/mnt/d/PointCloudsFiles/Pointcept/Refinement_PTv3/DataV2",
        split     = "val",
        loop      = 1,
        transform = [
            dict(type="GridSample", grid_size=0.05, hash_type="fnv",
                 mode="train", return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect",
                 keys=("coord", "grid_coord", "segment"),
                 feat_keys=("coord", "color")),
        ],
        test_mode = False,
    ),
)

# ==============================================================================
# 4. MODEL — same architecture, only in_channels changes from 4 to 5
# ==============================================================================
model = dict(
    type                  = "DefaultSegmentorV2",
    num_classes           = 10,
    backbone_out_channels = 64,

    backbone = dict(
        type        = "PT-v3m1",
        in_channels = 5,          # XYZ(3) + intensity(1) + v1_pred(1)

        order   = ("z", "z-trans", "hilbert", "hilbert-trans"),
        stride  = (2, 2, 2, 2),

        enc_depths      = (2, 2, 2, 6, 2),
        enc_channels    = (32, 64, 128, 256, 512),
        enc_num_head    = (2, 4, 8, 16, 32),
        enc_patch_size  = (48, 48, 48, 48, 48),

        dec_depths      = (2, 2, 2, 2),
        dec_channels    = (64, 64, 128, 256),
        dec_num_head    = (4, 4, 8, 16),
        dec_patch_size  = (48, 48, 48, 48),

        mlp_ratio        = 4,
        qkv_bias         = True,
        qk_scale         = None,
        attn_drop        = 0.0,
        proj_drop        = 0.0,
        drop_path        = 0.3,
        shuffle_orders   = True,
        pre_norm         = True,
        enable_rpe       = False,
        enable_flash     = False,
        upcast_attention = False,
        upcast_softmax   = False,
    ),

    criteria = [
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=255),
        dict(type="LovaszLoss",       mode="multiclass", loss_weight=1.0, ignore_index=255),
    ],
)

# ==============================================================================
# 5. OPTIMIZER & SCHEDULER
# ==============================================================================
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type             = "OneCycleLR",
    max_lr           = 0.001,
    pct_start        = 0.04,
    div_factor       = 10,
    final_div_factor = 100,
    total_steps      = 61000,  # 244 scenes * 10 loop / 2 batch * 50 epochs 
)

# ==============================================================================
# 6. HOOKS
# ==============================================================================
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer",    warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver",   save_freq=5),
]
