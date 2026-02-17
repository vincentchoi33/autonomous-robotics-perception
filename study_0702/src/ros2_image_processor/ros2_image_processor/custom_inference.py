_base_ = [
    '_base_/models/eva-clip+mask2former.py',
    '_base_/datasets/train/gta_512x512.py',
    '_base_/datasets/test/synth2cityscapes.py',
    '_base_/schedules/schedule_5k.py',
    '_base_/default_runtime.py'
]

# Custom dataset configuration
# data_root = '/data/choihy/study_0630/data/cropped'
data_root = '/data/choihy/study_0630/data/demo_data'
data_prefix = dict(img_path='', seg_map_path='')

# Load pretrained checkpoint
load_from = '/data/choihy/study_0630/VLTSeg/vltseg_checkpoint_mapillary+cityscapes_1.pth'

# Use the exact image size that matches the checkpoint's position embedding
# 5330 position embeddings ≈ 73x73 patches, with 14x14 patch size = 1022x1022
crop_size = (1022, 1022)
stride_size = (851, 851)

num_gpus = 1  # Changed to 1 for single GPU
num_samples_per_gpu_train = 8
num_workers_per_gpu_train = 1
num_samples_per_gpu_test = 1  # Changed to 1 for inference
num_workers_per_gpu_test = 1

model = dict(
    data_preprocessor=dict(size=crop_size),
    backbone=dict(
        img_size=crop_size[0],
        # Remove the pretrained EVA-CLIP requirement
        init_cfg=None
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=stride_size)
)

# Custom dataset configuration
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1022, 1022), keep_ratio=False),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=num_samples_per_gpu_test,
    num_workers=num_workers_per_gpu_test,
    dataset=dict(
        type='CustomDataset',
        data_root=data_root,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        test_mode=True,
        img_suffix='.png',
        seg_map_suffix='.png'
    )
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (# GPUs) x (# samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=num_gpus*num_samples_per_gpu_train)