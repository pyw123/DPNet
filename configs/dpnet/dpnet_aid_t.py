
_base_ = [
    '_base_/dpnet_default_runtime.py',
    '_base_/datasets/aid_bs64_pil_resize_autoaug.py',
    # '_base_/datasets/aid_dataset.py',
    '_base_/schedules/aid_schedule.py',
]

work_dir = 'work_dirs/dpnet_aid_t'

data_root = '/home/pyw/decdata/AID/AID/AID'
code_root = '/home/pyw/DPNet/datainfo/AID'

batch_size = 16
train_cfg = dict(max_epochs=700, val_interval=10)


vis_backends = [dict(type='LocalVisBackend'),
                ]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

num_classes = 30
data_preprocessor = dict(
    num_classes=num_classes,
)

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='Backbone_DPNET',
        depths=[1, 1, 2, 1],
        embed_dim=64,
        num_heads=[2, 4, 8, 16],
        simple_downsample=False,  # 'head', 'tail', 'head_tail', 'middle', 'none'
        simple_patch_embed=False,
        ssd_expansion=2,
        attn_types=['dpabl', 'dpabl', 'dpab', 'standard'],
        # attn_types=['standard', 'standard', 'standard', 'standard'],
        d_state=[64, 64, 64, 64]
    ),
    neck=None,
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=512,
        init_cfg=None,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
    ),
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ]),
)


train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        data_name='AID',
        data_root=data_root,
        ann_file=code_root+'/train.txt',

    ),
)

val_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        data_name='AID',
        data_root=data_root,
        ann_file=code_root+'/val.txt',
    )
)
test_dataloader = val_dataloader
