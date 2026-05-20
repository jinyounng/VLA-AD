_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

############### Training config ###############
# 2 GPUs, batch 4/GPU, cumulative_iters 2 => effective batch 16 (same as 1 GPU * 8 * cumul 2)
num_gpus = 2
batch_size = 8
num_epochs = 6
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
cumulative_iters = 1
lr = 2e-5

############### SpaceDrive Config ###############
vis_3d_pos = True
io_3d_pos = True

depth_model_type = 'unidepth'

# SmolVLM config
llm_lora_rank = 8  # unused in full finetuning
llm_path = 'ckpts/SmolVLM-256M-Instruct-with-new-special-tokens/'
tokenizer_path = 'ckpts/SmolVLM-256M-Instruct-with-new-special-tokens/'

# Loss config
loss_pos_lambda = 1
loss_pos_type = 'huber'

# PE encoding config
pe_scaling = 0.1
learnable_pe_scaling = True
pe_freq_coeff = 20000
pe_freq_scaling = 1
pe_type = 'transformer'
fone_dim = 8 * 3
input_pe_mlp = False

# PE decoding config
pe_decode_method = 'l2_coords_mlp_2layer'
use_vae_to_replace_mlp = False
with_cur = False

# training data config
planning_only = True
single_coords_only = False
pseudo_coords = False

# ego status
ego_status = ''
ego_status_len = 2
load_ego_command_in_question = True

# SmolVLM-specific: no M-RoPE support
use_rope = False

# other exps
load_high_level_command = False
single_token_output = False
enable_pe_input = False
include_semantic_posemb = False
supervise_semantic_posemb = False

# save paths
base_path = 'workspace/spacedrive_smolvlm_full/'
work_dir = base_path
results_path = base_path + '_results_planning_only/'
wb_run_name = base_path + 'debug_run'

# collect keys
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command',  'can_bus']
pc_keys= []


############### Model ###############
model = dict(
    type='SpaceDriveSmolVLM',
    save_path=results_path,
    frozen=False,
    use_lora=False,
    tokenizer=tokenizer_path,
    processor=llm_path,
    lm_head=llm_path,
    vis_3d_pos=vis_3d_pos,
    io_3d_pos=io_3d_pos,
    loss_pos_lambda=loss_pos_lambda,
    pe_decode_method=pe_decode_method,
    loss_pos_type=loss_pos_type,
    input_pe_mlp=input_pe_mlp,
    include_semantic_posemb=include_semantic_posemb,
    supervise_semantic_posemb=supervise_semantic_posemb,
    pe_freq_coeff=pe_freq_coeff,
    pe_freq_scaling=pe_freq_scaling,
    pe_type=pe_type,
    fone_dim=fone_dim,
    pe_scaling=pe_scaling,
    planning_only=planning_only,
    single_coords_only=single_coords_only,
    single_token_output=single_token_output,
    enable_pe_input=enable_pe_input,
    ego_status=ego_status,
    ego_status_len=ego_status_len,
    llm_lora_rank=llm_lora_rank,
    use_vae_to_replace_mlp=use_vae_to_replace_mlp,
    with_cur=with_cur,
    depth_model_type=depth_model_type,
    learnable_pe_scaling=learnable_pe_scaling,
    precomputed_depth_root=None,
    use_rope=use_rope,
    # SmolVLM: SigLIP 512x512, patch 16, scale_factor 4 -> 8x8 grid
    # stride = image_size(640) / grid_size(8) = 80
    stride=80,
)

############### data ###############
dataset_type = 'CustomNuScenesDataset'
data_root = '/data/jykim/projects/OpenDriveVLA/data/nuscenes/'

file_client_args = dict(backend='disk')

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True,
        with_label=True, with_bbox_depth=True),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type='LoadAnnoatationVQASmolVLM',
         load_3d_pos=True,
         load_ego_command_in_question=load_ego_command_in_question,
         load_high_level_command=load_high_level_command,
         single_token_output=single_token_output,
         pseudo_coords=pseudo_coords,
         planning_only=planning_only,
         base_vqa_path=data_root + 'vqa/train/',
         base_desc_path=data_root + 'desc/train/',
         base_conv_path=data_root + 'conv/train/',
         base_key_path=data_root + 'keywords/train/',
         tokenizer=tokenizer_path,
         processor=llm_path,
         max_length=131072,
         ignore_type=[],
         num_commands=-1,
         lane_objs_info=data_root + 'lane_obj_train.pkl'),
    dict(type='PETRFormatBundle3D', class_names=class_names, collect_keys=collect_keys + ['prev_exists']),
    dict(type='Collect3D', keys=['has_gt_planning', 'img', 'coords_pos_tensor','input_ids', 'vlm_labels', 'pixel_values','image_grid_thw',  'prev_exists', ] + collect_keys,
            meta_keys=('sample_idx', 'filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'img_norm_cfg', 'scene_token', ))
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='ResizeMultiview3D', img_scale=(640, 640), keep_ratio=False, multiscale_mode='value'),
    dict(type='LoadAnnoatationVQATestSmolVLM',
         load_3d_pos=True,
         load_ego_command_in_question=load_ego_command_in_question,
         base_vqa_path=data_root + 'vqa/val/',
         base_conv_path=data_root + 'conv/val/',
         base_counter_path=data_root + 'eval_cf/',
         load_type=["planning"],
         tokenizer=tokenizer_path,
         processor=llm_path,
         max_length=131072,),
    dict(
                type='PETRFormatBundle3D',
                collect_keys=collect_keys,
                class_names=class_names,
                with_label=False),
    dict(type='Collect3D', keys=['input_ids', 'img', 'pixel_values', 'image_grid_thw', 'attention_mask'] + collect_keys + pc_keys,
            meta_keys=('sample_idx', 'vlm_labels', 'filename', 'ori_shape', 'img_shape','pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token'))

]

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_train_with_command_desc.pkl',
        lane_path=data_root + 'data_dict_sample.pkl',
        lane_anno_file=data_root + 'data_dict_subset_B_val.pkl',
        seq_split_num=1,
        seq_mode=True,
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        filter_empty_gt=False,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        eval_mode=['lane', 'det'],
        pipeline=test_pipeline,
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl',
        lane_path=data_root + 'data_dict_sample.pkl',
        lane_anno_file=data_root + 'data_dict_subset_B_val.pkl',
        classes=class_names,
        modality=input_modality),
    test=dict(
        type=dataset_type,
        eval_mode=['lane', 'det'],
        pipeline=test_pipeline,
        ann_file=data_root + 'nuscenes2d_ego_temporal_infos_val_with_command_desc.pkl',
        lane_path=data_root + 'data_dict_sample.pkl',
        lane_anno_file=data_root + 'data_dict_subset_B_val.pkl',
        classes=class_names,
        modality=input_modality),
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
        seq_split_num=2,
        warmup_split_num=10,
        num_iters_to_seq=num_iters_per_epoch,
    ),
    nonshuffler_sampler=dict(type='DistributedSampler')
    )


optimizer = dict(constructor='LearningRateDecayOptimizerConstructor', type='AdamW',
                 lr=lr, betas=(0.9, 0.999), weight_decay=1e-4,
                 paramwise_cfg={'decay_rate': 0.9,
                                'head_decay_rate': 4.0,
                                'lm_head_decay_rate': 0.1,
                                'decay_type': 'vit_wise',
                                'num_layers': 24,
                                })

optimizer_config = dict(
    type='GradientCumulativeFp16OptimizerHook',
    cumulative_iters=cumulative_iters,
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2),
    distributed=True,
)
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )


evaluation = dict(interval=num_iters_per_epoch * num_epochs, pipeline=test_pipeline)

find_unused_parameters=True
checkpoint_config = dict(interval=num_iters_per_epoch//2, max_keep_ckpts=12)
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from=None
resume_from=None
