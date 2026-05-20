_base_ = [
    '../../../mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '../../../mmdetection3d/configs/_base_/default_runtime.py'
]
plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# This is used for online data generation
point_cloud_range = [-50, -50, -5.0, 50, 50, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

############### Training config ###############
num_gpus = 2
batch_size = 1
num_epochs = 6 
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
# Effective batch = num_gpus * batch_size * cumulative_iters (e.g. 2*1*4 = 8, same as 8*1).
# Optimizer steps per epoch = dataset / effective_batch; max_iters is micro-batch iters per epoch.
cumulative_iters = 4
lr = 1e-4

############### SpaceDrive Config ###############
vis_3d_pos = True # enable PE on vision tokens
io_3d_pos = True # enable PE in inputs/outputs

# depth model config
depth_model_type = 'unidepth' # 'depth_anything' or 'unidepth'

# llm config
llm_lora_rank=16 
# Extend with tools/token_additor.py for POS tokens (same workflow as Qwen2.5-VL).
llm_path = 'Qwen/Qwen3-VL-2B-Instruct'
tokenizer_path = 'Qwen/Qwen3-VL-2B-Instruct'

# Loss config
loss_pos_lambda = 1 
loss_pos_type = 'huber' #"l2" and 'huber' 

# PE encoding config
pe_scaling = 0.1 #  scale the final output
learnable_pe_scaling = True # if True, the pe_scaling is learnable, if False, the pe_scaling is fixed
## transformer PE config#
pe_freq_coeff = 20000 # frequency coefficient for 3d positional encoding
pe_freq_scaling = 1 # frequency scaling for 3d positional encoding
pe_type = 'transformer' # 'transformer' or 'fone' or 'nerf', type of 3d positional encoding, 'transformer' is the default, 'fone' is the fone positional encoding, 'nerf' is the nerf positional encoding
## fone PE config
# pe_freq_coeff = 2 # 
# pe_freq_scaling = 0.02 # 
# pe_type = 'fone' # 
fone_dim = 8 * 3 # change fone_dim here, default is 8 * 3, can be changed to 4 * 3 or 2 * 3 for better performance
## learnable PE config
input_pe_mlp = False # use mlp to get 3d positional encoding from input coordinates

# PE decoding config
pe_decode_method = 'l2_coords_mlp_2layer' # method for 3d positional encoding loss
## VAE configs
use_vae_to_replace_mlp = False # use vae to replace mlp for input coordinates to 3d positional decoding
with_cur = False # use current input embedding to assist vae decoding

# training data config
planning_only = True # only train planning task, not training detection task
single_coords_only = False # only train the first coordinate
pseudo_coords = False # use pseudo coordinates

# ego status
ego_status = '' # 'feature', 'PE'. 'feature' means using a mlp to generate the ego status feature, 'PE' means using the past ego status as PE
ego_status_len = 2 # ego status length
load_ego_command_in_question = True # load command in language format in question

# other exps
## ablations
load_high_level_command = False
single_token_output = False
enable_pe_input = False 
## sem posemb configs
include_semantic_posemb = False # include semantic posemb in 3d positional encoding 
supervise_semantic_posemb = False # use cross entropy loss to supervise semantic posemb, if True, the semantic posemb is supervised by the semantic labels

# save paths
base_path = 'workspace/spacedrive_qwen3vl/'
work_dir = base_path
results_path = base_path + '_results_planning_only/'
wb_run_name = base_path + 'debug_run'

# collect keys
collect_keys=['lidar2img', 'intrinsics', 'extrinsics','timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv', 'command',  'can_bus']
pc_keys= []


############### Model ###############
model = dict(
    type='SpaceDriveQwen3VL',
    save_path=results_path,  
    frozen=False,
    use_lora=True,
    tokenizer=tokenizer_path, 
    processor=llm_path,
    lm_head=llm_path,
    vis_3d_pos=vis_3d_pos, 
    io_3d_pos=io_3d_pos,
    loss_pos_lambda=loss_pos_lambda,
    pe_decode_method=pe_decode_method, 
    loss_pos_type=loss_pos_type,
    # PE config
    input_pe_mlp=input_pe_mlp, 
    include_semantic_posemb=include_semantic_posemb,
    supervise_semantic_posemb=supervise_semantic_posemb,
    pe_freq_coeff=pe_freq_coeff,
    pe_freq_scaling=pe_freq_scaling,
    pe_type=pe_type,
    fone_dim=fone_dim,
    pe_scaling=pe_scaling,
    # trainig part config
    planning_only=planning_only,
    single_coords_only=single_coords_only,
    # concept test config
    single_token_output=single_token_output,
    enable_pe_input=enable_pe_input,
    # ego status config 
    ego_status=ego_status,
    ego_status_len=ego_status_len,
    # llm config
    llm_lora_rank=llm_lora_rank,
    # decoder config
    use_vae_to_replace_mlp=use_vae_to_replace_mlp,
    with_cur=with_cur,
    # depth model config
    depth_model_type=depth_model_type,
    learnable_pe_scaling=learnable_pe_scaling,
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
    dict(type='LoadAnnoatationVQA',
         llm_type='qwen3vl', 
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
    dict(type='Collect3D', keys=['has_gt_planning', 'img', 'coords_pos_tensor','input_ids', 'vlm_labels', 'pixel_values','image_grid_thw',  'prev_exists', ] + collect_keys + pc_keys,
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
    dict(type='LoadAnnoatationVQATest',
         llm_type='qwen3vl',
         load_3d_pos=True,
         load_ego_command_in_question=load_ego_command_in_question, # load command in language format in question
         base_vqa_path=data_root + 'vqa/val/',
         base_conv_path=data_root + 'conv/val/',
         base_counter_path=data_root + 'eval_cf/',
         load_type=["planning"], # please don't test all the questions in single test, it requires quite long time
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
        seq_split_num=1, # streaming video training
        seq_mode=True, # streaming video training
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
        warmup_split_num=10, # lane det and vlm need short term temporal fusion in the early stage of training
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
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
    )


evaluation = dict(interval=num_iters_per_epoch * num_epochs, pipeline=test_pipeline)

find_unused_parameters=False #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_iters_per_epoch//2, max_keep_ckpts=12)
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
load_from=None
resume_from=None
