# ------------------------------------------------------------------------
# SpaceDrive
# Copyright (c) 2026 Zhenghao Zhang. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------

# general imports
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import os
import json
from torchvision.transforms.functional import to_pil_image
import numpy as np
import open3d as o3d

# mmcv, mmdet, mmdet3d imports
import mmcv
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector

# VLM imports
from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor, AutoModelForDepthEstimation

# model and utils imports
from ...datasets.utils.constants import IGNORE_INDEX , IMAGE_TOKEN_INDEX, VISION_START_TOKEN_INDEX, VISION_END_TOKEN_INDEX,POS_INDICATOR_TOKEN_INDEX, POS_EMBEDDING_TOKEN_INDEX, POS_EMBEDDING_TOKEN, POS_INDICATOR_TOKEN
from ..vlm_utils.misc_batch import load_model, locations
from ..vlm_utils.positional_encoding import PositionalEncoding3D
from ..vlm_utils.distributions import  VAEPEDecoder
from ..vlm_utils.precomputed_depth import PrecomputedDepthStore

# Unidepth imports
from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole


@DETECTORS.register_module()
class SpaceDriveBatch(MVXTwoStageDetector):
    """SpaceDrive."""
    def __init__(self,        
                 save_path='./results_vlm/',
                 lm_head=None,
                 tokenizer=None,
                 processor=None,
                 train_cfg=None,
                 test_cfg=None,
                 stride=14,
                 frozen=True,
                 use_lora=False,
                 vis_3d_pos=False,
                 io_3d_pos=False,
                 loss_pos_lambda=0.5,
                 pe_decode_method='cosine',
                 include_semantic_posemb = False,
                 supervise_semantic_posemb = False,
                 pe_freq_coeff=10000,
                 pe_freq_scaling=1,
                 pe_scaling=1,
                 pe_type='transformer',
                 fone_dim=8 * 3,
                 planning_only=False,
                 single_coords_only=False,
                 input_pe_mlp=False, # use mlp to get 3d positional encoding from input coordinates, if False, use a PositionalEncoding3D module to get 3d positional encoding from input coordinates
                 loss_pos_type ='huber', # 'huber' or 'l2', type of loss for 3d positional encoding, 'huber' is the default, 'l2' is the l2 loss
                 huber_delta=1.0, # delta for huber loss, default is 1.0
                 single_token_output=False, # use single output token to decode 6 coords
                 ego_status=None, # 'feature' or 'language' or 'PE', 'feature' means using a mlp to generate the ego status feature, 'language' means using natural language for ego status, 'PE' means using the ego status positional encoding + language
                 ego_status_len=2, # length of ego status, 2 for 2 frames
                 enable_pe_input=False, # enable the use of PE in autoregressive manner, if False, only use PE for supervision
                 llm_lora_rank=16,
                 use_vae_to_replace_mlp=False, # use vae to replace mlp for input coordinates to 3d positional decoding
                 with_cur=False,
                 learnable_pe_scaling=False, # if True, the pe_scaling is learnable, if False, the pe_scaling is fixed
                 depth_model_type = 'depth_anything', # 'depth_anything' or 'unidepth'
                 precomputed_depth_root=None, # optional root dir for precomputed depth .pt files
                 use_rope = False,
                 ):
        
        super(SpaceDriveBatch, self).__init__(train_cfg, test_cfg,)
        
        # ------------Configurations------------
        # general config
        self.vis_3d_pos = vis_3d_pos
        self.io_3d_pos = io_3d_pos

        # vlm config
        self.stride = stride
        self.lm_path = lm_head
        
        if 'llava' in processor:
            self.lm_type = 'llava'
        elif 'Qwen' in processor:
            self.lm_type = 'qwenvl25'

        # ego status configs
        self.ego_status = ego_status
        self.ego_status_len = ego_status_len
        if self.ego_status is not None:
            self.reset_memory()

        # ablations
        self.input_pe_mlp = input_pe_mlp
        self.use_rope = use_rope
        self.enable_pe_input = enable_pe_input
        self.use_vae_to_replace_mlp = use_vae_to_replace_mlp

        # paths
        self.save_path = save_path
        self.precomputed_depth_store = None
        if precomputed_depth_root is not None:
            self.precomputed_depth_store = PrecomputedDepthStore(precomputed_depth_root)

        # concept testing config
        self.single_token_output = single_token_output # use single output token to decode 6 coords


        # ------------Initialization------------

        # vlm init
        if processor is not None:
            self.processor = AutoProcessor.from_pretrained(processor)
            self.merge_size = self.processor.image_processor.merge_size if hasattr(self.processor.image_processor, 'merge_size') else 1

        if tokenizer is not None:
            self.tokenizer =  AutoTokenizer.from_pretrained(tokenizer,
                                        )
            self.processor.tokenizer = self.tokenizer
        else:
            self.tokenizer = None

        self.llm_hidden_dim = 4096
        if lm_head is not None:
            self.lm_head = load_model(lm_head, tokenizer, use_lora, frozen, llm_lora_rank=llm_lora_rank )
            self.lm_head.base_model.model.tokenizer = self.tokenizer
            if 'llava' in lm_head:
                self.llm_hidden_dim =4096
            elif 'Qwen' in lm_head:
                self.llm_hidden_dim = self.lm_head.base_model.model.config.hidden_size
            else:
                self.llm_hidden_dim = 4096
                print('Warning: llm_hidden_dim is set to 4096 by default, please check if this is correct for your lm_head:', lm_head)
        

        # PE init
        self.position_encoder = None
        if self.vis_3d_pos or self.io_3d_pos:
            # positional encoding
            self.pe_freq_coeff = pe_freq_coeff
            self.pe_freq_scaling = pe_freq_scaling
            self.pe_scaling = pe_scaling
            self.pe_type = pe_type
            self.fone_dim = fone_dim

            self.learnable_pe_scaling = learnable_pe_scaling
            if learnable_pe_scaling:
                self.pe_scaling = nn.Parameter(torch.tensor(pe_scaling).float(), requires_grad=True)

            self.position_encoder = PositionalEncoding3D(self.llm_hidden_dim, dtype_override=torch.float32, temperature=1e-4, freq_coeff=self.pe_freq_coeff, freq_scaling= self.pe_freq_scaling,  pe_type=self.pe_type, fone_dim = self.fone_dim, pe_scaling=self.pe_scaling) 

            if input_pe_mlp:
                # use mlp to get 3d positional encoding from input coordinates, if False, use a PositionalEncoding3D module to get 3d positional encoding from input coordinates
                if pe_decode_method  == 'l2_coords_mlp_2layer':
                    self.position_encoder_mlp = nn.Sequential(
                        nn.Linear(3, self.llm_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim),
                    )
                elif pe_decode_method  == 'l2_coords_mlp':
                    self.position_encoder_mlp = nn.Linear(3, self.llm_hidden_dim)

        # depth model init
        if self.vis_3d_pos:
            self.depth_model_type = depth_model_type
            self.depth_model = None
            # When precomputed depth is enabled, do not load heavy depth backbones.
            if self.precomputed_depth_store is None:
                if self.depth_model_type == 'depth_anything':
                    self.depth_anything_processor =  AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
                    self.depth_model  = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", device_map= {'':torch.cuda.current_device()}) 

                elif self.depth_model_type == 'unidepth':
                    type_ = "l"  # available types: s, b, l
                    name = f"unidepth-v2-vit{type_}14"
                    self.depth_model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}").eval()
                    self.depth_model.interpolation_mode = "bilinear"

                # freeze depth model
                for param in self.depth_model.parameters():
                    param.requires_grad = False


        # loss init
        self.pe_decode_method = None
        if self.io_3d_pos:
            self.loss_pos_lambda = torch.tensor(loss_pos_lambda)
            self.pe_decode_method = pe_decode_method # 'cosine', 'l2', 'l2_coords'
            self.include_semantic_posemb = include_semantic_posemb
            self.supervise_semantic_posemb = supervise_semantic_posemb
            self.single_coords_only = single_coords_only
            self.planning_only = planning_only

            self.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0] 
            self.voxel_size = [0.4, 0.4, 8]
            self.pos_emb_grid = None
            if 'mlp' not in self.pe_decode_method:
                self.pos_emb_grid = self.position_encoder.pos_grid_3d(
                    self.pc_range, 
                    voxel_size=self.voxel_size)
            if self.pe_decode_method == 'l2_coords_mlp':
                self.mlp_output_coords = nn.Linear(self.llm_hidden_dim, 3)
            elif self.pe_decode_method == 'l2_coords_mlp_2layer':
                if self.single_token_output:
                    self.mlp_output_coords = nn.Sequential(
                        nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.llm_hidden_dim, 6*3 ),
                        nn.Unflatten(-1, (6, 3)),
                    )
                else:
                    if self.use_vae_to_replace_mlp:
                        self.vae_output_coords = VAEPEDecoder(llm_hidden_dim=self.llm_hidden_dim, latent_dim=32, with_cur=with_cur)
                    else:
                        self.mlp_output_coords = nn.Sequential(
                            nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim),
                            nn.ReLU(),
                            nn.Linear(self.llm_hidden_dim, 3 )
                        )

            self.coords_l2_loss = nn.MSELoss(reduction='mean')
            self.huber_loss = nn.HuberLoss(reduction='mean', delta=huber_delta)
            self.l1_loss = nn.L1Loss(reduction='mean')

            self.loss_pos_type = loss_pos_type # 'huber' or 'l2', type of loss for 3d positional encoding, 'huber' is the default, 'l2' is the l2 loss
            if self.loss_pos_type == 'huber':
                self.loss_pos_func = self.huber_loss
            elif self.loss_pos_type == 'l2':
                self.loss_pos_func = self.coords_l2_loss
            elif self.loss_pos_type == 'l1':
                self.loss_pos_func = self.l1_loss



            if lm_head is not None:
                self.lm_head.base_model.model.position_encoder = self.position_encoder
                self.lm_head.base_model.model.pc_range = self.pc_range
                self.lm_head.base_model.model.voxel_size = self.voxel_size
                self.lm_head.base_model.model.pos_emb_grid = self.pos_emb_grid


        if ego_status is not None:
            if  'feature' in ego_status:
                self.ego_status_mlp = nn.Sequential(
                    nn.Linear(14*self.ego_status_len + 14 + 16*self.ego_status_len, self.llm_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(self.llm_hidden_dim, self.llm_hidden_dim),
                )



    @property
    def with_lm_head(self):
        """bool: Whether the detector has a lm head."""
        return hasattr(self,
                       'lm_head') and self.lm_head is not None

    def _extra_lm_forward_kwargs(self, input_ids):
        """Optional multimodal kwargs for `lm_head` / `generate` (see `SpaceDriveQwen3VL`)."""
        return {}

    ############### 3D position functions ###############
    def prepare_location(self,image_grid_thw , pixel_values):
        pad_h, pad_w = image_grid_thw[0][0][1:3] * self.stride
        bs = pixel_values.shape[0]
        n = 6 
        x = pixel_values.reshape(bs,n, image_grid_thw[0][0][1], image_grid_thw[0][0][2], -1) # (bs, 6, 46, 46, 1176)
        x = x.permute(0, 1, 4, 2, 3).flatten(0, 1) # (bs*n, 1176, 46, 46) # 1176 is channel(3) * temporal_patch_size(2, because it has to keep this consistent with vedio) * patch_size (14) * patch_size(14)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1) # NOTE: stride must match the resolution from qwen_utils
        return location

    def _extract_sample_ids(self, img_metas, batch_size):
        if img_metas is None:
            return None

        if isinstance(img_metas, (list, tuple)):
            if len(img_metas) == 0:
                return None
            # Test-time wrappers may add one more nesting level.
            if isinstance(img_metas[0], (list, tuple)):
                img_metas = [x[0] for x in img_metas if len(x) > 0]
            sample_ids = []
            for meta in img_metas:
                if not isinstance(meta, dict) or 'sample_idx' not in meta:
                    return None
                sample_ids.append(str(meta['sample_idx']))
            if len(sample_ids) == batch_size:
                return sample_ids
        return None

    def depth_prediction(self, img, intrinsics= None, img_metas=None):
        '''
        predict the depth of multi_view image 
        Args:
            img: torch.Tensor (batch_size, num_views, H, W, 3) # (640, 640, 3)
        Returns:
            depth: torch.Tensor (batch_size, num_views, H, W)
        '''
        B, N, C, H, W = img.shape
        device = img.device

        if self.precomputed_depth_store is not None:
            sample_ids = self._extract_sample_ids(img_metas, B)
            if sample_ids is None:
                raise RuntimeError(
                    "Precomputed depth is enabled, but sample_idx metadata is missing. "
                    "Please ensure img_metas contains sample_idx for every batch item."
                )
            cached = self.precomputed_depth_store.load_batch(
                sample_ids=sample_ids,
                num_views=N,
                height=H,
                width=W,
                device=device,
                dtype=img.dtype,
            )
            if cached is not None:
                return cached
            raise RuntimeError(
                "Precomputed depth is enabled, but cache file is missing or shape mismatched. "
                "Regenerate cache or fix precomputed_depth_root."
            )

        if self.depth_model_type == 'depth_anything':
            img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1]).permute(0, 2, 3, 1) # (batch_size * num_views, H, W, 3)
            img = img.cpu().numpy().astype('uint8')
            img = [to_pil_image(i) for i in img]
            inputs = self.depth_anything_processor(images=img, return_tensors="pt").to(self.depth_model.device).to(self.depth_model.dtype)
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
                predicted_depth = outputs.predicted_depth
        elif self.depth_model_type == 'unidepth':
            rgb_torch = img.reshape(B*N, img.shape[-3], img.shape[-2], img.shape[-1]).to(self.depth_model.device)
            intrinsics_torch = intrinsics[..., :3, :3].squeeze(0) # (B x num_views, 3, 3)
            camera = Pinhole(K=intrinsics_torch)
            predictions = self.depth_model.infer(rgb_torch)
            predicted_depth = predictions["depth"]

        depth = predicted_depth.view(B, N, predicted_depth.shape[-2], predicted_depth.shape[-1]).to(device)
        return depth

    def pool_downsample(self, img, output_size, mode='min'):
        '''
        Args:
            img: shape (B*num_views , c , h, w )
            mode: str,  'min', 'median', 'avg'
        '''

        B_views = img.shape[0]

        input_h, input_w = img.shape[-2:]
        output_h, output_w = output_size

        stride_h = input_h / output_h
        stride_w = input_w / output_w

        pooled_img = torch.zeros((B_views, 1, output_h,  output_w), dtype=img.dtype, device= img.device)
    
        for i in range(output_h):
            for j in range(output_w):

                start_h = round(i * stride_h)
                end_h = round((i + 1) * stride_h)
                start_w = round(j * stride_w)
                end_w = round((j + 1) * stride_w)

                end_h = min(end_h, input_h)
                end_w = min(end_w, input_w)

                window = img[:, :, start_h:end_h, start_w:end_w]

                if window.numel() > 0:
                    if mode == 'min':
                        pooled_img[:, :, i, j], _ = torch.min(window.reshape(window.shape[0], window.shape[1], -1 ), dim=-1)
                    elif mode == 'median':
                        pooled_img[:, :, i, j], _  = torch.median(window.reshape(window.shape[0], window.shape[1], -1 ), dim=-1)
                    elif mode == 'avg':
                        pooled_img[:, :, i, j] = torch.mean(window.reshape(window.shape[0], window.shape[1], -1 ), dim=-1)

        return pooled_img

    def position_embeding(self, data, memory_centers, img_metas, depth, image_grid_thw, visualize_pc=False, imgs=None, sample_idx=None):
        '''
        encode 3D position of each pixel in the image to a position embedding
        Args:
            data: dict, contains 'intrinsics', 'lidar2img'
            memory_centers: torch.Tensor (B,num_views, H, W, 2), the centers of each pixel in the image (relative position from 0 to 1, will be rescaled to the absolute position in this function)
            img_metas: list[dict], contains 'pad_shape' for each image
            depth: torch.Tensor (B, num_views, H, W), the depth of each pixel in the image
            image_grid_thw: torch.Tensor (B, num_views, 3), the grid of each pixel in the image
        '''
        B = data['intrinsics'].size(0)

        memory_centers = memory_centers.reshape(-1, memory_centers.shape[-3], memory_centers.shape[-2], 2).permute(0, 3, 1, 2)
        depth = depth.reshape(-1, depth.shape[-2], depth.shape[-1]).unsqueeze(1)
        
        img_token_h, img_token_w = int(image_grid_thw[0,0,1] / self.merge_size), int(image_grid_thw[0,0,2] / self.merge_size)

        memory_centers = torch.nn.functional.interpolate(
            memory_centers, size=(img_token_h, img_token_w), mode='bilinear', align_corners=False)

        depth = self.pool_downsample(depth, (img_token_h, img_token_w), 'min' ) 

        memory_centers = memory_centers.permute(0, 2, 3, 1).reshape(-1, img_token_h, img_token_w, 2)
        depth = depth.permute(0, 2, 3, 1).reshape(-1, img_token_h, img_token_w, 1)

        eps = 1e-5
        BN, H, W, _ = memory_centers.shape

        intrinsic = torch.stack([data['intrinsics'][..., 0, 0], data['intrinsics'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)

        pad_h, pad_w = image_grid_thw[0][0][1:3] * self.stride
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        memory_centers = memory_centers.detach().view(B, LEN, 2)

        coords_d = depth.view(B, LEN, 1)

        coords = torch.cat([memory_centers, coords_d], dim=-1)
        ones_col = torch.ones_like(coords[..., :1])
        coords = torch.cat((coords, ones_col), dim=-1)

        xy = coords[..., :2]
        depth_term = torch.maximum(coords[..., 2:3].clone(), torch.ones_like(coords[..., 2:3]) * eps)
        xy = xy * depth_term
        coords = torch.cat([xy, coords[..., 2:]], dim=-1)

        coords = coords.unsqueeze(-1)

        img2lidars = data['lidar2img'].inverse()
        img2lidars = img2lidars.view(BN, 1,  4, 4).repeat(1, H*W, 1, 1).view(B, LEN, 4, 4)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]

        coords3d = coords3d.reshape(B, -1, 3)

        if visualize_pc:
            self.vis_point_cloud(coords3d=coords3d, imgs=data['img'], sample_idx=img_metas[0]['sample_idx'], img_token_h=img_token_h, img_token_w=img_token_w, save_path='./vis/pc_vis')

        if self.input_pe_mlp: 
            coords_position_embeding = self.position_encoder_mlp(coords3d)
        else: 
            coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding, coords3d

    ############### Ego status ###############
    def reset_memory(self):
        self.memory_canbus = None
        self.memory_egopose = None
        self.sample_time = None
        self.memory_count = 0
    
    def memory_refresh(self, memory, prev_exist):
        memory_shape = memory.shape
        view_shape = [1 for _ in range(len(memory_shape))]
        prev_exist = prev_exist.view(-1, *view_shape[1:]) 
        return memory * prev_exist
    
    def pre_update_memory(self, data):
        B = data['intrinsics'].size(0)
        if self.memory_canbus is None:
            self.memory_egopose = data['intrinsics'].new_zeros(B, self.ego_status_len, 4, 4) 
            self.memory_canbus = data['intrinsics'].new_zeros(B, self.ego_status_len, 14)
            self.sample_time = data['intrinsics'].new_zeros(B)
        else:
            self.memory_count += 1
            self.sample_time += data['timestamp']
            prev_exist = (torch.abs(self.sample_time) < 2.0).to(data['intrinsics'].dtype)

            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose # world to local
            self.memory_egopose = self.memory_refresh(self.memory_egopose[:, :self.ego_status_len], prev_exist)
            self.memory_canbus = self.memory_refresh(self.memory_canbus[:, :self.ego_status_len], prev_exist)

            self.memory_count = self.memory_count * prev_exist

            self.sample_time = data['timestamp'].new_zeros(B)

    def post_update_memory(self, data, rec_ego_pose, rec_can_bus):
        self.memory_canbus = torch.cat([rec_can_bus, self.memory_canbus], dim=1)
        self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose # local to world
        self.sample_time -= data['timestamp']

    ############### Miscs ###############
    def format_number(self, n, decimal_places=1):
        if abs(round(n, decimal_places)) <= 1e-2:
            return 0.0
        else:
            format_string = f"{{n:+.{decimal_places}f}}"
            return format_string.format(n=n)
        
    def vis_depth(self, save_path, img, depth, sample_idx):
        '''
        save the image and depth map for visualization
        Args:
            save_path: str, the path to save the image and depth map
            img: torch.Tensor (B, num_views, 3, H, W)
            depth: torch.Tensor (B, num_views, H, W)
        '''
        os.makedirs(save_path, exist_ok=True)
        import matplotlib.pyplot as plt
        from torchvision.transforms import ToPILImage
        from PIL import Image, ImageDraw

        B = img.shape[0]

        for b in range(B):
            for n in range(6):
                img_pil = ToPILImage()(img[b,n])
                depth_map = depth[b,n].cpu().numpy()
                depth_map_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-5)
                depth_map_color = plt.get_cmap('plasma')(depth_map_norm)[:,:,:3]
                depth_map_color = (depth_map_color * 255).astype('uint8')
                depth_map_color = ToPILImage()(depth_map_color).resize((img[b,n].shape[1], img[b,n].shape[2]))
                blended = Image.blend(img_pil, depth_map_color, alpha=0.5)

                grid_size = img[b,n].shape[1] / depth_map.shape[0]
                draw = ImageDraw.Draw(blended)
                for i in range(depth_map.shape[0]):
                    for j in range(depth_map.shape[1]):
                        draw.text((j*grid_size, i*grid_size), f"{depth_map[i, j]:.1f}", fill="white")

                blended.save(os.path.join(save_path, f'sample{sample_idx}_view{n}_vis.png'))  

    def vis_point_cloud(self, coords3d, imgs, sample_idx, img_token_h, img_token_w, save_path ='./vis/pc_vis'):
        os.makedirs(save_path, exist_ok=True)
        for b in range(coords3d.shape[0]):
            pc = o3d.geometry.PointCloud()
            pc.points = o3d.utility.Vector3dVector(coords3d[b].detach().cpu().numpy())
            o3d.io.write_point_cloud(f'{save_path}/pc_{sample_idx}.ply', pc)
            print(f'save point cloud to {save_path}/pc_{sample_idx}.ply')

            if imgs is not None:
                img = imgs[b].permute(0, 2, 3, 1).cpu().numpy()
                import cv2
                img = np.array([cv2.resize(img[i], (img_token_h, img_token_w)) for i in range(img.shape[0])])
                img = img.reshape(-1, img_token_h, img_token_w, 3)
                pc.colors = o3d.utility.Vector3dVector(img.reshape(-1, 3)/255)
                o3d.io.write_point_cloud(f'{save_path}/pc_color_{sample_idx}.ply', pc)
                print(f'save point cloud with color to {save_path}/pc_color_{sample_idx}.ply')

    ##############################################
    ############### Main functions ###############
    ##############################################
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    ############### Train ###############
    def forward_train(self,
                      img_metas=None,
                      input_ids=None,
                      vlm_labels=None,
                      pixel_values=None,
                      image_grid_thw=None,
                      coords_pos_tensor=None,
                      **data):

        if self.ego_status is not None:
            self.pre_update_memory(data)

        if self.tokenizer is not None:
            input_ids = torch.nn.utils.rnn.pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.tokenizer.pad_token_id)
            
            vlm_labels = torch.nn.utils.rnn.pad_sequence(vlm_labels,
                                                    batch_first=True,
                                                    padding_value=IGNORE_INDEX)
            coords_pos_tensor = torch.cat(coords_pos_tensor, dim=0)
            
            vlm_attn_mask = input_ids.ne(self.tokenizer.pad_token_id)
        else:
            input_ids = None
            vlm_labels = None
            vlm_attn_mask = None

        losses = self.forward_train_vlm( img_metas, input_ids, vlm_labels,vlm_attn_mask,pixel_values,image_grid_thw,coords_pos_tensor, **data)


        if self.ego_status is not None:
            rec_can_bus = torch.cat([data['command'].unsqueeze(-1), data['can_bus']], dim=-1).unsqueeze(1) #shape (B, 1, 14)
            B = rec_can_bus.shape[0]
            rec_ego_pose = torch.eye(4, device=rec_can_bus.device).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1) # shape (B, 1, 4, 4)
            self.post_update_memory(data, rec_ego_pose, rec_can_bus)

        return losses

    def forward_train_vlm(self,
                          img_metas,
                          input_ids, 
                          vlm_labels, 
                          vlm_attn_mask,
                          pixel_values,
                          image_grid_thw,
                          coords_pos_tensor,
                          **data):

        B = pixel_values.shape[0]
        
        pos_embed = None
        if self.vis_3d_pos:
            depth = self.depth_prediction(data['img'], data['intrinsics'], img_metas=img_metas)

            location = self.prepare_location(image_grid_thw, pixel_values)

            pos_embed, coords3d = self.position_embeding(data, location, img_metas, depth, image_grid_thw, False, data['img'], sample_idx=img_metas[0]['sample_idx'])  # (6, 640, 640, 3) to (6,46,46). shape (B, seq_len, hidden_dim)

        io_coords_pos = None
        if self.io_3d_pos:
            io_coords = coords_pos_tensor # shape (num_coords, 2) only x,y  
            z_dim = torch.zeros(io_coords.shape[0]).unsqueeze(-1).to(io_coords.device) # shape (num_coords, 1) z=0
            gt_coords_xy = io_coords
            io_coords = torch.cat((io_coords, z_dim), dim=-1).unsqueeze(0) # shape (num_coords, 3) x,y,z

                
            if self.input_pe_mlp:
                # use mlp to get 3d positional encoding from input coordinates
                io_coords_pos = self.position_encoder_mlp(io_coords)
            else:
                io_coords_pos = self.position_encoder(io_coords)
                # no gradient for io_coords_pos
                io_coords_pos = io_coords_pos.detach() # shape (B, num_coords, 2048) 2048 is the embed_dims of position_encoder
                

            has_gt_planning = data.get('has_gt_planning', None)


        if self.ego_status is not None:
            rec_can_bus = torch.cat([data['command'].unsqueeze(-1), data['can_bus']], dim=-1)

            ego_feature = torch.empty(B, 0, self.llm_hidden_dim, device=rec_can_bus.device)

            if 'feature' in self.ego_status:
                ego_mlp_input = torch.cat([self.memory_canbus.reshape(B, -1), rec_can_bus.reshape(B, -1), self.memory_egopose.reshape(B, -1, 16).reshape(B, -1)], dim=-1)
                ego_token = self.ego_status_mlp(ego_mlp_input).unsqueeze(1) # shape (B, 1, hidden)
                ego_feature = torch.cat([ego_feature, ego_token], dim=1) # shape (B, 1, hidden)
                
                # add a extra image token id at the end of images in input_ids and a ignore index in labels
                if input_ids is not None:
                    if (input_ids[0] == VISION_END_TOKEN_INDEX).sum() == 0:
                        last_vision_end_token = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero().max()
                    else:
                        last_vision_end_token = (input_ids[0] == VISION_END_TOKEN_INDEX).nonzero().max()

                    # insert IMAGE_TOKEN_INDEX only — no vision_start/end wrapper to avoid
                    # get_rope_index treating ego token as a new image region (image_grid_thw mismatch)
                    insert_input_ids = torch.tensor([IMAGE_TOKEN_INDEX], device=input_ids.device).unsqueeze(0).repeat(B, 1)
                    insert_labels = torch.tensor([IGNORE_INDEX], device=vlm_labels.device).unsqueeze(0).repeat(B, 1)
                    insert_attn_mask = torch.tensor([1], device=vlm_attn_mask.device).unsqueeze(0).repeat(B, 1)

                    input_ids = torch.cat([input_ids[:, :last_vision_end_token+1], insert_input_ids, input_ids[:, last_vision_end_token+1:]], dim=-1)
                    vlm_labels = torch.cat([vlm_labels[:, :last_vision_end_token+1], insert_labels, vlm_labels[:, last_vision_end_token+1:]], dim=-1)
                    vlm_attn_mask = torch.cat([vlm_attn_mask[:, :last_vision_end_token+1], insert_attn_mask, vlm_attn_mask[:, last_vision_end_token+1:]], dim=-1)
            if 'PE' in self.ego_status:
                past_xyz = self.memory_egopose[:, :self.ego_status_len, :3, 3]
                encoded_past_xyz = self.position_encoder(past_xyz.reshape(B, -1, 3)).reshape(B, self.ego_status_len, -1)
                ego_feature = torch.cat([ego_feature, encoded_past_xyz], dim=1)

                if input_ids is not None:
                    # find the last vision end token
                    if (input_ids[0] == VISION_END_TOKEN_INDEX).sum() == 0: # llava
                        last_vision_end_token = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero().max()
                    else:
                        last_vision_end_token = (input_ids[0] == VISION_END_TOKEN_INDEX).nonzero().max()

                    len_past_pos = encoded_past_xyz.shape[1]

                    insert_input_ids = torch.tensor([POS_INDICATOR_TOKEN_INDEX, IMAGE_TOKEN_INDEX ]*len_past_pos, device = input_ids.device).unsqueeze(0).repeat(B, 1)
                    insert_labels = torch.tensor([IGNORE_INDEX]*len_past_pos *2, device = vlm_labels.device).unsqueeze(0).repeat(B, 1)
                    insert_attn_mask = torch.tensor([1]*len_past_pos*2, device = vlm_attn_mask.device).unsqueeze(0).repeat(B, 1)

                    input_ids = torch.cat([input_ids[:, :last_vision_end_token+1], insert_input_ids, input_ids[:, last_vision_end_token+1:]], dim=-1)
                    vlm_labels = torch.cat([vlm_labels[:, :last_vision_end_token+1], insert_labels, vlm_labels[:, last_vision_end_token+1:]], dim=-1)
                    vlm_attn_mask = torch.cat([vlm_attn_mask[:, :last_vision_end_token+1], insert_attn_mask, vlm_attn_mask[:, last_vision_end_token+1:]], dim=-1)

        losses = dict()
   
        if self.with_lm_head:

            lm_loss = self.lm_head(
                input_ids=input_ids, 
                attention_mask=vlm_attn_mask,
                labels=vlm_labels,
                pixel_values=pixel_values, 
                image_grid_thw=image_grid_thw, 
                pos_emb=pos_embed,
                io_coords_pos=io_coords_pos, 
                loss_pos_lambda=self.loss_pos_lambda if self.io_3d_pos else None,
                loss_for_pos=self.pe_decode_method,
                include_semantic_posemb= self.include_semantic_posemb if self.io_3d_pos else False,
                supervise_semantic_posemb=self.supervise_semantic_posemb if self.io_3d_pos else False,
                planning_only=self.planning_only if self.io_3d_pos else False,
                single_coords_only=self.single_coords_only if self.io_3d_pos else False,
                has_gt_planning=has_gt_planning if self.io_3d_pos else None,
                gt_coords_xy = gt_coords_xy if self.io_3d_pos else None,
                ego_feature = ego_feature if self.ego_status and ego_feature.numel() > 0  else None,
                enable_pe_input = self.enable_pe_input if self.io_3d_pos else False,
                pos_index = coords3d if self.use_rope else None,
                **self._extra_lm_forward_kwargs(input_ids),
            )

            losses.update(vlm_loss=lm_loss['loss'])
            if self.io_3d_pos: 
                # NOTE use mlp to decode the position embedding if self.pe_decode_method in ['l2_coords_mlp', 'l2_coords_mlp_2layer']
                if self.pe_decode_method in ['l2_coords_mlp', 'l2_coords_mlp_2layer'] and self.io_3d_pos:
                    # decode the position embedding to original coordinates+
                    if len(lm_loss['output_pos'].shape) == 2: # NOTE: because there is an unsqueeze inside plannning_only, so we don't PERFORM unsquezze again here
                        # if output_pos is 2D, we need to reshape it to 3D
                        lm_loss['output_pos'] = lm_loss['output_pos'].unsqueeze(0) # add batch dimension
                    if not has_gt_planning.any():
                        # NOTE: This is supposed to create a fake loss of 0
                        sampled_coords = torch.zeros(B, 1, 3).to(lm_loss['output_pos'].device) * (torch.tensor(self.pc_range[3:6]) - torch.tensor(self.pc_range[0:3])).to(lm_loss['output_pos'].device) + torch.tensor(self.pc_range[0:3]).to(lm_loss['output_pos'].device)
                        if self.input_pe_mlp:
                            sampled_coords_pos = self.position_encoder_mlp(sampled_coords)
                        else:
                            sampled_coords_pos = self.position_encoder(sampled_coords)

                        if self.use_vae_to_replace_mlp:
                            decoded_output_pos, loss_vae_gen = self.vae_output_coords(sampled_coords_pos, sampled_coords[:, :,:2] )
                            losses['loss_vae_gen'] = loss_vae_gen * 0
                            lm_loss['loss_pos'] = self.loss_pos_func(decoded_output_pos[:, :,:2], sampled_coords[:, :,:2]) * 0
                        else:
                            decoded_output_pos = self.mlp_output_coords(sampled_coords_pos).reshape(B, -1, 3)[:, :,:2]
                            lm_loss['loss_pos'] = self.loss_pos_func(decoded_output_pos, sampled_coords[:, :,:2]) * 0
                    else:
                        if self.use_vae_to_replace_mlp:
                            decoded_output_pos, loss_vae_gen = self.vae_output_coords(lm_loss['output_pos'], lm_loss['gt_coords_xy'])
                            losses['loss_vae_gen'] = loss_vae_gen 
                        else:
                            decoded_output_pos = self.mlp_output_coords(lm_loss['output_pos'])

                        if len(decoded_output_pos.shape) == 4: # i.e., batch, B, 6, 3 in single token output test
                            decoded_output_pos = decoded_output_pos.squeeze(1)  # remove the second dimension, shape (B, 6, 3)

                        decoded_output_pos = decoded_output_pos[:, :, :2]  # only take x,y coordinates

                        lm_loss['loss_pos'] = self.loss_pos_func(decoded_output_pos, lm_loss['gt_coords_xy']) * self.loss_pos_lambda

                elif self.pe_decode_method in ['cosine', 'l2']:
                    pass
                else:
                    raise NotImplementedError(f'pe_decode_method {self.pe_decode_method} not implemented')
                losses.update(loss_pos=lm_loss['loss_pos'])
        if self.learnable_pe_scaling:
            losses.update(pe_scaling=self.pe_scaling) # for logging the learnable PE scaling factor

        return losses

    ############### Test ###############
    def forward_test(self, img,  img_metas, rescale, **data):
        if self.ego_status is not None:
            self.pre_update_memory(data)
        for key in data:
             if key not in ['question_text']:
                data[key] = data[key][0].unsqueeze(0) 

        output = self.test_generation(img, img_metas, **data)

        if self.ego_status is not None:
            rec_can_bus = torch.cat([data['command'].unsqueeze(-1), data['can_bus']], dim=-1).unsqueeze(1) #shape (B, 1, 14)
            B = rec_can_bus.shape[0]
            rec_ego_pose = torch.eye(4, device=rec_can_bus.device).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1) # shape (B, 1, 4, 4)
            self.post_update_memory(data, rec_ego_pose, rec_can_bus)

        return output      

    def test_generation(self,  img, img_metas, **data):
        generated_text = self.test_generation_pts(
            img, img_metas, **data)
        return generated_text

    def test_generation_pts(self, img, img_metas, input_ids,pixel_values,image_grid_thw,attention_mask,  **data):
        """Test function of point cloud branch."""
        if 'question_text' in data:
            question_text = data['question_text']
        else:
            question_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
            if 'coords_pos_tensor' in data:
                formatted_points = ', '.join(f"({self.format_number(point[0], 2)}, {self.format_number(point[1], 2)})" for point in data['coords_pos_tensor'][0].cpu().numpy())
                question_text = question_text.replace(', , , , , ', str(formatted_points))

        B = pixel_values.shape[0]
        
        pos_embed = None
        if self.vis_3d_pos:

            depth = self.depth_prediction(img,  data['intrinsics'], img_metas=img_metas)

            location = self.prepare_location(image_grid_thw, pixel_values)

            pos_embed, coords3d = self.position_embeding(data, location, img_metas, depth, image_grid_thw) 

        io_coords_pos = None
        if self.io_3d_pos and 'coords_pos_tensor' in data:
            io_coords = data['coords_pos_tensor'] # shape (B, num_coords, 2) only x,y  
            z_dim = torch.zeros(io_coords.shape[1]).unsqueeze(-1).repeat(B,1,1).to(io_coords.device) # shape (B, num_coords, 1) z=0
            io_coords = torch.cat((io_coords, z_dim), dim=-1) # shape (num_coords, 3) x,y,z
            io_coords_pos = self.position_encoder(io_coords)
            # no gradient for io_coords_pos
            io_coords_pos = io_coords_pos.detach()
    	
        if self.ego_status is not None:
            rec_can_bus = torch.cat([data['command'].unsqueeze(-1), data['can_bus']], dim=-1)

            ego_feature = torch.empty(B, 0, self.llm_hidden_dim, device=rec_can_bus.device)

            if 'feature' in self.ego_status:
                ego_mlp_input = torch.cat([self.memory_canbus.reshape(B, -1), rec_can_bus.reshape(B, -1), self.memory_egopose.reshape(B, -1, 16).reshape(B, -1)], dim=-1)
                ego_token = self.ego_status_mlp(ego_mlp_input).unsqueeze(1) # shape (B, 1, hidden)
                ego_feature = torch.cat([ego_feature, ego_token], dim=1) # shape (B, 1, hidden
                
                # add a extra image token id at the end of images in input_ids and a ignore index in labels
                if input_ids is not None:
                    # find the last vision end token
                    if (input_ids[0] == VISION_END_TOKEN_INDEX).sum() == 0: # llava
                        last_vision_end_token = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero().max()
                    else:
                        last_vision_end_token = (input_ids[0] == VISION_END_TOKEN_INDEX).nonzero().max()

                    # insert IMAGE_TOKEN_INDEX only — no vision_start/end wrapper
                    insert_input_ids = torch.tensor([IMAGE_TOKEN_INDEX], device=input_ids.device).unsqueeze(0)
                    insert_attn_mask = torch.tensor([1], device=attention_mask.device).unsqueeze(0)

                    input_ids = torch.cat([input_ids[:, :last_vision_end_token+1], insert_input_ids, input_ids[:, last_vision_end_token+1:]], dim=-1)
                    attention_mask = torch.cat([attention_mask[:, :last_vision_end_token+1], insert_attn_mask, attention_mask[:, last_vision_end_token+1:]], dim=-1)
            if 'PE' in self.ego_status:
                past_xyz = self.memory_egopose[:, :self.ego_status_len, :3, 3]
                encoded_past_xyz = self.position_encoder(past_xyz.reshape(B, -1, 3)).reshape(B, self.ego_status_len, -1)
                ego_feature = torch.cat([ego_feature, encoded_past_xyz], dim=1)

                if input_ids is not None:
                    if (input_ids[0] == VISION_END_TOKEN_INDEX).sum() == 0:
                        last_vision_end_token = (input_ids[0] == IMAGE_TOKEN_INDEX).nonzero().max()
                    else:
                        last_vision_end_token = (input_ids[0] == VISION_END_TOKEN_INDEX).nonzero().max()

                    len_past_pos = encoded_past_xyz.shape[1]

                    # insert POS_IND and POS_EMB in between
                    insert_input_ids = torch.tensor([POS_INDICATOR_TOKEN_INDEX, IMAGE_TOKEN_INDEX ]*len_past_pos, device = input_ids.device).unsqueeze(0)
                    insert_attn_mask = torch.tensor([1]*len_past_pos*2, device = attention_mask.device).unsqueeze(0)

                    input_ids = torch.cat([input_ids[:, :last_vision_end_token+1], insert_input_ids, input_ids[:, last_vision_end_token+1:]], dim=-1)
                    attention_mask = torch.cat([attention_mask[:, :last_vision_end_token+1], insert_attn_mask, attention_mask[:, last_vision_end_token+1:]], dim=-1)


        generated_text = []
        if self.with_lm_head:
            mmcv.mkdir_or_exist(self.save_path)
            for i, input_ids in enumerate(input_ids): # batch level , is 1
                input_ids = input_ids.unsqueeze(0)
                len_input_ids = input_ids.shape[1]

                if self.lm_type == 'qwenvl25':
                    outputs = self.lm_head.generate( 
                        # forward args
                        input_ids=input_ids,
                        pixel_values=pixel_values, 
                        image_grid_thw=image_grid_thw, 
                        attention_mask=attention_mask,
                        # SpaceDrive args
                        pos_emb=pos_embed, # NOTE: this is visual pos embeds
                        loss_pos_lambda=self.loss_pos_lambda if self.io_3d_pos else None,
                        include_semantic_posemb= self.include_semantic_posemb if self.io_3d_pos else False,
                        planning_only=self.planning_only if self.io_3d_pos else False,
                        single_coords_only=self.single_coords_only if self.io_3d_pos else False,
                        ego_feature = ego_feature if self.ego_status and ego_feature.numel() > 0  else None,
                        enable_pe_input = self.enable_pe_input if self.io_3d_pos else False,
                        pos_index = coords3d if self.use_rope else None,
                        coords_encoder = self.position_encoder_mlp if (not self.single_token_output and self.pe_decode_method is not None and 'mlp' in self.pe_decode_method and self.input_pe_mlp) else self.position_encoder,
                        coords_decoder = self.mlp_output_coords  if  (not self.single_token_output and self.pe_decode_method is not None  and 'mlp' in self.pe_decode_method and not self.use_vae_to_replace_mlp) else  (self.vae_output_coords if self.use_vae_to_replace_mlp else None),
                        ## output args
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        ## hf generate args
                        # do_sample=True,
                        # temperature=0.1,
                        # top_p=0.75,
                        # num_beams=1,
                        ## others
                        max_new_tokens=100,
                        use_cache=True,
                        **self._extra_lm_forward_kwargs(input_ids),
                    )
                elif self.lm_type == 'llava': # NOTE: no adaption for rope as position encoding in llava, so we don't pass pos_index in this case
                    outputs = self.lm_head.generate( 
                        # forward args
                        input_ids=input_ids, 
                        pixel_values=pixel_values, 
                        attention_mask=attention_mask,
                        # SpaceDrive args
                        pos_emb=pos_embed, # NOTE: this is visual pos embeds
                        io_coords_pos=io_coords_pos,        
                        # inputs_embeds = inputs_embeds, # NOTE: this is not needed, we only need last hidden state in generation
                        loss_pos_lambda=self.loss_pos_lambda if self.io_3d_pos else None,
                        include_semantic_posemb= self.include_semantic_posemb if self.io_3d_pos else False,
                        planning_only=self.planning_only if self.io_3d_pos else False,
                        single_coords_only=self.single_coords_only if self.io_3d_pos else False,
                        ego_feature = ego_feature if self.ego_status and ego_feature.numel() > 0  else None,
                        enable_pe_input = self.enable_pe_input if self.io_3d_pos else False,
                        coords_encoder = self.position_encoder_mlp if (not self.single_token_output and self.pe_decode_method is not None and 'mlp' in self.pe_decode_method and self.input_pe_mlp) else self.position_encoder,
                        coords_decoder = self.mlp_output_coords  if  (not self.single_token_output and self.pe_decode_method is not None  and 'mlp' in self.pe_decode_method and not self.use_vae_to_replace_mlp) else  (self.vae_output_coords if self.use_vae_to_replace_mlp else None),
                        ## output args
                        output_hidden_states=True,
                        return_dict_in_generate=True,
                        ## hf generate args
                        # do_sample=True,
                        # temperature=0.1,
                        # top_p=0.75,
                        # num_beams=1,
                        ## others
                        max_new_tokens=100,
                        use_cache=True
                    )


                # for pure and vis_3d_pos, we need to decode the output_ids to original coordinates
                output_ids = outputs['sequences'][0][len_input_ids:].unsqueeze(0) # remove the input_ids part, only keep the generated part
                
                if self.io_3d_pos:
                    last_hidden_state = outputs['hidden_states_for_output'].reshape(1, -1, self.llm_hidden_dim)[0][len_input_ids-1:].unsqueeze(0) # NOTE -1 cause the input embeds is one pos less in the beginning  # this is the last hidden state of last iteration
                    # decode output_ids to text
                    
                    # find the <POS_INDICATOR> token and get the next token as 3d pos embedding
                    pos_indicator_mask = (output_ids[0] == POS_INDICATOR_TOKEN_INDEX)

                    # if two or more <POS_INDICATOR> occurs together, only select the first one
                    if pos_indicator_mask.sum() > 1:
                        pos_embedding_mask = torch.roll(pos_indicator_mask, shifts=1, dims=-1)
                        non_indicator_mask = torch.logical_not(pos_embedding_mask)
                        pos_indicator_mask = pos_indicator_mask & non_indicator_mask

                    if pos_indicator_mask.sum() == 0:
                        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
                        generated_text.append(
                            dict(
                            Q=question_text,
                            A=self.tokenizer.batch_decode(output_ids, skip_special_tokens=False),
                            ))
                        continue
                    
                    # print('output seq with no modification', self.tokenizer.batch_decode(output_ids, skip_special_tokens=False))
                    # shift pos_indicator_mask 1 to the right
                    pos_embedding_mask = torch.roll(pos_indicator_mask, shifts=1, dims=-1)
                    pos_embedding_mask[0] = False
                    pos_embedding_index = pos_embedding_mask.nonzero() # get the index of the next token after <POS_INDICATOR>

                    pos_embedding = last_hidden_state[0, pos_embedding_index, :].reshape(B, -1, self.llm_hidden_dim) #[1, N, 2048]

                    # replace all PE token pos_embedding_index with <POS_EMBEDDING> token_id in text
                    output_ids[0,pos_embedding_index ] = POS_EMBEDDING_TOKEN_INDEX


                    # decode output_ids to text
                    output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
                    # print(f'output_text: {output_text}')
                    if self.pe_decode_method in ['cosine', 'l2']:# Otherwise the decoding is already performed for the loss function
                        # decode output_pos and gt_pos to original coordinates
                        with torch.no_grad():
                            # decode pos_embedding to original coordinates
                            decoded_pos, interpolated_pos = self.position_encoder.decode_pos(pos_embedding, self.pos_emb_grid,
                                                                                    self.pc_range, self.voxel_size, sim_method=self.pe_decode_method)
                        decoded_pos, interpolated_pos = decoded_pos.squeeze(0), interpolated_pos.squeeze(0)
                    elif self.pe_decode_method == 'l2_coords_full_grid':
                        with torch.no_grad():
                            interpolated_pos = self.position_encoder.decode_pos_full_grid( pos_embedding, self.pos_emb_grid,
                                                                                        self.pc_range, self.voxel_size, sim_method='cosine')
                            interpolated_pos = interpolated_pos.squeeze(0)
                    elif 'mlp' in self.pe_decode_method:
                        # decode pos_embedding to original coordinates using mlp
                        with torch.no_grad():
                            pos_embedding = pos_embedding.to(torch.float32) # make sure the pos_embedding is in float32
                            if self.use_vae_to_replace_mlp:
                                interpolated_pos = self.vae_output_coords(pos_embedding).reshape(1, -1, 2)[0,:,:]
                            else:
                                interpolated_pos = self.mlp_output_coords(pos_embedding).reshape(1, -1, 3)[0,:,:]
                                
                                last_hidden_state = last_hidden_state.to(torch.float32) # make sure the last_hidden_state is in float32
                                sanity_check_decoded_pos = self.mlp_output_coords(last_hidden_state)[0,:,:]
                                # print(f'sanity_check_decoded_pos: {sanity_check_decoded_pos}, interpolated_pos: {interpolated_pos}')
                    

                    if self.single_token_output:
                        full_traj = ''
                        for n in range(interpolated_pos.shape[0]):
                            # Extract and format current position
                            if interpolated_pos[n].shape[0] == 2:
                                x, y = interpolated_pos[n]
                                coord_str = f'({float(x):.2f}, {float(y):.2f})'
                            elif interpolated_pos[n].shape[0] == 3:
                                x, y, z = interpolated_pos[n]
                                coord_str = f'({float(x):.2f}, {float(y):.2f})'

                            full_traj = full_traj + POS_INDICATOR_TOKEN +coord_str
                        
                        output_text = output_text.replace(POS_INDICATOR_TOKEN + POS_EMBEDDING_TOKEN, full_traj, 1)


                    else:
                        # replace coordinates in text with the decoded coordinates
                        # Process each coordinate one by one
                        for n in range(interpolated_pos.shape[0]):
                            # Extract and format current position
                            if interpolated_pos[n].shape[0] == 2:
                                x, y = interpolated_pos[n]
                                coord_str = f'({float(x):.2f}, {float(y):.2f})'
                            elif interpolated_pos[n].shape[0] == 3:
                                x, y, z = interpolated_pos[n]
                                coord_str = f'({float(x):.2f}, {float(y):.2f})'
                            
                            # Replace only the next occurrence of the token
                            output_text = output_text.replace(POS_EMBEDDING_TOKEN, coord_str, 1)

                    print(f'generated text: {output_text}')
                    generated_text.append(
                        dict(
                        Q=question_text,
                        A=output_text,
                        ))
                else:   
                    generated_text.append(
                        dict(
                        Q=question_text,
                        A=self.tokenizer.batch_decode(output_ids, skip_special_tokens=True),
                        ))
            with open(self.save_path+img_metas[0]['sample_idx'], 'w') as file:
                json.dump(generated_text, file)
        return generated_text
    
