# GT Attention Map 생성 명세서

## 목적
KLAL (KL Attention Loss) 적용을 위해, nuScenes의 3D bbox + HD map annotation으로부터 각 카메라 view별 GT attention map을 생성한다.

---

## Input
- `sample_token`: nuScenes sample token
- `cam_intrinsics`: (6, 3, 3) — 6개 카메라 intrinsic
- `cam_extrinsics`: (6, 4, 4) — 6개 카메라 extrinsic (lidar2cam 또는 world2cam)
- `annotations`: nuScenes 3D bbox list — 각 object의 center(x,y,z), size(w,l,h), rotation
- `hd_map`: nuScenes map API — lane, road_segment, crosswalk 등
- `patch_size`: int (default=14) — Qwen2.5-VL ViT patch 크기
- `merge_size`: int (default=2) — Qwen2.5-VL processor merge size
- `image_size`: (H, W) — 원본 이미지 크기 (e.g., 640x640)
- `sigma`: float (default=1.0) — Gaussian smoothing sigma

## Output
- `gt_attention_map`: (num_visual_tokens,) — 합이 1인 확률 분포
  - num_visual_tokens = 6 × (H // patch_size // merge_size) × (W // patch_size // merge_size)

---

## 처리 흐름

### Step 1: Visual Token Grid 크기 계산
```python
token_h = image_h // patch_size // merge_size  # e.g., 640//14//2 = 22
token_w = image_w // patch_size // merge_size  # e.g., 640//14//2 = 22
# 카메라 6개 → 총 visual token 수 = 6 * 22 * 22 = 2904
```

### Step 2: 각 카메라별 GT map 생성

카메라 6개에 대해 각각 (token_h, token_w) 크기의 map을 생성.

#### 2-1. 3D BBox → 2D Patch 매핑
```
for each annotation:
    1. 3D bbox의 8개 corner 계산 (center + size + rotation)
    2. cam_extrinsic으로 camera 좌표로 변환
    3. cam_intrinsic으로 pixel 좌표로 projection
    4. depth < 0인 점 제거 (카메라 뒤에 있는 경우)
    5. image 범위 밖의 점 clamp
    6. projected 2D bbox의 min/max로 사각형 영역 계산
    7. 해당 영역이 속하는 patch grid cell에 weight 할당
       - weight = 1.0 (모든 object 동일) 또는 distance 기반 가중치
```

#### 2-2. HD Map → 2D Patch 매핑
```
for each lane/road_segment:
    1. polyline의 3D point들을 가져옴
    2. cam_extrinsic → cam_intrinsic으로 pixel 좌표로 projection
    3. depth < 0인 점 제거
    4. 해당 pixel이 속하는 patch grid cell에 weight 할당
       - weight = 1.0
```

### Step 3: Gaussian Smoothing
```python
from scipy.ndimage import gaussian_filter
cam_map = gaussian_filter(cam_map, sigma=sigma)
```

### Step 4: 6개 카메라 concat + Normalize
```python
# 6개 카메라의 map을 SpaceDrive의 visual token 순서대로 concat
gt_map = np.concatenate([cam_maps[i].flatten() for i in range(6)])

# background에 작은 값 추가 (완전한 0 방지)
gt_map = gt_map + 1e-6

# 확률 분포로 normalize
gt_map = gt_map / gt_map.sum()
```

### Step 5: 저장
```python
torch.save(torch.tensor(gt_map, dtype=torch.float32), 
           f'gt_attention_maps/{sample_token}.pt')
```

---

## 주의사항

1. **카메라 순서**: SpaceDrive에서 6개 카메라 visual token이 concat되는 순서와 GT map의 카메라 순서가 일치해야 함. SpaceDrive dataset code에서 카메라 순서 확인 필수.

2. **Patch 좌표 매핑**: pixel (u, v)에서 patch index로 변환 시:
   ```python
   patch_i = int(v // (patch_size * merge_size))
   patch_j = int(u // (patch_size * merge_size))
   ```

3. **Occlusion 미처리**: 단순 projection이라 가려진 object도 map에 포함됨. 일단은 무시하고, 필요시 depth ordering으로 처리.

4. **빈 frame 처리**: annotation이 하나도 없는 frame은 uniform distribution 사용.

5. **Multi-view 중복**: 같은 object가 여러 카메라에 보이면 각 카메라 map에 중복 등록 — 이건 정상 동작.

---

## 검증 방법
- 생성된 GT map을 원본 이미지 위에 heatmap으로 overlay하여 시각화
- Object bbox와 road 영역이 high value인지 눈으로 확인
- 최소 10개 sample에서 sanity check 수행