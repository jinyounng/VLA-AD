# KLAL Loss 추가 명세서 (SpaceDrive)

## 개요
GT attention map과 모델 내부 attention map 간 KL divergence를 auxiliary loss로 추가한다.

```
L_total = L_NTP + L_pos + λ_klal · L_KLAL
```

---

## 변경 파일 목록

### 1. Dataset — GT map 로딩
**파일**: SpaceDrive dataset `__getitem__` 

```python
# GT attention map 로딩 추가
gt_attn_path = f'{self.klal_gt_dir}/{sample_token}.pt'
gt_attention_map = torch.load(gt_attn_path)  # (num_visual_tokens,)
```

- collate_fn에서 batch로 stack
- `forward_train`까지 `data['gt_attention_map']`으로 전달

---

### 2. Model forward — attention 추출 + loss 계산
**파일**: `spacedrive.py` → `forward_train_vlm`

#### Step 1: output_attentions=True 추가

```python
lm_loss = self.lm_head(
    input_ids=input_ids,
    attention_mask=vlm_attn_mask,
    labels=vlm_labels,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
    output_attentions=True,   # ← 추가
    ...
)
```

⚠️ `output_attentions=True`는 메모리를 많이 먹음. 모든 layer의 (B, H, seq, seq) attention을 저장하기 때문. 7B 모델 + 긴 시퀀스면 OOM 가능성 있음. 필요시 특정 layer만 추출하는 hook 방식으로 대체.

#### Step 2: Last answer token 위치 찾기

```python
# POS_INDICATOR token 위치 = answer token 역할
# 마지막 POS_INDICATOR의 위치를 last answer token으로 사용
pos_indicator_mask = (input_ids == POS_INDICATOR_TOKEN_INDEX)
last_answer_pos = pos_indicator_mask.nonzero()[:, 1].max()  # 마지막 위치
```

#### Step 3: Visual token 구간 파악

```python
# visual token 구간: IMAGE_TOKEN_INDEX인 위치들
# 또는 VISION_START ~ VISION_END 사이
vision_start = (input_ids == VISION_START_TOKEN_INDEX).nonzero()[:, 1].min()
vision_end = (input_ids == VISION_END_TOKEN_INDEX).nonzero()[:, 1].max()
```

#### Step 4: Attention 추출 + 평균

```python
attentions = lm_loss['attentions']  # tuple of (B, H, seq, seq) per layer

klal_losses = []
for l in range(num_layers):
    # last answer token → visual tokens
    attn = attentions[l][:, :, last_answer_pos, vision_start:vision_end+1]
    # (B, num_heads, num_visual_tokens)
    
    # head 평균
    attn = attn.mean(dim=1)  # (B, num_visual_tokens)
    
    # normalize to probability distribution
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
    
    # KL divergence with GT map
    gt_map = data['gt_attention_map']  # (B, num_visual_tokens)
    kl = (gt_map * (gt_map.log() - attn.log())).sum(dim=-1)  # (B,)
    klal_losses.append(kl.mean())

L_KLAL = sum(klal_losses) / len(klal_losses)
```

#### Step 5: 총 loss에 추가

```python
losses.update(vlm_loss=lm_loss['loss'])
losses.update(loss_pos=lm_loss['loss_pos'])
losses.update(loss_klal=self.klal_lambda * L_KLAL)  # ← 추가
```

---

## 추가 config

```python
# SpaceDrive __init__에 추가
self.klal_lambda = klal_lambda   # default: 0.1 (tuning 필요)
self.use_klal = use_klal         # default: False
```

---

## 주의사항

### 메모리
- `output_attentions=True`는 전 layer attention을 메모리에 올림
- 대안: register_forward_hook으로 특정 layer만 캡처
```python
# 예: 마지막 4개 layer만
target_layers = list(range(num_layers - 4, num_layers))
hooks = []
captured_attentions = {}

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        captured_attentions[layer_idx] = output[1]  # attention weights
    return hook_fn

for l in target_layers:
    h = model.layers[l].self_attn.register_forward_hook(make_hook(l))
    hooks.append(h)
```

### Visual token 순서
- GT map의 카메라 concat 순서와 모델 내부 visual token 순서가 일치하는지 반드시 확인
- SpaceDrive의 pixel_values reshape 순서 = GT map 생성 시 camera_order

### ego token 위치
- ego_status='feature'일 때 IMAGE_TOKEN_INDEX로 삽입되므로, visual token 구간에 ego token이 섞여 있을 수 있음
- ego token은 vision_end 다음에 삽입되므로 vision_start:vision_end 범위에서는 제외될 것으로 예상되나, 확인 필요

### Gradient
- GT attention map은 `.detach()` 불필요 (상수 tensor)
- Attention에서 backward가 정상적으로 흐르는지 확인 (일부 구현에서 attention weight가 detach될 수 있음)

---

## 검증 방법
1. λ_klal = 0으로 두고 기존 성능 재현 확인
2. L_KLAL 값만 로깅해서 학습 중 감소 추이 확인  
3. 학습 전/후 attention map 시각화 비교