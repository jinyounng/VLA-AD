# VSV → SpaceDrive 적용 코딩 명세서

## 목적
VISTA 논문의 VSV(Visual Steering Vector)를 SpaceDrive에 그대로 구현하여 nuScenes에서 L2, Collision Rate를 측정한다.

---

## VSV 구현 상세 (VISTA 코드 기준)

### 1단계: VSV 추출 (`steering_vector.py` → `obtain_vsv()`)

각 calibration 샘플에 대해 **이미지 있는 입력 / 이미지 없는 입력** 두 쌍을 만들고, 둘 다 forward pass해서 **각 layer의 마지막 token hidden state**를 추출한다.

```
for each sample:
    h_with_img = forward(이미지 O).hidden_states  # layer별 last token hidden state
    h_no_img   = forward(이미지 X).hidden_states  # layer별 last token hidden state
    diff = h_with_img - h_no_img                  # shape: [num_layers, hidden_dim]
```

모든 샘플의 diff를 모은 뒤, **PCA (rank=1)** 로 주성분 방향을 추출한다:

```python
fit_data = torch.stack(all_diffs)           # [N, num_layers * hidden_dim]
pca = PCA(n_components=1).fit(fit_data)
direction = (pca.components_.sum(dim=0) + pca.mean_).reshape(num_layers, hidden_dim)
```

→ 최종 output: `direction` (shape: `[num_layers, hidden_dim]`) — 이것이 VSV

**이미지 없는 입력 만드는 법** (`anchor.py` 참고):
- 이미지 있는 템플릿: `"USER: <ImageHere> <question> ASSISTANT:"`
- 이미지 없는 템플릿: `"USER: <question> ASSISTANT:"` (이미지 placeholder 자체를 제거)
- SpaceDrive에서는 multi-view visual token을 완전히 제거하거나 zero로 대체

### 2단계: VSV 적용 (`llm_layers.py` → `VSVLayer`, `add_vsv_layers()`)

각 transformer layer의 **MLP 뒤에** VSVLayer를 삽입한다 (Sequential로 감싸서):

```python
# add_vsv_layers() 핵심 로직:
for i, layer in enumerate(model_layers):
    original_mlp = layer.mlp
    layer.mlp = nn.Sequential(original_mlp, VSVLayer(vsv[i], lambda_val))
```

**VSVLayer forward 로직** (default mode, `simple_mode=False`):

```python
def forward(self, x):
    # x: [batch, seq_len, hidden_dim]
    original_norm = torch.norm(x, p=2, dim=-1, keepdim=True)

    # adaptive lambda: x가 VSV 반대 방향일수록 더 강하게 steering
    lambda_sim = 1.0 + max(0, cosine_similarity(x, -vsv))

    # normalized VSV를 lambda 스케일링하여 더함
    y = lambda_val * lambda_sim * F.normalize(vsv, dim=-1)

    # 더한 뒤 normalize하고 원래 norm으로 복원
    x = F.normalize(F.normalize(x, dim=-1) + y, dim=-1) * original_norm
    return x.half()
```

핵심 포인트:
- 단순 덧셈이 아니라 **direction만 바꾸고 norm은 보존**
- `cosine_similarity(x, -vsv)`로 adaptive weighting: visual과 반대 방향인 token일수록 더 강하게 보정
- 모든 token position에 동일하게 적용됨

### 3단계: 제거 (`remove_vsv_layers()`)

```python
for layer in model_layers:
    if isinstance(layer.mlp, nn.Sequential):
        layer.mlp = layer.mlp[0]  # 원래 MLP만 복원
```

---

## SpaceDrive 적용 시 해야 할 것

1. SpaceDrive의 VLM backbone (Qwen2.5-VL-7B)의 transformer layer 리스트 경로 파악
   - VISTA는 `find_longest_modulelist(model)`로 자동 탐색함 → 그대로 쓰면 됨
   - 또는 직접: `model.model.layers` (Qwen2.5-VL의 경우 확인 필요)

2. Calibration 데이터: nuScenes val set에서 200개 샘플
   - 이미지 O: 기존 SpaceDrive inference 입력 그대로
   - 이미지 X: visual token 제거 (또는 zero tensor 대체)

3. VSV 추출 후 `add_vsv_layers()`로 적용, 기존 eval 코드로 L2/Collision 측정

4. `--vsv-lambda` sweep: 0.1, 0.5, 1.0, 2.0 (VISTA 저자 가이드: 모델마다 적정 scale 다름)

---

## 참고 파일 (VISTA repo)
- `steering_vector.py`: VSV 추출 (`obtain_vsv()`, `get_hiddenstates()`)
- `llm_layers.py`: VSV 적용/제거 (`VSVLayer`, `add_vsv_layers()`, `remove_vsv_layers()`)
- `anchor.py`: 이미지 있는/없는 prompt 템플릿
- `model_loader.py`: 모델 로드

## 측정
- **Avg L2 (1s, 2s, 3s)**, **Collision Rate**
- baseline: SpaceDrive 원본 결과와 비교# VSV → SpaceDrive 적용 코딩 명세서

## 목적
VISTA 논문의 VSV(Visual Steering Vector)를 SpaceDrive에 그대로 구현하여 nuScenes에서 L2, Collision Rate를 측정한다.

---

## VSV 구현 상세 (VISTA 코드 기준)

### 1단계: VSV 추출 (`steering_vector.py` → `obtain_vsv()`)

각 calibration 샘플에 대해 **이미지 있는 입력 / 이미지 없는 입력** 두 쌍을 만들고, 둘 다 forward pass해서 **각 layer의 마지막 token hidden state**를 추출한다.

```
for each sample:
    h_with_img = forward(이미지 O).hidden_states  # layer별 last token hidden state
    h_no_img   = forward(이미지 X).hidden_states  # layer별 last token hidden state
    diff = h_with_img - h_no_img                  # shape: [num_layers, hidden_dim]
```

모든 샘플의 diff를 모은 뒤, **PCA (rank=1)** 로 주성분 방향을 추출한다:

```python
fit_data = torch.stack(all_diffs)           # [N, num_layers * hidden_dim]
pca = PCA(n_components=1).fit(fit_data)
direction = (pca.components_.sum(dim=0) + pca.mean_).reshape(num_layers, hidden_dim)
```

→ 최종 output: `direction` (shape: `[num_layers, hidden_dim]`) — 이것이 VSV

**이미지 없는 입력 만드는 법** (`anchor.py` 참고):
- 이미지 있는 템플릿: `"USER: <ImageHere> <question> ASSISTANT:"`
- 이미지 없는 템플릿: `"USER: <question> ASSISTANT:"` (이미지 placeholder 자체를 제거)
- SpaceDrive에서는 multi-view visual token을 완전히 제거하거나 zero로 대체

### 2단계: VSV 적용 (`llm_layers.py` → `VSVLayer`, `add_vsv_layers()`)

각 transformer layer의 **MLP 뒤에** VSVLayer를 삽입한다 (Sequential로 감싸서):

```python
# add_vsv_layers() 핵심 로직:
for i, layer in enumerate(model_layers):
    original_mlp = layer.mlp
    layer.mlp = nn.Sequential(original_mlp, VSVLayer(vsv[i], lambda_val))
```

**VSVLayer forward 로직** (default mode, `simple_mode=False`):

```python
def forward(self, x):
    # x: [batch, seq_len, hidden_dim]
    original_norm = torch.norm(x, p=2, dim=-1, keepdim=True)

    # adaptive lambda: x가 VSV 반대 방향일수록 더 강하게 steering
    lambda_sim = 1.0 + max(0, cosine_similarity(x, -vsv))

    # normalized VSV를 lambda 스케일링하여 더함
    y = lambda_val * lambda_sim * F.normalize(vsv, dim=-1)

    # 더한 뒤 normalize하고 원래 norm으로 복원
    x = F.normalize(F.normalize(x, dim=-1) + y, dim=-1) * original_norm
    return x.half()
```

핵심 포인트:
- 단순 덧셈이 아니라 **direction만 바꾸고 norm은 보존**
- `cosine_similarity(x, -vsv)`로 adaptive weighting: visual과 반대 방향인 token일수록 더 강하게 보정
- 모든 token position에 동일하게 적용됨

### 3단계: 제거 (`remove_vsv_layers()`)

```python
for layer in model_layers:
    if isinstance(layer.mlp, nn.Sequential):
        layer.mlp = layer.mlp[0]  # 원래 MLP만 복원
```

---

## SpaceDrive 적용 시 해야 할 것

1. SpaceDrive의 VLM backbone (Qwen2.5-VL-7B)의 transformer layer 리스트 경로 파악
   - VISTA는 `find_longest_modulelist(model)`로 자동 탐색함 → 그대로 쓰면 됨
   - 또는 직접: `model.model.layers` (Qwen2.5-VL의 경우 확인 필요)

2. Calibration 데이터: nuScenes val set에서 200개 샘플
   - 이미지 O: 기존 SpaceDrive inference 입력 그대로
   - 이미지 X: visual token 제거 (또는 zero tensor 대체)

3. VSV 추출 후 `add_vsv_layers()`로 적용, 기존 eval 코드로 L2/Collision 측정

4. `--vsv-lambda` sweep: 0.1, 0.5, 1.0, 2.0 (VISTA 저자 가이드: 모델마다 적정 scale 다름)

---

## 참고 파일 (VISTA repo)
- `steering_vector.py`: VSV 추출 (`obtain_vsv()`, `get_hiddenstates()`)
- `llm_layers.py`: VSV 적용/제거 (`VSVLayer`, `add_vsv_layers()`, `remove_vsv_layers()`)
- `anchor.py`: 이미지 있는/없는 prompt 템플릿
- `model_loader.py`: 모델 로드

## 측정
- **Avg L2 (1s, 2s, 3s)**, **Collision Rate**
- baseline: SpaceDrive 원본 결과와 비교
- 평가는 SpaceDrive 원본과 최대한 비슷하게 사용