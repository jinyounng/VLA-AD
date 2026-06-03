# VPER v3 구현 명세서

---

## 1. 구조

```
VLM(vision + command, ego 없음) → PE Decoder → base_wp
                                → Gate Head  → gate
Raw ego features → Delta MLP → raw_delta

final_wp = base_wp + gate * max_delta * tanh(raw_delta)
```

VLM forward 1회. ego는 VLM에 안 들어감.

---

## 2. 변경 사항

1. VLM 입력에서 ego token 제거 (feature+PE 을 ""로 수정하면 될 듯)
2. 기존 PE Decoder: base_wp 예측 역할로 전환
3. Gate Head 추가: VLM hidden state → gate (B, 2) ∈ [0, 1]
4. Delta MLP 추가: raw ego features + wp_index → raw_delta (B, 2)
5. 산술 합성: final_wp = base_wp + gate * max_delta * tanh(raw_delta)

---

## 3. 새 모듈

### 3.1 Gate Head

```python
class GateHead(nn.Module):
    def __init__(self, hidden_dim, output_dim=2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim),
        )

    def forward(self, hidden_state):
        # hidden_state: (B, D) - VLM output, |pos_ind| 다음 위치
        return torch.sigmoid(self.head(hidden_state))  # (B, 2)
```

### 3.2 Delta MLP

```python
class DeltaMLP(nn.Module):
    def __init__(self, ego_dim, hidden_dim, num_waypoints=6, output_dim=2):
        super().__init__()
        self.wp_embedding = nn.Embedding(num_waypoints, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(ego_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, ego_features, wp_index):
        # ego_features: (B, D_ego) - e_ego + phi(c^ego_tau) concat
        if isinstance(wp_index, int):
            wp_index = torch.full(
                (ego_features.size(0),), wp_index,
                device=ego_features.device, dtype=torch.long,
            )
        wp_emb = self.wp_embedding(wp_index)
        return self.head(torch.cat([ego_features, wp_emb], dim=-1))  # (B, 2)
```

---

## 4. Forward

```python
# 전처리 (기존과 동일)
H_v_tilde = vision_encoder(images) → mlp → + 3D PE
command_embedding = text_tokenizer(command)
e_ego = ego_encoder(ego_status)
phi_ego = pe_encoder(c_ego_tau)
ego_features = torch.cat([e_ego, phi_ego.flatten(1)], dim=-1)  # (B, D_ego)

# VLM forward (ego 없이)
output_seq = vlm(H_v_tilde, command_embedding)  # ego token 미포함

# max_delta buffer
self.register_buffer("vper_max_delta", torch.tensor([max_delta_x, max_delta_y]))

# waypoint 예측
wp_index = 0
for j in range(seq_len):
    if predicted_token[j] == POS_IND_TOKEN:
        h = output_seq[:, j + 1, :]

        base_wp   = self.pe_decoder(h)                          # (B, 2)
        gate      = self.gate_head(h)                           # (B, 2)
        raw_delta = self.delta_mlp(ego_features, wp_index)      # (B, 2)

        delta    = gate * self.vper_max_delta * torch.tanh(raw_delta)
        final_wp = base_wp + delta

        wp_index += 1
```

---

## 5. Loss

```python
L_total = huber_loss(final_wp, gt_wp) + 0.3 * huber_loss(base_wp, gt_wp)
```

Delta regularization 없음.

---

## 6. 구현 전 확인 사항

```
1. [ ] SpaceDrive+ 코드에서 ego token이 VLM에 어떻게 들어가는지 → 제거 방법 확인
2. [ ] e_ego, phi(c^ego_tau)의 shape → D_ego 결정
3. [ ] PE Decoder의 hidden_dim → Gate Head 입력 dim
4. [ ] x, y 축이 lateral/longitudinal 중 어느 쪽인지 → max_delta 값 결정
```

---

## 7. Logging

```python
log_dict = {
    "gate_mean": gate.mean().item(),
    "delta_norm": delta.norm(dim=-1).mean().item(),
    "base_l2": F.mse_loss(base_wp, gt_wp).item(),
}
```

---
