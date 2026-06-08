# Vision-Conditioned VIB for SpaceDrive+ — 구현 명세서

---

## 1. 핵심 변경

기존 명세서의 VIB는 `KL(q(z|ego) || N(0,I))`로 ego를 무조건 압축한다.
이를 `KL(q(z|ego) || p(z|vision))`으로 변경하여, **vision이 이미 아는 정보는 자동으로 걸러내고 vision이 모르는 complementary 정보만 남긴다.**

```
기존 (일반 VIB):
  ego → Encoder → μ_q, σ_q → z
  KL(q(z|ego) || N(0,I))          ← vision 무관, 무조건 압축

변경 (Vision-Conditioned VIB):
  ego → Encoder → μ_q, σ_q → z
  vision → PriorNet → μ_p, σ_p   ← 새로 추가
  KL(q(z|ego) || N(μ_p, σ_p))    ← vision이 아는 건 prior가 커버 → z에서 빠짐
```

---

## 2. 왜 이렇게 하면 vision-redundant가 제거되는가

KL(q || p) 를 최소화할 때:
- q(z|ego)가 p(z|vision)과 같은 정보를 담으면 → KL ≈ 0, 패널티 없음. 하지만 reconstruction(trajectory 예측)에도 도움 안 됨 (vision이 이미 아니까). → z에서 자연스럽게 빠짐.
- q(z|ego)가 p(z|vision)에 없는 정보를 담으면 → KL > 0, 패널티 발생. 하지만 reconstruction에 도움이 됨. → trade-off로 z에 남음.

결과: z에는 vision이 모르는 ego 정보(정확한 속도, 가속도 등)만 남고, vision이 이미 아는 정보(방향, 도로 구조 등)는 제거됨.

---

## 3. 모듈 구현

### 3.1 VisionConditionedVIB

```python
import torch
import torch.nn as nn


class VisionConditionedVIB(nn.Module):
    """
    Vision-Conditioned Variational Information Bottleneck.

    Ego feature를 vision-conditioned prior 기준으로 압축하여,
    vision이 이미 아는 정보는 제거하고 complementary 정보만 남긴다.

    Args:
        input_dim: ego token의 hidden dimension
        vision_dim: vision feature의 dimension
        z_dim: bottleneck latent dimension (None이면 input_dim 유지)
        beta: KL loss weight
    """
    def __init__(
        self,
        input_dim: int,
        vision_dim: int,
        z_dim: int = None,
        beta: float = 1e-3,
    ):
        super().__init__()
        if z_dim is None:
            z_dim = input_dim

        self.z_dim = z_dim
        self.beta = beta

        # Encoder: q(z|ego) — ego feature에서 posterior 생성
        self.enc_mu = nn.Linear(input_dim, z_dim)
        self.enc_logvar = nn.Linear(input_dim, z_dim)

        # PriorNet: p(z|vision) — vision feature에서 prior 생성
        self.prior_mu = nn.Sequential(
            nn.Linear(vision_dim, vision_dim // 2),
            nn.ReLU(),
            nn.Linear(vision_dim // 2, z_dim),
        )
        self.prior_logvar = nn.Sequential(
            nn.Linear(vision_dim, vision_dim // 2),
            nn.ReLU(),
            nn.Linear(vision_dim // 2, z_dim),
        )

        # z → 원래 dim으로 복원
        self.proj_back = nn.Linear(z_dim, input_dim) if z_dim != input_dim else nn.Identity()

    def forward(
        self,
        ego_feature: torch.Tensor,
        vision_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ego_feature: (B, num_ego_tokens, D_ego) — PE encoder 출력
            vision_summary: (B, D_vision) — vision feature의 pooled summary

        Returns:
            z_proj: (B, num_ego_tokens, D_ego) — 압축된 ego token
            kl_loss: scalar — vision-conditioned KL loss
        """
        # --- Encoder: q(z|ego) ---
        mu_q = self.enc_mu(ego_feature)        # (B, T, z_dim)
        logvar_q = self.enc_logvar(ego_feature) # (B, T, z_dim)

        # --- PriorNet: p(z|vision) ---
        mu_p = self.prior_mu(vision_summary)        # (B, z_dim)
        logvar_p = self.prior_logvar(vision_summary) # (B, z_dim)

        # ego token이 여러 개일 수 있으므로 prior를 token 차원에 맞춤
        if ego_feature.dim() == 3 and mu_p.dim() == 2:
            mu_p = mu_p.unsqueeze(1).expand_as(mu_q)         # (B, T, z_dim)
            logvar_p = logvar_p.unsqueeze(1).expand_as(logvar_q)  # (B, T, z_dim)

        # --- Reparameterization ---
        if self.training:
            std_q = torch.exp(0.5 * logvar_q)
            eps = torch.randn_like(std_q)
            z = mu_q + std_q * eps
        else:
            z = mu_q  # inference 시 deterministic

        z_proj = self.proj_back(z)  # (B, T, D_ego)

        # --- KL divergence: KL[N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)] ---
        kl_loss = self._kl_divergence(mu_q, logvar_q, mu_p, logvar_p)

        return z_proj, kl_loss

    @staticmethod
    def _kl_divergence(mu_q, logvar_q, mu_p, logvar_p):
        """
        KL[N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2)]
        = 0.5 * sum( logvar_p - logvar_q - 1
                      + (sigma_q^2 + (mu_q - mu_p)^2) / sigma_p^2 )
        """
        var_q = logvar_q.exp()
        var_p = logvar_p.exp()

        kl = 0.5 * (
            logvar_p - logvar_q
            - 1.0
            + var_q / var_p
            + (mu_q - mu_p).pow(2) / var_p
        )
        return kl.mean()
```

### 3.2 Vision Summary 생성

PriorNet에 넣을 vision summary는 VLM 입력 전 단계의 vision feature를 pooling해서 만든다.

```python
def _build_vision_summary(self, pixel_values, image_grid_thw, pos_embed=None, pos_index=None):
    """
    Vision encoder 출력을 mean pooling하여 (B, D) 벡터로 만든다.
    VLM 입력 전이므로 ego와 한 번도 섞이지 않은 순수 vision feature.
    """
    # 기존 _build_vision_features 호출
    vision_features = self._build_vision_features(
        pixel_values, image_grid_thw,
        pos_embed=pos_embed,
        pos_index=pos_index,
    )  # (B, N_v, D)

    # Mean pooling → (B, D)
    vision_summary = vision_features.mean(dim=1)
    return vision_summary, vision_features
```

---

## 4. Forward 흐름

```python
# 1. Vision features (기존과 동일)
vision_summary, vision_features = self._build_vision_summary(
    pixel_values, image_grid_thw, pos_embed, pos_index
)
flat_vision = vision_features.reshape(-1, vision_features.shape[-1])

# 2. Ego features (기존과 동일하게 생성)
ego_feature = self._build_ego_feature(B, data, input_ids, ...)

# 3. VIB 적용 (핵심 변경점)
ego_feature_compressed, scaled_kl, raw_kl = self.vib(
    ego_feature,             # (B, T, D)
    vision_summary,          # (B, D)
)

# 4. VLM forward (ego_feature 대신 compressed version 사용)
lm_loss = self.lm_head(
    ...
    ego_feature=ego_feature_compressed,  # VIB 통과한 ego
    precomputed_image_embeds=flat_vision,
    pixel_values=None,                   # vision encoder 재실행 방지
    pos_emb=None, pos_index=None,         # flat_vision에 이미 반영됨
    ...
)

# 5. Loss
total_loss = lm_loss["loss"] + scaled_kl
log_vars["vib_raw_kl"] = raw_kl  # logging only, not added to total loss
```

---

## 5. Loss

```python
L_total = L_planning + β * KL(q(z|ego) || p(z|vision))
```

- L_planning: 기존 SpaceDrive+ planning loss 그대로
- KL term: vision이 아는 정보를 z에서 제거하는 pressure
- β: 압축 강도 조절. 너무 크면 ego 정보 과도 손실 (mode collapse), 너무 작으면 shortcut 유지

---

## 6. 기존 명세서 대비 변경 요약

| | 기존 VIB 명세서 | 변경 (Vision-Conditioned) |
|---|---|---|
| Prior | N(0, I) | N(μ_p, σ_p) where μ_p, σ_p = PriorNet(vision) |
| KL 계산 | KL(q \|\| N(0,I)) | KL(q \|\| p(z\|vision)) |
| Vision 입력 | 없음 | PriorNet에 vision_summary 입력 |
| 새 모듈 | enc_mu, enc_logvar, proj_back | + prior_mu, prior_logvar (PriorNet) |
| 효과 | ego를 무조건 압축 | vision-redundant만 선택적 제거 |

---

## 7. 하이퍼파라미터

| 파라미터 | 초기값 | 설명 |
|---|---|---|
| beta | 1e-3 | KL weight. 핵심 튜닝 대상 |
| z_dim | None (=input_dim) | latent dim. 처음엔 줄이지 않고 시작 |
| vision_dim | llm_hidden_dim | vision summary dimension |
| PriorNet hidden | vision_dim // 2 | PriorNet 중간 layer dim |

### Beta sweep:
```
beta = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
```

---

## 8. 구현 체크리스트

```
1. [ ] VisionConditionedVIB 클래스 작성
2. [ ] _build_vision_summary 함수 추가
3. [ ] forward에서 VIB에 vision_summary 넘기도록 수정
4. [ ] KL 계산이 vision-conditioned인지 확인 (N(0,I) 아닌지)
5. [ ] kl_loss를 total loss에 추가
6. [ ] β=0으로 돌려서 기존 성능과 동일한지 확인
7. [ ] 강아지 테스트: 이미지 교체 시 성능 떨어지는지 확인
8. [ ] vision_summary에 detach 안 하기 (gradient 흘려야 함)
```

---

## 9. 주의사항

1. **vision_summary를 detach하지 말 것** — PriorNet으로 gradient가 vision encoder까지 흘러야 함
2. **PriorNet이 ego를 보면 안 됨** — vision feature만 입력. ego가 들어가면 prior가 ego 정보를 포함해서 KL이 무의미해짐
3. **Inference 시 z = mu_q** — sampling 안 함 (deterministic)
4. **β 튜닝이 가장 중요** — 과압축 시 SpaceDrive가 보고한 mode collapse 재현될 수 있음
5. **vision_features 이중 실행 주의** — _build_vision_features를 VIB용과 VLM용으로 두 번 호출하지 말고, 한 번 만들어서 `precomputed_image_embeds`로 공유할 것
