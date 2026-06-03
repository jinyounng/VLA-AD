# VIB (Variational Information Bottleneck) for SpaceDrive+ — 구현 명세서

## 1. 목적

SpaceDrive+의 ego-status shortcut 문제를 해결하기 위해, ego-status가 VLM에 입력되기 전에 **Variational Information Bottleneck (VIB)** 을 삽입한다.

### 배경
- SpaceDrive+에서 ego-status(속도, yaw rate, 과거 trajectory 등)는 PE encoder를 거쳐 text token 자리에 들어감
- 모델은 vision token 대신 ego token을 shortcut으로 사용 (vision token 대비 ~1000× 높은 attention)
- VIB로 ego 정보를 확률적으로 압축하여, trajectory 예측에 필수적인 정보만 통과시킴

## 2. 코드베이스 정보

- **SpaceDrive+ 루트**: `~/projects/SpaceDrive`
- **VIB 코드 작성 위치**: `/data/jykim/projects/SpaceDrive/projects/VIB`
- **공식 레포 그대로 사용 중** (수정 없음)
- **환경**: conda env `spacedrive_blackwell`

## 3. 사전 작업: 코드베이스 분석 (반드시 먼저 수행)

아래 순서로 SpaceDrive+ 코드를 분석하여 ego-status 처리 파이프라인을 파악할 것:

```bash
# 1. 프로젝트 구조 파악
find ~/projects/SpaceDrive/projects -name "*.py" | head -50

# 2. ego-status 관련 코드 위치 찾기
grep -rn "ego" ~/projects/SpaceDrive/projects --include="*.py" | grep -i "status\|state\|hist\|traj"

# 3. PE encoder 관련 코드 찾기
grep -rn "pe_encoder\|PE_encoder\|positional_encod\|pos_enc" ~/projects/SpaceDrive/projects --include="*.py"

# 4. 학습 loss 정의 위치 찾기
grep -rn "loss" ~/projects/SpaceDrive/projects --include="*.py" | grep -i "def \|total\|compute"
```

**중요**: 코드 분석 결과를 기반으로 아래 구현 계획의 정확한 삽입 위치를 결정할 것. 아래 설명은 논문 기반 추정이므로, 실제 변수명/함수명은 코드를 확인한 뒤 맞출 것.

## 4. 아키텍처 변경

### 4.1 현재 SpaceDrive+ 흐름 (ego 파이프라인)

```
ego-states (speed, yaw rate, past trajectory 등)
    ↓
[PE encoder φ]  — 3D sine-cosine positional encoding
    ↓
ego tokens → VLM에 text token과 함께 입력
```

### 4.2 VIB 삽입 후 흐름

```
ego-states
    ↓
[PE encoder φ]  — 기존과 동일, freeze하지 않음
    ↓
ego_pe  (shape: [B, num_ego_tokens, dim])
    ↓
[VIB Bottleneck]
    ├── mu_head: Linear(dim, z_dim)
    ├── logvar_head: Linear(dim, z_dim)
    ├── z = mu + exp(0.5 * logvar) * epsilon,  epsilon ~ N(0, I)
    └── proj_back: Linear(z_dim, dim)  ← 원래 dim으로 복원
    ↓
z_ego  (shape: [B, num_ego_tokens, dim])
    ↓
VLM에 입력 (기존 ego token 자리를 z_ego로 대체)
```

### 4.3 VIB 모듈 구현

```python
import torch
import torch.nn as nn

class VIBBottleneck(nn.Module):
    """
    Variational Information Bottleneck for ego-status tokens.
    
    Args:
        input_dim: ego token의 hidden dimension (PE encoder output dim)
        z_dim: bottleneck latent dimension (default: input_dim // 2 또는 별도 설정)
        beta: KL loss weight (핵심 하이퍼파라미터)
    """
    def __init__(self, input_dim: int, z_dim: int = None, beta: float = 1e-3):
        super().__init__()
        if z_dim is None:
            z_dim = input_dim  # 처음엔 dimension 줄이지 않고 시작
        
        self.mu_head = nn.Linear(input_dim, z_dim)
        self.logvar_head = nn.Linear(input_dim, z_dim)
        self.proj_back = nn.Linear(z_dim, input_dim) if z_dim != input_dim else nn.Identity()
        self.beta = beta
    
    def forward(self, ego_pe: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ego_pe: [B, num_tokens, dim] — PE encoder의 출력
        Returns:
            z_proj: [B, num_tokens, dim] — bottleneck 통과 후 토큰
            kl_loss: scalar — KL divergence loss
        """
        mu = self.mu_head(ego_pe)           # [B, T, z_dim]
        logvar = self.logvar_head(ego_pe)    # [B, T, z_dim]
        
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu  # inference 시 deterministic
        
        z_proj = self.proj_back(z)  # [B, T, dim]
        
        # KL divergence: KL[N(mu, sigma^2) || N(0, I)]
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return z_proj, kl_loss
```

## 5. Loss 수정

### 기존 loss:
```python
total_loss = trajectory_loss  # (+ 기타 auxiliary losses)
```

### VIB 적용 후:
```python
total_loss = trajectory_loss + beta * kl_loss
```

- `beta`는 VIBBottleneck 모듈에서 관리
- **beta 초기값: 1e-3** (논문 기반, 이후 sweep 필요)

## 6. 하이퍼파라미터

| 파라미터 | 초기값 | 설명 |
|---------|--------|------|
| `beta` | 1e-3 | KL loss weight. 핵심 튜닝 대상 |
| `z_dim` | input_dim (변경 없이 시작) | bottleneck dimension |
| 나머지 학습 설정 | SpaceDrive+ 기본값 유지 | lr, scheduler 등 건드리지 않기 |

### Beta sweep 계획 (1단계 실험):
```
beta = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
```

## 7. 구현 체크리스트

- [ ] **코드 분석**: ego-status → PE encoder → VLM 입력 경로에서 정확한 삽입 위치 파악
- [ ] **VIBBottleneck 모듈 작성**: `projects/VIB/vib_bottleneck.py`
- [ ] **모델 코드 수정**: ego token이 VLM에 들어가기 직전에 VIBBottleneck 통과하도록 수정 (원본 파일 복사 후 수정)
- [ ] **Loss 수정**: kl_loss를 total_loss에 추가
- [ ] **Config 수정**: beta, z_dim 등을 config에서 관리 가능하도록
- [ ] **검증**: VIB 없이 (beta=0) 돌렸을 때 기존 SpaceDrive+와 동일한 결과 나오는지 확인
- [ ] **Beta sweep 스크립트 작성**

## 8. 주의사항

1. **원본 코드 건드리지 말 것** — 수정이 필요한 파일은 복사 후 수정
2. **PE encoder는 수정하지 않음** — VIB는 PE encoder 출력 이후에 삽입
3. **Vision token pipeline은 건드리지 않음** — ego token에만 VIB 적용
4. **Inference 시 deterministic** — mu만 사용, sampling 안 함
5. **gradient가 PE encoder까지 흘러야 함** — detach하지 않기
6. **VRAM 주의** — (μ, σ) head 추가로 인한 메모리 증가 미미할 것이나, 확인 필요
7. **β=0으로 검증 먼저** — 구조 변경이 기존 성능에 영향 없는지 반드시 확인

## 9. 파일 구조

```
/data/jykim/projects/SpaceDrive/projects/VIB/
├── vib_bottleneck.py          # VIBBottleneck 모듈
├── model_with_vib.py          # VIB가 삽입된 모델 (원본 복사 후 수정)
├── config_vib.py              # VIB 관련 config
└── sweep_beta.sh              # beta sweep 스크립트

/data/jykim/projects/SpaceDrive/scripts/train
├── train_vib.sh               # 학습 스크립트
```
