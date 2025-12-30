# 학습 횟수 증가 및 하이퍼파라미터 튜닝 가이드

## 현재 설정 (ppo_trainer.py)

```python
total_timesteps=100000  # 기본값
learning_rate=3e-4
n_steps=2048
batch_size=64
gamma=0.99
```

## 권장 개선 사항

### 1. 학습 횟수 증가

**단계별 접근:**
- **1단계**: 100,000 → 300,000 timesteps
- **2단계**: 300,000 → 500,000 timesteps  
- **3단계**: 500,000 → 1,000,000 timesteps (최종)

**이유:**
- PPO는 안정적이지만 수렴에 시간이 걸림
- 더 많은 경험으로 다양한 시나리오 학습 가능
- K-ICS 방어 정확도 향상

### 2. 하이퍼파라미터 튜닝

#### A. Learning Rate 스케줄링
```python
# 고정 LR 대신 스케줄링 사용
from stable_baselines3.common.callbacks import EvalCallback

# Linear Decay
learning_rate = lambda progress: 3e-4 * (1 - progress)

# 또는 Cosine Annealing
learning_rate = lambda progress: 3e-4 * (1 + np.cos(np.pi * progress)) / 2
```

#### B. n_steps 조정
- **현재**: 2048
- **권장**: 4096 또는 8192
- **이유**: 더 긴 롤아웃으로 장기 의존성 학습

#### C. Batch Size 조정
- **현재**: 64
- **권장**: 128 또는 256
- **이유**: 더 안정적인 gradient 추정

#### D. Gamma (할인율) 조정
- **현재**: 0.99
- **권장**: 0.995 (더 장기적 보상 고려)
- **이유**: K-ICS 방어는 장기적 관점이 중요

### 3. Early Stopping 및 체크포인트

```python
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# 평가 콜백 (조기 종료)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./models/best_ppo/',
    log_path='./logs/',
    eval_freq=10000,
    deterministic=True,
    render=False
)

# 체크포인트 콜백
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path='./models/checkpoints/',
    name_prefix='ppo_kics'
)
```

### 4. 학습 스케줄 예시

```python
# 단계별 학습
configs = [
    {
        'total_timesteps': 100000,
        'learning_rate': 3e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'name': 'phase1_warmup'
    },
    {
        'total_timesteps': 200000,
        'learning_rate': 1e-4,  # 낮춤
        'n_steps': 4096,
        'batch_size': 128,
        'name': 'phase2_fine_tune'
    },
    {
        'total_timesteps': 300000,
        'learning_rate': 5e-5,  # 더 낮춤
        'n_steps': 8192,
        'batch_size': 256,
        'name': 'phase3_final'
    }
]

for config in configs:
    model.learn(
        total_timesteps=config['total_timesteps'],
        callback=[eval_callback, checkpoint_callback],
        reset_num_timesteps=False  # 이전 학습 이어서
    )
    model.save(f"models/{config['name']}")
```

### 5. 모니터링 지표

학습 중 확인할 지표:
- **Reward**: 점진적 증가 추세
- **K-ICS 비율**: 150% 이상 유지 비율
- **Hedge 비율 변동성**: 급격한 변화 없어야 함
- **Loss**: 안정적으로 감소

### 6. 검증 방법

각 단계마다:
1. **Backtest 실행**: `python -m src.validation.backtest`
2. **Stress Test 실행**: `python -m src.validation.stress_safety`
3. **Enhanced Validation 실행**: `python -m src.validation.enhanced_validation`

### 7. 예상 소요 시간

- **100K timesteps**: 약 1-2시간 (CPU 기준)
- **300K timesteps**: 약 3-6시간
- **500K timesteps**: 약 5-10시간
- **1M timesteps**: 약 10-20시간

**GPU 사용 시**: 약 1/3 ~ 1/5 시간 단축

### 8. 주의사항

- **오버피팅 방지**: Train/Test 분리 유지
- **메모리 관리**: 큰 batch_size 사용 시 메모리 부족 주의
- **체크포인트**: 정기적으로 저장하여 중단 시 복구 가능
- **TensorBoard**: 학습 과정 모니터링 필수

