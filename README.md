# Dynamic Shield: K-ICS 자본 최적화 AI 시스템

> **Capital Optimization, not Prediction**  
> 환율 예측이 아닌 자본 최적화 - K-ICS 규제 대응 동적 헤지 전략

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

---

## 📌 프로젝트 개요

**Dynamic Shield**는 보험사의 K-ICS(신지급여력제도) 자본 효율화를 위한 AI 기반 동적 헤지 시스템입니다.

### 핵심 발견: Risk Paradox (리스크 역설)
> **100% 헤지가 최선이 아니다!**  
> 주식-환율 간 음의 상관관계(Natural Hedge)를 활용하면 부분 헤지로 더 낮은 위험과 비용 달성 가능

---

## 🏆 주요 결과 (Synthetic Data)

### Phase 1: Risk Paradox 증명
| 상관계수 | 최적 헤지 비율 | 자본 절감률 |
|:---:|:---:|:---:|
| -0.6 | 0% | **17.69%** |
| -0.4 | 32% | **7.26%** |
| -0.2 | 67% | **1.63%** |

✅ **음의 상관관계에서 부분 헤지가 완전 헤지보다 효율적임을 증명**

### Phase 2: AI Surrogate Model
| 지표 | 결과 |
|---|---|
| MAPE (Mean Absolute Percentage Error) | **0.0385%** |
| 확장성 테스트 (10배 자산) | Pass |
| 추론 속도 | 실시간 가능 |

✅ **MLP 신경망이 K-ICS 계산을 0.04% 오차로 근사**

### Phase 3: Regime Detection (HMM)
| 시장 국면 | VIX 평균 | 헤지 전략 |
|---|---|---|
| Normal | 14.88 | 낮은 헤지 유지 |
| Transition | 15.21 | 점진적 조정 |
| Panic | 26.45 | 고헤지 전환 |

✅ **Hidden Markov Model로 3개 시장 국면 자동 분류**

### Phase 4: RL Training (PPO)
| 지표 | Q-Learning | **PPO** | 개선률 |
|---|---|---|---|
| Avg Reward | -1.69 | **+63.21** | **37배**|
| Min K-ICS | 4.2% | **341%** |  **81배** |
| Safety Layer 트리거 | - | 13,830회 | - |

✅ **PPO 에이전트가 K-ICS 341% 유지하며 학습 (규제 기준 100%의 3.4배)**

### Phase 5: Backtesting (5개 시나리오)
| 전략 | CAGR | MDD | RCR |
|---|---|---|---|
| 100% Hedge | -39.8% | -63.3% | -0.001 |
| 80% Fixed | -31.5% | -52.7% | 0.012 |
| Rule-based | -19.6% | -35.1% | 0.038 |
| **Dynamic Shield** | **-16.2%** | **-29.4%** | **0.047** |

✅ **Dynamic Shield가 모든 핵심 지표에서 1위**

### Efficient Frontier
| 전략 | Risk | Cost |
|---|---|---|
| 100% Hedge | 36.09% | 60.00% |
| **Dynamic Shield** | **33.84%** | **26.23%** |

✅ **리스크 2.25%p↓, 비용 33.77%p↓ 동시 달성**

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Dynamic Shield System                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ K-ICS Engine│  │ AI Surrogate│  │ Regime Detector     │ │
│  │ (Ground     │──│ (MLP Neural │──│ (Hidden Markov      │ │
│  │  Truth)     │  │  Network)   │  │  Model)             │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│         │                │                    │             │
│         └────────────────┼────────────────────┘             │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              PPO RL Agent (stable-baselines3)           ││
│  │  - State: [Hedge_Ratio, VIX, Correlation, SCR_Ratio]   ││
│  │  - Action: Continuous [-1, 1] → Hedge Adjustment       ││
│  │  - Reward: Capital Efficiency - Cost - K-ICS Penalty   ││
│  └─────────────────────────────────────────────────────────┘│
│                          │                                  │
│                          ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐│
│  │                    Safety Layer                         ││
│  │  - VIX > 30: Gradual De-risking                        ││
│  │  - K-ICS < 100%: Force 100% Hedge                      ││
│  │  - Max Step: ±10% per period (급발진 방지)              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 프로젝트 구조

```
한화/
├── src/
│   ├── core/                      # 핵심 모듈
│   │   ├── kics_real.py           # K-ICS 엔진 (Ground Truth)
│   │   ├── kics_surrogate.py      # AI Surrogate 모델
│   │   ├── regime.py              # HMM 시장 국면 탐지
│   │   ├── environment.py         # RL 환경
│   │   ├── agent.py               # Dynamic Shield 에이전트
│   │   ├── gym_environment.py     # Gymnasium 호환 환경
│   │   ├── ppo_trainer.py         # PPO 훈련 (stable-baselines3)
│   │   ├── rl_trainer.py          # Q-Learning 훈련
│   │   ├── system.py              # 통합 시스템
│   │   └── realistic_data.py      # 현실적 데이터 생성기
│   │
│   ├── validation/                # 검증 모듈
│   │   ├── backtest.py            # 백테스팅
│   │   ├── proof_risk_paradox.py  # Risk Paradox 증명
│   │   ├── solvency_visualizer.py # K-ICS 방어 시각화
│   │   ├── stress_safety.py       # Safety Layer 스트레스 테스트
│   │   ├── advanced_viz.py        # Efficient Frontier 시각화
│   │   └── shap_analysis.py       # Why Not 100% Hedge 분석
│   │
│   ├── phase6_final_review.py     # 최종 검토 스크립트
│   └── requirements.txt           # 의존성
│
├── models/                        # 학습된 모델
│   └── ppo_kics.zip
│
└── tensorboard_logs/              # 학습 로그
```

---

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
# Conda 환경 생성 (Python 3.11 권장)
conda create -n quant python=3.11 pytorch cpuonly -c pytorch -y
conda activate quant

# 의존성 설치
pip install -r src/requirements.txt
conda install -c conda-forge hmmlearn -y

# Jupyter 커널 등록 (선택)
python -m ipykernel install --user --name quant --display-name "(Quant)"
```

### 2. 전체 파이프라인 실행
```bash
cd src/core

# Phase 1: K-ICS Engine
python kics_real.py

# Phase 2: AI Surrogate + Regime Detection
python kics_surrogate.py
python regime.py

# Phase 3: System Integration
python system.py

# Phase 4: RL Training
python ppo_trainer.py  # PPO (권장)
# python rl_trainer.py  # Q-Learning (대안)

# Phase 5: Validation
cd ../validation
python proof_risk_paradox.py
python solvency_visualizer.py
python backtest.py
python advanced_viz.py

# Phase 6: Final Review
cd ..
python phase6_final_review.py
```

### 3. TensorBoard 모니터링
```bash
tensorboard --logdir=./tensorboard_logs/
# 브라우저에서 http://localhost:6006 접속
```

---

## 📊 생성되는 시각화 파일

| 파일명 | 설명 |
|---|---|
| `risk_paradox_proof.png` | Risk Paradox 증명 그래프 |
| `kics_defense_result.png` | COVID-19 시나리오 K-ICS 방어 |
| `ppo_training_result.png` | PPO 학습 진행 그래프 |
| `backtest_results.png` | 백테스팅 성과 비교 |
| `efficient_frontier.png` | 효율적 투자선 |
| `counterfactual_dashboard.png` | 의사결정 경계 |
| `shap_why_not_analysis.png` | Why Not 100% Hedge 분석 |

---

## 🔮 향후 계획: 실제 데이터 적용

### Phase 1: 데이터 수집
| 데이터 | 소스 | 용도 |
|---|---|---|
| VIX 지수 | Yahoo Finance / Bloomberg | 시장 변동성 |
| USD/KRW 환율 | 한국은행 ECOS API | FX 데이터 |
| KOSPI 지수 | KRX | 주식-환율 상관관계 |
| 금리 (한/미) | FRED / 한국은행 | 스왑 포인트 계산 |
| 실제 K-ICS 데이터 | 보험사 내부 | Ground Truth |

### Phase 2: 데이터 전처리
- **일별 수익률 계산**: `log(P_t / P_{t-1})`
- **롤링 상관관계**: 60일/120일 윈도우
- **GARCH 변동성 추정**: 클러스터링 효과 반영
- **레짐 라벨링**: VIX 기반 Normal/Transition/Panic 구분

### Phase 3: 모델 재학습
```python
# 실제 데이터로 HMM 재학습
from core.regime import MarketRegimeDetector
detector = MarketRegimeDetector(n_regimes=3)
detector.fit(real_market_data)

# PPO 에이전트 재학습
from core.ppo_trainer import PPOTrainer
trainer = PPOTrainer(total_timesteps=500000)  # 더 많은 학습
trainer.train()
```

### Phase 4: 백테스팅
- **기간**: 2015-2024 (10년)
- **시나리오**: 2015 중국발 폭락, 2018 금리인상, 2020 COVID, 2022 금리쇼크
- **벤치마크**: 실제 보험사 헤지 전략 vs Dynamic Shield

### Phase 5: 실시간 시스템 구축 (2-3주)
```
┌─────────────────────────────────────────────────────────┐
│                  Production System                       │
├─────────────────────────────────────────────────────────┤
│  [Data Pipeline]                                        │
│   Bloomberg/Reuters → Kafka → Feature Store             │
│                                                         │
│  [Inference Engine]                                     │
│   AI Surrogate (ONNX) → PPO Agent → Safety Layer        │
│                                                         │
│  [Execution]                                            │
│   Hedge Signal → Risk Manager Approval → FX Desk        │
│                                                         │
│  [Monitoring]                                           │
│   Grafana Dashboard + Alert System                      │
└─────────────────────────────────────────────────────────┘
```

---

## 📚 기술 스택

| 영역 | 기술 |
|---|---|
| 언어 | Python 3.11 |
| RL Framework | stable-baselines3, Gymnasium |
| 딥러닝 | PyTorch 2.0+ |
| ML | scikit-learn, hmmlearn |
| 시각화 | Matplotlib, TensorBoard |
| 데이터 | NumPy, Pandas, SciPy |


