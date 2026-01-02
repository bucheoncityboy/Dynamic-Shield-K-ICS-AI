# Config 폴더 구조 구축 가이드

## 📋 목차
1. [과정 개요](#과정-개요)
2. [1단계: YAML 설정 파일 설계](#1단계-yaml-설정-파일-설계)
3. [2단계: Config 로더 유틸리티 구현](#2단계-config-로더-유틸리티-구현)
4. [3단계: 기존 코드 리팩토링](#3단계-기존-코드-리팩토링)
5. [4단계: 테스트 및 검증](#4단계-테스트-및-검증)

---

## 과정 개요

### 🎯 목표
**하드코딩된 설정값을 YAML 파일로 분리하여 관리**

### 📊 현재 문제점
```
❌ 하드코딩된 값들이 여러 파일에 산재
❌ 실험 설정 변경 시 코드 수정 필요
❌ 재현성 확보 어려움
❌ 설정 일관성 관리 어려움
```

### ✅ 해결 방법
```
✅ YAML 파일로 모든 설정 중앙 관리
✅ Config 로더로 설정 자동 로드
✅ 코드는 설정 파일만 참조
✅ 실험별 설정 파일 분리 가능
```

---

## 1단계: YAML 설정 파일 설계

### 1.1 폴더 구조 생성

```bash
# 프로젝트 루트에서 실행
mkdir -p config
```

### 1.2 `config/base_config.yaml` 설계

이 파일은 **모든 하이퍼파라미터**를 포함합니다.

```yaml
# config/base_config.yaml
# Dynamic Shield v3.0 - 기본 설정 파일

# ==========================================
# TimeGAN 설정
# ==========================================
timegan:
  training:
    epochs: 300
    batch_size: 128
    sequence_length: 24
    learning_rate: 0.001
    noise_dim: 32
    layers_dim: 128
    latent_dim: 24
    gamma: 1
  
  data:
    feature_cols:
      - VIX
      - FX
      - Correlation
    n_samples_default: 1000
  
  model:
    save_path: "models/timegan"
    files:
      - timegan_model.pkl
      - scaler.pkl
      - params.pkl

# ==========================================
# PPO (강화학습) 설정
# ==========================================
ppo:
  algorithm: "PPO"  # 또는 "A2C"
  total_timesteps: 100000
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  gamma: 0.99
  tensorboard_log: "./tensorboard_logs/"

# ==========================================
# Gym Environment 설정
# ==========================================
gym_env:
  lambda1: 0.1           # 거래 비용 페널티 가중치
  lambda2: 1000          # K-ICS 위반 페널티 (강력!)
  scr_target: 0.35       # 목표 SCR 비율
  hedge_cost_rate: 0.002 # 일일 헤지 비용률
  max_steps: 500         # 에피소드 최대 길이

# ==========================================
# K-ICS 엔진 설정
# ==========================================
kics:
  initial_assets: 10000.0
  initial_liabilities: 9000.0
  
  # 포트폴리오 비중
  portfolio_weights:
    equity: 0.3   # 주식 30%
    bond: 0.5     # 채권 50%
    fx: 0.2       # 외화 20%
  
  # 듀레이션 설정 (년)
  duration:
    asset: 8.0    # 자산 듀레이션
    liability: 10.0  # 부채 듀레이션
  
  # 규제 충격 시나리오
  stress_scenarios:
    equity_shock: 0.30    # 주식 충격 30%
    fx_shock: 0.10       # 환율 충격 10%
    rate_shock: 0.01     # 금리 충격 1%

# ==========================================
# Agent (Safety Layer) 설정
# ==========================================
agent:
  vix_panic_threshold: 30        # 패닉 VIX 임계값
  vix_transition_threshold: 20   # 전환 VIX 임계값
  kics_danger_threshold: 120     # K-ICS 위험 임계값 (%)
  kics_critical_threshold: 100   # K-ICS 치명적 임계값 (%)
  max_hedge_change: 0.15         # 최대 1회 헤지 변동 (Gradual)
  min_hedge: 0.3                 # 최소 헤지 비율 (30%)
  max_hedge: 1.0                 # 최대 헤지 비율 (100%)

# ==========================================
# 데이터 경로 설정
# ==========================================
paths:
  data_root: "DATA"
  real_data: "DATA/data/Dynamic_Shield_Data_v4.csv"
  synthetic_stress_dir: "DATA/synthetic_stress"
  models_dir: "models"
  timegan_model_dir: "models/timegan"
  tensorboard_logs: "tensorboard_logs"
  validation_output: "src/validation"
```

### 1.3 `config/scenarios.yaml` 설계

이 파일은 **스트레스 테스트 시나리오**를 관리합니다.

```yaml
# config/scenarios.yaml
# Dynamic Shield v3.0 - 스트레스 테스트 시나리오 설정

# ==========================================
# Historical Stress 시나리오 (30%)
# ==========================================
historical_stress:
  directory: "DATA/synthetic_stress"
  ratio: 0.3  # 하이브리드 데이터셋에서 30% 차지
  
  scenarios:
    - name: "Stagflation"
      file: "Scenario_A_Stagflation.csv"
      description: "스태그플레이션 시나리오"
      enabled: true
    
    - name: "Correlation_Breakdown"
      file: "Scenario_B_Correlation_Breakdown.csv"
      description: "상관관계 붕괴 시나리오"
      enabled: true
    
    - name: "Interest_Rate_Shock"
      file: "Scenario_C_Interest_Rate_Shock.csv"
      description: "금리 충격 시나리오"
      enabled: true
    
    - name: "COVID19"
      file: "Scenario_COVID19.csv"
      description: "COVID-19 팬데믹 시나리오"
      enabled: true
    
    - name: "Swap_Point_Extreme"
      file: "Scenario_D_Swap_Point_Extreme.csv"
      description: "스왑 포인트 극단 시나리오"
      enabled: true
    
    - name: "Regime_Transition"
      file: "Scenario_E_Regime_Transition.csv"
      description: "국면 전환 시나리오"
      enabled: true
    
    - name: "Tail_Risk"
      file: "Scenario_Tail_Risk.csv"
      description: "꼬리 위험 시나리오"
      enabled: true

# ==========================================
# TimeGAN 생성 데이터 (70%)
# ==========================================
timegan_generated:
  ratio: 0.7  # 하이브리드 데이터셋에서 70% 차지
  n_samples: 1000
  sequence_length: 24  # base_config.yaml과 동기화
  
  # 생성 옵션
  options:
    use_historical_seed: true  # Historical 데이터를 seed로 사용
    diversity_weight: 0.5      # 다양성 가중치

# ==========================================
# 하이브리드 데이터셋 구성
# ==========================================
hybrid:
  historical_ratio: 0.3
  generated_ratio: 0.7
  total_days: 5000  # 목표 총 일수
  
  # 검증 옵션
  validation:
    t_sne: true
    discriminative_score: true
    save_visualization: true
```

---

## 2단계: Config 로더 유틸리티 구현

### 2.1 `src/core/config_loader.py` 생성

이 모듈은 YAML 파일을 로드하고 Python 객체로 변환합니다.

```python
"""
Config Loader - YAML 설정 파일 로더
===================================
설정 파일을 로드하고 검증하는 유틸리티 모듈
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    YAML 설정 파일 로더
    
    사용 예:
        loader = ConfigLoader()
        config = loader.load_base_config()
        timegan_epochs = config['timegan']['training']['epochs']
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Args:
            config_dir: 설정 파일 디렉토리 경로 (기본값: 프로젝트 루트/config)
        """
        if config_dir is None:
            # 프로젝트 루트 자동 탐색
            script_dir = Path(__file__).parent  # src/core/
            project_root = script_dir.parent.parent  # 프로젝트 루트
            config_dir = project_root / 'config'
        
        self.config_dir = Path(config_dir)
        
        if not self.config_dir.exists():
            raise FileNotFoundError(
                f"Config 디렉토리가 없습니다: {self.config_dir}\n"
                f"다음 명령으로 생성하세요: mkdir -p {self.config_dir}"
            )
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        """
        YAML 파일 로드
        
        Args:
            filename: YAML 파일명 (예: 'base_config.yaml')
        
        Returns:
            설정 딕셔너리
        """
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(
                f"설정 파일이 없습니다: {filepath}\n"
                f"다음 파일을 생성하세요: {filepath}"
            )
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def load_base_config(self) -> Dict[str, Any]:
        """
        base_config.yaml 로드
        
        Returns:
            기본 설정 딕셔너리
        """
        return self.load_yaml('base_config.yaml')
    
    def load_scenarios(self) -> Dict[str, Any]:
        """
        scenarios.yaml 로드
        
        Returns:
            시나리오 설정 딕셔너리
        """
        return self.load_yaml('scenarios.yaml')
    
    def get_timegan_config(self) -> Dict[str, Any]:
        """
        TimeGAN 설정만 추출
        
        Returns:
            TimeGAN 설정 딕셔너리
        """
        config = self.load_base_config()
        return config.get('timegan', {})
    
    def get_ppo_config(self) -> Dict[str, Any]:
        """
        PPO 설정만 추출
        
        Returns:
            PPO 설정 딕셔너리
        """
        config = self.load_base_config()
        return config.get('ppo', {})
    
    def get_kics_config(self) -> Dict[str, Any]:
        """
        K-ICS 설정만 추출
        
        Returns:
            K-ICS 설정 딕셔너리
        """
        config = self.load_base_config()
        return config.get('kics', {})
    
    def get_paths(self) -> Dict[str, str]:
        """
        경로 설정만 추출
        
        Returns:
            경로 설정 딕셔너리
        """
        config = self.load_base_config()
        paths = config.get('paths', {})
        
        # 상대 경로를 절대 경로로 변환
        script_dir = Path(__file__).parent  # src/core/
        project_root = script_dir.parent.parent  # 프로젝트 루트
        
        absolute_paths = {}
        for key, value in paths.items():
            if isinstance(value, str) and not os.path.isabs(value):
                # 상대 경로인 경우 프로젝트 루트 기준으로 변환
                absolute_paths[key] = str(project_root / value)
            else:
                absolute_paths[key] = value
        
        return absolute_paths


# 전역 인스턴스 (선택적)
_default_loader = None

def get_config_loader() -> ConfigLoader:
    """전역 ConfigLoader 인스턴스 반환"""
    global _default_loader
    if _default_loader is None:
        _default_loader = ConfigLoader()
    return _default_loader


# 편의 함수
def load_base_config() -> Dict[str, Any]:
    """기본 설정 로드 (편의 함수)"""
    return get_config_loader().load_base_config()

def load_scenarios() -> Dict[str, Any]:
    """시나리오 설정 로드 (편의 함수)"""
    return get_config_loader().load_scenarios()
```

### 2.2 `requirements.txt`에 PyYAML 추가

```bash
# requirements.txt에 추가
PyYAML>=6.0
```

---

## 3단계: 기존 코드 리팩토링

### 3.1 `colab_timegan_training.py` 리팩토링

**변경 전:**
```python
# 설정
SEQUENCE_LENGTH = 24
EPOCHS = 300
BATCH_SIZE = 128
FEATURE_COLS = ['VIX', 'FX', 'Correlation']
```

**변경 후:**
```python
# Config 로더 추가
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'core'))
from config_loader import ConfigLoader

# 설정 로드
loader = ConfigLoader()
config = loader.load_base_config()
timegan_config = config['timegan']

# 설정값 사용
SEQUENCE_LENGTH = timegan_config['training']['sequence_length']
EPOCHS = timegan_config['training']['epochs']
BATCH_SIZE = timegan_config['training']['batch_size']
FEATURE_COLS = timegan_config['data']['feature_cols']

# 모델 저장 경로도 설정에서 가져오기
paths = loader.get_paths()
save_dir = paths['timegan_model_dir']
```

### 3.2 `hybrid_scenarios.py` 리팩토링

**변경 전:**
```python
def train_timegan(self, training_data, epochs=300, batch_size=128, sequence_length=24):
    feature_cols = ['VIX', 'FX', 'Correlation']
    # ...
```

**변경 후:**
```python
from config_loader import ConfigLoader

class HybridScenarioBuilder:
    def __init__(self, historical_stress_dir=None, timegan_model_path=None):
        # Config 로더 초기화
        self.config_loader = ConfigLoader()
        self.base_config = self.config_loader.load_base_config()
        self.scenarios_config = self.config_loader.load_scenarios()
        
        # 경로 설정
        paths = self.config_loader.get_paths()
        if historical_stress_dir is None:
            historical_stress_dir = paths['synthetic_stress_dir']
        # ...
    
    def train_timegan(self, training_data, epochs=None, batch_size=None, sequence_length=None):
        """
        TimeGAN 모델 학습 (설정 파일에서 기본값 로드)
        """
        # 설정 파일에서 기본값 로드
        timegan_config = self.base_config['timegan']
        
        if epochs is None:
            epochs = timegan_config['training']['epochs']
        if batch_size is None:
            batch_size = timegan_config['training']['batch_size']
        if sequence_length is None:
            sequence_length = timegan_config['training']['sequence_length']
        
        feature_cols = timegan_config['data']['feature_cols']
        # ...
```

### 3.3 `ppo_trainer.py` 리팩토링

**변경 전:**
```python
def __init__(self, 
             algorithm='PPO',
             total_timesteps=100000,
             learning_rate=3e-4,
             n_steps=2048,
             batch_size=64,
             gamma=0.99):
```

**변경 후:**
```python
from config_loader import ConfigLoader

class PPOTrainer:
    def __init__(self, 
                 algorithm=None,
                 total_timesteps=None,
                 learning_rate=None,
                 n_steps=None,
                 batch_size=None,
                 gamma=None,
                 config_path=None):
        """
        Args:
            config_path: 설정 파일 경로 (기본값: config/base_config.yaml)
            다른 파라미터가 None이면 설정 파일에서 로드
        """
        # Config 로드
        loader = ConfigLoader()
        ppo_config = loader.get_ppo_config()
        
        # 설정 파일에서 기본값 로드 (인자로 전달된 값이 우선)
        self.algorithm = algorithm or ppo_config.get('algorithm', 'PPO')
        self.total_timesteps = total_timesteps or ppo_config.get('total_timesteps', 100000)
        self.learning_rate = learning_rate or ppo_config.get('learning_rate', 3e-4)
        self.n_steps = n_steps or ppo_config.get('n_steps', 2048)
        self.batch_size = batch_size or ppo_config.get('batch_size', 64)
        self.gamma = gamma or ppo_config.get('gamma', 0.99)
        
        # ...
```

### 3.4 `gym_environment.py` 리팩토링

**변경 전:**
```python
def __init__(self, 
             lambda1=0.1,
             lambda2=1000,
             scr_target=0.35,
             hedge_cost_rate=0.002,
             max_steps=500):
```

**변경 후:**
```python
from config_loader import ConfigLoader

class KICSGymEnv(gym.Env):
    def __init__(self, 
                 lambda1=None,
                 lambda2=None,
                 scr_target=None,
                 hedge_cost_rate=None,
                 max_steps=None):
        # Config 로드
        loader = ConfigLoader()
        gym_config = loader.load_base_config()['gym_env']
        
        # 설정 파일에서 기본값 로드
        self.lambda1 = lambda1 or gym_config.get('lambda1', 0.1)
        self.lambda2 = lambda2 or gym_config.get('lambda2', 1000)
        self.scr_target = scr_target or gym_config.get('scr_target', 0.35)
        self.hedge_cost_rate = hedge_cost_rate or gym_config.get('hedge_cost_rate', 0.002)
        self.max_steps = max_steps or gym_config.get('max_steps', 500)
        # ...
```

### 3.5 `kics_real.py` 리팩토링

**변경 전:**
```python
def __init__(self, initial_assets=10000.0, initial_liabilities=9000.0):
    self.w_equity = 0.3
    self.w_bond = 0.5
    self.w_fx = 0.2
    self.dur_asset = 8.0
    self.dur_liab = 10.0
```

**변경 후:**
```python
from config_loader import ConfigLoader

class KICSCalculator:
    def __init__(self, initial_assets=None, initial_liabilities=None):
        # Config 로드
        loader = ConfigLoader()
        kics_config = loader.get_kics_config()
        
        # 설정 파일에서 기본값 로드
        self.initial_assets = initial_assets or kics_config.get('initial_assets', 10000.0)
        self.initial_liabilities = initial_liabilities or kics_config.get('initial_liabilities', 9000.0)
        
        # 포트폴리오 비중
        weights = kics_config.get('portfolio_weights', {})
        self.w_equity = weights.get('equity', 0.3)
        self.w_bond = weights.get('bond', 0.5)
        self.w_fx = weights.get('fx', 0.2)
        
        # 듀레이션
        duration = kics_config.get('duration', {})
        self.dur_asset = duration.get('asset', 8.0)
        self.dur_liab = duration.get('liability', 10.0)
        # ...
```

---

## 4단계: 테스트 및 검증

### 4.1 설정 파일 검증 스크립트

```python
# test_config.py
"""
설정 파일 로드 테스트
"""

from src.core.config_loader import ConfigLoader

def test_config_loading():
    """설정 파일 로드 테스트"""
    print("=" * 60)
    print("Config 로드 테스트")
    print("=" * 60)
    
    loader = ConfigLoader()
    
    # 기본 설정 로드
    print("\n[1] base_config.yaml 로드")
    base_config = loader.load_base_config()
    print(f"  ✓ 로드 성공")
    print(f"  - TimeGAN epochs: {base_config['timegan']['training']['epochs']}")
    print(f"  - PPO learning_rate: {base_config['ppo']['learning_rate']}")
    print(f"  - K-ICS initial_assets: {base_config['kics']['initial_assets']}")
    
    # 시나리오 설정 로드
    print("\n[2] scenarios.yaml 로드")
    scenarios = loader.load_scenarios()
    print(f"  ✓ 로드 성공")
    print(f"  - Historical ratio: {scenarios['historical_stress']['ratio']}")
    print(f"  - TimeGAN ratio: {scenarios['timegan_generated']['ratio']}")
    print(f"  - 시나리오 수: {len(scenarios['historical_stress']['scenarios'])}")
    
    # 경로 확인
    print("\n[3] 경로 설정 확인")
    paths = loader.get_paths()
    for key, value in paths.items():
        print(f"  - {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✓ 모든 테스트 통과!")
    print("=" * 60)

if __name__ == "__main__":
    test_config_loading()
```

### 4.2 실행 순서

```bash
# 1. 폴더 생성
mkdir -p config

# 2. YAML 파일 생성 (위의 내용을 복사하여 생성)
# config/base_config.yaml
# config/scenarios.yaml

# 3. Config 로더 구현
# src/core/config_loader.py

# 4. PyYAML 설치
pip install PyYAML

# 5. 테스트 실행
python test_config.py

# 6. 기존 코드 리팩토링 (위의 예시 참고)
```

---

## 📝 구현 체크리스트

### 필수 작업
- [ ] `config/` 폴더 생성
- [ ] `config/base_config.yaml` 작성
- [ ] `config/scenarios.yaml` 작성
- [ ] `src/core/config_loader.py` 구현
- [ ] `requirements.txt`에 PyYAML 추가
- [ ] `colab_timegan_training.py` 리팩토링
- [ ] `hybrid_scenarios.py` 리팩토링
- [ ] `ppo_trainer.py` 리팩토링
- [ ] `gym_environment.py` 리팩토링
- [ ] `kics_real.py` 리팩토링
- [ ] 테스트 스크립트 작성 및 실행

### 선택 작업
- [ ] `agent.py` 리팩토링
- [ ] `environment.py` 리팩토링
- [ ] 설정 파일 검증 로직 추가
- [ ] 실험별 설정 파일 분리 (예: `config/experiment_001.yaml`)

---

## 🎯 기대 효과

### Before (하드코딩)
```python
# 여러 파일에 산재된 설정
# colab_timegan_training.py
EPOCHS = 300

# hybrid_scenarios.py
def train_timegan(..., epochs=300, ...):

# ppo_trainer.py
def __init__(..., total_timesteps=100000, ...):
```

### After (YAML 설정)
```python
# 모든 설정이 YAML에 중앙 집중
# config/base_config.yaml
timegan:
  training:
    epochs: 300

# 코드는 설정만 참조
loader = ConfigLoader()
config = loader.load_base_config()
epochs = config['timegan']['training']['epochs']
```

### 장점
1. ✅ **중앙 관리**: 모든 설정이 한 곳에
2. ✅ **재현성**: 설정 파일만 공유하면 동일한 실험 재현
3. ✅ **유연성**: 코드 수정 없이 설정만 변경
4. ✅ **일관성**: 여러 파일 간 설정 일관성 보장
5. ✅ **버전 관리**: Git으로 설정 변경 이력 추적

---

## 다음 단계

설정 파일 구조가 완성되면:
1. **2단계**: `main.py` 통합 실행 스크립트 구현
2. **3단계**: `src/safety/risk_control.py` 독립 모듈 구현

이렇게 단계적으로 진행하면 제안서 구조에 맞는 완전한 시스템을 구축할 수 있습니다.

