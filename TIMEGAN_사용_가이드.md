# TimeGAN 모델 활용 가이드

Colab에서 학습한 TimeGAN 모델을 로컬에서 활용하는 방법입니다.

## 📁 현재 상태

현재 `timegan_model/` 폴더가 프로젝트 루트에 있습니다:
```
timegan_model/
├── timegan_model.pkl    # 학습된 TimeGAN 모델
├── scaler.pkl           # 정규화 스케일러
└── params.pkl           # 하이퍼파라미터
```

## 🚀 빠른 시작

### 방법 1: 예제 스크립트 실행

```bash
python use_timegan_model.py
```

이 스크립트는:
- TimeGAN 모델을 자동으로 로드
- 샘플 데이터 생성
- Historical Stress 데이터와 혼합하여 하이브리드 데이터셋 생성

### 방법 2: Python 코드에서 직접 사용

```python
from src.core.hybrid_scenarios import HybridScenarioBuilder

# 모델 로드
builder = HybridScenarioBuilder()
builder.load_timegan_model('timegan_model')  # 현재 위치

# 데이터 생성
generated_data = builder.generate_timegan_data(n_samples=1000)

# 결과 확인
print(generated_data.head())
```

## 📋 상세 사용법

### 1. 모델 로드

```python
from src.core.hybrid_scenarios import HybridScenarioBuilder

builder = HybridScenarioBuilder()

# 방법 A: 직접 경로 지정
builder.load_timegan_model('timegan_model')

# 방법 B: models/timegan/ 폴더로 이동 후 자동 로드
# mkdir -p models/timegan
# cp -r timegan_model/* models/timegan/
builder = HybridScenarioBuilder()  # 자동으로 models/timegan/에서 로드
```

### 2. 데이터 생성

```python
# 기본 생성
generated_data = builder.generate_timegan_data(n_samples=1000)

# 시퀀스 길이 지정
generated_data = builder.generate_timegan_data(
    n_samples=1000,
    sequence_length=24
)
```

### 3. 하이브리드 데이터셋 구축

```python
# Historical Stress 데이터 로드
builder.load_historical_stress()

# 하이브리드 데이터셋 구축 (70% 생성, 30% Historical)
builder.build_hybrid_dataset(
    generated_ratio=0.7,
    historical_ratio=0.3
)

# 결과 확인
print(builder.hybrid_data.head())
```

### 4. 전체 파이프라인 실행

```python
# 한 번에 모든 작업 수행
hybrid_data = builder.run_full_pipeline(
    n_generated=2000,
    epochs=100,  # 이미 학습 완료되었으므로 무시됨
    save_dir='output'
)
```

## 🔧 모델 정보 확인

```python
# 모델 파라미터 확인
if builder.timegan_params:
    print(f"Sequence Length: {builder.timegan_params['sequence_length']}")
    print(f"Epochs: {builder.timegan_params['epochs']}")
    print(f"Batch Size: {builder.timegan_params['batch_size']}")
    print(f"Feature Columns: {builder.timegan_params['feature_cols']}")
```

## 📊 생성된 데이터 활용

생성된 데이터는 다음과 같은 컬럼을 가집니다:
- `VIX`: 변동성 지수
- `FX`: 환율
- `Correlation`: 상관관계

```python
# 생성된 데이터 저장
generated_data.to_csv('generated_data.csv', index=False)

# 통계 확인
print(generated_data.describe())

# 시각화
import matplotlib.pyplot as plt
generated_data.plot(subplots=True, figsize=(12, 8))
plt.show()
```

## 🗂️ 폴더 구조 정리 (선택사항)

코드가 `models/timegan/` 폴더를 기본 경로로 사용하므로, 이동하는 것을 권장합니다:

```bash
# 프로젝트 루트에서 실행
mkdir -p models/timegan
cp -r timegan_model/* models/timegan/
```

이후에는 경로 지정 없이 자동으로 로드됩니다:

```python
builder = HybridScenarioBuilder()  # 자동으로 models/timegan/에서 로드
```

## ⚠️ 주의사항

1. **패키지 설치**: `ydata-synthetic` 패키지가 필요합니다
   ```bash
   pip install ydata-synthetic
   ```

2. **Python 버전**: Python 3.9-3.11만 지원합니다

3. **메모리**: 대량의 데이터 생성 시 메모리 사용량에 주의하세요

## 🔍 문제 해결

### 모델 로드 실패

```python
# 경로 확인
import os
print(os.path.exists('timegan_model/timegan_model.pkl'))

# 수동 로드
builder.load_timegan_model('timegan_model')
```

### 데이터 생성 실패

- 모델이 제대로 로드되었는지 확인
- `n_samples`를 줄여서 시도
- 모델이 학습된 시퀀스 길이와 동일한지 확인

## 📝 예제 스크립트

`use_timegan_model.py` 파일을 참고하세요. 이 스크립트는:
- 모델 로드
- 데이터 생성
- 하이브리드 데이터셋 구축
- 결과 저장

을 모두 수행합니다.

