# TimeGAN 모델 학습 가이드 (Google Colab)

이 가이드는 Google Colab에서 TimeGAN 모델을 학습하고, 로컬 프로젝트에서 사용하는 방법을 설명합니다.

## 📋 개요

- **문제**: 로컬 환경에서 TimeGAN 학습 시 저장공간 부족
- **해결**: Google Colab에서 학습 후 모델 다운로드
- **결과**: 로컬에서 저장된 모델을 로드하여 사용

## 🚀 사용 방법

### 1단계: Google Colab에서 모델 학습

1. **Colab 노트북 생성**
   - Google Colab 접속: https://colab.research.google.com/
   - 새 노트북 생성

2. **스크립트 업로드**
   - 프로젝트의 `colab_timegan_training.py` 파일을 Colab에 업로드
   - 또는 Colab에서 직접 코드 복사

3. **데이터 준비**
   - 옵션 A: Historical Stress CSV 파일들을 Colab에 업로드
   - 옵션 B: Google Drive에 데이터 저장 후 마운트
   - 옵션 C: 스크립트 내 샘플 데이터 사용 (테스트용)

4. **스크립트 실행**
   ```python
   # Colab에서 실행
   !python colab_timegan_training.py
   ```
   
   또는 노트북 셀에서 단계별 실행

5. **파일 다운로드**
   - 생성된 `timegan_model.zip` 파일을 다운로드
   - 또는 Google Drive에 저장

### 2단계: 로컬 프로젝트에 모델 복사

1. **압축 해제**
   ```bash
   unzip timegan_model.zip
   ```

2. **프로젝트에 복사**
   ```bash
   # 프로젝트 루트에서
   mkdir -p models/timegan
   cp -r timegan_model/* models/timegan/
   ```

3. **폴더 구조 확인**
   ```
   models/timegan/
   ├── timegan_model.pkl    # 학습된 TimeGAN 모델
   ├── scaler.pkl           # 정규화 스케일러
   └── params.pkl           # 하이퍼파라미터
   ```

### 3단계: 로컬에서 사용

로컬 프로젝트에서 `hybrid_scenarios.py`를 실행하면 자동으로 저장된 모델을 로드합니다:

```python
from core.hybrid_scenarios import HybridScenarioBuilder

# 모델 자동 로드 (models/timegan/ 폴더에서)
builder = HybridScenarioBuilder()

# 전체 파이프라인 실행
hybrid_data = builder.run_full_pipeline(
    n_generated=2000,
    epochs=100,  # 학습은 이미 완료되었으므로 무시됨
    save_dir='hybrid_output'
)
```

## 📁 파일 구조

```
프로젝트 루트/
├── colab_timegan_training.py    # Colab용 학습 스크립트
├── models/
│   └── timegan/                 # 저장된 모델 (Colab에서 다운로드)
│       ├── timegan_model.pkl
│       ├── scaler.pkl
│       └── params.pkl
└── src/
    └── core/
        └── hybrid_scenarios.py   # 로컬 실행 스크립트
```

## ⚙️ 설정 변경

### Colab 스크립트에서 변경 가능한 설정

`colab_timegan_training.py` 파일에서:

```python
SEQUENCE_LENGTH = 24  # 시퀀스 길이 (일 단위)
EPOCHS = 300          # 학습 에포크 수
BATCH_SIZE = 128       # 배치 크기
```

### 데이터 소스 변경

스크립트의 데이터 준비 부분을 수정:

```python
# 옵션 A: CSV 파일 로드
training_data = pd.read_csv('your_data.csv')

# 옵션 B: Google Drive에서 로드
from google.colab import drive
drive.mount('/content/drive')
training_data = pd.read_csv('/content/drive/MyDrive/your_data.csv')
```

## 🔍 문제 해결

### 모델 로드 실패

1. **경로 확인**
   ```python
   # models/timegan/ 폴더가 올바른 위치에 있는지 확인
   import os
   print(os.path.exists('models/timegan/timegan_model.pkl'))
   ```

2. **수동 로드**
   ```python
   builder = HybridScenarioBuilder()
   builder.load_timegan_model('models/timegan')
   ```

### Colab에서 메모리 부족

- 배치 크기 줄이기: `BATCH_SIZE = 64`
- 에포크 수 줄이기: `EPOCHS = 100` (테스트용)
- 데이터 크기 줄이기

### 생성 품질이 낮음

- 에포크 수 증가: `EPOCHS = 500+`
- 더 많은 학습 데이터 사용
- 하이퍼파라미터 튜닝

## 📊 결과 확인

학습 완료 후:

1. **Discriminative Score 확인**
   - 0.5에 가까울수록 좋음 (구분 어려움 = 품질 좋음)
   - 1.0에 가까울수록 나쁨 (쉽게 구분됨 = 품질 나쁨)

2. **t-SNE 시각화 확인**
   - 생성된 데이터와 Historical 데이터의 분포 비교
   - 다양성 확인

## 💡 팁

- Colab Pro 사용 시 더 빠른 학습 가능
- Google Drive에 모델 저장하여 백업
- 여러 버전의 모델을 비교하여 최적 선택

## 📝 참고

- TimeGAN 라이브러리: https://github.com/ydataai/ydata-synthetic
- 프로젝트 문서: `README.md`

