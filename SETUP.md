# Dynamic Shield - 빠른 시작 가이드

이 문서는 프로젝트를 처음 실행하는 분을 위한 단계별 설치 가이드입니다.

## 📋 사전 준비

1. **Python 3.9 이상** 설치 확인
   ```bash
   python3 --version
   ```

2. **프로젝트 다운로드/클론**
   - 프로젝트 폴더를 본인의 컴퓨터에 다운로드하세요

## 🔧 설치 단계

### 방법 1: 자동 설치 스크립트 사용 (권장)

#### macOS/Linux
```bash
cd /path/to/Dynamic-Shield-K-ICS-AI
./install.sh
```

#### Windows
```cmd
cd C:\path\to\Dynamic-Shield-K-ICS-AI
install.bat
```

### 방법 2: 수동 설치

#### Step 1: 프로젝트 디렉토리로 이동
```bash
cd /path/to/Dynamic-Shield-K-ICS-AI
```

#### Step 2: 가상환경 생성 및 활성화

#### Conda 사용 시 (권장)
```bash
conda create -n dynamic_shield python=3.11 -y
conda activate dynamic_shield
```

#### venv 사용 시
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate     # Windows
```

### Step 3: PyTorch 설치
```bash
# Conda 사용 시
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# pip 사용 시
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: 나머지 의존성 설치
```bash
pip install -r requirements.txt
```

### Step 5: 설치 확인
```bash
cd src
python -c "import torch; import stable_baselines3; print('✓ 설치 완료!')"
```

## 🎯 빠른 실행 테스트

### 1. K-ICS 엔진 테스트
```bash
cd src
python core/kics_real.py
```

### 2. 백테스트 실행
```bash
python main.py --mode backtest
```

### 3. 전체 파이프라인 실행
```bash
python main.py --mode all
```

## ❓ 문제 해결

### 문제: `ModuleNotFoundError: No module named 'xxx'`
**해결**: 가상환경이 활성화되어 있는지 확인하고, `pip install -r requirements.txt` 다시 실행

### 문제: PyTorch 설치 실패
**해결**: 
- Conda 사용 시: `conda install pytorch torchvision torchaudio cpuonly -c pytorch -y`
- pip 사용 시: 공식 PyTorch 사이트에서 OS에 맞는 명령어 확인

### 문제: 경로 오류
**해결**: 항상 `src` 디렉토리에서 스크립트를 실행하세요
```bash
cd src
python main.py --mode backtest
```

## 📚 더 자세한 정보

- 전체 실행 가이드: `src/실행_가이드.md`
- 프로젝트 개요: `README.md`

## 💡 팁

- 매번 실행 전에 가상환경을 활성화하세요
- `src` 디렉토리에서 스크립트를 실행하세요
- 문제가 발생하면 `src/실행_가이드.md`를 참고하세요

