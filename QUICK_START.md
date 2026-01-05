# 🚀 빠른 시작 가이드

Dynamic Shield를 빠르게 실행하는 방법입니다.

## 1️⃣ 설치 (최초 1회)

### 자동 설치 (권장)
```bash
# macOS/Linux
./install.sh

# Windows
install.bat
```

### 수동 설치
자세한 내용은 [SETUP.md](SETUP.md)를 참고하세요.

## 2️⃣ 실행

### 가상환경 활성화
```bash
# Conda 사용 시
conda activate dynamic_shield

# venv 사용 시
source venv/bin/activate  # macOS/Linux
# 또는
venv\Scripts\activate     # Windows
```

### 프로젝트 디렉토리로 이동
```bash
cd src
```

## 3️⃣ 빠른 테스트

### 백테스트 실행 (가장 간단)
```bash
python main.py --mode backtest
```

### 전체 파이프라인 실행
```bash
python main.py --mode all
```

## 4️⃣ 주요 명령어

```bash
# 백테스트만 실행
python main.py --mode backtest

# PPO 학습
python main.py --mode train --timesteps 50000

# 시스템 검증
python main.py --mode validate

# 실시간 모드 (시뮬레이션)
python main.py --mode live
```

## 📚 더 자세한 정보

- 전체 실행 가이드: `src/실행_가이드.md`
- 설치 가이드: `SETUP.md`
- 프로젝트 개요: `README.md`

## ❓ 문제 해결

### 가상환경이 활성화되지 않음
```bash
# Conda 사용 시
conda activate dynamic_shield

# venv 사용 시
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 모듈을 찾을 수 없음
```bash
# src 디렉토리에서 실행하는지 확인
cd src
python main.py --mode backtest
```

### PyTorch 설치 오류
```bash
# CPU 버전으로 재설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

