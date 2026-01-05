#!/bin/bash
# Dynamic Shield - 자동 설치 스크립트 (macOS/Linux)

set -e  # 오류 발생 시 중단

echo "=========================================="
echo "Dynamic Shield 설치 스크립트"
echo "=========================================="
echo ""

# Python 버전 확인
echo "[1/5] Python 버전 확인..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3가 설치되어 있지 않습니다."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✓ Python $PYTHON_VERSION 발견"

# 가상환경 생성
echo ""
echo "[2/5] 가상환경 생성..."
if [ -d "venv" ]; then
    echo "⚠️  venv 폴더가 이미 존재합니다. 기존 환경을 사용합니다."
else
    python3 -m venv venv
    echo "✓ 가상환경 생성 완료"
fi

# 가상환경 활성화
echo ""
echo "[3/5] 가상환경 활성화..."
source venv/bin/activate
echo "✓ 가상환경 활성화 완료"

# PyTorch 설치
echo ""
echo "[4/5] PyTorch 설치 (CPU 버전)..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo "✓ PyTorch 설치 완료"

# 나머지 의존성 설치
echo ""
echo "[5/5] 나머지 의존성 설치..."
pip install -r requirements.txt
echo "✓ 의존성 설치 완료"

# 설치 확인
echo ""
echo "=========================================="
echo "설치 확인 중..."
echo "=========================================="
cd src
python -c "import torch; import stable_baselines3; print('✓ 모든 패키지 정상 설치됨')" || {
    echo "❌ 설치 확인 실패"
    exit 1
}

echo ""
echo "=========================================="
echo "✅ 설치 완료!"
echo "=========================================="
echo ""
echo "다음 명령어로 실행하세요:"
echo "  source venv/bin/activate"
echo "  cd src"
echo "  python main.py --mode backtest"
echo ""

