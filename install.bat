@echo off
REM Dynamic Shield - 자동 설치 스크립트 (Windows)

echo ==========================================
echo Dynamic Shield 설치 스크립트
echo ==========================================
echo.

REM Python 버전 확인
echo [1/5] Python 버전 확인...
python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python이 설치되어 있지 않습니다.
    pause
    exit /b 1
)
python --version
echo [확인] Python 발견됨

REM 가상환경 생성
echo.
echo [2/5] 가상환경 생성...
if exist venv (
    echo [경고] venv 폴더가 이미 존재합니다. 기존 환경을 사용합니다.
) else (
    python -m venv venv
    echo [확인] 가상환경 생성 완료
)

REM 가상환경 활성화
echo.
echo [3/5] 가상환경 활성화...
call venv\Scripts\activate.bat
echo [확인] 가상환경 활성화 완료

REM PyTorch 설치
echo.
echo [4/5] PyTorch 설치 (CPU 버전)...
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo [확인] PyTorch 설치 완료

REM 나머지 의존성 설치
echo.
echo [5/5] 나머지 의존성 설치...
pip install -r requirements.txt
echo [확인] 의존성 설치 완료

REM 설치 확인
echo.
echo ==========================================
echo 설치 확인 중...
echo ==========================================
cd src
python -c "import torch; import stable_baselines3; print('[확인] 모든 패키지 정상 설치됨')" || (
    echo [오류] 설치 확인 실패
    pause
    exit /b 1
)

echo.
echo ==========================================
echo 설치 완료!
echo ==========================================
echo.
echo 다음 명령어로 실행하세요:
echo   venv\Scripts\activate
echo   cd src
echo   python main.py --mode backtest
echo.
pause

