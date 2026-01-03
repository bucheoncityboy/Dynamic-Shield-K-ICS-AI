"""
TimeGAN 모델 사용 스크립트 (Google Colab용)

로컬에서 학습한 TimeGAN 모델을 Colab에서 사용합니다.
디스크 용량이 부족한 로컬 환경 대신 Colab에서 실행하세요.
"""

# ==========================================
# 1. 필수 패키지 설치
# ==========================================
print("=" * 70)
print("필수 패키지 설치")
print("=" * 70)

import subprocess
import sys

def install_package(package_name):
    """패키지 설치"""
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', package_name, '--quiet'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        return True
    except:
        return False

# ydata-synthetic 설치
try:
    import ydata_synthetic
    print("✓ ydata-synthetic 이미 설치됨")
except ImportError:
    print("📦 ydata-synthetic 설치 중...")
    if install_package('ydata-synthetic'):
        print("✓ ydata-synthetic 설치 완료")
    else:
        print("❌ ydata-synthetic 설치 실패")
        print("수동으로 설치하세요: !pip install ydata-synthetic")

# ==========================================
# 2. 모델 파일 업로드
# ==========================================
print("\n" + "=" * 70)
print("모델 파일 준비")
print("=" * 70)
print("다음 중 하나를 선택하세요:")
print("\n[방법 1] Google Drive에서 로드:")
print("  from google.colab import drive")
print("  drive.mount('/content/drive')")
print("  # timegan_model 폴더를 Google Drive에 업로드한 후")
print("  model_path = '/content/drive/MyDrive/timegan_model'")
print("\n[방법 2] 직접 업로드:")
print("  from google.colab import files")
print("  files.upload()  # timegan_model.zip 업로드")
print("  !unzip timegan_model.zip")
print("  model_path = 'timegan_model'")

# ==========================================
# 3. 모델 사용 예제
# ==========================================
print("\n" + "=" * 70)
print("모델 사용 예제")
print("=" * 70)

example_code = '''
# 예제 코드 (Colab에서 실행)

import sys
import os

# 프로젝트 파일 업로드 (또는 GitHub에서 클론)
# 방법 1: 파일 업로드
from google.colab import files
uploaded = files.upload()  # hybrid_scenarios.py 등 필요한 파일 업로드

# 방법 2: GitHub에서 클론
# !git clone https://github.com/your-repo/your-project.git
# sys.path.insert(0, 'your-project/src/core')

# 모델 경로 설정
model_path = 'timegan_model'  # 또는 Google Drive 경로

# 모델 로드 및 사용
from hybrid_scenarios import HybridScenarioBuilder

builder = HybridScenarioBuilder()
builder.load_timegan_model(model_path)

# 데이터 생성
generated_data = builder.generate_timegan_data(n_samples=1000)
print(f"생성된 데이터: {len(generated_data)}일")
print(generated_data.head())

# 하이브리드 데이터셋 구축
builder.load_historical_stress()
builder.build_hybrid_dataset(generated_ratio=0.7, historical_ratio=0.3)

# 결과 저장
builder.hybrid_data.to_csv('hybrid_dataset.csv', index=False)
files.download('hybrid_dataset.csv')  # 다운로드
'''

print(example_code)

print("\n" + "=" * 70)
print("완료!")
print("=" * 70)
print("\n위 예제 코드를 Colab에서 실행하세요.")

