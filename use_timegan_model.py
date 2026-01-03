"""
TimeGAN 모델 활용 예제 스크립트

Colab에서 학습한 TimeGAN 모델을 로컬에서 사용하는 방법을 보여줍니다.
"""

import os
import sys
import subprocess

# Python 버전 확인
python_version = sys.version_info
print("=" * 70)
print("시스템 확인")
print("=" * 70)
print(f"Python 버전: {python_version.major}.{python_version.minor}.{python_version.micro}")

# ydata-synthetic는 Python 3.9-3.11만 지원
if python_version.major == 3 and python_version.minor >= 12:
    print("⚠️  경고: Python 3.12 이상입니다.")
    print("   ydata-synthetic는 Python 3.9-3.11만 지원합니다.")
    print("   일부 기능이 작동하지 않을 수 있습니다.")
elif python_version.major == 3 and python_version.minor < 9:
    print("⚠️  경고: Python 3.9 미만입니다.")
    print("   ydata-synthetic는 Python 3.9-3.11만 지원합니다.")
    print("   일부 기능이 작동하지 않을 수 있습니다.")

print("\n필수 패키지 확인 및 설치")
print("=" * 70)

def install_package(package_name):
    """패키지 자동 설치"""
    try:
        print(f"📦 {package_name} 설치 중...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package_name],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            print(f"✓ {package_name} 설치 완료")
            return True
        else:
            print(f"⚠️  {package_name} 설치 경고 (계속 시도): {result.stderr[:100]}")
            # 경고가 있어도 계속 진행
            return True
    except Exception as e:
        print(f"❌ {package_name} 설치 실패: {e}")
        return False

missing_packages = []

# pyyaml 확인
try:
    import yaml
    print("✓ pyyaml 설치됨")
except ImportError:
    missing_packages.append('pyyaml')
    print("⚠️  pyyaml 패키지가 없습니다.")

# ydata-synthetic 확인 (선택적 - 모델 사용 시에만 필요)
ydata_synthetic_available = False
try:
    import ydata_synthetic
    ydata_synthetic_available = True
    print("✓ ydata-synthetic 설치됨")
except ImportError:
    print("⚠️  ydata-synthetic 패키지가 없습니다.")
    print("   (모델 파일 확인은 가능하지만, 실제 사용은 Colab에서 하세요)")
    # ydata-synthetic는 용량이 크므로 자동 설치하지 않음
    # missing_packages.append('ydata-synthetic')  # 주석 처리

# 자동 설치
if missing_packages:
    print(f"\n📦 {len(missing_packages)}개 패키지 자동 설치 중...")
    print("-" * 70)
    
    for package in missing_packages:
        if not install_package(package):
            print(f"\n❌ {package} 설치에 실패했습니다.")
            print(f"수동으로 설치하세요: pip install {package}")
            sys.exit(1)
    
    # 설치 후 다시 import 시도
    print("\n설치된 패키지 로드 중...")
    import importlib
    
    # 모듈 캐시 무효화
    if 'yaml' in sys.modules:
        del sys.modules['yaml']
    if 'ydata_synthetic' in sys.modules:
        del sys.modules['ydata_synthetic']
    
    importlib.invalidate_caches()
    
    # 재확인 및 재시도
    failed = []
    max_retries = 3
    
    for package in missing_packages:
        loaded = False
        for retry in range(max_retries):
            try:
                if package == 'pyyaml':
                    import yaml
                    print("✓ pyyaml 로드 성공")
                    loaded = True
                    break
                elif package == 'ydata-synthetic':
                    import ydata_synthetic
                    print("✓ ydata-synthetic 로드 성공")
                    loaded = True
                    break
            except ImportError:
                if retry < max_retries - 1:
                    importlib.invalidate_caches()
                    import time
                    time.sleep(0.5)  # 잠시 대기
                else:
                    failed.append(package)
        
        if not loaded and package not in failed:
            failed.append(package)
    
    if failed:
        print(f"\n⚠️  다음 패키지 로드 실패: {', '.join(failed)}")
        print("스크립트를 계속 실행하지만 일부 기능이 작동하지 않을 수 있습니다.")
        print("커널/터미널을 재시작한 후 다시 시도하세요.")
        # sys.exit(1) 제거 - 계속 진행하도록

print("\n✓ 모든 필수 패키지 준비 완료\n")

import pandas as pd
import numpy as np

# 프로젝트 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'core'))

from hybrid_scenarios import HybridScenarioBuilder

def main():
    print("=" * 70)
    print("TimeGAN 모델 활용 예제")
    print("=" * 70)
    
    # 방법 1: 자동 로드 (models/timegan/ 폴더에 있는 경우)
    # 현재 timegan_model 폴더가 루트에 있으므로 경로 지정 필요
    print("\n[방법 1] 직접 경로 지정")
    print("-" * 70)
    
    # 프로젝트 루트 경로
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # 모델 경로 확인 (여러 위치 시도)
    possible_model_paths = [
        os.path.join(project_root, 'timegan_model'),  # 현재 위치
        os.path.join(project_root, 'models', 'timegan'),  # 표준 위치
    ]
    
    model_path = None
    for path in possible_model_paths:
        model_file = os.path.join(path, 'timegan_model.pkl')
        if os.path.exists(model_file):
            model_path = path
            print(f"✓ 모델 파일 발견: {model_file}")
            break
    
    if model_path is None:
        print("❌ 모델 파일을 찾을 수 없습니다.")
        print("\n다음 위치를 확인하세요:")
        for path in possible_model_paths:
            print(f"  - {path}/timegan_model.pkl")
        print("\n해결 방법:")
        print("1. timegan_model 폴더가 프로젝트 루트에 있는지 확인")
        print("2. 또는 models/timegan/ 폴더로 이동:")
        print("   mkdir -p models/timegan")
        print("   cp -r timegan_model/* models/timegan/")
        return
    
    # HybridScenarioBuilder 생성
    try:
        builder = HybridScenarioBuilder()
    except Exception as e:
        print(f"❌ HybridScenarioBuilder 생성 실패: {e}")
        print("\n해결 방법:")
        print("1. 필수 패키지가 모두 설치되었는지 확인")
        print("2. Config 파일이 올바른 위치에 있는지 확인")
        print("3. 스크립트를 다시 실행하세요")
        return
    
    # 모델 파일 확인 (ydata-synthetic 없이도 가능)
    print(f"\n모델 파일 확인: {model_path}")
    
    # 파라미터 파일만 먼저 확인
    params_path = os.path.join(model_path, 'params.pkl')
    if os.path.exists(params_path):
        try:
            import pickle
            with open(params_path, 'rb') as f:
                params = pickle.load(f)
            print("✓ 모델 파라미터 확인:")
            for key, value in params.items():
                print(f"  - {key}: {value}")
        except Exception as e:
            print(f"⚠️  파라미터 파일 읽기 실패: {e}")
    
    # ydata-synthetic가 없으면 모델 로드 불가
    if not ydata_synthetic_available:
        print("\n" + "=" * 70)
        print("⚠️  ydata-synthetic가 설치되지 않아 모델을 로드할 수 없습니다")
        print("=" * 70)
        print("\n해결 방법:")
        print("\n[방법 1] Colab에서 사용 (권장):")
        print("  1. timegan_model 폴더를 zip으로 압축")
        print("  2. Colab에 업로드")
        print("  3. Colab에서 다음 코드 실행:")
        print()
        print("     !pip install ydata-synthetic")
        print("     from src.core.hybrid_scenarios import HybridScenarioBuilder")
        print("     builder = HybridScenarioBuilder()")
        print("     builder.load_timegan_model('timegan_model')")
        print("     generated_data = builder.generate_timegan_data(n_samples=1000)")
        print()
        print("[방법 2] 디스크 공간 확보 후 설치:")
        print("  pip install ydata-synthetic")
        print("  python use_timegan_model.py")
        print()
        print("현재 모델 파일은 정상적으로 저장되어 있습니다.")
        print("모델 사용은 Colab에서 하시는 것을 권장합니다.")
        return
    
    # ydata-synthetic가 있으면 모델 로드 시도
    print(f"\n모델 로드 시도: {model_path}")
    try:
        load_success = builder.load_timegan_model(model_path)
    except Exception as e:
        print(f"❌ 모델 로드 중 오류 발생: {e}")
        print("\n가능한 원인:")
        print("1. ydata-synthetic 패키지가 제대로 설치되지 않음")
        print("2. 모델 파일이 손상됨")
        print("3. Python 버전 호환성 문제 (ydata-synthetic는 Python 3.9-3.11만 지원)")
        return
    
    if load_success:
        print("✓ 모델 로드 성공!")
        
        # 모델 정보 확인
        if builder.timegan_params:
            print(f"\n모델 정보:")
            print(f"  - Sequence Length: {builder.timegan_params.get('sequence_length', 'N/A')}")
            print(f"  - Epochs: {builder.timegan_params.get('epochs', 'N/A')}")
            print(f"  - Batch Size: {builder.timegan_params.get('batch_size', 'N/A')}")
            print(f"  - Feature Columns: {builder.timegan_params.get('feature_cols', 'N/A')}")
        
        # 데이터 생성 예제
        print("\n[데이터 생성 예제]")
        print("-" * 70)
        n_samples = 100  # 생성할 시퀀스 수
        print(f"TimeGAN으로 {n_samples}개의 시퀀스 생성 중...")
        
        try:
            generated_data = builder.generate_timegan_data(n_samples=n_samples)
        except Exception as e:
            print(f"❌ 데이터 생성 실패: {e}")
            print("\n가능한 원인:")
            print("1. 모델이 제대로 로드되지 않음")
            print("2. ydata-synthetic 패키지 문제")
            print("3. 메모리 부족")
            generated_data = None
        
        if generated_data is not None and len(generated_data) > 0:
            print(f"\n✓ 생성 완료: {len(generated_data)}일의 데이터")
            print(f"\n생성된 데이터 샘플:")
            print(generated_data.head())
            print(f"\n데이터 통계:")
            print(generated_data.describe())
            
            # 저장 예제
            output_path = 'generated_timegan_data.csv'
            generated_data.to_csv(output_path, index=False)
            print(f"\n✓ 데이터 저장: {output_path}")
        else:
            print("⚠️  데이터 생성 실패")
    else:
        print("❌ 모델 로드 실패")
        print("\n해결 방법:")
        print("1. timegan_model 폴더가 프로젝트 루트에 있는지 확인")
        print("2. 또는 models/timegan/ 폴더로 이동:")
        print("   mkdir -p models/timegan")
        print("   cp -r timegan_model/* models/timegan/")
        return
    
    # 방법 2: 전체 파이프라인 실행 (Historical Stress + TimeGAN 생성)
    if 'generated_data' in locals() and generated_data is not None:
        print("\n\n[방법 2] 전체 하이브리드 파이프라인 실행")
        print("-" * 70)
        print("Historical Stress 데이터와 TimeGAN 생성 데이터를 혼합합니다.")
        
        # Historical Stress 데이터 로드
        try:
            builder.load_historical_stress()
        except Exception as e:
            print(f"⚠️  Historical Stress 데이터 로드 실패: {e}")
            builder.historical_data = None
        
        if builder.historical_data is not None and len(builder.historical_data) > 0:
            # 하이브리드 데이터셋 구축
            print("\n하이브리드 데이터셋 구축 중...")
            try:
                builder.build_hybrid_dataset(generated_ratio=0.7, historical_ratio=0.3)
            except Exception as e:
                print(f"⚠️  하이브리드 데이터셋 구축 실패: {e}")
                builder.hybrid_data = None
            
            if builder.hybrid_data is not None and len(builder.hybrid_data) > 0:
                print(f"\n✓ 하이브리드 데이터셋 생성 완료: {len(builder.hybrid_data)}일")
                print(f"\n데이터 구성:")
                print(f"  - Historical Stress: {len(builder.historical_data)}일 (30%)")
                print(f"  - TimeGAN 생성: {len(generated_data)}일 (70%)")
                print(f"  - 총합: {len(builder.hybrid_data)}일")
                
                # 저장
                try:
                    hybrid_output_path = 'hybrid_dataset.csv'
                    builder.hybrid_data.to_csv(hybrid_output_path, index=False)
                    print(f"\n✓ 하이브리드 데이터셋 저장: {hybrid_output_path}")
                except Exception as e:
                    print(f"⚠️  파일 저장 실패: {e}")
            else:
                print("⚠️  하이브리드 데이터셋 생성 실패")
        else:
            print("⚠️  Historical Stress 데이터를 찾을 수 없습니다.")
            print("   DATA/synthetic_stress/ 폴더를 확인하세요.")
    else:
        print("\n⚠️  데이터 생성이 완료되지 않아 하이브리드 파이프라인을 건너뜁니다.")
    
    print("\n" + "=" * 70)
    print("완료!")
    print("=" * 70)

if __name__ == "__main__":
    main()

