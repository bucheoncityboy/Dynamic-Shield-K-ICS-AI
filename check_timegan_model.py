"""
TimeGAN 모델 확인 스크립트 (ydata-synthetic 설치 불필요)

디스크 용량이 부족한 환경에서도 모델 파일과 파라미터를 확인할 수 있습니다.
실제 모델 사용은 Colab에서 하세요.
"""

import os
import pickle
import sys

def check_model_files(model_dir='timegan_model'):
    """모델 파일 확인 (모델 로드 없이)"""
    print("=" * 70)
    print("TimeGAN 모델 파일 확인")
    print("=" * 70)
    
    # 프로젝트 루트 경로
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(project_root, model_dir)
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 디렉토리를 찾을 수 없습니다: {model_path}")
        return False
    
    print(f"✓ 모델 디렉토리 발견: {model_path}\n")
    
    # 파일 확인
    files_to_check = {
        'timegan_model.pkl': '학습된 TimeGAN 모델',
        'scaler.pkl': '정규화 스케일러',
        'params.pkl': '하이퍼파라미터'
    }
    
    all_exist = True
    for filename, description in files_to_check.items():
        filepath = os.path.join(model_path, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            size_mb = size / (1024 * 1024)
            print(f"✓ {filename:20s} - {description:20s} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {filename:20s} - {description:20s} (없음)")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  일부 파일이 없습니다.")
        return False
    
    # 파라미터 파일 로드 (ydata-synthetic 불필요)
    print("\n" + "=" * 70)
    print("모델 파라미터 정보")
    print("=" * 70)
    
    params_path = os.path.join(model_path, 'params.pkl')
    try:
        with open(params_path, 'rb') as f:
            params = pickle.load(f)
        
        print("\n하이퍼파라미터:")
        for key, value in params.items():
            print(f"  - {key}: {value}")
        
        print("\n✓ 모델 파일 확인 완료!")
        print("\n" + "=" * 70)
        print("다음 단계")
        print("=" * 70)
        print("이 모델을 사용하려면:")
        print("1. Colab에서 use_timegan_model.py를 실행하세요")
        print("2. 또는 Colab에서 다음 코드를 사용하세요:")
        print()
        print("   from google.colab import files")
        print("   files.upload()  # timegan_model.zip 업로드")
        print("   !unzip timegan_model.zip")
        print("   # 이후 모델 사용")
        print()
        print("3. 로컬에서 사용하려면 디스크 공간을 확보한 후")
        print("   pip install ydata-synthetic")
        print("   python use_timegan_model.py")
        
        return True
        
    except Exception as e:
        print(f"❌ 파라미터 파일 로드 실패: {e}")
        return False

def main():
    # 여러 위치 확인
    possible_paths = [
        'timegan_model',
        'models/timegan',
        os.path.join(os.path.dirname(__file__), 'timegan_model'),
        os.path.join(os.path.dirname(__file__), 'models', 'timegan'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.exists(os.path.join(path, 'timegan_model.pkl')):
            check_model_files(path)
            return
    
    print("❌ 모델 파일을 찾을 수 없습니다.")
    print("\n다음 위치를 확인하세요:")
    for path in possible_paths:
        print(f"  - {path}")

if __name__ == "__main__":
    main()

