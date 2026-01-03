"""
TimeGAN 모델 학습 스크립트 (Google Colab용)

이 스크립트를 Google Colab에서 실행하여 TimeGAN 모델을 학습하고 저장합니다.
저장된 모델은 로컬 프로젝트에서 사용할 수 있습니다.

⚠️ 중요 사항:

1. Python 버전 요구사항:
   - ydata-synthetic 패키지는 Python 3.9-3.11만 지원합니다
   - Colab에서 Python 버전을 확인하고 필요시 변경하세요:
     런타임 > 런타임 유형 변경 > Python 버전: 3.11 (또는 3.10)

2. 패키지 설치:
   별도의 셀에서 다음 명령을 먼저 실행하세요:
     !pip install ydata-synthetic

사용 방법:
1. Colab에서 Python 버전을 3.11 (또는 3.10)로 설정
2. 이 파일을 Google Colab에 업로드
3. 별도 셀에서 '!pip install ydata-synthetic' 실행
4. 데이터 파일 준비 (다음 중 하나 선택):
   
   [방법 A] 직접 업로드:
   - Colab에서 파일 아이콘 클릭 > 파일 업로드
   - 'Dynamic_Shield_Data_v4.csv' 파일 업로드
   
   [방법 B] Google Drive 마운트:
   - 별도 셀에서 실행:
     from google.colab import drive
     drive.mount('/content/drive')
   - CSV 파일을 Google Drive에 업로드
   - 스크립트가 자동으로 '/content/drive/MyDrive/' 경로에서 찾음
   
   [방법 C] GitHub에서 다운로드:
   - 별도 셀에서 실행:
     !wget https://your-repo-url/Dynamic_Shield_Data_v4.csv
   
   [방법 D] 샘플 데이터 사용:
   - CSV 파일이 없으면 자동으로 샘플 데이터 생성
   
5. 스크립트 실행
6. 생성된 timegan_model.zip 파일을 다운로드
7. 로컬 프로젝트의 models/timegan/ 폴더에 압축 해제
"""

# ==========================================
# 0. 필수 패키지 설치 (Colab용)
# ==========================================
print("=" * 60)
print("0. 필수 패키지 확인 및 설치")
print("=" * 60)

import sys
import subprocess
import importlib
import pkg_resources

# Python 버전 확인
python_version = sys.version_info
print(f"Python 버전: {python_version.major}.{python_version.minor}.{python_version.micro}")

# ydata-synthetic는 Python 3.9-3.11만 지원
if python_version.major == 3 and python_version.minor >= 12:
    print("\n" + "=" * 60)
    print("⚠️  Python 버전 호환성 경고")
    print("=" * 60)
    print("현재 Python 버전이 3.12 이상입니다.")
    print("ydata-synthetic 패키지는 Python 3.9-3.11만 지원합니다.")
    print("\n해결 방법:")
    print("1. Colab 런타임을 Python 3.11로 변경하세요:")
    print("   런타임 > 런타임 유형 변경 > Python 버전: 3.11")
    print("2. 또는 다음 명령으로 Python 3.11을 설치하고 사용하세요:")
    print("   !apt-get update && apt-get install -y python3.11 python3.11-venv")
    print("=" * 60)
    raise RuntimeError(
        "Python 버전이 호환되지 않습니다.\n"
        "Colab에서 Python 3.11로 런타임을 변경한 후 다시 시도하세요."
    )

def check_package_installed(package_name):
    """패키지가 설치되어 있는지 확인"""
    try:
        pkg_resources.get_distribution(package_name)
        return True
    except:
        return False

def install_package(package_name, version=None, force_reinstall=False):
    """패키지 설치 시도"""
    try:
        if version:
            package_spec = f"{package_name}=={version}"
        else:
            package_spec = package_name
        
        cmd = [sys.executable, '-m', 'pip', 'install', package_spec]
        if force_reinstall:
            cmd.append('--force-reinstall')
        else:
            cmd.append('--upgrade')
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"설치 오류: {e}")
        return False

def fix_numpy_compatibility():
    """numpy 버전 호환성 문제 해결"""
    print("🔧 numpy 버전 호환성 확인 중...")
    try:
        import numpy as np
        numpy_version = np.__version__
        print(f"  현재 numpy 버전: {numpy_version}")
        
        # numpy 재설치로 호환성 문제 해결
        print("  numpy 재설치 중 (호환성 문제 해결)...")
        if install_package('numpy', force_reinstall=True):
            # 모듈 리로드
            importlib.reload(np)
            print("  ✓ numpy 재설치 완료")
            return True
        else:
            print("  ⚠️  numpy 재설치 실패")
            return False
    except Exception as e:
        print(f"  ⚠️  numpy 호환성 확인 실패: {e}")
        # numpy가 없으면 설치
        print("  numpy 설치 중...")
        return install_package('numpy')

# 먼저 패키지 설치 여부 확인
package_installed = check_package_installed('ydata-synthetic')
print(f"패키지 설치 확인: {'✓ 설치됨' if package_installed else '✗ 미설치'}")

# ydata_synthetic 패키지 import 시도
TimeGAN = None
TIMEGAN_IMPORT_METHOD = None  # import 방법 저장

def try_import_timegan():
    """여러 방법으로 TimeGAN import 시도"""
    import_methods = [
        # 방법 1: timegan 모듈에서 직접 import
        lambda: __import__('ydata_synthetic.synthesizers.timeseries.timegan', fromlist=['TimeGAN']).TimeGAN,
        # 방법 2: 표준 경로
        lambda: __import__('ydata_synthetic.synthesizers.timeseries', fromlist=['TimeGAN']).TimeGAN,
        # 방법 3: 직접 import
        lambda: __import__('ydata_synthetic.synthesizers.timeseries').synthesizers.timeseries.TimeGAN,
        # 방법 4: 전체 모듈에서 찾기
        lambda: getattr(__import__('ydata_synthetic.synthesizers.timeseries'), 'TimeGAN'),
    ]
    
    for i, method in enumerate(import_methods, 1):
        try:
            timegan = method()
            return timegan, f"방법 {i}"
        except (ImportError, AttributeError) as e:
            continue
    
    # 패키지 구조 확인
    try:
        import ydata_synthetic
        import ydata_synthetic.synthesizers
        import ydata_synthetic.synthesizers.timeseries as ts_module
        
        # 모듈의 속성 확인
        print("  📋 패키지 구조 확인:")
        print(f"    - ydata_synthetic: {dir(ydata_synthetic)[:5]}...")
        print(f"    - synthesizers: {dir(ydata_synthetic.synthesizers)[:5]}...")
        print(f"    - timeseries 모듈 속성: {[x for x in dir(ts_module) if not x.startswith('_')][:10]}")
        
        # timegan 모듈 확인 (우선순위 1: 직접 클래스)
        if hasattr(ts_module, 'timegan'):
            timegan_module = getattr(ts_module, 'timegan')
            print(f"    - timegan 모듈 발견, 내부 속성 확인 중...")
            timegan_attrs = [x for x in dir(timegan_module) if not x.startswith('_')]
            print(f"    - timegan 모듈 속성: {timegan_attrs}")
            
            # timegan 모듈 안에서 클래스 찾기
            for attr in timegan_attrs:
                obj = getattr(timegan_module, attr)
                # 클래스인지 확인
                if isinstance(obj, type):
                    if 'TimeGAN' in attr or attr == 'TimeGAN' or attr.lower() == 'timegan':
                        print(f"    - 발견된 클래스: {attr}")
                        return obj, f"timegan 모듈의 {attr} 클래스"
        
        # TimeSeriesSynthesizer 확인 (우선순위 2: 래퍼 클래스)
        if hasattr(ts_module, 'TimeSeriesSynthesizer'):
            ts_synth = getattr(ts_module, 'TimeSeriesSynthesizer')
            if isinstance(ts_synth, type):
                print(f"    - TimeSeriesSynthesizer 클래스 발견 (modelname='timegan' 필요)")
                return ts_synth, "TimeSeriesSynthesizer"
        
        # TimeGAN이 다른 이름으로 있을 수 있음
        for attr in dir(ts_module):
            if not attr.startswith('_'):
                obj = getattr(ts_module, attr)
                # 클래스인지 확인
                if isinstance(obj, type) and ('TimeGAN' in attr or ('time' in attr.lower() and 'gan' in attr.lower())):
                    print(f"    - 발견된 클래스: {attr}")
                    return obj, f"대체 클래스: {attr}"
    except Exception as e:
        print(f"  ⚠️  구조 확인 실패: {e}")
    
    return None, None

# numpy 호환성 문제 해결 시도
numpy_fixed = False
try:
    TimeGAN, method = try_import_timegan()
    if TimeGAN:
        TIMEGAN_IMPORT_METHOD = method
        print(f"✓ ydata_synthetic import 성공 ({method})")
    else:
        raise ImportError("TimeGAN을 찾을 수 없습니다")
except (ImportError, ValueError) as e:
    # ValueError는 numpy 호환성 문제일 수 있음
    if isinstance(e, ValueError) and ('numpy' in str(e).lower() or 'dtype' in str(e).lower()):
        print(f"⚠️  numpy 버전 호환성 문제 감지: {e}")
        print("   numpy 재설치를 시도합니다...")
        if fix_numpy_compatibility():
            numpy_fixed = True
            # 재시도
            importlib.invalidate_caches()
            try:
                TimeGAN, method = try_import_timegan()
                if TimeGAN:
                    TIMEGAN_IMPORT_METHOD = method
                    print(f"✓ ydata_synthetic import 성공 ({method}) - numpy 재설치 후")
                else:
                    raise ImportError("TimeGAN을 찾을 수 없습니다")
            except Exception as e2:
                print(f"⚠️  재시도 실패: {e2}")
                # ImportError로 처리 계속
                e = ImportError(f"numpy 재설치 후에도 실패: {e2}")
    
    # ImportError 처리
    if isinstance(e, ImportError):
        print(f"⚠️  import 실패: {e}")
        
        # 패키지가 설치되어 있다고 나오지만 import가 안 되는 경우
        if package_installed:
            print("📦 패키지는 설치되어 있지만 import가 실패했습니다.")
            print("   패키지 구조를 확인하고 다른 방법을 시도합니다...")
            
            # 모듈 캐시 무효화
            importlib.invalidate_caches()
            
            # 패키지 구조 확인
            try:
                import ydata_synthetic.synthesizers.timeseries as ts
                print(f"  모듈 위치: {ts.__file__}")
                print(f"  사용 가능한 속성: {[x for x in dir(ts) if not x.startswith('_')]}")
            except Exception as e2:
                print(f"  구조 확인 실패: {e2}")
            
            # 다시 import 시도
            TimeGAN, method = try_import_timegan()
            if TimeGAN:
                TIMEGAN_IMPORT_METHOD = method
                print(f"✓ 재시도 후 import 성공 ({method})")
            else:
                print("❌ 재시도 실패. 패키지를 재설치합니다...")
                package_installed = False
        else:
            package_installed = False
        
        # 패키지가 설치되지 않은 경우 설치 시도
        if not package_installed:
        print("📦 ydata-synthetic 패키지 설치 중...")
        
        # Python 버전에 맞는 패키지 버전 선택
        # Python 3.9-3.11 지원
        if python_version.minor == 9:
            # Python 3.9: 최신 버전 시도
            package_versions = [None, '1.3.2', '1.2.0']
        elif python_version.minor == 10:
            # Python 3.10: 최신 버전 시도
            package_versions = [None, '1.4.0', '1.3.2']
        elif python_version.minor == 11:
            # Python 3.11: 최신 버전 시도
            package_versions = [None, '1.4.0', '1.3.2']
        else:
            package_versions = [None]
        
        installed_success = False
        for version in package_versions:
            if version:
                print(f"  버전 {version} 설치 시도 중...")
            else:
                print("  최신 버전 설치 시도 중...")
            
            if install_package('ydata-synthetic', version):
                # 설치 확인
                if check_package_installed('ydata-synthetic'):
                    # 모듈 캐시 무효화
                    importlib.invalidate_caches()
                    # import 시도 (여러 방법 시도)
                    timegan_class, method = try_import_timegan()
                    if timegan_class:
                        TimeGAN = timegan_class
                        TIMEGAN_IMPORT_METHOD = method
                        print(f"✓ 패키지 설치 및 import 성공{' (버전: ' + version + ', ' + method + ')' if version else ' (' + method + ')'}")
                        installed_success = True
                        break
                    else:
                        print(f"  ⚠️  설치되었지만 import 실패")
                        continue
                else:
                    print(f"  ⚠️  설치 확인 실패")
            else:
                print(f"  ⚠️  설치 실패")
        
        if not installed_success:
            print("\n" + "=" * 60)
            print("❌ 패키지 설치 실패")
            print("=" * 60)
            print("다음 방법을 시도하세요:")
            print("\n[방법 1] Python 버전 확인 및 변경:")
            print("  런타임 > 런타임 유형 변경 > Python 버전: 3.11 (또는 3.10)")
            print("\n[방법 2] 별도 셀에서 수동 설치:")
            print("  !pip install ydata-synthetic")
            print("\n[방법 3] 특정 버전 설치 시도:")
            print("  !pip install ydata-synthetic==1.4.0")
            print("=" * 60)
            raise ImportError(
                "ydata_synthetic 패키지 설치에 실패했습니다.\n"
                "Python 버전이 3.9-3.11인지 확인하고, "
                "Colab에서 별도 셀을 만들어 '!pip install ydata-synthetic'를 실행하세요."
            )
    else:
        # ValueError인데 numpy 문제가 아니면 다시 raise
        raise

# 최종 확인
if TimeGAN is None:
    raise ImportError("TimeGAN을 import할 수 없습니다.")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
import os
import zipfile
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 설정 (Config 파일에서 로드)
# ==========================================
# Colab 환경에서는 config 파일이 없을 수 있으므로 폴백 처리
try:
    # 프로젝트 구조에서 config 로더 찾기 시도
    import sys
    
    # Colab 환경에서 __file__이 없을 수 있으므로 현재 작업 디렉토리 사용
    try:
        current_file = __file__
    except NameError:
        # Colab 환경: 현재 작업 디렉토리 사용
        current_file = os.getcwd()
    
    # 여러 경로 시도
    possible_paths = [
        os.path.join(os.path.dirname(current_file), 'src', 'core'),
        os.path.join(os.path.dirname(current_file), '..', 'src', 'core'),
        'src/core',
        '/content/src/core',  # Colab 기본 경로
        os.path.join(os.getcwd(), 'src', 'core'),  # 현재 작업 디렉토리 기준
    ]
    
    config_loader = None
    for path in possible_paths:
        config_loader_path = os.path.join(path, 'config_loader.py')
        if os.path.exists(config_loader_path):
            sys.path.insert(0, path)
            try:
                from config_loader import ConfigLoader
                config_loader = ConfigLoader()
                print(f"[Config 로드] 경로에서 발견: {path}")
                break
            except ImportError as e:
                continue
    
    if config_loader:
        config = config_loader.load_base_config()
        timegan_config = config['timegan']
        SEQUENCE_LENGTH = timegan_config['training']['sequence_length']
        EPOCHS = timegan_config['training']['epochs']
        BATCH_SIZE = timegan_config['training']['batch_size']
        FEATURE_COLS = timegan_config['data']['feature_cols']
        paths = config_loader.get_paths()
        DEFAULT_SAVE_DIR = paths.get('timegan_model_dir', 'timegan_model')
        print(f"[Config 로드] 설정 파일에서 로드 완료")
        print(f"  - Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}, Sequence Length: {SEQUENCE_LENGTH}")
    else:
        raise ImportError("Config 로더를 찾을 수 없음")
        
except (ImportError, FileNotFoundError, KeyError, NameError) as e:
    # 폴백: 기본값 사용 (Colab 환경 등)
    print(f"[경고] Config 파일을 찾을 수 없어 기본값 사용: {e}")
    SEQUENCE_LENGTH = 24  # 시퀀스 길이 (일 단위)
    EPOCHS = 300  # 학습 에포크
    BATCH_SIZE = 128
    FEATURE_COLS = ['VIX', 'FX', 'Correlation']
    DEFAULT_SAVE_DIR = 'timegan_model'

# ==========================================
# 1. 데이터 준비
# ==========================================
print("=" * 60)
print("1. 데이터 준비")
print("=" * 60)

# Google Drive 마운트 시도 (Colab 환경에서만)
try:
    from google.colab import drive
    if not os.path.exists('/content/drive/MyDrive'):
        print("📁 Google Drive 마운트 시도 중...")
        print("   (인증이 필요할 수 있습니다)")
        drive.mount('/content/drive', force_remount=False)
        print("✓ Google Drive 마운트 완료")
    else:
        print("✓ Google Drive 이미 마운트됨")
except ImportError:
    # Colab이 아닌 환경
    pass
except Exception as e:
    print(f"⚠️  Google Drive 마운트 실패 (무시하고 계속): {e}")

# 데이터 파일 경로 설정 (여러 위치 시도)
possible_data_paths = [
    # 직접 업로드된 파일 (Colab)
    'Dynamic_Shield_Data_v4.csv',
    'data.csv',
    'training_data.csv',
    # Google Drive 마운트 경로
    '/content/drive/MyDrive/Dynamic_Shield_Data_v4.csv',
    '/content/drive/MyDrive/data/Dynamic_Shield_Data_v4.csv',
    '/content/drive/MyDrive/DATA/data/Dynamic_Shield_Data_v4.csv',
    # 로컬 프로젝트 경로 (Colab에 업로드한 경우)
    'DATA/data/Dynamic_Shield_Data_v4.csv',
    '../DATA/data/Dynamic_Shield_Data_v4.csv',
    # Config에서 경로 가져오기 (있는 경우)
]

training_data = None
data_source = None

# 옵션 A: CSV 파일에서 로드 시도
for data_path in possible_data_paths:
    if os.path.exists(data_path):
        try:
            print(f"📂 CSV 파일 발견: {data_path}")
            loaded_data = pd.read_csv(data_path)
            
            # 필수 컬럼 확인
            required_cols = set(FEATURE_COLS)
            available_cols = set(loaded_data.columns)
            
            if required_cols.issubset(available_cols):
                # 필요한 컬럼만 선택
                training_data = loaded_data[FEATURE_COLS].copy()
                data_source = data_path
                print(f"✓ CSV 파일 로드 성공: {len(training_data)}행")
                break
            else:
                missing_cols = required_cols - available_cols
                print(f"⚠️  필수 컬럼 누락: {missing_cols}")
                print(f"   사용 가능한 컬럼: {list(available_cols)}")
                
                # 누락된 컬럼을 생성하거나 대체 시도
                training_data = loaded_data.copy()
                data_fixed = False
                
                # Correlation 컬럼이 없으면 계산하거나 대체
                if 'Correlation' in missing_cols:
                    if 'VIX' in available_cols and 'FX' in available_cols:
                        # VIX와 FX의 정규화된 차이로 Correlation 근사
                        print("   → Correlation 컬럼을 VIX와 FX의 정규화된 관계로 계산합니다...")
                        vix_norm = (training_data['VIX'] - training_data['VIX'].mean()) / training_data['VIX'].std()
                        fx_norm = (training_data['FX'] - training_data['FX'].mean()) / training_data['FX'].std()
                        # 정규화된 값의 곱을 상관관계 근사로 사용
                        training_data['Correlation'] = (vix_norm * fx_norm).clip(-1, 1)
                        data_fixed = True
                    elif 'SPX' in available_cols and 'KOSPI' in available_cols:
                        # SPX와 KOSPI의 정규화된 관계 사용
                        print("   → Correlation 컬럼을 SPX와 KOSPI의 정규화된 관계로 계산합니다...")
                        spx_norm = (training_data['SPX'] - training_data['SPX'].mean()) / training_data['SPX'].std()
                        kospi_norm = (training_data['KOSPI'] - training_data['KOSPI'].mean()) / training_data['KOSPI'].std()
                        training_data['Correlation'] = (spx_norm * kospi_norm).clip(-1, 1)
                        data_fixed = True
                    else:
                        # 랜덤 값으로 생성
                        print("   → Correlation 컬럼을 랜덤 값으로 생성합니다...")
                        training_data['Correlation'] = np.random.uniform(-0.6, 0.8, len(training_data))
                        data_fixed = True
                
                # VIX나 FX가 없으면 대체 컬럼 찾기
                if 'VIX' in missing_cols:
                    if 'VIX_Change' in available_cols:
                        print("   → VIX 컬럼을 VIX_Change로 대체합니다...")
                        training_data['VIX'] = training_data['VIX_Change'].abs() * 20 + 20  # 스케일 조정
                        data_fixed = True
                
                if 'FX' in missing_cols:
                    if 'Swap_Point_Proxy' in available_cols:
                        print("   → FX 컬럼을 Swap_Point_Proxy로 대체합니다...")
                        training_data['FX'] = training_data['Swap_Point_Proxy'] * 10 + 1200  # 스케일 조정
                        data_fixed = True
                
                # 필요한 컬럼이 모두 있으면 사용
                if set(FEATURE_COLS).issubset(set(training_data.columns)):
                    training_data = training_data[FEATURE_COLS].copy()
                    data_source = f"{data_path} (컬럼 보정됨)"
                    print(f"✓ CSV 파일 로드 성공 (컬럼 보정): {len(training_data)}행")
                    break
                else:
                    print("   ⚠️  필수 컬럼을 생성/대체할 수 없어 다음 파일을 시도합니다...")
                    training_data = None
                    
        except Exception as e:
            print(f"⚠️  파일 로드 실패 ({data_path}): {e}")
            training_data = None
            continue

# 옵션 B: 샘플 데이터 생성 (CSV 파일이 없는 경우 또는 로드 실패)
if training_data is None:
    print("📊 CSV 파일을 찾을 수 없어 샘플 데이터를 생성합니다.")
    print("   (실제 데이터를 사용하려면 다음 중 하나를 수행하세요:)")
    print("   1. Colab에 CSV 파일 업로드")
    print("   2. Google Drive 마운트 후 파일 경로 설정")
    print("   3. GitHub에서 파일 다운로드")
    print()
    
    np.random.seed(42)
    n_days = 2000

    vix = np.random.uniform(10, 60, n_days)
    fx = 1200 + np.cumsum(np.random.normal(0, 5, n_days))
    correlation = np.random.uniform(-0.6, 0.8, n_days)

    training_data = pd.DataFrame({
        'VIX': vix,
        'FX': fx,
        'Correlation': correlation
    })
    data_source = "샘플 데이터 (생성)"

# 최종 확인: training_data가 None이면 오류
if training_data is None:
    raise ValueError("데이터를 로드할 수 없습니다. CSV 파일을 확인하거나 샘플 데이터 생성에 실패했습니다.")

# data_source가 정의되지 않았을 경우
if 'data_source' not in locals():
    data_source = "알 수 없음"

print(f"\n학습 데이터: {len(training_data)}일")
print(f"데이터 소스: {data_source}")
print(f"컬럼: {list(training_data.columns)}")
print(training_data.head())

# ==========================================
# 2. 데이터 전처리
# ==========================================
print("\n" + "=" * 60)
print("2. 데이터 전처리")
print("=" * 60)

data = training_data[FEATURE_COLS].copy()

# 정규화
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

print(f"정규화 완료. 형태: {data_scaled.shape}")

# ==========================================
# 3. 시퀀스 데이터 생성
# ==========================================
print("\n" + "=" * 60)
print("3. 시퀀스 데이터 생성")
print("=" * 60)

n_samples = len(data_scaled) - SEQUENCE_LENGTH + 1
sequences = []

for i in range(n_samples):
    seq = data_scaled[i:i+SEQUENCE_LENGTH]
    sequences.append(seq)

sequences = np.array(sequences)

print(f"시퀀스 수: {len(sequences)}")
print(f"시퀀스 형태: {sequences.shape}")

# ==========================================
# 4. TimeGAN 모델 학습
# ==========================================
print("\n" + "=" * 60)
print("4. TimeGAN 모델 학습")
print("=" * 60)
print(f"Epochs: {EPOCHS}, Batch Size: {BATCH_SIZE}")
print("학습 중... (시간이 다소 걸릴 수 있습니다)")

# TimeSeriesSynthesizer인 경우 ModelParameters와 TrainParameters 사용
if TIMEGAN_IMPORT_METHOD == "TimeSeriesSynthesizer":
    print("  (TimeSeriesSynthesizer 사용, ModelParameters/TrainParameters 설정)")
    from ydata_synthetic.synthesizers import ModelParameters, TrainParameters
    
    # 모델 파라미터 설정
    # number_sequences는 feature 수를 의미 (3개: VIX, FX, Correlation)
    model_params = ModelParameters(
        batch_size=BATCH_SIZE,
        lr=0.001,
        noise_dim=32,
        layers_dim=128,
        latent_dim=SEQUENCE_LENGTH,  # sequence_length와 동일
        gamma=1,
    )
    
    # 학습 파라미터 설정
    # number_sequences는 feature 수 (컬럼 수)
    train_params = TrainParameters(
        epochs=EPOCHS,
        sequence_length=SEQUENCE_LENGTH,
        number_sequences=len(FEATURE_COLS),  # feature 수: 3
    )
    
    # 모델 생성
    timegan_model = TimeGAN(
        modelname='timegan',
        model_parameters=model_params
    )
    
    # TimeSeriesSynthesizer는 원본 DataFrame을 기대함 (정규화된 데이터를 DataFrame으로 변환)
    data_for_fit = pd.DataFrame(data_scaled, columns=FEATURE_COLS)
    
    # fit 메서드에 train_arguments와 컬럼 정보 전달
    # num_cols: 숫자형 컬럼 리스트, cat_cols: 범주형 컬럼 리스트 (없으면 빈 리스트)
    timegan_model.fit(
        data_for_fit, 
        train_params,
        num_cols=FEATURE_COLS,  # 모든 컬럼이 숫자형
        cat_cols=[]  # 범주형 컬럼 없음
    )
else:
    timegan_model = TimeGAN(
        sequence_length=SEQUENCE_LENGTH,
        number_sequences=len(sequences),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )
    timegan_model.fit(sequences)

print("✓ TimeGAN 모델 학습 완료!")

# ==========================================
# 5. 모델 및 스케일러 저장
# ==========================================
print("\n" + "=" * 60)
print("5. 모델 저장")
print("=" * 60)

# 저장 디렉토리 설정 (Config에서 로드했으면 사용, 아니면 기본값)
save_dir = DEFAULT_SAVE_DIR if 'DEFAULT_SAVE_DIR' in globals() else 'timegan_model'
os.makedirs(save_dir, exist_ok=True)

# 모델 저장
model_path = os.path.join(save_dir, 'timegan_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(timegan_model, f)
print(f"✓ 모델 저장: {model_path}")

# 스케일러 저장
scaler_path = os.path.join(save_dir, 'scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ 스케일러 저장: {scaler_path}")

# 하이퍼파라미터 저장
params = {
    'sequence_length': SEQUENCE_LENGTH,
    'feature_cols': FEATURE_COLS,
    'epochs': EPOCHS,
    'batch_size': BATCH_SIZE
}
params_path = os.path.join(save_dir, 'params.pkl')
with open(params_path, 'wb') as f:
    pickle.dump(params, f)
print(f"✓ 하이퍼파라미터 저장: {params_path}")

# ==========================================
# 6. 테스트 생성
# ==========================================
print("\n" + "=" * 60)
print("6. 테스트 생성")
print("=" * 60)

n_test = 10
test_sequences = timegan_model.sample(n_test)

# sample()이 리스트를 반환할 수 있으므로 numpy 배열로 변환
if isinstance(test_sequences, list):
    test_sequences = np.array(test_sequences)
elif not isinstance(test_sequences, np.ndarray):
    test_sequences = np.array(test_sequences)

print(f"생성된 시퀀스 형태: {test_sequences.shape}")
print("✓ 테스트 생성 성공!")

# ==========================================
# 7. ZIP 파일 생성 (다운로드용)
# ==========================================
print("\n" + "=" * 60)
print("7. ZIP 파일 생성")
print("=" * 60)

zip_path = 'timegan_model.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(save_dir):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, os.path.dirname(save_dir))
            zipf.write(file_path, arcname)

print(f"✓ ZIP 파일 생성: {zip_path}")

# ==========================================
# 8. 완료 메시지
# ==========================================
print("\n" + "=" * 60)
print("완료!")
print("=" * 60)
print("\n다음 단계:")
print("1. timegan_model.zip 파일을 다운로드")
print("2. 로컬 프로젝트의 models/timegan/ 폴더에 압축 해제")
print("3. 로컬에서 hybrid_scenarios.py 실행")
print("\n파일 구조:")
print("  models/timegan/")
print("    ├── timegan_model.pkl")
print("    ├── scaler.pkl")
print("    └── params.pkl")

