"""
Phase 3.2: Hybrid Scenarios Construction (하이브리드 시나리오 구축)
===================================================================
제안서 3.2 데이터 증강: 하이브리드 시나리오 구축

[핵심 목적]
- TimeGAN 생성 데이터의 "Hallucination" 및 "Optimistic Bias" 방지
- 보수적 데이터셋 구축: Historical Stress (30%) + TimeGAN 생성 (70%)

[구현 기능]
1. TimeGAN 모델 학습
2. Historical Stress 데이터 로드 (2008, 2022 등)
3. 70:30 혼합 로직
4. t-SNE 시각화 (데이터 다양성 검증)
5. Discriminative Score 계산 (생성 품질 정량화)

핵심 철학: Capital Optimization, not Prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

# TimeGAN 관련 (선택적 - 없으면 폴백)
try:
    from ydata_synthetic.synthesizers.timeseries import TimeGAN
    TIMEGAN_AVAILABLE = True
except ImportError:
    TIMEGAN_AVAILABLE = False
    print("[경고] ydata-synthetic이 설치되지 않았습니다. TimeGAN 대신 기존 생성기를 사용합니다.")
    print("       설치: pip install ydata-synthetic")

# 머신러닝
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 기존 모듈
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from realistic_data import RealisticMarketGenerator
except ImportError:
    from .realistic_data import RealisticMarketGenerator


class HybridScenarioBuilder:
    """
    하이브리드 시나리오 구축기
    
    Historical Stress (30%) + TimeGAN 생성 (70%) 혼합
    """
    
    def __init__(self, historical_stress_dir=None, timegan_model_path=None):
        """
        Args:
            historical_stress_dir: Historical Stress CSV 파일들이 있는 디렉토리
            timegan_model_path: 저장된 TimeGAN 모델 경로 (선택적)
        """
        # Config 로더 초기화
        try:
            from config_loader import ConfigLoader
            self.config_loader = ConfigLoader()
            self.base_config = self.config_loader.load_base_config()
            self.scenarios_config = self.config_loader.load_scenarios()
            paths = self.config_loader.get_paths()
            
            # 경로 설정 (Config에서 로드)
            if historical_stress_dir is None:
                historical_stress_dir = paths.get('synthetic_stress_dir', 
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'DATA', 'synthetic_stress'))
                )
            
            if timegan_model_path is None:
                timegan_model_path = paths.get('timegan_model_dir', 
                    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models', 'timegan')
                )
            
            print(f"[Config 로드] 설정 파일에서 경로 로드 완료")
        except (ImportError, FileNotFoundError) as e:
            # 폴백: 기본 경로 사용
            print(f"[경고] Config 파일을 찾을 수 없어 기본 경로 사용: {e}")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            
            if historical_stress_dir is None:
                historical_stress_dir = os.path.join(project_root, 'DATA', 'synthetic_stress')
            
            if timegan_model_path is None:
                timegan_model_path = os.path.join(project_root, 'models', 'timegan')
            
            self.config_loader = None
            self.base_config = None
            self.scenarios_config = None
        
        self.historical_stress_dir = historical_stress_dir
        
        # TimeGAN 모델
        self.timegan_model = None
        self.timegan_trained = False
        self.timegan_scaler = None  # 역정규화용 스케일러
        self.timegan_params = None  # 하이퍼파라미터
        
        self.timegan_model_path = timegan_model_path
        
        # 저장된 모델이 있으면 자동 로드 시도
        if self.timegan_model_path and os.path.exists(self.timegan_model_path):
            try:
                self.load_timegan_model(self.timegan_model_path)
                print(f"[자동 로드] TimeGAN 모델 로드 완료: {self.timegan_model_path}")
            except Exception as e:
                print(f"[경고] 저장된 모델 자동 로드 실패: {e}")
        
        # 데이터 저장
        self.historical_data = None
        self.generated_data = None
        self.hybrid_data = None
        
        # 검증 결과
        self.tsne_result = None
        self.discriminative_score = None
        
    def load_historical_stress(self):
        """
        Historical Stress 데이터 로드
        
        로드 대상:
        - 2008 금융위기 (Scenario_B_Correlation_Breakdown.csv 등)
        - 2020 COVID-19 (Scenario_COVID19.csv)
        - 기타 스트레스 시나리오
        """
        print("=" * 60)
        print("Historical Stress 데이터 로드")
        print("=" * 60)
        
        if not os.path.exists(self.historical_stress_dir):
            print(f"[경고] 디렉토리 없음: {self.historical_stress_dir}")
            return None
        
        # CSV 파일 목록
        csv_files = [f for f in os.listdir(self.historical_stress_dir) if f.endswith('.csv')]
        
        if len(csv_files) == 0:
            print(f"[경고] CSV 파일 없음: {self.historical_stress_dir}")
            return None
        
        print(f"\n[발견된 시나리오 파일] {len(csv_files)}개")
        all_data = []
        
        for csv_file in csv_files:
            file_path = os.path.join(self.historical_stress_dir, csv_file)
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                
                # 필수 컬럼 확인 및 생성
                required_cols = ['VIX', 'FX', 'Correlation']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    print(f"  [{csv_file}] 필수 컬럼 누락: {missing_cols}, 추정 생성 중...")
                    
                    # VIX가 없으면 추정
                    if 'VIX' not in df.columns:
                        # KOSPI 변동성 기반 추정
                        if 'KOSPI' in df.columns:
                            kospi_returns = df['KOSPI'].pct_change().fillna(0)
                            vix_estimate = 15 + (kospi_returns.rolling(20).std() * 100).fillna(0) * 50
                            df['VIX'] = np.clip(vix_estimate, 10, 80)
                        else:
                            df['VIX'] = 20  # 기본값
                    
                    # Correlation이 없으면 추정
                    if 'Correlation' not in df.columns:
                        # VIX 기반 추정
                        vix = df['VIX'].values
                        corr = np.where(vix >= 30, np.random.uniform(0.5, 0.8, len(vix)),
                                np.where(vix >= 20, np.random.uniform(-0.2, 0.5, len(vix)),
                                         np.random.uniform(-0.6, -0.2, len(vix))))
                        df['Correlation'] = corr
                    
                    # FX가 없으면 추정
                    if 'FX' not in df.columns:
                        df['FX'] = 1200  # 기본값
                
                # 시나리오 이름 추가
                scenario_name = csv_file.replace('.csv', '').replace('Scenario_', '')
                df['Scenario'] = scenario_name
                
                all_data.append(df[required_cols + ['Scenario']])
                print(f"  ✓ {csv_file}: {len(df)}일 로드 완료")
                
            except Exception as e:
                print(f"  ✗ {csv_file} 로드 실패: {e}")
        
        if len(all_data) == 0:
            print("[경고] 로드된 데이터 없음")
            return None
        
        # 통합
        self.historical_data = pd.concat(all_data, ignore_index=True)
        print(f"\n[통합 완료] 총 {len(self.historical_data)}일의 Historical Stress 데이터")
        print(f"  시나리오: {self.historical_data['Scenario'].unique().tolist()}")
        
        return self.historical_data
    
    def load_timegan_model(self, model_dir=None):
        """
        저장된 TimeGAN 모델 로드 (Colab에서 학습한 모델)
        
        Args:
            model_dir: 모델이 저장된 디렉토리 경로
                      (기본값: models/timegan/)
        """
        import pickle
        
        if model_dir is None:
            model_dir = self.timegan_model_path
        
        if model_dir is None or not os.path.exists(model_dir):
            print(f"[경고] 모델 디렉토리 없음: {model_dir}")
            return False
        
        try:
            # 모델 파일 경로
            model_path = os.path.join(model_dir, 'timegan_model.pkl')
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            params_path = os.path.join(model_dir, 'params.pkl')
            
            # 모델 로드
            if not os.path.exists(model_path):
                print(f"[오류] 모델 파일 없음: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                self.timegan_model = pickle.load(f)
            
            # 스케일러 로드
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.timegan_scaler = pickle.load(f)
            else:
                print("[경고] 스케일러 파일 없음. 역정규화 시 대략적 변환 사용")
            
            # 하이퍼파라미터 로드
            if os.path.exists(params_path):
                with open(params_path, 'rb') as f:
                    self.timegan_params = pickle.load(f)
            
            self.timegan_trained = True
            
            print(f"[완료] TimeGAN 모델 로드 성공: {model_dir}")
            if self.timegan_params:
                print(f"  Sequence Length: {self.timegan_params.get('sequence_length', 'N/A')}")
                print(f"  Epochs: {self.timegan_params.get('epochs', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"[오류] 모델 로드 실패: {e}")
            return False
    
    def train_timegan(self, training_data, epochs=None, batch_size=None, sequence_length=None):
        """
        TimeGAN 모델 학습
        
        Args:
            training_data: 학습용 데이터 (DataFrame with VIX, FX, Correlation)
            epochs: 학습 에포크 수 (None이면 Config에서 로드)
            batch_size: 배치 크기 (None이면 Config에서 로드)
            sequence_length: 시퀀스 길이 (None이면 Config에서 로드)
        """
        print("=" * 60)
        print("TimeGAN 모델 학습")
        print("=" * 60)
        
        if not TIMEGAN_AVAILABLE:
            print("[경고] TimeGAN 사용 불가. 기존 생성기로 대체합니다.")
            return None
        
        # Config에서 기본값 로드 (인자로 전달된 값이 우선)
        if self.base_config:
            timegan_config = self.base_config.get('timegan', {})
            training_config = timegan_config.get('training', {})
            data_config = timegan_config.get('data', {})
            
            epochs = epochs or training_config.get('epochs', 300)
            batch_size = batch_size or training_config.get('batch_size', 128)
            sequence_length = sequence_length or training_config.get('sequence_length', 24)
            feature_cols = data_config.get('feature_cols', ['VIX', 'FX', 'Correlation'])
        else:
            # 폴백: 기본값 사용
            epochs = epochs or 300
            batch_size = batch_size or 128
            sequence_length = sequence_length or 24
            feature_cols = ['VIX', 'FX', 'Correlation']
        
        # 데이터 전처리
        data = training_data[feature_cols].copy()
        
        # 정규화
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # 시퀀스 데이터 생성
        n_samples = len(data_scaled) - sequence_length + 1
        sequences = []
        
        for i in range(n_samples):
            seq = data_scaled[i:i+sequence_length]
            sequences.append(seq)
        
        sequences = np.array(sequences)
        print(f"\n[데이터 준비] 시퀀스 수: {len(sequences)}, 형태: {sequences.shape}")
        
        # TimeGAN 모델 생성 및 학습
        print(f"\n[학습 시작] Epochs: {epochs}, Batch Size: {batch_size}")
        print("  (시간이 다소 걸릴 수 있습니다...)")
        
        try:
            self.timegan_model = TimeGAN(
                sequence_length=sequence_length,
                number_sequences=len(sequences),
                epochs=epochs,
                batch_size=batch_size
            )
            
            # 학습
            self.timegan_model.fit(sequences)
            self.timegan_trained = True
            
            # 스케일러 저장
            self.timegan_scaler = scaler
            self.timegan_params = {
                'sequence_length': sequence_length,
                'feature_cols': feature_cols,
                'epochs': epochs,
                'batch_size': batch_size
            }
            
            print("[완료] TimeGAN 모델 학습 완료")
            
            # 모델 저장
            if self.timegan_model_path:
                import pickle
                os.makedirs(self.timegan_model_path, exist_ok=True)
                
                # 모델 저장
                model_path = os.path.join(self.timegan_model_path, 'timegan_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(self.timegan_model, f)
                
                # 스케일러 저장
                scaler_path = os.path.join(self.timegan_model_path, 'scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.timegan_scaler, f)
                
                # 하이퍼파라미터 저장
                params_path = os.path.join(self.timegan_model_path, 'params.pkl')
                with open(params_path, 'wb') as f:
                    pickle.dump(self.timegan_params, f)
                
                print(f"[저장] 모델 저장 완료: {self.timegan_model_path}")
            
            return self.timegan_model
            
        except Exception as e:
            print(f"[오류] TimeGAN 학습 실패: {e}")
            print("       기존 생성기로 대체합니다.")
            return None
    
    def generate_timegan_data(self, n_samples=None, sequence_length=None):
        """
        TimeGAN으로 데이터 생성
        
        Args:
            n_samples: 생성할 시퀀스 수 (None이면 Config에서 로드)
            sequence_length: 시퀀스 길이 (None이면 Config에서 로드)
        """
        print("=" * 60)
        print("TimeGAN 데이터 생성")
        print("=" * 60)
        
        if not self.timegan_trained or self.timegan_model is None:
            print("[경고] TimeGAN 모델이 학습되지 않았습니다.")
            print("       기존 생성기(RealisticMarketGenerator)로 대체합니다.")
            return self._generate_fallback_data(n_samples or 1000)
        
        try:
            # Config에서 기본값 로드
            if self.scenarios_config:
                timegan_config = self.scenarios_config.get('timegan_generated', {})
                n_samples = n_samples or timegan_config.get('n_samples', 1000)
                sequence_length = sequence_length or timegan_config.get('sequence_length', 24)
            elif self.timegan_params:
                sequence_length = sequence_length or self.timegan_params.get('sequence_length', 24)
                n_samples = n_samples or 1000
            else:
                # 폴백
                sequence_length = sequence_length or 24
                n_samples = n_samples or 1000
            
            # 생성
            print(f"\n[생성 중] {n_samples}개 시퀀스 생성...")
            generated_sequences = self.timegan_model.sample(n_samples)
            
            # 시퀀스를 평탄화하여 일일 데이터로 변환
            # 각 시퀀스의 마지막 시점만 사용 (또는 전체를 펼침)
            # 여기서는 각 시퀀스의 마지막 시점만 사용
            flat_data = generated_sequences[:, -1, :]  # 마지막 시점만
            
            # 역정규화 (저장된 스케일러 사용)
            if self.timegan_scaler is not None:
                # 스케일러로 역정규화
                data_denormalized = self.timegan_scaler.inverse_transform(flat_data)
                df = pd.DataFrame(data_denormalized, columns=['VIX', 'FX', 'Correlation'])
            else:
                # 스케일러가 없으면 대략적 변환
                df = pd.DataFrame(flat_data, columns=['VIX', 'FX', 'Correlation'])
                df['VIX'] = df['VIX'] * 20 + 20  # 대략 0~60 범위
                df['FX'] = df['FX'] * 100 + 1200  # 대략 1100~1300 범위
                df['Correlation'] = np.clip(df['Correlation'] * 0.5, -0.8, 0.9)
            
            # 값 범위 제한
            df['VIX'] = np.clip(df['VIX'], 10, 80)
            df['FX'] = np.clip(df['FX'], 1000, 1500)
            df['Correlation'] = np.clip(df['Correlation'], -0.8, 0.9)
            
            # 시나리오 라벨
            df['Scenario'] = 'TimeGAN_Generated'
            
            # 샘플링 (너무 많으면 일부만 사용)
            if len(df) > n_samples:
                df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
            
            self.generated_data = df
            print(f"[완료] {len(df)}일의 TimeGAN 생성 데이터")
            
            return df
            
        except Exception as e:
            print(f"[오류] TimeGAN 생성 실패: {e}")
            print("       기존 생성기로 대체합니다.")
            return self._generate_fallback_data(n_samples)
    
    def _generate_fallback_data(self, n_samples=1000):
        """TimeGAN 대체용: 기존 생성기 사용"""
        print("[대체] RealisticMarketGenerator 사용")
        
        generator = RealisticMarketGenerator(seed=42)
        df = generator.generate(n_samples, scenario='mixed')
        
        # 필요한 컬럼만 선택
        df_result = pd.DataFrame({
            'VIX': df['VIX'],
            'FX': df['FX'],
            'Correlation': df['Correlation'],
            'Scenario': 'Fallback_Generated'
        })
        
        self.generated_data = df_result
        return df_result
    
    def build_hybrid_dataset(self, generated_ratio=0.7, historical_ratio=0.3):
        """
        하이브리드 데이터셋 구축 (70% 생성 + 30% Historical)
        
        Args:
            generated_ratio: 생성 데이터 비율 (기본 0.7)
            historical_ratio: Historical 데이터 비율 (기본 0.3)
        """
        print("=" * 60)
        print("하이브리드 데이터셋 구축")
        print("=" * 60)
        
        # 데이터 확인
        if self.generated_data is None:
            print("[경고] 생성 데이터 없음. 먼저 generate_timegan_data()를 호출하세요.")
            return None
        
        if self.historical_data is None:
            print("[경고] Historical 데이터 없음. 먼저 load_historical_stress()를 호출하세요.")
            return None
        
        # 비율 정규화
        total_ratio = generated_ratio + historical_ratio
        generated_ratio = generated_ratio / total_ratio
        historical_ratio = historical_ratio / total_ratio
        
        # 샘플링
        n_generated = int(len(self.generated_data) * generated_ratio / (1 - generated_ratio))
        n_historical = len(self.historical_data)
        
        # Historical 데이터 샘플링 (비율 맞추기)
        if n_historical > n_generated * historical_ratio / generated_ratio:
            n_historical_sample = int(n_generated * historical_ratio / generated_ratio)
            historical_sample = self.historical_data.sample(n=min(n_historical_sample, n_historical), 
                                                           random_state=42)
        else:
            historical_sample = self.historical_data
        
        # Generated 데이터 샘플링
        n_generated_sample = int(len(historical_sample) * generated_ratio / historical_ratio)
        generated_sample = self.generated_data.sample(n=min(n_generated_sample, len(self.generated_data)), 
                                                      random_state=42)
        
        # 혼합
        hybrid_data = pd.concat([generated_sample, historical_sample], ignore_index=True)
        hybrid_data = hybrid_data.sample(frac=1, random_state=42).reset_index(drop=True)  # 셔플
        
        self.hybrid_data = hybrid_data
        
        print(f"\n[혼합 완료]")
        print(f"  생성 데이터: {len(generated_sample)}일 ({len(generated_sample)/len(hybrid_data)*100:.1f}%)")
        print(f"  Historical: {len(historical_sample)}일 ({len(historical_sample)/len(hybrid_data)*100:.1f}%)")
        print(f"  총 데이터: {len(hybrid_data)}일")
        
        return hybrid_data
    
    def visualize_tsne(self, n_samples=1000, perplexity=30, random_state=42, save_path=None):
        """
        t-SNE 시각화 (데이터 다양성 검증)
        
        Args:
            n_samples: 시각화할 샘플 수
            perplexity: t-SNE perplexity 파라미터
            random_state: 랜덤 시드
            save_path: 저장 경로
        """
        print("=" * 60)
        print("t-SNE 시각화 (데이터 다양성 검증)")
        print("=" * 60)
        
        if self.hybrid_data is None:
            print("[경고] 하이브리드 데이터 없음. 먼저 build_hybrid_dataset()를 호출하세요.")
            return None
        
        # 데이터 준비 (시나리오 정보 포함)
        if 'Scenario' in self.hybrid_data.columns:
            data = self.hybrid_data[['VIX', 'FX', 'Correlation', 'Scenario']].copy()
        else:
            data = self.hybrid_data[['VIX', 'FX', 'Correlation']].copy()
            data['Scenario'] = 'Unknown'
        
        # 샘플링 (너무 많으면 일부만)
        if len(data) > n_samples:
            data = data.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
        
        # 정규화 (시나리오 제외)
        feature_data = data[['VIX', 'FX', 'Correlation']].values
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(feature_data)
        
        # t-SNE
        print(f"\n[t-SNE 계산 중] 샘플 수: {len(data_scaled)}, Perplexity: {perplexity}...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                   n_iter=1000, verbose=0)
        tsne_result = tsne.fit_transform(data_scaled)
        
        self.tsne_result = tsne_result
        
        # 시각화
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 시나리오별 색상
        scenario_colors = {
            'TimeGAN_Generated': 'blue',
            'Fallback_Generated': 'lightblue',
            'COVID19': 'red',
            'Correlation_Breakdown': 'orange',
            'Interest_Rate_Shock': 'purple',
            'Stagflation': 'brown',
            'Swap_Point_Extreme': 'pink',
            'Regime_Transition': 'green',
            'Tail_Risk': 'gray',
            'Unknown': 'black'
        }
        
        # 시나리오별 플롯
        scenarios = data['Scenario'].unique()
        for scenario in scenarios:
            mask = data['Scenario'] == scenario
            color = scenario_colors.get(scenario, 'black')
            label = scenario.replace('_', ' ')
            ax.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                      c=color, label=label, alpha=0.6, s=20)
        
        ax.set_xlabel('t-SNE Component 1', fontsize=12)
        ax.set_ylabel('t-SNE Component 2', fontsize=12)
        ax.set_title('t-SNE Visualization: Hybrid Dataset Diversity', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n[저장] t-SNE 시각화: {save_path}")
        else:
            plt.savefig('hybrid_scenarios_tsne.png', dpi=150, bbox_inches='tight')
            print(f"\n[저장] t-SNE 시각화: hybrid_scenarios_tsne.png")
        
        plt.show()
        
        return tsne_result
    
    def calculate_discriminative_score(self, test_size=0.3, random_state=42):
        """
        Discriminative Score 계산 (생성 품질 정량화)
        
        실제 데이터와 생성 데이터를 구분할 수 있는지 측정
        - 낮을수록 좋음 (구분하기 어려움 = 생성 품질 좋음)
        - 높을수록 나쁨 (쉽게 구분됨 = 생성 품질 나쁨)
        
        Args:
            test_size: 테스트 데이터 비율
            random_state: 랜덤 시드
        """
        print("=" * 60)
        print("Discriminative Score 계산")
        print("=" * 60)
        
        if self.hybrid_data is None:
            print("[경고] 하이브리드 데이터 없음.")
            return None
        
        if self.historical_data is None or self.generated_data is None:
            print("[경고] Historical 또는 Generated 데이터 없음.")
            return None
        
        # 데이터 준비
        historical_features = self.historical_data[['VIX', 'FX', 'Correlation']].values
        generated_features = self.generated_data[['VIX', 'FX', 'Correlation']].values
        
        # 라벨 생성
        historical_labels = np.zeros(len(historical_features))  # 0 = 실제
        generated_labels = np.ones(len(generated_features))     # 1 = 생성
        
        # 통합
        X = np.vstack([historical_features, generated_features])
        y = np.hstack([historical_labels, generated_labels])
        
        # 정규화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train/Test 분리
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # 분류기 학습 (Random Forest)
        print(f"\n[분류기 학습] Train: {len(X_train)}, Test: {len(X_test)}")
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        # 예측
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Discriminative Score = Accuracy
        # 0.5에 가까울수록 좋음 (랜덤 추측 수준 = 구분 불가)
        # 1.0에 가까울수록 나쁨 (완벽 구분 = 생성 품질 나쁨)
        discriminative_score = accuracy
        
        self.discriminative_score = discriminative_score
        
        print(f"\n[결과]")
        print(f"  Discriminative Score: {discriminative_score:.4f}")
        print(f"  해석:")
        print(f"    - 0.5에 가까울수록 좋음 (구분하기 어려움 = 생성 품질 좋음)")
        print(f"    - 1.0에 가까울수록 나쁨 (쉽게 구분됨 = 생성 품질 나쁨)")
        
        if discriminative_score < 0.6:
            print(f"  ✓ 생성 품질 우수 (구분 어려움)")
        elif discriminative_score < 0.75:
            print(f"  ⚠ 생성 품질 보통 (일부 구분 가능)")
        else:
            print(f"  ✗ 생성 품질 낮음 (쉽게 구분됨)")
        
        return discriminative_score
    
    def run_full_pipeline(self, n_generated=2000, epochs=300, save_dir=None):
        """
        전체 파이프라인 실행
        
        Args:
            n_generated: 생성할 데이터 수
            epochs: TimeGAN 학습 에포크
            save_dir: 결과 저장 디렉토리
        """
        print("=" * 70)
        print("하이브리드 시나리오 구축 전체 파이프라인")
        print("=" * 70)
        
        # 1. Historical Stress 로드
        self.load_historical_stress()
        
        # 2. TimeGAN 모델 로드 또는 학습
        # 먼저 저장된 모델이 있는지 확인
        if self.timegan_model_path and os.path.exists(self.timegan_model_path):
            print("\n[저장된 모델 발견] 로드 시도...")
            if self.load_timegan_model(self.timegan_model_path):
                print("✓ 저장된 TimeGAN 모델 로드 완료")
            else:
                print("[경고] 모델 로드 실패. 새로 학습합니다.")
                self._train_timegan_if_needed(epochs)
        elif TIMEGAN_AVAILABLE and self.historical_data is not None:
            # 저장된 모델이 없으면 학습
            self._train_timegan_if_needed(epochs)
        else:
            print("[경고] TimeGAN 사용 불가. 폴백 생성기 사용")
    
    def _train_timegan_if_needed(self, epochs=300):
        """TimeGAN 학습 (내부 헬퍼 메서드)"""
        if not TIMEGAN_AVAILABLE:
            return
        
        if self.historical_data is not None:
            # 학습용 데이터 준비 (Historical 일부 사용)
            training_data = self.historical_data[['VIX', 'FX', 'Correlation']].copy()
            if len(training_data) < 100:
                # Historical이 부족하면 기존 생성기 데이터 추가
                generator = RealisticMarketGenerator(seed=42)
                fallback_data = generator.generate(500, scenario='mixed')
                training_data = pd.concat([
                    training_data,
                    fallback_data[['VIX', 'FX', 'Correlation']]
                ], ignore_index=True)
            
            self.train_timegan(training_data, epochs=epochs)
        
        # 3. 데이터 생성
        self.generate_timegan_data(n_samples=n_generated)
        
        # 4. 하이브리드 구축
        self.build_hybrid_dataset(generated_ratio=0.7, historical_ratio=0.3)
        
        # 5. t-SNE 시각화
        save_path = os.path.join(save_dir, 'hybrid_tsne.png') if save_dir else None
        self.visualize_tsne(save_path=save_path)
        
        # 6. Discriminative Score
        self.calculate_discriminative_score()
        
        # 7. 결과 저장
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, 'hybrid_dataset.csv')
            self.hybrid_data.to_csv(output_path, index=False)
            print(f"\n[저장] 하이브리드 데이터셋: {output_path}")
        
        print("\n" + "=" * 70)
        print("파이프라인 완료!")
        print("=" * 70)
        
        return self.hybrid_data


if __name__ == "__main__":
    # 사용 예시
    builder = HybridScenarioBuilder()
    
    # 전체 파이프라인 실행
    hybrid_data = builder.run_full_pipeline(
        n_generated=2000,
        epochs=100,  # 빠른 테스트용 (실제로는 300+ 권장)
        save_dir='hybrid_output'
    )
    
    print(f"\n[최종 결과]")
    print(f"  하이브리드 데이터: {len(hybrid_data)}일")
    print(f"  Discriminative Score: {builder.discriminative_score:.4f}")

