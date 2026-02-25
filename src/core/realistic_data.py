"""
Realistic Market Data Generator
================================
실제 역사적 데이터 기반 + 합성 데이터 생성기

[v4.0 개선 사항 - Anti-Overfitting]
1. 실제 데이터(Dynamic_Shield_Data_v4.csv) 사용
2. 명확한 Train/Test 분리 (시간 기반 분할)
3. 합성 데이터는 보조용으로만 사용

핵심 철학: Capital Optimization, not Prediction
(환율 예측이 아닌 자본 최적화)
"""

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import os

# ==============================================
# 실제 데이터 설정 (Anti-Overfitting)
# ==============================================
# 데이터 경로 (현재 파일 기준 상대 경로)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_SCRIPT_DIR, '..', '..', 'DATA', 'data')
REAL_DATA_PATH = os.path.join(_DATA_DIR, 'Dynamic_Shield_Data_v4.csv')

# Train/Test 분리 비율 (시간 기준) — 논문 5,292일 기준 70% 학습 / 30% 테스트
TRAIN_RATIO = 0.70  # 70% Train, 30% Test (학습/평가 데이터 누수 방지)
TRAIN_SEED_RANGE = (1, 1000)      # 학습용 seed 범위
TEST_SEED_RANGE = (10001, 20000)  # 테스트용 seed 범위 (명확히 분리)

# 실제 데이터 캐시 (메모리 효율)
_REAL_DATA_CACHE = None
_TRAIN_DATA_CACHE = None
_TEST_DATA_CACHE = None


class RealisticMarketGenerator:
    """
    현실적인 시장 데이터 생성기
    
    Features:
    - GARCH(1,1) 변동성 클러스터링
    - Markov Chain 기반 점진적 Regime 전환
    - 금리차 기반 스왑 포인트
    - Student-t 분포 Fat Tail
    """
    
    def __init__(self, seed=42):
        np.random.seed(seed)
        
        # GARCH(1,1) 파라미터
        self.garch_omega = 0.00001   # 기저 변동성
        self.garch_alpha = 0.08      # ARCH 계수
        self.garch_beta = 0.90       # GARCH 계수
        
        # Student-t 자유도 (낮을수록 Fat Tail)
        self.df_fx = 5               # 환율: Fat Tail
        self.df_stock = 4            # 주식: 더 Fat Tail
        
        # Regime 전환 확률 (점진적)
        self.trans_matrix = np.array([
            [0.98, 0.015, 0.005],  # Normal -> Normal/Transition/Panic
            [0.10, 0.85, 0.05],    # Transition -> ...
            [0.05, 0.20, 0.75]     # Panic -> ...
        ])
        
        # 금리 설정 (스왑 포인트 계산용)
        self.us_rate_base = 0.05     # 미국 금리 5%
        self.kr_rate_base = 0.035    # 한국 금리 3.5%
        
    def _generate_garch_volatility(self, n_days, base_vol=0.01):
        """GARCH(1,1) 변동성 시계열 생성"""
        variance = np.zeros(n_days)
        variance[0] = base_vol ** 2
        
        returns = np.zeros(n_days)
        returns[0] = np.random.normal(0, base_vol)
        
        for t in range(1, n_days):
            # GARCH(1,1): σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}
            variance[t] = (self.garch_omega + 
                          self.garch_alpha * returns[t-1]**2 + 
                          self.garch_beta * variance[t-1])
            
            vol_t = np.sqrt(variance[t])
            returns[t] = np.random.normal(0, vol_t)
        
        return np.sqrt(variance), returns
    
    def _generate_fat_tail_returns(self, n_days, scale=1.0, df=5):
        """Student-t 분포로 Fat Tail 수익률 생성"""
        return student_t.rvs(df, loc=0, scale=scale, size=n_days)
    
    def _generate_regime_sequence(self, n_days):
        """Markov Chain 기반 점진적 Regime 전환"""
        states = np.zeros(n_days, dtype=int)
        states[0] = 0  # Normal에서 시작
        
        for t in range(1, n_days):
            current = states[t-1]
            states[t] = np.random.choice([0, 1, 2], p=self.trans_matrix[current])
        
        return states
    
    def _calculate_swap_points(self, fx_rate, regime, n_days):
        """
        금리차 기반 스왑 포인트 계산
        Swap Point = FX * (US Rate - KR Rate) * (Days/360)
        
        위기 시: 달러 수요 급증 -> 스왑 포인트 마이너스 심화
        """
        swap_points = np.zeros(n_days)
        
        for t in range(n_days):
            # 기본 금리차
            rate_diff = self.us_rate_base - self.kr_rate_base
            
            # Regime에 따른 조정
            if regime[t] == 2:  # Panic
                rate_diff -= 0.02  # 달러 프리미엄 (스왑 마이너스 심화)
            elif regime[t] == 1:  # Transition
                rate_diff -= 0.01
            
            # 1개월 스왑 포인트 (원 단위)
            swap_points[t] = fx_rate[t] * rate_diff * (30/360)
        
        return swap_points
    
    def _generate_vix(self, regime, n_days):
        """Regime 기반 VIX 생성 (점진적 변화)"""
        vix = np.zeros(n_days)
        
        regime_vix = {0: 15, 1: 28, 2: 50}  # 목표 VIX
        regime_std = {0: 2, 1: 5, 2: 10}    # 변동성
        
        vix[0] = regime_vix[regime[0]]
        
        for t in range(1, n_days):
            target = regime_vix[regime[t]]
            std = regime_std[regime[t]]
            
            # 점진적 이동 (0.3 계수로 부드럽게)
            vix[t] = vix[t-1] * 0.7 + target * 0.3 + np.random.normal(0, std * 0.5)
        
        vix = np.clip(vix, 10, 80)
        return vix
    
    def _generate_correlation(self, regime, n_days):
        """Regime 기반 동적 상관계수"""
        corr = np.zeros(n_days)
        
        regime_corr = {0: -0.4, 1: 0.1, 2: 0.7}  # 목표 상관계수
        regime_std = {0: 0.1, 1: 0.15, 2: 0.1}
        
        corr[0] = regime_corr[regime[0]]
        
        for t in range(1, n_days):
            target = regime_corr[regime[t]]
            std = regime_std[regime[t]]
            
            # 점진적 이동
            corr[t] = corr[t-1] * 0.8 + target * 0.2 + np.random.normal(0, std * 0.3)
        
        corr = np.clip(corr, -0.8, 0.95)
        return corr
    
    def generate(self, n_days=500, scenario='mixed'):
        """
        현실적인 시장 데이터 생성
        
        Args:
            n_days: 생성할 일수
            scenario: 'normal', 'crisis', 'mixed'
        
        Returns:
            DataFrame with VIX, FX, Correlation, SwapPoints, Regime
        """
        # 1. Regime 시퀀스 생성
        if scenario == 'normal':
            regime = np.zeros(n_days, dtype=int)
        elif scenario == 'crisis':
            regime = np.concatenate([
                np.zeros(n_days // 4, dtype=int),
                np.ones(n_days // 4, dtype=int),
                np.full(n_days // 4, 2, dtype=int),
                np.ones(n_days - 3 * (n_days // 4), dtype=int)
            ])
        else:  # mixed
            regime = self._generate_regime_sequence(n_days)
        
        # 2. GARCH 변동성 기반 환율 생성
        fx_vol, fx_returns = self._generate_garch_volatility(n_days, base_vol=0.005)
        
        # Fat Tail 적용
        fat_tail_shock = self._generate_fat_tail_returns(n_days, scale=0.5, df=self.df_fx)
        
        # 환율 생성 (기준 1200원)
        fx = np.zeros(n_days)
        fx[0] = 1200
        
        for t in range(1, n_days):
            # Regime에 따른 drift
            if regime[t] == 2:
                drift = 0.003  # 위기 시 원화 약세
            elif regime[t] == 1:
                drift = 0.001
            else:
                drift = 0.0
            
            # GARCH 변동성 + Fat Tail shock
            shock = fx_returns[t] + fat_tail_shock[t] * 0.002
            fx[t] = fx[t-1] * (1 + drift + shock)
        
        fx = np.clip(fx, 1000, 1500)
        
        # 3. VIX 생성 (점진적)
        vix = self._generate_vix(regime, n_days)
        
        # 4. 동적 상관계수
        correlation = self._generate_correlation(regime, n_days)
        
        # 5. 스왑 포인트
        swap_points = self._calculate_swap_points(fx, regime, n_days)
        
        # 6. GARCH 변동성 저장
        fx_volatility = fx_vol * 100  # % 단위
        
        # DataFrame 생성
        df = pd.DataFrame({
            'VIX': vix,
            'FX': fx,
            'Correlation': correlation,
            'SwapPoints': swap_points,
            'FX_Volatility': fx_volatility,
            'Regime': regime
        })
        
        # Regime 라벨
        df['Regime_Label'] = df['Regime'].map({0: 'Normal', 1: 'Transition', 2: 'Panic'})
        
        return df
    
    def summary(self, df):
        """데이터 요약 통계"""
        print("=" * 60)
        print("Realistic Market Data Summary")
        print("=" * 60)
        
        print("\n[Regime Distribution]")
        print(df['Regime_Label'].value_counts())
        
        print("\n[Statistics by Regime]")
        stats = df.groupby('Regime_Label').agg({
            'VIX': ['mean', 'std', 'min', 'max'],
            'FX': ['mean', 'std'],
            'Correlation': ['mean', 'std'],
            'SwapPoints': ['mean', 'std']
        }).round(2)
        print(stats)
        
        print("\n[Fat Tail Check: Kurtosis]")
        from scipy.stats import kurtosis
        fx_returns = df['FX'].pct_change().dropna()
        print(f"  FX Returns Kurtosis: {kurtosis(fx_returns):.2f} (Normal=0, Fat Tail>0)")
        
        print("\n[GARCH Effect: Volatility Clustering]")
        print(f"  FX Volatility Autocorrelation: {fx_returns.rolling(5).std().autocorr():.2f}")


def generate_realistic_scenario(n_days=500, scenario='mixed', seed=42):
    """편의 함수: 현실적인 시장 데이터 생성"""
    generator = RealisticMarketGenerator(seed=seed)
    df = generator.generate(n_days, scenario)
    return df


# 기존 함수와 호환되는 인터페이스 (v4.0 개선)
def generate_market_scenario(n_days=500, scenario='normal', use_real_data=True, is_training=False):
    """
    시장 데이터 생성 (Anti-Overfitting 버전)
    
    [v4.0 변경 사항]
    - use_real_data=True: 실제 역사적 데이터 사용 (권장)
    - is_training: True=학습용(앞쪽 70%), False=테스트용(뒤쪽 30%)
    
    Args:
        n_days: 필요한 일수 (실제 데이터 길이보다 작아야 함)
        scenario: 시나리오 타입 (실제 데이터 사용 시 무시됨)
        use_real_data: True면 실제 CSV 데이터 사용
        is_training: True면 학습용 데이터, False면 테스트용 데이터
        
    Returns:
        DataFrame with VIX, FX, Correlation columns
    """
    global _REAL_DATA_CACHE, _TRAIN_DATA_CACHE, _TEST_DATA_CACHE
    
    if use_real_data:
        # 실제 데이터 로드 (캐시 사용)
        if _REAL_DATA_CACHE is None:
            if os.path.exists(REAL_DATA_PATH):
                _REAL_DATA_CACHE = pd.read_csv(REAL_DATA_PATH)
                
                # 데이터 전처리: 실제 Correlation 계산 (KOSPI와 FX의 실제 상관계수)
                if 'Correlation' not in _REAL_DATA_CACHE.columns:
                    print("  [정보] Correlation 컬럼 없음. 실제 데이터로부터 계산합니다.")
                    
                    # KOSPI와 FX의 수익률 계산
                    if 'KOSPI' in _REAL_DATA_CACHE.columns and 'FX' in _REAL_DATA_CACHE.columns:
                        # 수익률 계산 (pct_change)
                        kospi_returns = _REAL_DATA_CACHE['KOSPI'].pct_change().fillna(0)
                        fx_returns = _REAL_DATA_CACHE['FX'].pct_change().fillna(0)
                        
                        # 롤링 윈도우로 상관계수 계산 (예: 60일 윈도우)
                        window = 60
                        correlations = []
                        for i in range(len(_REAL_DATA_CACHE)):
                            start_idx = max(0, i - window + 1)
                            end_idx = i + 1
                            if end_idx - start_idx >= 10:  # 최소 10일 필요
                                corr = np.corrcoef(
                                    kospi_returns.iloc[start_idx:end_idx],
                                    fx_returns.iloc[start_idx:end_idx]
                                )[0, 1]
                                # NaN 처리
                                if np.isnan(corr):
                                    corr = 0.0
                                correlations.append(corr)
                            else:
                                correlations.append(0.0)  # 초기값
                        
                        _REAL_DATA_CACHE['Correlation'] = correlations
                        print(f"  ✓ 실제 Correlation 계산 완료 (롤링 {window}일 윈도우)")
                        print(f"    - Correlation 범위: [{np.min(correlations):.3f}, {np.max(correlations):.3f}]")
                    else:
                        print("  [경고] KOSPI 또는 FX 컬럼 없음. VIX 기반 추정 사용.")
                        # 폴백: VIX 기반 추정
                        vix = _REAL_DATA_CACHE['VIX'].values
                        corr = np.where(vix >= 30, np.random.uniform(0.5, 0.8, len(vix)),
                                np.where(vix >= 20, np.random.uniform(-0.2, 0.5, len(vix)),
                                         np.random.uniform(-0.6, -0.2, len(vix))))
                        _REAL_DATA_CACHE['Correlation'] = corr
                
                # Train/Test 분리 (시간 기반)
                split_idx = int(len(_REAL_DATA_CACHE) * TRAIN_RATIO)
                _TRAIN_DATA_CACHE = _REAL_DATA_CACHE.iloc[:split_idx].reset_index(drop=True)
                _TEST_DATA_CACHE = _REAL_DATA_CACHE.iloc[split_idx:].reset_index(drop=True)
                
                print(f"[실제 데이터 로드] 총 {len(_REAL_DATA_CACHE)}일")
                print(f"  -> 학습용: {len(_TRAIN_DATA_CACHE)}일 ({TRAIN_RATIO*100:.0f}%)")
                print(f"  -> 테스트용: {len(_TEST_DATA_CACHE)}일 ({(1-TRAIN_RATIO)*100:.0f}%)")
            else:
                print(f"[경고] 실제 데이터 파일 없음: {REAL_DATA_PATH}")
                print("       합성 데이터로 대체합니다.")
                use_real_data = False
        
        if use_real_data and _REAL_DATA_CACHE is not None:
            # Train 또는 Test 데이터 선택
            data_source = _TRAIN_DATA_CACHE if is_training else _TEST_DATA_CACHE
            
            if n_days > len(data_source):
                print(f"[경고] 요청 일수({n_days})가 가용 데이터({len(data_source)})보다 많음. 최대치로 조정.")
                n_days = len(data_source)
            
            # 시나리오에 따른 구간 선택
            if scenario in ['2008_crisis', 'crisis']:
                # 위기 구간 찾기 (VIX > 30인 구간)
                crisis_mask = data_source['VIX'] > 25
                crisis_indices = np.where(crisis_mask)[0]
                if len(crisis_indices) >= n_days:
                    start_idx = crisis_indices[0]
                else:
                    start_idx = 0
            elif scenario == '2020_pandemic':
                # 팬데믹 유사 구간 (VIX 급등)
                start_idx = len(data_source) // 2  # 중간 지점
            else:
                # 랜덤 시작점 (하지만 Train/Test 내에서)
                max_start = max(0, len(data_source) - n_days)
                start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            
            end_idx = min(start_idx + n_days, len(data_source))
            df_slice = data_source.iloc[start_idx:end_idx].copy().reset_index(drop=True)
            
            return pd.DataFrame({
                'VIX': df_slice['VIX'],
                'FX': df_slice['FX'],
                'Correlation': df_slice['Correlation']
            })
    
    # 합성 데이터 폴백 (use_real_data=False 또는 파일 없음)
    # 테스트용일 때는 다른 seed 범위 사용
    if is_training:
        seed = np.random.randint(*TRAIN_SEED_RANGE)
    else:
        seed = np.random.randint(*TEST_SEED_RANGE)
    
    generator = RealisticMarketGenerator(seed=seed)
    df = generator.generate(n_days, scenario)
    
    return pd.DataFrame({
        'VIX': df['VIX'],
        'FX': df['FX'],
        'Correlation': df['Correlation']
    })


def load_real_data_for_training(n_days=None):
    """
    PPO 학습용 실제 데이터 로드 (Anti-Overfitting)
    
    Returns:
        학습용 데이터 (앞쪽 70%)
    """
    return generate_market_scenario(n_days=n_days or 3000, use_real_data=True, is_training=True)


def load_real_data_for_testing(n_days=None):
    """
    백테스트용 실제 데이터 로드 (Anti-Overfitting)
    
    Returns:
        테스트용 데이터 (뒤쪽 30%)
    """
    return generate_market_scenario(n_days=n_days or 1500, use_real_data=True, is_training=False)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    generator = RealisticMarketGenerator()
    
    # 다양한 시나리오 생성
    df = generator.generate(n_days=500, scenario='mixed')
    generator.summary(df)
    
    # 시각화
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    # VIX with Regime
    colors = {'Normal': 'green', 'Transition': 'orange', 'Panic': 'red'}
    for label, color in colors.items():
        mask = df['Regime_Label'] == label
        axes[0, 0].scatter(df[mask].index, df[mask]['VIX'], c=color, s=5, label=label, alpha=0.6)
    axes[0, 0].set_title('VIX with Regime (Gradual Transitions)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # FX with GARCH Volatility
    axes[0, 1].plot(df['FX'], 'b-', lw=1)
    axes[0, 1].set_title('FX Rate (GARCH Volatility Clustering)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # FX Volatility (GARCH effect)
    axes[1, 0].plot(df['FX_Volatility'], 'r-', lw=1)
    axes[1, 0].set_title('FX Volatility (GARCH Clustering)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Correlation (Dynamic)
    axes[1, 1].plot(df['Correlation'], 'purple', lw=1)
    axes[1, 1].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Dynamic Correlation')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Swap Points
    axes[2, 0].plot(df['SwapPoints'], 'orange', lw=1)
    axes[2, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
    axes[2, 0].set_title('Swap Points (Interest Rate Differential)')
    axes[2, 0].grid(True, alpha=0.3)
    
    # FX Returns Distribution (Fat Tail)
    fx_returns = df['FX'].pct_change().dropna() * 100
    axes[2, 1].hist(fx_returns, bins=50, density=True, alpha=0.7, color='blue', label='FX Returns')
    # Normal overlay
    from scipy.stats import norm
    x = np.linspace(fx_returns.min(), fx_returns.max(), 100)
    axes[2, 1].plot(x, norm.pdf(x, fx_returns.mean(), fx_returns.std()), 'r--', label='Normal')
    axes[2, 1].set_title('FX Returns Distribution (Fat Tail vs Normal)')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Realistic Market Data Generator', y=1.02, fontsize=14, fontweight='bold')
    plt.savefig('realistic_data_demo.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n[Saved] realistic_data_demo.png")
