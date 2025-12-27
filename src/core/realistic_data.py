"""
Realistic Market Data Generator
================================
개선된 합성 데이터 생성기

개선 사항:
1. GARCH(1,1) - 환율 변동성 클러스터링
2. 점진적 Regime 전환
3. 스왑 포인트 (금리차 기반)
4. Fat Tail (Student-t 분포)

핵심 철학: Capital Optimization, not Prediction
(환율 예측이 아닌 자본 최적화)
"""

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


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


# 기존 함수와 호환되는 인터페이스
def generate_market_scenario(n_days=500, scenario='normal'):
    """
    기존 backtest.py와 호환되는 인터페이스
    """
    generator = RealisticMarketGenerator()
    df = generator.generate(n_days, scenario)
    
    return pd.DataFrame({
        'VIX': df['VIX'],
        'FX': df['FX'],
        'Correlation': df['Correlation']
    })


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
