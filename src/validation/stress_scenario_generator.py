"""
스트레스 시나리오 생성기 및 검증 프레임워크
===========================================
기존 3개 시나리오 외 추가 시나리오 생성 및 체계적 검증

핵심 철학: Capital Optimization, not Prediction
(환율 예측이 아닌 자본 최적화)
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta


class StressScenarioGenerator:
    """
    추가 스트레스 시나리오 생성기
    
    기존 시나리오:
    1. Scenario_A_Stagflation (스태그플레이션)
    2. Scenario_B_Correlation_Breakdown (상관관계 붕괴)
    3. Scenario_COVID19 (팬데믹)
    4. Scenario_Tail_Risk (꼬리 위험)
    
    추가 시나리오:
    5. Scenario_C_Interest_Rate_Shock (금리 쇼크)
    6. Scenario_D_Swap_Point_Extreme (스왑포인트 극단 마이너스)
    7. Scenario_E_Regime_Transition (Regime 전이 구간)
    """
    
    def __init__(self, base_date='2025-12-30', n_days=250):
        self.base_date = pd.to_datetime(base_date)
        self.n_days = n_days
        self.dates = pd.date_range(self.base_date, periods=n_days, freq='D')
        
    def generate_interest_rate_shock(self):
        """
        시나리오 5: 금리 쇼크 특화
        - 국고채 10Y 급등 (스프레드 확대)
        - 주가/환율은 중립 유지
        - 목적: 금리 리스크에만 취약한 상황에서 K-ICS 방어력 검증
        """
        df = pd.DataFrame(index=self.dates)
        
        # 기본값 설정
        df['SPX'] = 7000.0
        df['FX'] = 1445.0
        df['KOSPI'] = 600.0
        df['VIX'] = 15.0
        df['VIX_Change'] = 0.0
        df['Returns_SPX'] = 0.0
        
        # 금리 쇼크: 국고채 10Y가 3.37% → 5.5%로 급등 (약 2.13%p 상승)
        df['KR_10Y'] = np.linspace(3.37, 5.5, self.n_days)
        df['US_10Y'] = np.linspace(4.134, 4.5, self.n_days)  # 미국 금리는 소폭 상승
        
        # Yield Spread 계산
        df['Yield_Spread'] = df['KR_10Y'] - df['US_10Y']
        
        # 스왑포인트는 금리차 확대로 인해 변화
        df['Swap_Point_Proxy'] = df['FX'] * df['Yield_Spread'] * (30/360)
        
        # FX_MA_Divergence는 중립
        df['FX_MA_Divergence'] = np.random.normal(0, 0.01, self.n_days)
        
        # Correlation은 약간 음의 상관 유지 (금리 쇼크가 주식/환율에 직접 영향 없음)
        df['Correlation'] = np.random.uniform(-0.3, -0.1, self.n_days)
        
        return df
    
    def generate_swap_point_extreme(self):
        """
        시나리오 6: 스왑포인트 극단 마이너스
        - Swap Point가 매우 마이너스로 확대 (해지 비용 극대화)
        - 목적: 비용 절감 vs 리스크 트레이드오프 검증
        """
        df = pd.DataFrame(index=self.dates)
        
        # 기본값
        df['SPX'] = 7000.0
        df['FX'] = 1445.0
        df['KOSPI'] = 600.0
        df['VIX'] = 20.0  # 약간 높은 VIX
        df['VIX_Change'] = 0.0
        df['Returns_SPX'] = 0.0
        
        # 금리 설정: 미국 금리는 유지, 한국 금리는 급락
        df['US_10Y'] = 4.134
        df['KR_10Y'] = np.linspace(3.37, 1.5, self.n_days)  # 급락
        
        # Yield Spread 계산
        df['Yield_Spread'] = df['KR_10Y'] - df['US_10Y']
        
        # 스왑포인트가 극단적으로 마이너스
        df['Swap_Point_Proxy'] = df['FX'] * df['Yield_Spread'] * (30/360)
        # 마이너스 값이 더 커짐
        
        # FX_MA_Divergence는 중립
        df['FX_MA_Divergence'] = np.random.normal(0, 0.01, self.n_days)
        
        # Correlation은 약간 양의 상관 (불안정)
        df['Correlation'] = np.random.uniform(0.0, 0.3, self.n_days)
        
        return df
    
    def generate_regime_transition(self):
        """
        시나리오 7: Regime 전이 구간
        - VIX 중간 수준, 스프레드 확대, 환율 변동성 증가
        - Normal → Transition → Panic 직전 상태
        - 목적: 패닉 직전 상황에서의 헤지 비율 path 확인
        """
        df = pd.DataFrame(index=self.dates)
        
        # 기본값
        df['SPX'] = 7000.0
        df['FX'] = 1445.0
        df['KOSPI'] = 600.0
        df['Returns_SPX'] = 0.0
        
        # VIX가 점진적으로 상승 (15 → 28 → 35)
        n1, n2 = self.n_days // 3, 2 * self.n_days // 3
        vix_values = np.zeros(self.n_days)
        vix_values[:n1] = np.linspace(15, 20, n1)
        vix_values[n1:n2] = np.linspace(20, 28, n2 - n1)
        vix_values[n2:] = np.linspace(28, 35, self.n_days - n2)
        df['VIX'] = vix_values
        
        # VIX_Change 계산
        df['VIX_Change'] = df['VIX'].diff().fillna(0)
        
        # Yield Spread 확대
        df['US_10Y'] = 4.134
        df['KR_10Y'] = np.linspace(3.37, 4.0, self.n_days)
        df['Yield_Spread'] = df['KR_10Y'] - df['US_10Y']
        
        # 스왑포인트
        df['Swap_Point_Proxy'] = df['FX'] * df['Yield_Spread'] * (30/360)
        
        # FX_MA_Divergence 증가 (변동성 확대)
        df['FX_MA_Divergence'] = np.random.normal(0, 0.02, self.n_days) * np.linspace(1, 2, self.n_days)
        
        # Correlation이 점진적으로 양의 상관으로 전환
        df['Correlation'] = np.linspace(-0.3, 0.5, self.n_days)
        
        return df
    
    def save_scenario(self, df, scenario_name):
        """시나리오를 CSV로 저장"""
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'DATA', 'synthetic_stress'
        )
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f'{scenario_name}.csv')
        df.to_csv(output_path, index=True)
        print(f"[저장 완료] {output_path}")
        return output_path


def generate_all_additional_scenarios():
    """모든 추가 시나리오 생성"""
    print("=" * 60)
    print("추가 스트레스 시나리오 생성")
    print("=" * 60)
    
    generator = StressScenarioGenerator()
    
    # 시나리오 5: 금리 쇼크
    print("\n[시나리오 5] Interest Rate Shock 생성 중...")
    df5 = generator.generate_interest_rate_shock()
    generator.save_scenario(df5, 'Scenario_C_Interest_Rate_Shock')
    
    # 시나리오 6: 스왑포인트 극단 마이너스
    print("\n[시나리오 6] Swap Point Extreme 생성 중...")
    df6 = generator.generate_swap_point_extreme()
    generator.save_scenario(df6, 'Scenario_D_Swap_Point_Extreme')
    
    # 시나리오 7: Regime 전이
    print("\n[시나리오 7] Regime Transition 생성 중...")
    df7 = generator.generate_regime_transition()
    generator.save_scenario(df7, 'Scenario_E_Regime_Transition')
    
    print("\n" + "=" * 60)
    print("모든 추가 시나리오 생성 완료!")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_additional_scenarios()

