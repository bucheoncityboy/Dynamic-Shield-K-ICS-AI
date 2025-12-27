"""
Phase 1: K-ICS Engine (Ground Truth Calculator)
================================================
자산 규모에 의존하지 않는 '비율(Ratio)' 기반의 순수 리스크 엔진.
Input: Hedge Ratio, Correlation
Output: SCR Ratio (Ex: 0.035 = 3.5%)

핵심 철학: Capital Optimization, not Prediction
(환율 예측이 아닌 자본 최적화)
"""

import numpy as np
import matplotlib.pyplot as plt

class RatioKICSEngine:
    def __init__(self):
        self.shock_equity = 0.35  # 주가 35% 충격
        self.shock_fx = 0.20      # 환율 20% 충격
        self.ir_risk_rate = 0.05  # 금리 5% 충격
        
        # 기본 상관계수 틀 (Broadcasting용)
        self.base_corr = np.array([
            [ 1.00, -0.40,  0.15],
            [-0.40,  1.00,  0.10],
            [ 0.15,  0.10,  1.00]
        ])

    def calculate_scr_ratio_batch(self, hedge_ratios, correlations):
        """K-ICS 표준모형 기반 SCR Ratio 계산 (제곱근 합산)"""
        ones = np.ones_like(hedge_ratios)
        
        # 1. 개별 리스크 비율 산출
        risk_equity = ones * self.shock_equity
        risk_fx = (ones * (1 - hedge_ratios)) * self.shock_fx
        risk_ir = ones * self.ir_risk_rate
        
        risks = np.column_stack((risk_equity, risk_fx, risk_ir))
        
        # 2. 동적 상관계수 적용
        batch_size = len(hedge_ratios)
        corr_tensor = np.tile(self.base_corr, (batch_size, 1, 1))
        corr_tensor[:, 0, 1] = correlations
        corr_tensor[:, 1, 0] = correlations
        
        # 3. Batch 행렬 연산: sqrt( R.T @ Corr @ R )
        temp = np.einsum('ni,nij->nj', risks, corr_tensor)
        total_risk_squared = np.einsum('nj,nj->n', temp, risks)
        
        return np.sqrt(total_risk_squared)


def run_phase1_risk_paradox():
    """Phase 1: Risk Paradox Proof"""
    print("=== Phase 1: Risk Paradox Proof ===")
    engine = RatioKICSEngine()
    
    # 시뮬레이션: 주식-환율 상관계수 -0.4 (Natural Hedge)
    test_ratios = np.linspace(0, 1, 101)
    fixed_corr = np.full_like(test_ratios, -0.4)
    
    scr_ratios = engine.calculate_scr_ratio_batch(test_ratios, fixed_corr)
    
    # 결과 해석 (1억 원 기준)
    example_asset = 100_000_000
    scr_values = scr_ratios * example_asset
    
    opt_idx = np.argmin(scr_values)
    opt_ratio = test_ratios[opt_idx]
    opt_scr = scr_values[opt_idx]
    scr_100 = scr_values[-1]
    
    print(f"[-] Optimal Hedge Ratio: {opt_ratio*100:.1f}%")
    print(f"[-] SCR at 100% Hedge: {scr_100:,.0f} KRW")
    print(f"[-] SCR at Sweet Spot: {opt_scr:,.0f} KRW")
    print(f"[-] Capital Saved: {scr_100 - opt_scr:,.0f} KRW")
    
    plt.figure(figsize=(10, 5))
    plt.plot(test_ratios*100, scr_values, label='K-ICS Required Capital', lw=2)
    plt.axvline(opt_ratio*100, color='red', linestyle='--', label=f'Optimal: {opt_ratio*100:.1f}%')
    plt.title(f"Phase 1: Risk Paradox Proof (Correlation -0.4)")
    plt.xlabel("Hedge Ratio (%)")
    plt.ylabel("Required Capital (KRW)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return engine


if __name__ == "__main__":
    engine = run_phase1_risk_paradox()
