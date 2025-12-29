"""
Phase 5.1: Risk Paradox Proof (이론적 타당성 검증)
===================================================
"헤지를 덜 했는데(80~90%) 왜 총 위험액은 줄어드는가?"에 대한 수학적 증명
주식-환율 음의 상관관계(Natural Hedge)로 인한 분산 효과 수치 입증
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.kics_real import RatioKICSEngine


def prove_risk_paradox():
    """
    Risk Paradox 증명:
    100% 헤지보다 80% 헤지가 더 낮은 총 위험액을 보이는 현상 입증
    """
    print("=" * 60)
    print("Phase 5.1: Risk Paradox Proof")
    print("=" * 60)
    
    engine = RatioKICSEngine()
    
    # 다양한 상관계수에서 테스트
    correlations_to_test = [-0.6, -0.4, -0.2, 0.0, 0.2]
    
    results = []
    
    for corr in correlations_to_test:
        hedge_ratios = np.linspace(0, 1, 101)
        fixed_corr = np.full_like(hedge_ratios, corr)
        
        scr_ratios = engine.calculate_scr_ratio_batch(hedge_ratios, fixed_corr)
        
        # 최적점 찾기
        # 최적점 찾기 (지급여력비율 최대화)
        opt_idx = np.argmax(scr_ratios)
        opt_ratio = hedge_ratios[opt_idx]
        opt_scr_ratio = scr_ratios[opt_idx]
        ratio_100 = scr_ratios[-1]  # 100% 헤지 시 비율
        
        # 자본 절감률 계산 (Risk = Capital / Ratio)
        # Savings = (Risk_100 - Risk_opt) / Risk_100 = 1 - (Ratio_100 / Ratio_opt)
        if opt_scr_ratio > 0:
            savings = (1 - (ratio_100 / opt_scr_ratio)) * 100
        else:
            savings = 0.0

        # Risk Paradox 확인 (부분 헤지가 완전 헤지보다 비율이 높으면 성공)
        is_paradox = opt_ratio < 1.0 and opt_scr_ratio > ratio_100
        
        results.append({
            'correlation': corr,
            'optimal_hedge': opt_ratio,
            'optimal_scr_ratio': opt_scr_ratio,
            'scr_ratio_100_hedge': ratio_100,
            'savings': savings,
            'paradox_proven': is_paradox
        })
        
        print(f"\n[Correlation: {corr:.1f}]")
        print(f"  Optimal Hedge Ratio: {opt_ratio*100:.1f}%")
        print(f"  Solvency Ratio (Opt): {opt_scr_ratio*100:.2f}%") # 정규화 가정 무시하고 비율로 표시
        print(f"  Solvency Ratio (100%): {ratio_100*100:.2f}%")
        print(f"  Risk Capital Savings: {savings:.2f}%")
        print(f"  Paradox Proven: {'YES ✓' if is_paradox else 'NO'}")
    
    # 전체 결과 확인
    paradox_count = sum(1 for r in results if r['paradox_proven'])
    
    print("\n" + "=" * 60)
    if paradox_count >= 3:
        print("[SUCCESS] Risk Paradox Proven!")
        print(f"  {paradox_count}/{len(results)} scenarios show the paradox")
    else:
        print("[PARTIAL] Risk Paradox partially demonstrated")
    
    # 시각화
    plt.figure(figsize=(12, 6))
    
    for corr in [-0.6, -0.4, 0.0]:
        hedge_ratios = np.linspace(0, 1, 101)
        fixed_corr = np.full_like(hedge_ratios, corr)
        scr_ratios = engine.calculate_scr_ratio_batch(hedge_ratios, fixed_corr)
        
        plt.plot(hedge_ratios * 100, scr_ratios * 100, label=f'Corr = {corr}', lw=2)
        
        # 최적점 표시
        opt_idx = np.argmin(scr_ratios)
        plt.scatter(hedge_ratios[opt_idx] * 100, scr_ratios[opt_idx] * 100, s=100, zorder=5)
    
    plt.axhline(y=36, color='red', linestyle='--', label='100% Hedge SCR', alpha=0.7)
    plt.xlabel('Hedge Ratio (%)')
    plt.ylabel('SCR Ratio (%)')
    plt.title('Risk Paradox: Why Less Hedge = Less Risk?')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('risk_paradox_proof.png', dpi=150)
    plt.show()
    
    print("\n[Saved] risk_paradox_proof.png")
    
    return results


if __name__ == "__main__":
    results = prove_risk_paradox()
