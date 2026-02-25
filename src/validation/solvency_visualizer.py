"""
Phase 5.2: Solvency Analysis (자본 적정성 시각화)
=================================================
2020년 3월 코로나 팬데믹 시나리오 시뮬레이션
- Line A (100% Hedge): K-ICS 비율 급락
- Line B (Dynamic Shield): K-ICS 비율 안정권 유지
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.kics_real import RatioKICSEngine


def generate_pandemic_scenario(n_days=252):
    """2020년 3월 코로나 팬데믹 시나리오 생성"""
    np.random.seed(2020)
    
    # Phase 1: 평온 (1~60일)
    vix_p1 = np.random.normal(14, 2, 60)
    
    # Phase 2: 폭락 시작 (61~80일) - VIX 급등
    vix_p2 = np.linspace(14, 65, 20) + np.random.normal(0, 3, 20)
    
    # Phase 3: 공포 정점 (81~100일)
    vix_p3 = np.random.normal(60, 8, 20)
    
    # Phase 4: 회복 시작 (101~150일)
    vix_p4 = np.linspace(60, 30, 50) + np.random.normal(0, 4, 50)
    
    # Phase 5: 안정화 (151~252일)
    vix_p5 = np.random.normal(25, 5, 102)
    
    vix = np.concatenate([vix_p1, vix_p2, vix_p3, vix_p4, vix_p5])
    
    # 환율 (VIX와 양의 상관: 공포 시 KRW 약세)
    fx_rate = [1180]
    for v in vix[:-1]:
        if v > 40:
            change = np.random.normal(8, 10)  # 급등
        elif v > 25:
            change = np.random.normal(3, 5)
        else:
            change = np.random.normal(0, 3)
        fx_rate.append(fx_rate[-1] + change)
    
    # 주가 (VIX와 음의 상관: 공포 시 하락)
    stock_idx = [2200]
    for v in vix[:-1]:
        if v > 40:
            change = np.random.normal(-30, 20)
        elif v > 25:
            change = np.random.normal(-5, 10)
        else:
            change = np.random.normal(5, 8)
        stock_idx.append(stock_idx[-1] + change)
    
    return pd.DataFrame({
        'VIX': vix,
        'FX': fx_rate,
        'Stock': stock_idx
    })


def simulate_kics_ratio(market_data, strategy='100_hedge'):
    """
    K-ICS 비율 시뮬레이션
    K-ICS Ratio = Available Capital / Required Capital
    """
    engine = RatioKICSEngine()
    
    # 초기 자본
    initial_capital = 10_000_000_000  # 100억
    available_capital = 15_000_000_000  # 150억 (초기 K-ICS 150%)
    
    kics_ratios = []
    hedge_ratios = []
    current_hedge = 0.5 if strategy == 'dynamic' else 1.0 if strategy == '100_hedge' else 0.8
    
    for i, row in market_data.iterrows():
        vix = row['VIX']
        
        # 상관계수 추정 (VIX 기반)
        if vix >= 40:
            corr = 0.8  # Panic
        elif vix >= 25:
            corr = 0.2  # Transition
        else:
            corr = -0.4  # Normal
        
        # 전략별 헤지 비율 결정
        if strategy == 'dynamic':
            if vix >= 40:
                current_hedge = min(current_hedge + 0.1, 1.0)
            elif vix >= 25:
                current_hedge = min(max(current_hedge, 0.7), 0.9)
            else:
                current_hedge = max(current_hedge - 0.03, 0.4)
        elif strategy == '100_hedge':
            current_hedge = 1.0
        else:  # 80_fixed
            current_hedge = 0.8
        
        # SCR 계산
        scr_ratio = engine.calculate_scr_ratio_batch(
            np.array([current_hedge]), 
            np.array([corr])
        )[0]
        
        required_capital = scr_ratio * initial_capital
        
        # 헤지 비용 차감 (일일 0.02%)
        hedge_cost = current_hedge * initial_capital * 0.0002
        
        # 자산 변동 (주가 변동 반영)
        if i > 0:
            stock_return = (row['Stock'] - market_data.iloc[i-1]['Stock']) / market_data.iloc[i-1]['Stock']
            asset_change = initial_capital * stock_return * (1 - current_hedge)  # 비헤지 부분만 영향
            available_capital += asset_change
        
        # 환차익/손 (비헤지 부분)
        if i > 0:
            fx_change = (row['FX'] - market_data.iloc[i-1]['FX']) / market_data.iloc[i-1]['FX']
            fx_gain_loss = initial_capital * fx_change * (1 - current_hedge) * 0.3  # 환익 일부 반영
            available_capital += fx_gain_loss
        
        available_capital -= hedge_cost
        
        # K-ICS 비율 계산
        kics_ratio = (available_capital / required_capital) * 100
        kics_ratios.append(kics_ratio)
        hedge_ratios.append(current_hedge)
    
    return kics_ratios, hedge_ratios


def run_solvency_analysis():
    """자본 적정성 분석 실행"""
    print("=" * 60)
    print("Phase 5.2: Solvency Analysis (COVID-19 Scenario)")
    print("=" * 60)
    
    # 시나리오 생성
    market_data = generate_pandemic_scenario()
    
    # 각 전략별 시뮬레이션
    kics_100, hedge_100 = simulate_kics_ratio(market_data, '100_hedge')
    kics_80, hedge_80 = simulate_kics_ratio(market_data, '80_fixed')
    kics_dynamic, hedge_dynamic = simulate_kics_ratio(market_data, 'dynamic')
    
    # 결과 분석
    print(f"\n[100% Hedge]")
    print(f"  Min K-ICS: {min(kics_100):.1f}%")
    print(f"  Final K-ICS: {kics_100[-1]:.1f}%")
    
    print(f"\n[80% Fixed]")
    print(f"  Min K-ICS: {min(kics_80):.1f}%")
    print(f"  Final K-ICS: {kics_80[-1]:.1f}%")
    
    print(f"\n[Dynamic Shield]")
    print(f"  Min K-ICS: {min(kics_dynamic):.1f}%")
    print(f"  Final K-ICS: {kics_dynamic[-1]:.1f}%")
    
    # 위기 방어 성공 여부
    if min(kics_dynamic) > 100 and min(kics_dynamic) > min(kics_100):
        print("\n[SUCCESS] Dynamic Shield maintained K-ICS > 100% during crisis!")
    
    # 시각화 (The Money Shot!)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: VIX
    axes[0].plot(market_data['VIX'], color='red', alpha=0.7)
    axes[0].axhline(30, linestyle='--', color='orange', label='Panic Threshold')
    axes[0].set_title('VIX Index (2020 COVID-19 Scenario)')
    axes[0].set_ylabel('VIX')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: K-ICS Ratio (핵심 그래프!)
    axes[1].plot(kics_100, label='100% Hedge (Benchmark)', color='gray', lw=2)
    axes[1].plot(kics_80, label='80% Fixed', color='blue', lw=2, alpha=0.7)
    axes[1].plot(kics_dynamic, label='Dynamic Shield', color='green', lw=2)
    axes[1].axhline(100, linestyle='--', color='red', lw=2, label='K-ICS 100% (Danger)')
    axes[1].axhline(150, linestyle='--', color='orange', alpha=0.5, label='K-ICS 150% (Safe)')
    axes[1].fill_between(range(len(kics_dynamic)), 0, 100, alpha=0.1, color='red')
    axes[1].set_title('K-ICS Ratio Comparison (The Money Shot!)')
    axes[1].set_ylabel('K-ICS Ratio (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(50, 200)
    
    # Plot 3: Hedge Ratio
    axes[2].plot(hedge_100, label='100% Hedge', color='gray', lw=2)
    axes[2].plot(hedge_dynamic, label='Dynamic Shield', color='green', lw=2)
    axes[2].set_title('Hedge Ratio Over Time')
    axes[2].set_ylabel('Hedge Ratio')
    axes[2].set_xlabel('Days')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kics_defense_result.png', dpi=150)
    plt.show()
    
    print("\n[Saved] kics_defense_result.png")
    
    return {
        'kics_100': kics_100,
        'kics_dynamic': kics_dynamic,
        'market_data': market_data
    }


if __name__ == "__main__":
    results = run_solvency_analysis()
