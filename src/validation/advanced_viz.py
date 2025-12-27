"""
Phase 5.5: Advanced Visualization (XAI & Efficient Frontier)
=============================================================
AI 판단의 투명성(Why Not)과 비용 대비 효율성(Cost-Efficiency) 시각화
1. Counterfactual Dashboard (Why Not 분석)
2. Efficient Frontier (효율적 투자선)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.kics_real import RatioKICSEngine
from validation.backtest import generate_market_scenario, BacktestEngine


def plot_decision_boundary():
    """
    Plot 1: Counterfactual Dashboard - Why Not 분석
    "만약 VIX가 10% 추가 상승했다면 AI는 어떻게 반응했을까?"
    """
    print("\n[Plot 1] Counterfactual Dashboard (Decision Boundary)")
    print("-" * 50)
    
    engine = RatioKICSEngine()
    
    # VIX vs Hedge Ratio 의사결정 경계선
    vix_range = np.linspace(10, 60, 50)
    hedge_ratios_normal = []
    hedge_ratios_alternative = []
    
    current_hedge = 0.5
    
    for vix in vix_range:
        # 현재 로직 (Normal Path)
        if vix >= 30:
            target = min(current_hedge + 0.15, 1.0)
        elif vix >= 20:
            target = min(current_hedge + 0.05, 0.8)
        else:
            target = max(current_hedge - 0.03, 0.4)
        hedge_ratios_normal.append(target)
        
        # 대안 로직 (VIX +10% 시나리오)
        vix_alt = vix * 1.1
        if vix_alt >= 30:
            target_alt = min(current_hedge + 0.15, 1.0)
        elif vix_alt >= 20:
            target_alt = min(current_hedge + 0.05, 0.8)
        else:
            target_alt = max(current_hedge - 0.03, 0.4)
        hedge_ratios_alternative.append(target_alt)
        
        current_hedge = target
    
    # 시각화
    plt.figure(figsize=(12, 6))
    
    plt.plot(vix_range, hedge_ratios_normal, 'b-', lw=2, label='Current Decision Path')
    plt.plot(vix_range, hedge_ratios_alternative, 'r--', lw=2, label='If VIX +10% (Counterfactual)')
    plt.fill_between(vix_range, hedge_ratios_normal, hedge_ratios_alternative, alpha=0.2, color='orange')
    
    # 의사결정 경계
    plt.axvline(20, color='orange', linestyle=':', label='Transition Threshold')
    plt.axvline(30, color='red', linestyle=':', label='Panic Threshold')
    
    plt.xlabel('VIX Index')
    plt.ylabel('Hedge Ratio')
    plt.title('Counterfactual Analysis: AI Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 주석 추가
    plt.annotate('현재: 80% 유지\n만약 VIX +10% → 즉시 95%',
                 xy=(25, 0.8), xytext=(35, 0.65),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('counterfactual_dashboard.png', dpi=150)
    plt.show()
    
    print("[Saved] counterfactual_dashboard.png")


def plot_efficient_frontier():
    """
    Plot 2: Efficient Frontier - 효율적 투자선
    X축: Total Risk (SCR), Y축: Total Hedge Cost
    Dynamic Shield가 Sweet Spot (Low Risk, Low Cost)에 위치해야 함
    """
    print("\n[Plot 2] Efficient Frontier (Risk vs Cost)")
    print("-" * 50)
    
    engine = BacktestEngine()
    
    # 여러 시나리오에서 데이터 수집
    scenarios = ['normal', '2008_crisis', '2020_pandemic']
    
    strategy_data = {
        '100% Hedge': {'risks': [], 'costs': []},
        '80% Fixed': {'risks': [], 'costs': []},
        'Rule-based': {'risks': [], 'costs': []},
        'Dynamic Shield': {'risks': [], 'costs': []}
    }
    
    for scenario in scenarios:
        market_data = generate_market_scenario(300, scenario)
        results = engine.run_all_strategies(market_data)
        
        for strategy_name, df in results.items():
            avg_scr = df['SCR_Ratio'].mean()
            total_cost = df['Hedge_Cost'].sum()
            strategy_data[strategy_name]['risks'].append(avg_scr)
            strategy_data[strategy_name]['costs'].append(total_cost)
    
    # 평균값 계산
    avg_data = {}
    for strategy, data in strategy_data.items():
        avg_data[strategy] = {
            'risk': np.mean(data['risks']),
            'cost': np.mean(data['costs'])
        }
    
    # 시각화
    plt.figure(figsize=(10, 8))
    
    colors = {'100% Hedge': 'gray', '80% Fixed': 'blue', 'Rule-based': 'orange', 'Dynamic Shield': 'green'}
    markers = {'100% Hedge': 's', '80% Fixed': '^', 'Rule-based': 'D', 'Dynamic Shield': 'o'}
    
    for strategy, vals in avg_data.items():
        plt.scatter(vals['risk'] * 100, vals['cost'] * 100, 
                   s=300, c=colors[strategy], marker=markers[strategy],
                   label=strategy, edgecolors='black', linewidths=2)
    
    # Sweet Spot 영역 표시
    plt.axvspan(33, 36, alpha=0.1, color='green', label='Sweet Spot Zone')
    
    # 축 설정
    plt.xlabel('Average SCR Ratio (= Total Risk) %', fontsize=12)
    plt.ylabel('Total Hedge Cost %', fontsize=12)
    plt.title('Efficient Frontier: Risk vs Cost Trade-off', fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 화살표로 Dynamic Shield 강조
    ds = avg_data['Dynamic Shield']
    plt.annotate('SWEET SPOT\n(Low Risk, Low Cost)',
                 xy=(ds['risk'] * 100, ds['cost'] * 100),
                 xytext=(ds['risk'] * 100 - 2, ds['cost'] * 100 + 0.1),
                 arrowprops=dict(arrowstyle='->', color='green', lw=2),
                 fontsize=11, fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig('efficient_frontier.png', dpi=150)
    plt.show()
    
    print("[Saved] efficient_frontier.png")
    
    # 결과 출력
    print("\n[Efficient Frontier Summary]")
    for strategy, vals in avg_data.items():
        print(f"  {strategy:15s}: Risk={vals['risk']*100:.2f}%, Cost={vals['cost']*100:.2f}%")
    
    # Dynamic Shield가 Sweet Spot에 있는지 확인
    ds = avg_data['Dynamic Shield']
    benchmark = avg_data['100% Hedge']
    
    if ds['risk'] < benchmark['risk'] and ds['cost'] < benchmark['cost']:
        print("\n[SUCCESS] Dynamic Shield is in the SWEET SPOT!")
        print("  → Lower risk AND lower cost than 100% Hedge!")
    
    return avg_data


def run_advanced_visualization():
    """전체 고급 시각화 실행"""
    print("=" * 60)
    print("Phase 5.5: Advanced Visualization (XAI)")
    print("=" * 60)
    
    plot_decision_boundary()
    frontier_data = plot_efficient_frontier()
    
    print("\n" + "=" * 60)
    print("[COMPLETE] All advanced visualizations generated!")
    print("  1. counterfactual_dashboard.png")
    print("  2. efficient_frontier.png")
    print("=" * 60)
    
    return frontier_data


if __name__ == "__main__":
    results = run_advanced_visualization()
