"""
Phase 5.4: Walk-Forward Backtesting & Performance Analysis
===========================================================
Dynamic Shield v3.0의 정량적 성과 검증
- Walk-Forward Backtesting
- 4가지 전략 비교
- Stress Test (2008, 2020, 가상 시나리오)
- 성과 지표: CAGR, MDD, Sharpe, RCR

핵심 철학: Capital Optimization, not Prediction
(환율 예측이 아닌 자본 최적화)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Phase 1: K-ICS Engine 가져오기
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.kics_real import RatioKICSEngine


# ==========================================
# 1. Strategy Definitions (전략 정의)
# ==========================================

def strategy_100_hedge(vix, current_ratio):
    """① 100% 완전 헤지 (기존 관행)"""
    return 1.0

def strategy_80_fixed(vix, current_ratio):
    """② 80% 고정 헤지"""
    return 0.8

def strategy_rule_based(vix, current_ratio):
    """③ Rule-based (VIX 연동)"""
    if vix > 30:
        return 1.0
    elif vix > 20:
        return 0.7
    else:
        return 0.5

def strategy_dynamic_shield(vix, current_ratio):
    """④ Dynamic Shield (AI + Regime 기반)"""
    # Gradual adjustment based on regime
    if vix >= 30:  # Panic
        target = 1.0
        step = 0.15
        return min(current_ratio + step, target)
    elif vix >= 20:  # Transition
        target = 0.7
        if current_ratio < target:
            return current_ratio + 0.05
        else:
            return current_ratio - 0.03
    else:  # Normal
        target = 0.4
        step = 0.05
        return max(current_ratio - step, target)


# ==========================================
# 2. Market Data Generator (시장 데이터 생성)
# ==========================================

# 현실적인 데이터 생성기 사용 (GARCH, Fat Tail, Swap Points)
try:
    from core.realistic_data import generate_market_scenario, RealisticMarketGenerator
    USE_REALISTIC_DATA = True
except ImportError:
    USE_REALISTIC_DATA = False

def generate_market_scenario_legacy(n_days=500, scenario='normal'):
    """기존 단순 데이터 생성기 (백업용)"""
    np.random.seed(42)
    
    if scenario == 'normal':
        vix = np.random.normal(15, 3, n_days)
        vix = np.clip(vix, 10, 25)
        
    elif scenario == '2008_crisis':
        phase1 = np.random.normal(15, 2, 150)
        phase2 = np.linspace(15, 80, 50) + np.random.normal(0, 5, 50)
        phase3 = np.random.normal(60, 10, 100)
        phase4 = np.linspace(60, 25, 100) + np.random.normal(0, 5, 100)
        phase5 = np.random.normal(20, 3, 100)
        vix = np.concatenate([phase1, phase2, phase3, phase4, phase5])
        
    elif scenario == '2020_pandemic':
        phase1 = np.random.normal(14, 2, 200)
        phase2 = np.linspace(14, 65, 30) + np.random.normal(0, 5, 30)
        phase3 = np.linspace(65, 25, 70) + np.random.normal(0, 3, 70)
        phase4 = np.random.normal(22, 4, 200)
        vix = np.concatenate([phase1, phase2, phase3, phase4])
        
    elif scenario == 'stagflation':
        vix = np.random.normal(35, 10, n_days)
        vix = np.clip(vix, 20, 60)
        
    elif scenario == 'correlation_breakdown':
        vix = np.abs(np.sin(np.linspace(0, 8*np.pi, n_days))) * 40 + 20
        vix += np.random.normal(0, 5, n_days)
        
    else:
        vix = np.random.normal(18, 5, n_days)
    
    fx = [1200]
    for v in vix[:-1]:
        if v > 30:
            change = np.random.normal(5, 10)
        elif v > 20:
            change = np.random.normal(2, 5)
        else:
            change = np.random.normal(0, 3)
        fx.append(fx[-1] + change)
    
    correlations = []
    for v in vix:
        if v >= 30:
            corr = np.random.uniform(0.5, 0.9)
        elif v >= 20:
            corr = np.random.uniform(-0.2, 0.5)
        else:
            corr = np.random.uniform(-0.6, -0.2)
        correlations.append(corr)
    
    return pd.DataFrame({
        'VIX': vix[:len(fx)],
        'FX': fx,
        'Correlation': correlations[:len(fx)]
    })


# ==========================================
# 3. Backtest Engine
# ==========================================

class BacktestEngine:
    def __init__(self):
        self.engine = RatioKICSEngine()
        self.strategies = {
            '100% Hedge': strategy_100_hedge,
            '80% Fixed': strategy_80_fixed,
            'Rule-based': strategy_rule_based,
            'Dynamic Shield': strategy_dynamic_shield
        }
        
    def run_backtest(self, market_data, strategy_name):
        """단일 전략 백테스트"""
        strategy_func = self.strategies[strategy_name]
        
        results = []
        current_ratio = 0.5  # 초기 헤지 비율
        
        for i, row in market_data.iterrows():
            vix = row['VIX']
            corr = row['Correlation']
            
            # 전략에 따른 헤지 비율 결정
            new_ratio = strategy_func(vix, current_ratio)
            
            # SCR 계산
            scr = self.engine.calculate_scr_ratio_batch(
                np.array([new_ratio]), 
                np.array([corr])
            )[0]
            
            # 헤지 비용 (헤지 비율에 비례, 간단화)
            hedge_cost = new_ratio * 0.002  # 연 0.2%를 일 단위로 환산 근사
            
            results.append({
                'Day': i,
                'VIX': vix,
                'Hedge_Ratio': new_ratio,
                'SCR_Ratio': scr,
                'Hedge_Cost': hedge_cost,
                'Correlation': corr
            })
            
            current_ratio = new_ratio
            
        return pd.DataFrame(results)
    
    def run_all_strategies(self, market_data):
        """모든 전략 비교"""
        all_results = {}
        for name in self.strategies.keys():
            all_results[name] = self.run_backtest(market_data, name)
        return all_results


# ==========================================
# 4. Performance Analyzer
# ==========================================

class PerformanceAnalyzer:
    @staticmethod
    def calculate_metrics(results_df, initial_capital=10_000_000_000):
        """
        성과 지표 산출 (개선된 버전)
        
        수익 계산 로직:
        - 자본 효율성 수익 = 절감된 요구자본 - 헤지 비용
        - 100% 헤지 대비 절감된 요구자본을 수익으로 간주
        """
        n_days = len(results_df)
        
        # 기준: 100% 헤지 시 평균 SCR (약 36%)
        baseline_scr = 0.36
        
        # 일별 자본 효율성 수익 계산
        # Capital Efficiency = (Baseline SCR - Strategy SCR) - Hedge Cost
        # 이는 "100% 헤지 대비 절감한 자본 - 지불한 비용"을 의미
        daily_capital_savings = (baseline_scr - results_df['SCR_Ratio']) * 0.01  # 스케일 조정
        daily_hedge_cost = results_df['Hedge_Cost']
        daily_efficiency = daily_capital_savings - daily_hedge_cost
        
        # 누적 수익률
        cumulative_returns = (1 + daily_efficiency).cumprod()
        
        # CAGR (연환산 수익률)
        total_return = cumulative_returns.iloc[-1] - 1
        cagr = (1 + total_return) ** (252 / n_days) - 1
        
        # Volatility (연환산)
        volatility = daily_efficiency.std() * np.sqrt(252)
        
        # Sharpe Ratio (무위험 이자율 3% 가정)
        # 0으로 나누기 방지
        risk_free_daily = 0.03 / 252
        excess_return = daily_efficiency.mean() - risk_free_daily
        if volatility > 0.0001:  # 충분한 변동성이 있을 때만 계산
            sharpe = excess_return / (daily_efficiency.std() + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0  # 변동성이 거의 없으면 Sharpe = 0
        
        # MDD (최대 낙폭)
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / (peak + 1e-8)
        mdd = drawdown.min()
        
        # RCR (비용 대비 위험 감축 효율) - 핵심 지표
        avg_scr = results_df['SCR_Ratio'].mean()
        hedge_cost_total = results_df['Hedge_Cost'].sum() * initial_capital
        saved_capital = (baseline_scr - avg_scr) * initial_capital
        rcr = saved_capital / (hedge_cost_total + 1) if hedge_cost_total > 0 else 0
        
        # 총 비용 절감액 (백만원 단위)
        net_benefit = saved_capital - hedge_cost_total
        
        return {
            'CAGR': cagr,
            'Volatility': volatility,
            'Sharpe': sharpe,
            'MDD': mdd,
            'Avg_SCR': avg_scr,
            'Total_Hedge_Cost': hedge_cost_total / 1e8,  # 억원 단위
            'Net_Benefit': net_benefit / 1e8,  # 억원 단위
            'RCR': rcr
        }


# ==========================================
# 5. Report Generator
# ==========================================

def run_full_analysis():
    print("=" * 60)
    print("Phase 4: Backtesting & Performance Analysis")
    print("=" * 60)
    
    engine = BacktestEngine()
    analyzer = PerformanceAnalyzer()
    
    scenarios = ['normal', '2008_crisis', '2020_pandemic', 'stagflation', 'correlation_breakdown']
    
    all_metrics = []
    
    for scenario in scenarios:
        print(f"\n[Scenario: {scenario.upper()}]")
        market_data = generate_market_scenario(500, scenario)
        results = engine.run_all_strategies(market_data)
        
        for strategy_name, df in results.items():
            metrics = analyzer.calculate_metrics(df)
            metrics['Scenario'] = scenario
            metrics['Strategy'] = strategy_name
            all_metrics.append(metrics)
            
    # Summary Table
    summary_df = pd.DataFrame(all_metrics)
    
    print("\n" + "=" * 60)
    print("Performance Summary (All Scenarios)")
    print("=" * 60)
    
    # Pivot by Strategy
    pivot = summary_df.groupby('Strategy').agg({
        'CAGR': 'mean',
        'Sharpe': 'mean',
        'MDD': 'mean',
        'RCR': 'mean',
        'Avg_SCR': 'mean',
        'Net_Benefit': 'mean'
    }).round(4)
    
    print(pivot.to_string())
    
    # Winner Analysis
    print("\n[Winner by Metric]")
    print(f"  Best CAGR:   {pivot['CAGR'].idxmax()} ({pivot['CAGR'].max():.4f})")
    print(f"  Best Sharpe: {pivot['Sharpe'].idxmax()} ({pivot['Sharpe'].max():.4f})")
    print(f"  Best MDD:    {pivot['MDD'].idxmax()} ({pivot['MDD'].max():.4f})")  # least negative
    print(f"  Best RCR:    {pivot['RCR'].idxmax()} ({pivot['RCR'].max():.4f})")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    strategies = pivot.index.tolist()
    x = np.arange(len(strategies))
    
    # CAGR
    axes[0, 0].bar(x, pivot['CAGR'] * 100, color=['gray', 'blue', 'orange', 'green'])
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(strategies, rotation=15)
    axes[0, 0].set_title('Average CAGR (%)')
    axes[0, 0].axhline(0, color='black', lw=0.5)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Sharpe
    axes[0, 1].bar(x, pivot['Sharpe'], color=['gray', 'blue', 'orange', 'green'])
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(strategies, rotation=15)
    axes[0, 1].set_title('Average Sharpe Ratio')
    axes[0, 1].axhline(0, color='black', lw=0.5)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # MDD
    axes[1, 0].bar(x, pivot['MDD'] * 100, color=['gray', 'blue', 'orange', 'green'])
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(strategies, rotation=15)
    axes[1, 0].set_title('Average MDD (%)')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # RCR (핵심 지표)
    axes[1, 1].bar(x, pivot['RCR'], color=['gray', 'blue', 'orange', 'green'])
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(strategies, rotation=15)
    axes[1, 1].set_title('RCR (Risk-Cost Ratio) - KEY METRIC')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Phase 4: Strategy Comparison', y=1.02, fontsize=14, fontweight='bold')
    plt.show()
    
    return summary_df


if __name__ == "__main__":
    summary = run_full_analysis()
