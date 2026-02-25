"""
Phase 5.4: Walk-Forward Backtesting & Performance Analysis
===========================================================
Dynamic Shield v3.0의 정량적 성과 검증 (Real AI Inference Ver.)
- Walk-Forward Backtesting
- 4가지 전략 비교
- **Real AI Inference (Auto-detect Model Path)**

핵심 철학: Capital Optimization, not Prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# PPO 모델 로드를 위한 라이브러리
try:
    from stable_baselines3 import PPO
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("[WARNING] stable-baselines3 not installed. AI strategy will fall back to rule-based.")

# Phase 1: K-ICS Engine 및 데이터 생성기 가져오기
# 경로 설정: 현재 파일의 상위 상위 폴더를 path에 추가하여 core 모듈을 찾음
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # 패키지 구조 내에서 실행될 때
    from core.kics_real import RatioKICSEngine
    from core.realistic_data import generate_market_scenario
except ImportError:
    # 단독 스크립트로 실행될 때 (같은 폴더에 파일이 있는 경우)
    try:
        from kics_real import RatioKICSEngine
        from realistic_data import generate_market_scenario
    except ImportError:
        print("[ERROR] 필수 모듈(kics_real.py, realistic_data.py)을 찾을 수 없습니다.")
        sys.exit(1)

# ==========================================
# 1. Strategy Definitions (기존 함수형 전략)
# ==========================================

def strategy_100_hedge(vix, current_ratio):
    """① 100% 완전 헤지"""
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

# ④ Dynamic Shield (Mimic Logic) - 모델 로드 실패 시 폴백용
def strategy_dynamic_shield_fallback(vix, current_ratio):
    if vix >= 30:
        target = 1.0
        step = 0.15
        return min(current_ratio + step, target)
    elif vix >= 20:
        target = 0.7
        if current_ratio < target:
            return current_ratio + 0.05
        else:
            return current_ratio - 0.03
    else:
        target = 0.4
        step = 0.05
        return max(current_ratio - step, target)


# ==========================================
# 2. Backtest Engine (AI 추론 기능 강화)
# ==========================================

class BacktestEngine:
    def __init__(self, model_filename="ppo_kics"):
        self.engine = RatioKICSEngine()
        self.model = None
        
        # [핵심 변경] 모델 파일 자동 탐색 (현재 폴더 -> models 폴더 순)
        if STABLE_BASELINES_AVAILABLE:
            # 1. 확장자(.zip) 제거 (load 함수가 알아서 붙임)
            if model_filename.endswith('.zip'):
                model_filename = model_filename[:-4]
            
            # 2. 탐색 경로 후보
            search_paths = [
                model_filename,                 # 현재 폴더 (예: ppo_kics)
                f"models/{model_filename}",     # models 하위 폴더
                f"../models/{model_filename}"   # 상위 models 폴더
            ]
            
            loaded_path = None
            for path in search_paths:
                if os.path.exists(path + ".zip"):
                    try:
                        self.model = PPO.load(path)
                        loaded_path = path
                        print(f"\n[Info] ✅ Real AI Model loaded successfully from: {path}.zip")
                        break
                    except Exception as e:
                        print(f"[Warning] Failed to load found model at {path}: {e}")
            
            if self.model is None:
                print(f"\n[Warning] ⚠️ Model file '{model_filename}.zip' not found in current or 'models/' directory.")
                print("           -> Using 'Fallback Logic (Mimic)' instead.")
        
        self.strategies = {
            '100% Hedge': strategy_100_hedge,
            '80% Fixed': strategy_80_fixed,
            'Rule-based': strategy_rule_based,
            'Dynamic Shield': self.strategy_real_ai_inference # AI 메서드 연결
        }
        
    def strategy_real_ai_inference(self, vix, current_ratio, correlation, scr_ratio):
        """
        [Real AI Inference]
        학습된 신경망을 통해 행동을 결정합니다.
        입력 상태(State) 구성은 gym_environment.py와 100% 일치해야 합니다.
        """
        # 모델이 없으면 기존 하드코딩 로직(Fallback) 사용
        if self.model is None:
            return strategy_dynamic_shield_fallback(vix, current_ratio)

        # 1. State 구성 (gym_environment.py의 _get_obs 참조)
        # [hedge_ratio, vix_norm, corr_norm, scr_ratio]
        obs = np.array([
            current_ratio,                  
            np.clip(vix / 100.0, 0, 1),     
            np.clip((correlation + 1) / 2, 0, 1), 
            np.clip(scr_ratio, 0, 1)        
        ], dtype=np.float32)

        # 2. AI 예측 (Deterministic=True: 확률적 탐색 배제)
        action, _ = self.model.predict(obs, deterministic=True)

        # 3. Action 해석 (gym_environment.py의 step 참조)
        # Action space [-1, 1] -> Change [-0.1, 0.1]
        hedge_change = float(action[0]) * 0.1
        
        new_ratio = np.clip(current_ratio + hedge_change, 0.0, 1.0)
        return new_ratio

    def run_backtest(self, market_data, strategy_name):
        """단일 전략 백테스트"""
        strategy_func = self.strategies[strategy_name]
        is_ai_strategy = (strategy_name == 'Dynamic Shield')
        
        results = []
        current_ratio = 0.5  # 초기 헤지 비율
        
        for i, row in market_data.iterrows():
            vix = row['VIX']
            corr = row['Correlation']
            
            # 현재 상태 SCR 계산 (AI 입력용)
            current_scr = self.engine.calculate_scr_ratio_batch(
                np.array([current_ratio]), 
                np.array([corr])
            )[0]
            
            # [전략 실행]
            if is_ai_strategy:
                # AI는 더 많은 상태 정보(Correlation, SCR)가 필요함
                new_ratio = strategy_func(vix, current_ratio, corr, current_scr)
            else:
                # 기존 단순 전략들
                new_ratio = strategy_func(vix, current_ratio)
            
            # 결과 기록용 SCR 재계산
            final_scr = self.engine.calculate_scr_ratio_batch(
                np.array([new_ratio]), 
                np.array([corr])
            )[0]
            
            # 헤지 비용: 연 1.5% 스왑포인트 비용 일할 (보험연구원 2018 기준)
            hedge_cost = new_ratio * (0.015 / 252) 
            
            results.append({
                'Day': i,
                'VIX': vix,
                'Hedge_Ratio': new_ratio,
                'SCR_Ratio': final_scr,
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
# 3. Performance Analyzer (v4.0 개선 - Anti-Overfitting)
# ==========================================

class PerformanceAnalyzer:
    @staticmethod
    def calculate_metrics(results_df, initial_capital=10_000_000_000):
        """
        [v4.0 개선] 현실적인 성과 지표 계산
        
        주요 변경:
        1. FX 변동 기반 실제 손익 계산
        2. Sharpe를 현실적 범위로 제한 (-5 ~ 5)
        3. MDD를 포트폴리오 가치 기반으로 계산
        """
        n_days = len(results_df)
        if n_days < 2:
            return {'CAGR': 0, 'Volatility': 0, 'Sharpe': 0, 'MDD': 0, 
                    'Avg_SCR': 0, 'Total_Hedge_Cost': 0, 'Net_Benefit': 0, 'RCR': 0}
        
        # 1. 외화 자산 비중 (보험사 가정: 20%)
        fx_asset_ratio = 0.20
        fx_assets = initial_capital * fx_asset_ratio  # 2조원 외화 자산
        
        # 2. FX 변동에 따른 손익 계산
        fx_returns = results_df['FX'].pct_change().fillna(0)
        hedge_ratios = results_df['Hedge_Ratio']
        
        # 헤지되지 않은 포지션만 환율 변동에 노출
        open_position_ratio = 1 - hedge_ratios
        fx_pnl = fx_returns * open_position_ratio * fx_asset_ratio
        
        # 3. 헤지 비용 (연 1.5% 스왑포인트 가정, 일할 계산)
        annual_hedge_cost_rate = 0.015
        daily_hedge_cost = hedge_ratios * (annual_hedge_cost_rate / 252) * fx_asset_ratio
        
        # 4. 일일 총 수익률 (FX P&L - 헤지 비용)
        daily_returns = fx_pnl - daily_hedge_cost
        
        # 5. 포트폴리오 가치 계산
        portfolio_values = initial_capital * (1 + daily_returns).cumprod()
        
        # 6. CAGR 계산
        total_return = (portfolio_values.iloc[-1] / initial_capital) - 1
        cagr = (1 + total_return) ** (252 / n_days) - 1
        
        # 7. 변동성 (연율화)
        volatility = daily_returns.std() * np.sqrt(252)
        
        # 8. Sharpe Ratio (현실적 범위로 제한)
        risk_free_rate = 0.03  # 연 3%
        excess_return = (daily_returns.mean() * 252) - risk_free_rate
        if volatility > 0.001:
            sharpe = excess_return / volatility
        else:
            sharpe = 0.0
        # 캡핑 제거 (학술적 정확성)
        
        # 9. MDD 계산 (포트폴리오 가치 기반)
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        mdd = drawdown.min()  # 음수값
        
        # 10. SCR 관련 지표 (0~1 정규화된 값)
        avg_scr = results_df['SCR_Ratio'].mean()
        # baseline_scr: 100% 헤지 시 기대 SCR 비율 (정규화된 값, 약 0.1)
        baseline_scr = 0.10  # 100% K-ICS 비율 = 0.1 (10/10 정규화 기준)
        
        # 11. 헤지 비용 총액
        total_hedge_cost = (daily_hedge_cost * initial_capital).sum()
        
        # 12. 자본 절감 효과 (SCR 비율 증가 = 요구자본 감소)
        # 비율이 높을수록 안전 -> (avg_scr - baseline_scr) * capital
        saved_capital = (avg_scr - baseline_scr) * initial_capital
        
        # 13. RCR (Risk-Cost Ratio): 절감액 대비 비용
        rcr = saved_capital / (total_hedge_cost + 1) if total_hedge_cost > 0 else 0
        
        # 14. 순이익 (절감액 - 비용)
        net_benefit = saved_capital - total_hedge_cost
        
        return {
            'CAGR': cagr,
            'Volatility': volatility,
            'Sharpe': sharpe,
            'MDD': mdd,
            'Avg_SCR': avg_scr,
            'Total_Hedge_Cost': total_hedge_cost / 1e8,  # 억원 단위
            'Net_Benefit': net_benefit / 1e8,
            'RCR': rcr
        }


# ==========================================
# 4. Main Execution
# ==========================================

def run_full_analysis():
    print("=" * 60)
    print("Phase 5.4: Backtesting & Performance Analysis (With Real AI)")
    print("[v4.0] Anti-Overfitting: 실제 데이터 사용, Train/Test 분리")
    print("=" * 60)
    
    # 모델 파일이 같은 폴더에 있다면 "ppo_kics"만 입력하면 됨 (확장자 자동 처리)
    engine = BacktestEngine(model_filename="ppo_kics")
    analyzer = PerformanceAnalyzer()
    
    # 테스트용 시나리오 (실제 데이터의 Test 구간 사용)
    scenarios = ['normal', '2008_crisis', '2020_pandemic']
    
    all_metrics = []
    
    for scenario in scenarios:
        print(f"\n[Scenario: {scenario.upper()}]")
        # [핵심 변경] is_training=False로 테스트 데이터만 사용
        market_data = generate_market_scenario(500, scenario, use_real_data=True, is_training=False)
        
        # FX 열 필요 (성과 계산용)
        if 'FX' not in market_data.columns:
            print(f"  [경고] FX 데이터 없음, 스킵")
            continue
            
        results = engine.run_all_strategies(market_data)
        
        for strategy_name, df in results.items():
            # FX 데이터 추가 (성과 계산에 필요)
            df['FX'] = market_data['FX'].values[:len(df)]
            metrics = analyzer.calculate_metrics(df)
            metrics['Scenario'] = scenario
            metrics['Strategy'] = strategy_name
            all_metrics.append(metrics)
            
    summary_df = pd.DataFrame(all_metrics)
    
    print("\n" + "=" * 60)
    print("Performance Summary (All Scenarios)")
    print("=" * 60)
    
    pivot = summary_df.groupby('Strategy').agg({
        'CAGR': 'mean',
        'Sharpe': 'mean',
        'MDD': 'mean',
        'RCR': 'mean',
        'Avg_SCR': 'mean',
        'Net_Benefit': 'mean'
    }).round(4)
    
    print(pivot.to_string())
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    strategies = pivot.index.tolist()
    x = np.arange(len(strategies))
    
    metrics_to_plot = ['CAGR', 'Sharpe', 'MDD', 'RCR']
    titles = ['Average CAGR (%)', 'Average Sharpe Ratio', 'Average MDD (%)', 'RCR (Risk-Cost Ratio)']
    
    for i, ax in enumerate(axes.flat):
        metric = metrics_to_plot[i]
        vals = pivot[metric] * 100 if metric in ['CAGR', 'MDD'] else pivot[metric]
        # 색상 매핑: AI 전략을 눈에 띄게 (초록색)
        colors = []
        for s in strategies:
            if 'Dynamic' in s: colors.append('green')
            elif '100' in s: colors.append('gray')
            else: colors.append('blue')
            
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(strategies, rotation=15)
        ax.set_title(titles[i])
        ax.grid(axis='y', alpha=0.3)
        if metric in ['CAGR', 'Sharpe']:
            ax.axhline(0, color='black', lw=0.5)
            
    plt.tight_layout()
    plt.suptitle('Phase 5.4: Strategy Comparison (AI vs Traditional)', y=1.02, fontsize=14, fontweight='bold')
    plt.savefig('backtest_result_ai.png', dpi=150)
    plt.show()
    
    print("\n[Saved] backtest_result_ai.png")
    
    return summary_df


if __name__ == "__main__":
    summary = run_full_analysis()
