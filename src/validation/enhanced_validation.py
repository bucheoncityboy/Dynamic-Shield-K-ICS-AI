"""
강화된 검증론 프레임워크
=======================
Walk-Forward, Bootstrap, RCR 등 고급 검증 기법 적용

핵심 철학: Capital Optimization, not Prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kics_real import RatioKICSEngine
from core.realistic_data import load_real_data_for_testing


class EnhancedValidator:
    """
    강화된 검증론 적용 클래스
    """
    
    def __init__(self):
        self.engine = RatioKICSEngine()
        
    def walk_forward_validation(self, 
                                 data, 
                                 window_sizes=[252, 504, 756],  # 1년, 2년, 3년
                                 step_sizes=[63, 126]):  # 3개월, 6개월
        """
        Walk-Forward Backtesting (롤링 윈도우)
        
        Args:
            data: 시계열 데이터
            window_sizes: 학습 윈도우 크기 (일수)
            step_sizes: 윈도우 이동 간격 (일수)
        
        Returns:
            results: 각 윈도우별 성과 지표
        """
        print("=" * 60)
        print("Walk-Forward Validation")
        print("=" * 60)
        
        results = []
        
        for window_size in window_sizes:
            for step_size in step_sizes:
                print(f"\n[윈도우: {window_size}일, 스텝: {step_size}일]")
                
                n_windows = (len(data) - window_size) // step_size
                
                window_metrics = []
                
                for i in range(n_windows):
                    train_start = i * step_size
                    train_end = train_start + window_size
                    test_start = train_end
                    test_end = min(test_start + step_size, len(data))
                    
                    if test_end - test_start < 10:  # 최소 테스트 구간
                        continue
                    
                    # Train 구간 (학습용)
                    train_data = data.iloc[train_start:train_end]
                    
                    # Test 구간 (검증용)
                    test_data = data.iloc[test_start:test_end]
                    
                    # 간단한 성과 계산 (예시)
                    # 실제로는 여기서 모델 학습 및 평가
                    train_scr_mean = self._calculate_avg_scr(train_data)
                    test_scr_mean = self._calculate_avg_scr(test_data)
                    
                    window_metrics.append({
                        'window': window_size,
                        'step': step_size,
                        'train_scr': train_scr_mean,
                        'test_scr': test_scr_mean,
                        'window_idx': i
                    })
                
                if window_metrics:
                    avg_test_scr = np.mean([m['test_scr'] for m in window_metrics])
                    std_test_scr = np.std([m['test_scr'] for m in window_metrics])
                    
                    results.append({
                        'window_size': window_size,
                        'step_size': step_size,
                        'avg_test_scr': avg_test_scr,
                        'std_test_scr': std_test_scr,
                        'n_windows': len(window_metrics)
                    })
                    
                    print(f"  평균 Test SCR: {avg_test_scr:.4f} ± {std_test_scr:.4f}")
        
        return pd.DataFrame(results)
    
    def bootstrap_validation(self, 
                            returns_series, 
                            kics_ratios,
                            n_bootstrap=1000,
                            confidence_level=0.95):
        """
        Bootstrap 검증 (블록 부트스트랩)
        
        시계열 데이터의 블록 부트스트랩을 통해
        CAGR, MDD, K-ICS 최소값의 분포를 추정
        
        Args:
            returns_series: 수익률 시계열
            kics_ratios: K-ICS 비율 시계열
            n_bootstrap: 부트스트랩 반복 횟수
            confidence_level: 신뢰구간 수준
        
        Returns:
            bootstrap_stats: 부트스트랩 통계량
        """
        print("=" * 60)
        print("Bootstrap Validation")
        print("=" * 60)
        
        # 블록 크기 (예: 20일)
        block_size = 20
        n_blocks = len(returns_series) // block_size
        
        bootstrap_cagr = []
        bootstrap_mdd = []
        bootstrap_min_kics = []
        
        for _ in range(n_bootstrap):
            # 블록 부트스트랩 샘플링
            sampled_returns = []
            sampled_kics = []
            
            for _ in range(n_blocks):
                block_idx = np.random.randint(0, n_blocks)
                start_idx = block_idx * block_size
                end_idx = start_idx + block_size
                
                sampled_returns.extend(returns_series[start_idx:end_idx].values)
                sampled_kics.extend(kics_ratios[start_idx:end_idx].values)
            
            # 지표 계산
            sampled_returns = np.array(sampled_returns)
            sampled_kics = np.array(sampled_kics)
            
            # CAGR 계산
            total_return = np.prod(1 + sampled_returns) - 1
            n_years = len(sampled_returns) / 252
            cagr = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
            bootstrap_cagr.append(cagr)
            
            # MDD 계산
            cumulative = np.cumprod(1 + sampled_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            mdd = np.min(drawdown)
            bootstrap_mdd.append(mdd)
            
            # 최소 K-ICS 비율
            min_kics = np.min(sampled_kics)
            bootstrap_min_kics.append(min_kics)
        
        # 통계량 계산
        alpha = 1 - confidence_level
        results = {
            'cagr': {
                'mean': np.mean(bootstrap_cagr),
                'std': np.std(bootstrap_cagr),
                'ci_lower': np.percentile(bootstrap_cagr, 100 * alpha/2),
                'ci_upper': np.percentile(bootstrap_cagr, 100 * (1 - alpha/2))
            },
            'mdd': {
                'mean': np.mean(bootstrap_mdd),
                'std': np.std(bootstrap_mdd),
                'ci_lower': np.percentile(bootstrap_mdd, 100 * alpha/2),
                'ci_upper': np.percentile(bootstrap_mdd, 100 * (1 - alpha/2))
            },
            'min_kics': {
                'mean': np.mean(bootstrap_min_kics),
                'std': np.std(bootstrap_min_kics),
                'ci_lower': np.percentile(bootstrap_min_kics, 100 * alpha/2),
                'ci_upper': np.percentile(bootstrap_min_kics, 100 * (1 - alpha/2))
            }
        }
        
        print(f"\nCAGR: {results['cagr']['mean']:.4f} "
              f"[{results['cagr']['ci_lower']:.4f}, {results['cagr']['ci_upper']:.4f}]")
        print(f"MDD: {results['mdd']['mean']:.4f} "
              f"[{results['mdd']['ci_lower']:.4f}, {results['mdd']['ci_upper']:.4f}]")
        print(f"Min K-ICS: {results['min_kics']['mean']:.2f}% "
              f"[{results['min_kics']['ci_lower']:.2f}%, {results['min_kics']['ci_upper']:.2f}%]")
        
        return results
    
    def calculate_rcr_metrics(self, 
                             hedge_costs, 
                             kics_improvements,
                             baseline_kics=150.0):
        """
        RCR (Risk-Cost Ratio) 지표 계산
        
        RCR = (K-ICS 개선량) / (헤지 비용)
        -> 1원의 비용으로 얼마의 자본을 절감했는지
        
        Args:
            hedge_costs: 헤지 비용 시계열
            kics_improvements: K-ICS 개선량 시계열 (Baseline 대비)
            baseline_kics: 기준 K-ICS 비율
        
        Returns:
            rcr_stats: RCR 통계량
        """
        print("=" * 60)
        print("RCR (Risk-Cost Ratio) Analysis")
        print("=" * 60)
        
        # RCR 계산 (비용이 0인 경우 제외)
        valid_mask = hedge_costs > 0
        rcr_values = kics_improvements[valid_mask] / hedge_costs[valid_mask]
        
        results = {
            'mean_rcr': np.mean(rcr_values),
            'median_rcr': np.median(rcr_values),
            'std_rcr': np.std(rcr_values),
            'min_rcr': np.min(rcr_values),
            'max_rcr': np.max(rcr_values),
            'n_valid': len(rcr_values)
        }
        
        print(f"\n평균 RCR: {results['mean_rcr']:.4f}")
        print(f"중앙값 RCR: {results['median_rcr']:.4f}")
        print(f"표준편차: {results['std_rcr']:.4f}")
        print(f"범위: [{results['min_rcr']:.4f}, {results['max_rcr']:.4f}]")
        
        return results
    
    def kics_quantile_analysis(self, kics_ratios, quantiles=[0.01, 0.05, 0.10, 0.25]):
        """
        K-ICS 비율의 하위 Quantile 분석
        
        위험 구간(100% 미만)에서의 방어력 확인
        
        Args:
            kics_ratios: K-ICS 비율 시계열
            quantiles: 분석할 분위수
        
        Returns:
            quantile_stats: 분위수별 통계량
        """
        print("=" * 60)
        print("K-ICS Quantile Analysis")
        print("=" * 60)
        
        results = {}
        
        for q in quantiles:
            quantile_value = np.percentile(kics_ratios, 100 * q)
            below_100 = np.sum(kics_ratios < 100)
            below_quantile = np.sum(kics_ratios < quantile_value)
            
            results[q] = {
                'quantile_value': quantile_value,
                'below_100_count': below_100,
                'below_100_pct': below_100 / len(kics_ratios) * 100,
                'below_quantile_count': below_quantile,
                'below_quantile_pct': below_quantile / len(kics_ratios) * 100
            }
            
            print(f"\n{q*100:.0f}% 분위수: {quantile_value:.2f}%")
            print(f"  100% 미만: {below_100}일 ({results[q]['below_100_pct']:.2f}%)")
            print(f"  {quantile_value:.2f}% 미만: {below_quantile}일 ({results[q]['below_quantile_pct']:.2f}%)")
        
        return results
    
    def _calculate_avg_scr(self, data):
        """평균 SCR 계산 (간단 버전)"""
        # 실제로는 모델을 사용하지만, 여기서는 간단히
        if 'Correlation' in data.columns:
            hedge_ratios = np.ones(len(data)) * 0.8  # 예시
            correlations = data['Correlation'].values
            scr_values = self.engine.calculate_scr_ratio_batch(hedge_ratios, correlations)
            return np.mean(scr_values)
        return 0.35  # 기본값


def run_enhanced_validation():
    """강화된 검증 실행"""
    print("=" * 60)
    print("Enhanced Validation Framework")
    print("=" * 60)
    
    validator = EnhancedValidator()
    
    # 테스트 데이터 로드
    test_data = load_real_data_for_testing(n_days=1000)
    
    if len(test_data) < 100:
        print("[경고] 테스트 데이터가 부족합니다.")
        return
    
    # 1. Walk-Forward Validation
    print("\n[1] Walk-Forward Validation")
    wf_results = validator.walk_forward_validation(test_data)
    
    # 2. Bootstrap Validation (예시 데이터)
    print("\n[2] Bootstrap Validation")
    returns_example = np.random.normal(0.0005, 0.01, len(test_data))  # 예시
    kics_example = np.random.uniform(120, 180, len(test_data))  # 예시
    bootstrap_results = validator.bootstrap_validation(
        pd.Series(returns_example),
        pd.Series(kics_example)
    )
    
    # 3. RCR Metrics
    print("\n[3] RCR Metrics")
    hedge_costs_example = np.random.uniform(0.001, 0.005, len(test_data))
    kics_improvements_example = np.random.uniform(5, 20, len(test_data))
    rcr_results = validator.calculate_rcr_metrics(
        pd.Series(hedge_costs_example),
        pd.Series(kics_improvements_example)
    )
    
    # 4. K-ICS Quantile Analysis
    print("\n[4] K-ICS Quantile Analysis")
    quantile_results = validator.kics_quantile_analysis(
        pd.Series(kics_example)
    )
    
    print("\n" + "=" * 60)
    print("Enhanced Validation 완료!")
    print("=" * 60)
    
    return {
        'walk_forward': wf_results,
        'bootstrap': bootstrap_results,
        'rcr': rcr_results,
        'quantile': quantile_results
    }


if __name__ == "__main__":
    results = run_enhanced_validation()

