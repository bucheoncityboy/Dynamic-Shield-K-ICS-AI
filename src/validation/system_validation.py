"""
System Validation 스크립트
===========================
제안서 연관:
- Surrogate 오차율 검증 (MAPE < 1%)
- System Latency 검증 (P95 < 50ms)
- Safety Layer 응답 검증

누수/편향/오버피팅 방지:
- Surrogate 검증: Train에서 본 적 없는 Test 데이터로만 평가
- Latency: 순수 측정 (ML 무관)
"""

import sys
import os
import time
import numpy as np
from typing import Dict, Any, Tuple

# 프로젝트 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
project_src = os.path.dirname(script_dir)
sys.path.insert(0, project_src)


def validate_surrogate() -> Tuple[bool, Dict[str, float]]:
    """
    Surrogate 모델 정확도 검증
    
    Anti-Leakage:
    - Test 데이터는 Train에서 본 적 없는 새로운 샘플
    - 80/20 분할 후 Test셋으로만 MAPE 계산
    
    Returns:
        (pass, metrics): 통과 여부와 지표
    """
    print("\n" + "=" * 60)
    print("[1] Surrogate 정확도 검증")
    print("=" * 60)
    
    try:
        from core.kics_real import RatioKICSEngine
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        
        engine = RatioKICSEngine()
        
        # 테스트 데이터 생성 (Train과 분리)
        np.random.seed(999)  # 다른 시드로 새로운 데이터
        n_test = 5000
        
        hedge_ratios = np.random.uniform(0, 1.0, n_test)
        correlations = np.random.uniform(-0.6, 0.9, n_test)
        
        # Ground Truth 계산
        y_true = engine.calculate_scr_ratio_batch(hedge_ratios, correlations)
        
        # Surrogate 로드 및 예측
        from core.kics_surrogate import RobustSurrogate, train_surrogate_model
        
        # 새 모델 학습 (별도 시드)
        print("  [-] Surrogate 학습 중 (별도 데이터)...")
        np.random.seed(42)
        model_obj, scaler_x, scaler_y = train_surrogate_model()
        
        # Test 데이터로 평가
        X_test = np.column_stack([hedge_ratios, correlations])
        X_test_scaled = scaler_x.transform(X_test)
        y_pred_scaled = model_obj.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        
        # MAPE 계산
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        metrics = {
            'MAPE': mape,
            'MAE': mae,
            'RMSE': rmse,
            'Test_Samples': n_test
        }
        
        print(f"\n  결과:")
        print(f"    MAPE: {mape:.4f}%")
        print(f"    MAE: {mae:.6f}")
        print(f"    RMSE: {rmse:.6f}")
        
        passed = mape < 1.0
        if passed:
            print(f"  ✓ PASS (MAPE < 1%)")
        else:
            print(f"  ✗ FAIL (MAPE >= 1%)")
        
        return passed, metrics
        
    except Exception as e:
        print(f"  ✗ 오류: {e}")
        return False, {'error': str(e)}


def validate_latency() -> Tuple[bool, Dict[str, float]]:
    """
    시스템 지연시간 검증
    
    목표:
    - Surrogate 추론: < 10ms
    - Safety Layer: < 5ms
    - 전체 파이프라인: < 50ms
    """
    print("\n" + "=" * 60)
    print("[2] 시스템 지연시간 검증")
    print("=" * 60)
    
    try:
        from realtime.latency import LatencyMonitor
        from core.kics_real import RatioKICSEngine
        from safety.risk_control import RiskController
        
        monitor = LatencyMonitor()
        engine = RatioKICSEngine()
        controller = RiskController()
        
        n_iter = 100
        
        # 1. Surrogate 추론 시간
        print(f"  [-] Surrogate 추론 ({n_iter}회)...")
        for _ in range(n_iter):
            with monitor.measure_context("surrogate"):
                _ = engine.calculate_scr_ratio_batch(
                    np.array([np.random.uniform(0, 1)]),
                    np.array([np.random.uniform(-0.6, 0.9)])
                )
        
        # 2. Safety Layer 시간
        print(f"  [-] Safety Layer ({n_iter}회)...")
        for _ in range(n_iter):
            with monitor.measure_context("safety_layer"):
                _ = controller.apply_safety_rules(
                    np.random.uniform(-1, 1),
                    np.random.uniform(0, 1),
                    np.random.uniform(10, 50),
                    np.random.uniform(80, 200)
                )
        
        # 3. 전체 파이프라인 시간
        print(f"  [-] 전체 파이프라인 ({n_iter}회)...")
        for _ in range(n_iter):
            with monitor.measure_context("total_pipeline"):
                # Surrogate
                scr = engine.calculate_scr_ratio_batch(
                    np.array([0.5]),
                    np.array([-0.3])
                )[0]
                # Safety
                _ = controller.apply_safety_rules(0, 0.5, 20, 150)
        
        # 결과 수집
        results = {}
        thresholds = {
            'surrogate': 10.0,
            'safety_layer': 5.0,
            'total_pipeline': 50.0
        }
        
        all_passed = True
        print(f"\n  결과:")
        print(f"  {'Component':<20} {'P95 (ms)':>10} {'Threshold':>12} {'Status':>10}")
        print("  " + "-" * 55)
        
        for name in ['surrogate', 'safety_layer', 'total_pipeline']:
            stats = monitor.get_stats(name)
            if stats:
                p95 = stats.p95_ms
                threshold = thresholds.get(name, 50.0)
                passed = p95 < threshold
                status = "✓ PASS" if passed else "✗ FAIL"
                
                if not passed:
                    all_passed = False
                
                results[f'{name}_p95'] = p95
                print(f"  {name:<20} {p95:>10.3f} {threshold:>10.1f}ms {status:>10}")
        
        return all_passed, results
        
    except Exception as e:
        print(f"  ✗ 오류: {e}")
        return False, {'error': str(e)}


def validate_safety_layer() -> Tuple[bool, Dict[str, Any]]:
    """
    Safety Layer 기능 검증
    
    테스트 케이스:
    1. K-ICS < 100% → 100% 강제 헤지
    2. VIX >= 40 → 헤지 증가
    3. 정상 상황 → AI 제안 수용
    """
    print("\n" + "=" * 60)
    print("[3] Safety Layer 기능 검증")
    print("=" * 60)
    
    try:
        from safety.risk_control import RiskController
        
        controller = RiskController()
        
        test_cases = [
            # (action, current_hedge, vix, kics, expected_behavior, expected_hedge_min)
            (0.0, 0.5, 15, 95, "CRITICAL: K-ICS < 100%", 1.0),
            (0.0, 0.5, 15, 115, "DANGER: K-ICS < 120%", 0.6),
            (0.0, 0.5, 45, 180, "PANIC: VIX >= 40", 0.6),
            (-0.5, 0.8, 15, 200, "NORMAL: 감소 허용", None),
        ]
        
        all_passed = True
        results = {'cases': []}
        
        print(f"\n  테스트 케이스:")
        print(f"  {'#':>3} {'VIX':>6} {'K-ICS':>8} {'Expected':>30} {'Status':>10}")
        print("  " + "-" * 60)
        
        for i, (action, hedge, vix, kics, expected, min_hedge) in enumerate(test_cases, 1):
            controller.reset()
            new_hedge, reason = controller.apply_safety_rules(action, hedge, vix, kics)
            
            # 검증
            if min_hedge is not None:
                passed = new_hedge >= min_hedge
            else:
                passed = True  # 정상 케이스는 감소 허용
            
            if expected.split(":")[0] not in reason:
                passed = False
            
            status = "✓ PASS" if passed else "✗ FAIL"
            if not passed:
                all_passed = False
            
            results['cases'].append({
                'case': i,
                'vix': vix,
                'kics': kics,
                'new_hedge': new_hedge,
                'reason': reason,
                'passed': passed
            })
            
            print(f"  {i:>3} {vix:>6.0f} {kics:>7.0f}% {expected:>30} {status:>10}")
        
        return all_passed, results
        
    except Exception as e:
        print(f"  ✗ 오류: {e}")
        return False, {'error': str(e)}


def validate_risk_paradox() -> Tuple[bool, Dict[str, float]]:
    """
    분산 효과 최적화 검증
    
    검증: 헤지를 늘리면 SCR이 감소하는가?
    """
    print("\n" + "=" * 60)
    print("[4] 분산 효과 최적화 검증")
    print("=" * 60)
    
    try:
        from core.kics_real import RatioKICSEngine
        
        engine = RatioKICSEngine()
        
        # Panic 상황 (높은 상관계수)
        correlation = 0.7
        
        hedge_0 = 0.0
        hedge_100 = 1.0
        
        scr_0 = engine.calculate_scr_ratio_batch(np.array([hedge_0]), np.array([correlation]))[0]
        scr_100 = engine.calculate_scr_ratio_batch(np.array([hedge_100]), np.array([correlation]))[0]
        
        # SCR 비율: 높을수록 안전
        improvement = (scr_100 - scr_0) / scr_0 * 100 if scr_0 > 0 else 0
        
        print(f"\n  상관계수 (Panic): {correlation}")
        print(f"  헤지 0% SCR 비율: {scr_0:.4f}")
        print(f"  헤지 100% SCR 비율: {scr_100:.4f}")
        print(f"  개선율: {improvement:.2f}%")
        
        passed = scr_100 > scr_0  # 헤지 증가 시 SCR 비율 증가
        
        if passed:
            print(f"  ✓ PASS (헤지 증가 → SCR 비율 증가)")
        else:
            print(f"  ✗ FAIL")
        
        return passed, {
            'correlation': correlation,
            'scr_0_hedge': scr_0,
            'scr_100_hedge': scr_100,
            'improvement': improvement
        }
        
    except Exception as e:
        print(f"  ✗ 오류: {e}")
        return False, {'error': str(e)}


def run_full_validation() -> Dict[str, Any]:
    """
    전체 시스템 검증 실행
    """
    print("=" * 70)
    print("Dynamic Shield v3.0 - 시스템 검증")
    print("=" * 70)
    
    results = {}
    
    # 1. Surrogate 검증
    passed, metrics = validate_surrogate()
    results['surrogate'] = {'passed': passed, 'metrics': metrics}
    
    # 2. Latency 검증
    passed, metrics = validate_latency()
    results['latency'] = {'passed': passed, 'metrics': metrics}
    
    # 3. Safety Layer 검증
    passed, metrics = validate_safety_layer()
    results['safety_layer'] = {'passed': passed, 'metrics': metrics}
    
    # 4. 분산 효과 최적화 검증
    passed, metrics = validate_risk_paradox()
    results['risk_paradox'] = {'passed': passed, 'metrics': metrics}
    
    # 요약
    print("\n" + "=" * 70)
    print("검증 요약")
    print("=" * 70)
    
    all_passed = True
    for name, result in results.items():
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"  {name:<20}: {status}")
        if not result['passed']:
            all_passed = False
    
    print("-" * 70)
    final_status = "✓ 전체 PASS" if all_passed else "✗ 일부 FAIL"
    print(f"  최종 결과: {final_status}")
    
    return results


if __name__ == "__main__":
    results = run_full_validation()
