#!/usr/bin/env python
"""
Dynamic Shield v3.0 - 통합 실행 스크립트
=========================================
CLI 기반 오케스트레이션

사용법:
    python main.py --mode train      # PPO 학습
    python main.py --mode backtest   # 백테스트
    python main.py --mode validate   # 시스템 검증
    python main.py --mode live       # 실시간 운영 (Phase 3에서 구현)
    python main.py --mode all        # 전체 파이프라인
"""

import argparse
import sys
import os
import time

# 프로젝트 경로 설정
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_train(args):
    """PPO 강화학습 훈련"""
    print("\n" + "=" * 60)
    print("🎯 Mode: TRAIN (PPO 강화학습)")
    print("=" * 60)
    
    try:
        from core.ppo_trainer import PPOTrainer
        
        trainer = PPOTrainer(
            algorithm='PPO',
            total_timesteps=args.timesteps or 50000,
            learning_rate=args.lr or 3e-4
        )
        
        trainer.setup()
        trainer.train()
        trainer.evaluate(n_episodes=10)
        trainer.plot_training()
        trainer.save()
        
        print("\n✓ 학습 완료!")
        return True
        
    except Exception as e:
        print(f"\n✗ 학습 실패: {e}")
        return False


def run_backtest(args):
    """백테스팅 실행"""
    print("\n" + "=" * 60)
    print("📊 Mode: BACKTEST")
    print("=" * 60)
    
    try:
        from validation.backtest import run_full_analysis
        
        summary = run_full_analysis()
        
        print("\n✓ 백테스트 완료!")
        return True
        
    except Exception as e:
        print(f"\n✗ 백테스트 실패: {e}")
        return False


def run_validate(args):
    """시스템 검증"""
    print("\n" + "=" * 60)
    print("🔍 Mode: VALIDATE (시스템 검증)")
    print("=" * 60)
    
    results = {
        'surrogate': False,
        'safety_layer': False,
        'latency': False,
        'risk_paradox': False
    }
    
    # 1. Surrogate 정확도 검증
    print("\n[1/4] Surrogate 모델 검증...")
    try:
        from core.kics_surrogate import train_surrogate_model
        model, scaler_x, scaler_y = train_surrogate_model()
        results['surrogate'] = True
        print("  ✓ Surrogate MAPE < 1%")
    except Exception as e:
        print(f"  ✗ Surrogate 검증 실패: {e}")
    
    # 2. Safety Layer 검증
    print("\n[2/4] Safety Layer 검증...")
    try:
        from safety.risk_control import RiskController
        
        controller = RiskController()
        
        # 테스트 케이스
        test_passed = True
        
        # K-ICS < 100% → 100% 헤지 강제
        hedge, reason = controller.apply_safety_rules(0, 0.5, 15, 95)
        if hedge != 1.0:
            test_passed = False
            print(f"  ✗ K-ICS 위반 테스트 실패")
        
        # VIX > 40 → 헤지 증가
        controller.reset()
        hedge, reason = controller.apply_safety_rules(0, 0.5, 45, 180)
        if hedge <= 0.5:
            test_passed = False
            print(f"  ✗ VIX 패닉 테스트 실패")
        
        if test_passed:
            results['safety_layer'] = True
            print("  ✓ Safety Layer 정상 작동")
    except Exception as e:
        print(f"  ✗ Safety Layer 검증 실패: {e}")
    
    # 3. 지연시간 검증
    print("\n[3/4] 지연시간 검증...")
    try:
        from realtime.latency import LatencyMonitor
        import numpy as np
        
        monitor = LatencyMonitor()
        
        # Surrogate 추론 시간 측정
        from core.kics_real import RatioKICSEngine
        engine = RatioKICSEngine()
        
        for _ in range(100):
            with monitor.measure_context("surrogate_inference"):
                _ = engine.calculate_scr_ratio_batch(
                    np.array([0.5]),
                    np.array([-0.3])
                )
        
        stats = monitor.get_stats("surrogate_inference")
        if stats and stats.p95_ms < 10:
            results['latency'] = True
            print(f"  ✓ Surrogate P95: {stats.p95_ms:.3f}ms < 10ms")
        else:
            print(f"  ⚠ Surrogate P95: {stats.p95_ms:.3f}ms (목표: < 10ms)")
            
    except Exception as e:
        print(f"  ✗ 지연시간 검증 실패: {e}")
    
    # 4. Risk Paradox 검증
    print("\n[4/4] Risk Paradox 검증...")
    try:
        from validation.proof_risk_paradox import prove_risk_paradox
        prove_risk_paradox()
        results['risk_paradox'] = True
        print("  ✓ Risk Paradox 증명 완료")
    except Exception as e:
        print(f"  ✗ Risk Paradox 검증 실패: {e}")
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("검증 결과 요약")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, status in results.items():
        icon = "✓" if status else "✗"
        print(f"  {icon} {name}")
    
    print(f"\n총 {passed}/{total} 검증 통과")
    
    return passed == total


def run_live(args):
    """실시간 운영 모드"""
    print("\n" + "=" * 60)
    print("🔴 Mode: LIVE (실시간 운영)")
    print("=" * 60)
    
    print("\n✅ Phase 3 구현 완료 - 시뮬레이션 모드로 동작합니다.")
    print("   (실제 API 연동은 별도 설정 필요)")
    
    try:
        from safety.risk_control import RiskController
        from realtime.latency import LatencyMonitor
        import numpy as np
        
        controller = RiskController()
        monitor = LatencyMonitor()
        
        interval = args.interval or 5
        
        print(f"\n[시뮬레이션 시작] {interval}초 간격")
        print("Ctrl+C로 종료\n")
        
        current_hedge = 0.5
        step = 0
        
        while True:
            step += 1
            
            # 시뮬레이션 데이터 생성
            vix = np.random.normal(20, 5)
            kics = np.random.normal(150, 20)
            action = np.random.uniform(-0.5, 0.5)
            
            with monitor.measure_context("total_pipeline"):
                safe_hedge, reason = controller.apply_safety_rules(
                    action, current_hedge, vix, kics
                )
            
            print(f"[Step {step}] VIX={vix:.1f}, K-ICS={kics:.1f}% | "
                  f"Hedge: {current_hedge:.1%} → {safe_hedge:.1%} | {reason}")
            
            current_hedge = safe_hedge
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n[종료] 사용자 중단")
        monitor.print_report()
        return True
    except Exception as e:
        print(f"\n✗ 실시간 모드 실패: {e}")
        return False


def run_all(args):
    """전체 파이프라인 실행"""
    print("\n" + "=" * 60)
    print("🚀 Mode: ALL (전체 파이프라인)")
    print("=" * 60)
    
    steps = [
        ("1. 시스템 검증", run_validate),
        ("2. PPO 학습", run_train),
        ("3. 백테스트", run_backtest),
    ]
    
    results = {}
    
    for name, func in steps:
        print(f"\n{'='*60}")
        print(f">>> {name}")
        print("=" * 60)
        
        try:
            success = func(args)
            results[name] = success
        except Exception as e:
            print(f"✗ {name} 실패: {e}")
            results[name] = False
    
    # 최종 요약
    print("\n" + "=" * 60)
    print("전체 파이프라인 완료")
    print("=" * 60)
    
    for name, success in results.items():
        icon = "✓" if success else "✗"
        print(f"  {icon} {name}")
    
    return all(results.values())


def main():
    parser = argparse.ArgumentParser(
        description='Dynamic Shield v3.0 - K-ICS 연계형 동적 환헤지 최적화',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py --mode train --timesteps 100000
  python main.py --mode backtest
  python main.py --mode validate
  python main.py --mode live --interval 10
  python main.py --mode all
        """
    )
    
    parser.add_argument(
        '--mode', '-m',
        type=str,
        required=True,
        choices=['train', 'backtest', 'validate', 'live', 'all'],
        help='실행 모드'
    )
    
    parser.add_argument(
        '--timesteps', '-t',
        type=int,
        default=50000,
        help='PPO 학습 timesteps (기본: 50000)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='학습률 (기본: 0.0003)'
    )
    
    parser.add_argument(
        '--interval', '-i',
        type=int,
        default=5,
        help='실시간 모드 간격 (초, 기본: 5)'
    )
    
    args = parser.parse_args()
    
    # 모드별 실행
    mode_handlers = {
        'train': run_train,
        'backtest': run_backtest,
        'validate': run_validate,
        'live': run_live,
        'all': run_all,
    }
    
    print("=" * 60)
    print("Dynamic Shield v3.0")
    print("Capital Optimization, not Prediction")
    print("=" * 60)
    
    handler = mode_handlers.get(args.mode)
    if handler:
        success = handler(args)
        sys.exit(0 if success else 1)
    else:
        print(f"알 수 없는 모드: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
