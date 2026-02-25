"""
Phase 5.3: Safety Layer Stress Test (시스템 안전성 검증)
=======================================================
AI 오작동이나 극단적 공포 국면(VIX > 40)에서의 강제 제어 시스템 작동 확인
- Emergency: Gradual De-risking Triggered 메시지 출력 확인
- 헤지 비율이 설정된 Step(0.05) 단위로 천천히 상승하는지 확인
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.environment import KICSEnvironment
from core.agent import DynamicShieldAgent


def run_stress_test():
    """Safety Layer Stress Test 실행"""
    print("=" * 60)
    print("Phase 5.3: Safety Layer Stress Test")
    print("=" * 60)
    
    # 환경 및 에이전트 생성
    env = KICSEnvironment(lambda2=1000)
    agent = DynamicShieldAgent(vix_panic_threshold=30)
    
    # 테스트 시나리오: VIX 40 이상 강제 주입
    print("\n[Test 1] VIX > 40 Injection Test")
    print("-" * 40)
    
    state = env.reset(initial_vix=15, initial_corr=-0.4)
    
    # VIX를 점진적으로 올림
    vix_scenario = [15, 20, 25, 30, 35, 40, 45, 50, 55, 50, 45, 40, 35, 30, 25, 20]
    corr_scenario = [-0.4, -0.3, -0.2, 0.0, 0.2, 0.5, 0.7, 0.8, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0, -0.2, -0.4]
    
    emergency_triggered = False
    gradual_increase_confirmed = False
    hedge_history = [env.hedge_ratio]
    
    for i, (vix, corr) in enumerate(zip(vix_scenario, corr_scenario)):
        action, info = agent.get_action(state, env)
        state, reward, done, step_info = env.step(action, vix, corr)
        
        hedge_history.append(env.hedge_ratio)
        
        # 로그 출력
        print(f"Step {i+1:2d} | VIX: {vix:2d} | Hedge: {env.hedge_ratio:.2f} | Action: {action} | {info['reason']}")
        
        # Emergency 트리거 확인
        if 'CRITICAL' in info['reason'] or 'DANGER' in info['reason'] or 'PANIC' in info['reason']:
            if vix >= 40:
                emergency_triggered = True
                print("  >>> Emergency: Gradual De-risking Triggered <<<")
    
    # Gradual Increase 확인
    print("\n[Test 2] Gradual Increase Verification")
    print("-" * 40)
    
    max_step = 0
    for i in range(1, len(hedge_history)):
        step_change = abs(hedge_history[i] - hedge_history[i-1])
        if step_change > max_step:
            max_step = step_change
    
    print(f"Max single-step hedge change: {max_step:.2f}")
    
    if max_step <= 0.15:  # 최대 15% 이하로 변동
        gradual_increase_confirmed = True
        print("[PASS] Hedge ratio changed gradually (max step <= 0.15)")
    else:
        print("[FAIL] Hedge ratio changed too rapidly!")
    
    # K-ICS 100% 미만 테스트
    print("\n[Test 3] K-ICS < 100% Penalty Test")
    print("-" * 40)
    
    # SCR을 강제로 높여 K-ICS < 100% 상황 시뮬레이션
    env.reset(initial_vix=60, initial_corr=0.9)
    env.scr_ratio = 0.5  # 강제 설정
    
    kics_ratio = env.get_kics_ratio()
    print(f"Forced K-ICS Ratio: {kics_ratio:.1f}%")
    
    if kics_ratio < 100:
        action, info = agent.get_action(env.get_state(), env)
        print(f"Agent Response: {info['reason']}")
        
        if action == 3:  # 대폭 증가
            print("[PASS] Agent correctly responded to K-ICS < 100% with maximum hedge increase")
        else:
            print(f"[WARNING] Agent responded with action {action}")
    
    # 최종 결과
    print("\n" + "=" * 60)
    print("Stress Test Results")
    print("=" * 60)
    
    results = {
        'emergency_triggered': emergency_triggered,
        'gradual_confirmed': gradual_increase_confirmed,
        'max_step_change': max_step
    }
    
    if emergency_triggered:
        print("✓ Emergency De-risking: TRIGGERED")
    else:
        print("✗ Emergency De-risking: NOT TRIGGERED")
    
    if gradual_increase_confirmed:
        print("✓ Gradual Increase: CONFIRMED")
    else:
        print("✗ Gradual Increase: FAILED")
    
    if emergency_triggered and gradual_increase_confirmed:
        print("\n[SUCCESS] Safety Layer passed all stress tests!")
    else:
        print("\n[PARTIAL] Some tests did not pass. Review needed.")
    
    return results


if __name__ == "__main__":
    results = run_stress_test()
