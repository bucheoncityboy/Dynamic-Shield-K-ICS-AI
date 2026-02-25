"""
Phase 4.2: RL Agent with Safety Layer
======================================
Gradual De-risking 프로토콜이 내장된 에이전트
- AI가 패닉에 빠지거나 급발진하는 것을 막는 하드코딩 룰
- K-ICS 비율이 100% 미만이면 강제 헤지 증가
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from environment import KICSEnvironment
except ImportError:
    from .environment import KICSEnvironment


class DynamicShieldAgent:
    """
    Dynamic Shield 에이전트
    
    구조:
    1. Generator: 최적 액션 제안 (AI 또는 Rule-based)
    2. Safety Layer: 위험 상황 시 하드코딩 룰로 오버라이드
    """
    
    def __init__(self, 
                 vix_panic_threshold=None,      # 패닉 VIX 임계값
                 vix_transition_threshold=None, # 전환 VIX 임계값
                 kics_danger_threshold=None,   # K-ICS 위험 임계값 (%)
                 kics_critical_threshold=None, # K-ICS 치명적 임계값 (%)
                 max_hedge_change=None):      # 최대 1회 헤지 변동 (Gradual)
        
        # Config 로드 시도
        try:
            from config_loader import ConfigLoader
            loader = ConfigLoader()
            agent_config = loader.get_agent_config()
            
            # 설정 파일에서 기본값 로드 (인자로 전달된 값이 우선)
            self.vix_panic = vix_panic_threshold or agent_config.get('vix_panic_threshold', 30)
            self.vix_transition = vix_transition_threshold or agent_config.get('vix_transition_threshold', 20)
            self.kics_danger = kics_danger_threshold or agent_config.get('kics_danger_threshold', 120)
            self.kics_critical = kics_critical_threshold or agent_config.get('kics_critical_threshold', 100)
            self.max_change = max_hedge_change or agent_config.get('max_hedge_change', 0.15)
            self.min_hedge = agent_config.get('min_hedge', 0.3)
            self.max_hedge = agent_config.get('max_hedge', 1.0)
        except (ImportError, FileNotFoundError, KeyError):
            # 폴백: 기본값 사용
            self.vix_panic = vix_panic_threshold or 30
            self.vix_transition = vix_transition_threshold or 20
            self.kics_danger = kics_danger_threshold or 120
            self.kics_critical = kics_critical_threshold or 100
            self.max_change = max_hedge_change or 0.15
            self.min_hedge = 0.3
            self.max_hedge = 1.0
        
        # Gradual De-risking state
        self.is_derisking = False
        self.derisking_target = 1.0
        
    def get_action(self, state, env):
        """
        Safety Layer가 적용된 행동 결정
        
        Args:
            state: 환경 상태 [hedge_ratio, vix_norm, corr_norm, scr_ratio]
            env: 환경 객체 (K-ICS 비율 조회용)
        
        Returns:
            action: 0=감소, 1=유지, 2=증가, 3=대폭 증가
            info: 행동 결정 정보
        """
        hedge_ratio = state[0]
        vix = state[1] * 100  # 역정규화
        kics_ratio = env.get_kics_ratio()
        
        # =============================================
        # Safety Layer 1: K-ICS 위반 방지 (최우선!)
        # =============================================
        if kics_ratio < self.kics_critical:
            # 치명적 상황: 즉시 100% 헤지
            self.is_derisking = True
            self.derisking_target = 1.0
            return 3, {'reason': 'CRITICAL: K-ICS < 100%, FORCE HEDGE 100%'}
        
        elif kics_ratio < self.kics_danger:
            # 위험 상황: 단계적 헤지 증가
            self.is_derisking = True
            self.derisking_target = min(hedge_ratio + 0.1, 1.0)
            return 2, {'reason': f'DANGER: K-ICS={kics_ratio:.1f}%, Increasing Hedge'}
        
        # =============================================
        # Safety Layer 2: Gradual De-risking 진행 중
        # =============================================
        if self.is_derisking:
            if hedge_ratio >= self.derisking_target:
                # 목표 도달, De-risking 종료
                self.is_derisking = False
                return 1, {'reason': 'De-risking Complete'}
            else:
                # 계속 증가
                return 2, {'reason': 'Gradual De-risking in Progress'}
        
        # =============================================
        # Safety Layer 3: VIX 기반 Regime 대응
        # =============================================
        if vix >= self.vix_panic:
            # 패닉 국면: 헤지 증가 (단, 점진적으로)
            if hedge_ratio < 0.9:
                return 3, {'reason': f'PANIC: VIX={vix:.1f}, Rapid Hedge Increase'}
            else:
                return 1, {'reason': f'PANIC: VIX={vix:.1f}, Hedge Already High'}
        
        elif vix >= self.vix_transition:
            # 전환 국면: 헤지 유지 또는 소폭 증가
            if hedge_ratio < 0.7:
                return 2, {'reason': f'TRANSITION: VIX={vix:.1f}, Gradual Increase'}
            else:
                return 1, {'reason': f'TRANSITION: VIX={vix:.1f}, Maintain'}
        
        else:
            # 평온 국면: 헤지 비용 절감
            if hedge_ratio > 0.5:
                return 0, {'reason': f'NORMAL: VIX={vix:.1f}, Reducing Hedge Cost'}
            else:
                return 1, {'reason': f'NORMAL: VIX={vix:.1f}, Maintain Low Hedge'}
    
    def apply_hard_constraints(self, hedge_ratio, action):
        """
        Hard Constraints 적용 (헤지 비율 범위 제한)
        """
        if action == 0:
            new_ratio = hedge_ratio - 0.05
        elif action == 2:
            new_ratio = hedge_ratio + 0.05
        elif action == 3:
            new_ratio = hedge_ratio + 0.10
        else:
            new_ratio = hedge_ratio
        
        # Clamp to [min, max]
        return np.clip(new_ratio, self.min_hedge, self.max_hedge)


class RLTrainer:
    """
    RL 훈련/시뮬레이션 러너 (Simple Policy Gradient 방식)
    """
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.episode_rewards = []
        
    def run_episode(self, market_data):
        """
        시장 데이터를 기반으로 한 에피소드 실행
        """
        state = self.env.reset(
            initial_vix=market_data['VIX'].iloc[0],
            initial_corr=market_data['Correlation'].iloc[0]
        )
        
        episode_log = []
        
        for i in range(len(market_data)):
            # Agent 행동 결정 (Safety Layer 적용)
            action, info = self.agent.get_action(state, self.env)
            
            # 다음 시장 상태
            if i + 1 < len(market_data):
                next_vix = market_data['VIX'].iloc[i + 1]
                next_corr = market_data['Correlation'].iloc[i + 1]
            else:
                next_vix = None
                next_corr = None
            
            # 환경 스텝
            next_state, reward, done, step_info = self.env.step(action, next_vix, next_corr)
            
            episode_log.append({
                'step': i,
                'vix': self.env.vix,
                'hedge_ratio': self.env.hedge_ratio,
                'scr_ratio': self.env.scr_ratio,
                'kics_ratio': self.env.get_kics_ratio(),
                'action': action,
                'reward': reward,
                'reason': info['reason']
            })
            
            state = next_state
            
            if done:
                break
        
        self.episode_rewards.append(self.env.total_reward)
        return episode_log


# ==========================================
# Test Code
# ==========================================
if __name__ == "__main__":
    import pandas as pd
    from validation.backtest import generate_market_scenario
    
    print("=== RL Agent with Safety Layer Test ===\n")
    
    # 환경 및 에이전트 생성
    env = KICSEnvironment(lambda2=1000)  # 강력한 K-ICS 페널티
    agent = DynamicShieldAgent()
    trainer = RLTrainer(env, agent)
    
    # 2008 금융위기 시나리오 테스트
    print("[Scenario: 2008 Financial Crisis]")
    market_data = generate_market_scenario(200, '2008_crisis')
    
    episode_log = trainer.run_episode(market_data)
    
    # 결과 출력
    df = pd.DataFrame(episode_log)
    
    print("\n[Sample Steps]")
    print(df[['step', 'vix', 'hedge_ratio', 'kics_ratio', 'action', 'reason']].iloc[::20].to_string())
    
    print(f"\n[Episode Summary]")
    print(f"  Total Reward: {env.total_reward:.4f}")
    print(f"  Final Hedge Ratio: {env.hedge_ratio:.2f}")
    print(f"  Final K-ICS Ratio: {env.get_kics_ratio():.2f}%")
    
    # Safety Layer 작동 확인
    panic_steps = df[df['vix'] >= 30]
    if not panic_steps.empty:
        avg_hedge_panic = panic_steps['hedge_ratio'].mean()
        print(f"\n[Safety Layer Check]")
        print(f"  Panic Steps: {len(panic_steps)}")
        print(f"  Avg Hedge during Panic: {avg_hedge_panic:.2f}")
        if avg_hedge_panic > 0.8:
            print("  [PASS] Safety Layer correctly increased hedge during panic!")
        else:
            print("  [FAIL] Safety Layer did not respond adequately.")
