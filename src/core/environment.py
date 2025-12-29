"""
Phase 4.1: RL Environment Setup
================================
K-ICS 연계형 강화학습 환경
- Reward Function: 자본 효율성 - 페널티(K-ICS < 100%)
- State: [Hedge_Ratio, VIX, Correlation, SCR_Ratio]
- Action: 헤지 비율 조정
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from kics_real import RatioKICSEngine
except ImportError:
    from .kics_real import RatioKICSEngine


class KICSEnvironment:
    """
    K-ICS 연계형 강화학습 환경
    
    State Space:
        - hedge_ratio: 현재 헤지 비율 (0~1)
        - vix: 현재 VIX 지수 (0~100)
        - correlation: 주식-환율 상관계수 (-1~1)
        - scr_ratio: 현재 SCR 비율 (0~1)
    
    Action Space:
        - 0: 헤지 비율 5% 감소
        - 1: 유지
        - 2: 헤지 비율 5% 증가
        - 3: 헤지 비율 10% 증가 (패닉 대응)
    
    Reward Function:
        R_t = (r_portfolio - r_benchmark) 
              - λ1 * |h_t - h_{t-1}|        # 거래 비용 페널티
              - λ2 * max(0, SCR_target - SCR_t)  # K-ICS 위반 페널티
    """
    
    def __init__(self, 
                 lambda1=0.1,      # 거래 비용 페널티 가중치
                 lambda2=1000,     # K-ICS 위반 페널티 (강력!)
                 scr_target=0.35,  # 목표 SCR 비율 (100% K-ICS 비율에 해당)
                 hedge_cost_rate=0.002):  # 일일 헤지 비용률
        
        self.engine = RatioKICSEngine()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.scr_target = scr_target
        self.hedge_cost_rate = hedge_cost_rate
        
        # State variables
        self.hedge_ratio = 0.5
        self.vix = 15.0
        self.correlation = -0.4
        self.scr_ratio = 0.35
        self.prev_hedge_ratio = 0.5
        
        # Episode tracking
        self.step_count = 0
        self.total_reward = 0
        self.done = False
        
        # Baseline for benchmark
        self.baseline_scr = 0.36  # 100% 헤지 시 평균 SCR
        
    def reset(self, initial_vix=15.0, initial_corr=-0.4):
        """환경 초기화"""
        self.hedge_ratio = 0.5
        self.prev_hedge_ratio = 0.5
        self.vix = initial_vix
        self.correlation = initial_corr
        self.scr_ratio = self._calculate_scr()
        self.step_count = 0
        self.total_reward = 0
        self.done = False
        
        return self.get_state()
    
    def get_state(self):
        """현재 상태 반환 (정규화된 형태)"""
        return np.array([
            self.hedge_ratio,              # 0~1
            self.vix / 100.0,              # 정규화 (0~1)
            (self.correlation + 1) / 2,    # 정규화 (-1~1 -> 0~1)
            self.scr_ratio                 # 0~1
        ])
    
    def _calculate_scr(self):
        """K-ICS SCR 비율 계산"""
        return self.engine.calculate_scr_ratio_batch(
            np.array([self.hedge_ratio]),
            np.array([self.correlation])
        )[0]
    
    def step(self, action, new_vix=None, new_corr=None):
        """
        한 스텝 진행
        
        Args:
            action: 0=감소, 1=유지, 2=증가, 3=대폭 증가
            new_vix: 새로운 VIX 값 (시장 데이터에서 제공)
            new_corr: 새로운 상관계수 (시장 데이터에서 제공)
        
        Returns:
            state, reward, done, info
        """
        self.step_count += 1
        self.prev_hedge_ratio = self.hedge_ratio
        
        # 1. Action 적용 (헤지 비율 조정)
        if action == 0:
            self.hedge_ratio = max(0.0, self.hedge_ratio - 0.05)
        elif action == 1:
            pass  # 유지
        elif action == 2:
            self.hedge_ratio = min(1.0, self.hedge_ratio + 0.05)
        elif action == 3:
            self.hedge_ratio = min(1.0, self.hedge_ratio + 0.10)
        
        # 2. 시장 상태 업데이트
        if new_vix is not None:
            self.vix = new_vix
        if new_corr is not None:
            self.correlation = new_corr
        
        # 3. SCR 계산
        self.scr_ratio = self._calculate_scr()
        
        # 4. Reward 계산
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # 5. 종료 조건
        if self.step_count >= 500:  # 에피소드 최대 길이
            self.done = True
        
        info = {
            'hedge_ratio': self.hedge_ratio,
            'scr_ratio': self.scr_ratio,
            'vix': self.vix,
            'correlation': self.correlation,
            'step': self.step_count
        }
        
        return self.get_state(), reward, self.done, info
    
    def _calculate_reward(self):
        """
        Reward Function 계산
        
        R_t = Capital_Efficiency - Transaction_Cost - KICS_Penalty
        """
        # 1. 자본 효율성 (Baseline 대비 SCR 절감)
        capital_efficiency = (self.baseline_scr - self.scr_ratio) * 10  # 스케일링
        
        # 2. 헤지 비용
        hedge_cost = self.hedge_ratio * self.hedge_cost_rate
        
        # 3. 거래 비용 페널티 (포지션 변경 비용)
        transaction_penalty = self.lambda1 * abs(self.hedge_ratio - self.prev_hedge_ratio)
        
        # 4. K-ICS 위반 페널티 (핵심!)
        # SCR이 목표치를 초과하면 큰 페널티
        # (SCR이 높을수록 요구자본이 많음 = K-ICS 비율 하락)
        if self.scr_ratio > self.scr_target * 1.2:  # 20% 초과 시 페널티
            kics_penalty = self.lambda2 * (self.scr_ratio - self.scr_target)
        else:
            kics_penalty = 0
        
        # 최종 보상
        reward = capital_efficiency - hedge_cost - transaction_penalty - kics_penalty
        
        return reward
    
    def get_kics_ratio(self):
        """
        현재 K-ICS 비율 추정 (가용자본 / 요구자본)
        높을수록 좋음. 100% 미만이면 위험.
        """
        # 간단화: K-ICS 비율 ≈ 1.5 / SCR_Ratio (역수 관계)
        if self.scr_ratio > 0:
            return 1.5 / self.scr_ratio
        return 999


# ==========================================
# Test Code
# ==========================================
if __name__ == "__main__":
    env = KICSEnvironment()
    
    print("=== RL Environment Test ===")
    state = env.reset()
    print(f"Initial State: {state}")
    print(f"Initial K-ICS Ratio: {env.get_kics_ratio():.2f}%")
    
    # Simulate a few steps
    actions = [2, 2, 2, 1, 1, 0, 0]  # 증가 -> 유지 -> 감소
    vix_changes = [15, 18, 25, 35, 40, 30, 20]
    corr_changes = [-0.4, -0.3, 0.0, 0.5, 0.7, 0.4, -0.2]
    
    for i, (action, vix, corr) in enumerate(zip(actions, vix_changes, corr_changes)):
        state, reward, done, info = env.step(action, vix, corr)
        print(f"\nStep {i+1}:")
        print(f"  Action: {action}, VIX: {vix}, Corr: {corr:.1f}")
        print(f"  Hedge Ratio: {info['hedge_ratio']:.2f}")
        print(f"  SCR Ratio: {info['scr_ratio']:.4f}")
        print(f"  Reward: {reward:.4f}")
        print(f"  K-ICS Ratio: {env.get_kics_ratio():.2f}%")
    
    print(f"\nTotal Reward: {env.total_reward:.4f}")
