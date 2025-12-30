"""
Phase 4.1: RL Environment Setup (Gym Compatible)
=================================================
stable-baselines3 호환 Gym 환경
- Reward Function: 자본 효율성 - 페널티(K-ICS < 100%)
- State: [Hedge_Ratio, VIX, Correlation, SCR_Ratio]
- Action: Continuous [-1, 1] -> 헤지 비율 조정

핵심 철학: Capital Optimization, not Prediction
(환율 예측이 아닌 자본 최적화)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from kics_real import RatioKICSEngine
except ImportError:
    from .kics_real import RatioKICSEngine


class KICSGymEnv(gym.Env):
    """
    K-ICS 연계형 강화학습 환경 (Gymnasium Compatible)
    
    State Space (Box):
        - hedge_ratio: 현재 헤지 비율 (0~1)
        - vix_normalized: 현재 VIX 지수 정규화 (0~1)
        - corr_normalized: 상관계수 정규화 (0~1)
        - scr_ratio: 현재 SCR 비율 (0~1)
    
    Action Space (Box):
        - Continuous: [-1, 1]
        - -1: 헤지 비율 최대 감소 (-10%)
        - +1: 헤지 비율 최대 증가 (+10%)
    
    Reward Function:
        R_t = Capital_Efficiency - Hedge_Cost - Transaction_Cost - KICS_Penalty
        
        Constraint: K-ICS 비율이 100% 미만이면 λ2 페널티 (-1000점)
    """
    
    metadata = {'render_modes': ['human']}
    
    def __init__(self, 
                 lambda1=0.1,           # 거래 비용 페널티 가중치
                 lambda2=1000,          # K-ICS 위반 페널티 (강력!)
                 scr_target=0.35,       # 목표 SCR 비율
                 hedge_cost_rate=0.002, # 일일 헤지 비용률
                 max_steps=500):        # 에피소드 최대 길이
        
        super().__init__()
        
        self.engine = RatioKICSEngine()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.scr_target = scr_target
        self.hedge_cost_rate = hedge_cost_rate
        self.max_steps = max_steps
        
        # Observation space: [hedge_ratio, vix_norm, corr_norm, scr_ratio]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Action space: continuous [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # State variables
        self.hedge_ratio = 0.5
        self.vix = 15.0
        self.correlation = -0.4
        self.scr_ratio = 0.35
        self.prev_hedge_ratio = 0.5
        
        # Episode tracking
        self.step_count = 0
        self.total_reward = 0
        
        # Baseline for benchmark (0.10 = 100% K-ICS, normalized)
        self.baseline_scr = 0.10
        
        # Market scenario data (will be set during reset)
        self.market_data = None
        self.current_idx = 0
        
        # [v4.0] 실제 데이터 캐시 (Anti-Overfitting)
        self._real_data_cache = None
        
    def _generate_market_data(self, n_steps):
        """
        [v4.0 개선] 실제 데이터 기반 시장 데이터 생성
        
        Anti-Overfitting:
        - 학습 시에는 실제 데이터의 앞쪽 70% (Train set)만 사용
        - 합성 데이터는 폴백용으로만 사용
        """
        try:
            # 실제 데이터 로드 시도
            from realistic_data import load_real_data_for_training
            
            if self._real_data_cache is None:
                self._real_data_cache = load_real_data_for_training(n_days=5000)
            
            # 랜덤 시작점에서 n_steps만큼 추출
            max_start = max(0, len(self._real_data_cache) - n_steps)
            start_idx = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            end_idx = start_idx + n_steps
            
            df_slice = self._real_data_cache.iloc[start_idx:end_idx]
            
            return {
                'vix': df_slice['VIX'].values,
                'correlation': df_slice['Correlation'].values
            }
        except Exception as e:
            # 폴백: 합성 데이터 사용
            print(f"[경고] 실제 데이터 로드 실패, 합성 데이터 사용: {e}")
            return self._generate_synthetic_market_data(n_steps)
    
    def _generate_synthetic_market_data(self, n_steps):
        """합성 데이터 폴백 (기존 로직)"""
        scenario = np.random.choice(['normal', 'crisis', 'pandemic'])
        
        if scenario == 'normal':
            vix = np.random.normal(15, 3, n_steps)
            vix = np.clip(vix, 10, 25)
        elif scenario == 'crisis':
            phase1 = np.random.normal(15, 2, n_steps // 3)
            phase2 = np.linspace(15, 60, n_steps // 3) + np.random.normal(0, 5, n_steps // 3)
            phase3 = np.linspace(60, 25, n_steps - 2 * (n_steps // 3)) + np.random.normal(0, 3, n_steps - 2 * (n_steps // 3))
            vix = np.concatenate([phase1, phase2, phase3])
        else:  # pandemic
            phase1 = np.random.normal(14, 2, n_steps // 2)
            phase2 = np.linspace(14, 65, n_steps // 4) + np.random.normal(0, 3, n_steps // 4)
            phase3 = np.random.normal(30, 8, n_steps - n_steps // 2 - n_steps // 4)
            vix = np.concatenate([phase1, phase2, phase3])
        
        # VIX 기반 상관계수 생성
        correlations = []
        for v in vix:
            if v >= 30:
                corr = np.random.uniform(0.5, 0.9)
            elif v >= 20:
                corr = np.random.uniform(-0.2, 0.5)
            else:
                corr = np.random.uniform(-0.6, -0.2)
            correlations.append(corr)
        
        return {'vix': vix, 'correlation': np.array(correlations)}
    
    def _calculate_scr(self):
        """K-ICS SCR 비율 계산"""
        return self.engine.calculate_scr_ratio_batch(
            np.array([self.hedge_ratio]),
            np.array([self.correlation])
        )[0]
    
    def _get_obs(self):
        """현재 상태 반환 (정규화된 형태)"""
        return np.array([
            self.hedge_ratio,
            np.clip(self.vix / 100.0, 0, 1),
            np.clip((self.correlation + 1) / 2, 0, 1),
            np.clip(self.scr_ratio, 0, 1)
        ], dtype=np.float32)
    
    def _get_kics_ratio(self):
        """K-ICS 비율 추정"""
        if self.scr_ratio > 0:
            return 1.5 / self.scr_ratio * 100
        return 999
    
    def reset(self, seed=None, options=None):
        """환경 초기화"""
        super().reset(seed=seed)
        
        self.hedge_ratio = 0.5
        self.prev_hedge_ratio = 0.5
        self.step_count = 0
        self.total_reward = 0
        self.current_idx = 0
        
        # 새로운 시장 시나리오 생성
        self.market_data = self._generate_market_data(self.max_steps + 10)
        self.vix = self.market_data['vix'][0]
        self.correlation = self.market_data['correlation'][0]
        self.scr_ratio = self._calculate_scr()
        
        return self._get_obs(), {}
    
    def step(self, action):
        """
        한 스텝 진행
        
        Args:
            action: [-1, 1] continuous action
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.step_count += 1
        self.prev_hedge_ratio = self.hedge_ratio
        
        # 1. Action 적용 (헤지 비율 조정)
        # action: [-1, 1] -> hedge change: [-0.1, 0.1]
        hedge_change = float(action[0]) * 0.1
        self.hedge_ratio = np.clip(self.hedge_ratio + hedge_change, 0.0, 1.0)
        
        # 2. 시장 상태 업데이트
        self.current_idx += 1
        if self.current_idx < len(self.market_data['vix']):
            self.vix = self.market_data['vix'][self.current_idx]
            self.correlation = self.market_data['correlation'][self.current_idx]
        
        # 3. SCR 계산
        self.scr_ratio = self._calculate_scr()
        
        # 4. Reward 계산
        reward = self._calculate_reward()
        self.total_reward += reward
        
        # 5. 종료 조건
        terminated = False
        truncated = self.step_count >= self.max_steps
        
        info = {
            'hedge_ratio': self.hedge_ratio,
            'scr_ratio': self.scr_ratio,
            'vix': self.vix,
            'kics_ratio': self._get_kics_ratio(),
            'step': self.step_count
        }
        
        return self._get_obs(), reward, terminated, truncated, info
    
    def _calculate_reward(self):
        """
        Reward Function 계산
        
        R_t = Capital_Efficiency - Hedge_Cost - Transaction_Cost - KICS_Penalty
        
        Constraint: K-ICS < 100%이면 -1000점 강력 페널티
        """
        # 1. 자본 효율성 (비율이 높을수록 안전 = 보상)
        # scr_ratio가 baseline보다 높으면 양수 보상
        capital_efficiency = (self.scr_ratio - self.baseline_scr) * 100
        
        # 2. 헤지 비용
        hedge_cost = self.hedge_ratio * self.hedge_cost_rate
        
        # 3. 거래 비용 페널티 (포지션 변경 비용)
        transaction_penalty = self.lambda1 * abs(self.hedge_ratio - self.prev_hedge_ratio)
        
        # 4. K-ICS 위반 페널티 (핵심!)
        kics_ratio = self._get_kics_ratio()
        if kics_ratio < 100:
            # K-ICS 100% 미만: 강력 페널티
            kics_penalty = self.lambda2
        elif kics_ratio < 120:
            # K-ICS 100-120%: 경고 페널티
            kics_penalty = (120 - kics_ratio) * 10
        else:
            kics_penalty = 0
        
        # 최종 보상
        reward = capital_efficiency - hedge_cost - transaction_penalty - kics_penalty
        
        return reward
    
    def render(self, mode='human'):
        """렌더링"""
        if mode == 'human':
            print(f"Step {self.step_count}: Hedge={self.hedge_ratio:.2f}, "
                  f"VIX={self.vix:.1f}, SCR={self.scr_ratio:.4f}, "
                  f"K-ICS={self._get_kics_ratio():.1f}%")


# 환경 등록
def register_kics_env():
    """Gymnasium에 환경 등록"""
    gym.register(
        id='KICS-v0',
        entry_point='src.core.gym_environment:KICSGymEnv',
        max_episode_steps=500,
    )
