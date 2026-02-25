import pandas as pd
import numpy as np
import os

# 커스텀 모듈 임포트
from kics_real import KICSCalculator
from regime import RegimeClassifier

class DynamicShieldSystem:
    """
    Dynamic Shield 통합 시뮬레이션 환경 (Environment)
    - 데이터: Dynamic_Shield_Data_v5.csv (Real Data)
    - 엔진: K-ICS Calculator, Regime Classifier
    """
    def __init__(self, data_path=None):
        # 1. 데이터 파일 자동 탐색 (프로젝트 구조 기반)
        if data_path is None:
            # 프로젝트 루트/DATA/data/Dynamic_Shield_Data_v4.csv 자동 탐색
            script_dir = os.path.dirname(os.path.abspath(__file__))  # src/core/
            project_root = os.path.dirname(os.path.dirname(script_dir))  # 한화/
            data_path = os.path.join(project_root, 'DATA', 'data', 'Dynamic_Shield_Data_v4.csv')
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"데이터 파일이 없습니다: {data_path}")
            
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.dates = self.df.index
        self.total_steps = len(self.df)
        
        # 2. 엔진 초기화
        self.kics_engine = KICSCalculator()
        self.regime_engine = RegimeClassifier()
        
        # 3. Regime 모델 사전 학습 (Pre-training)
        self.regime_engine.fit(self.df)
        
        # 내부 상태 변수
        self.current_step = 0
        self.current_kics = 150.0 # 초기값 가정
        self.current_regime = 0 
        
    def reset(self):
        """환경 초기화"""
        self.current_step = 0
        self.kics_engine.reset()
        
        first_row = self.df.iloc[0]
        state = self._get_observation(first_row)
        return state

    def step(self, action):
        """
        하루 진행 (Step)
        Input: action (헤지 비율 0.0 ~ 1.0)
        Returns: next_state, reward, done, info
        """
        curr_row = self.df.iloc[self.current_step]
        prev_row = self.df.iloc[self.current_step - 1] if self.current_step > 0 else None
        
        # 1. Action Clipping
        hedge_ratio = np.clip(action, 0.0, 1.0)
        
        # 2. K-ICS 엔진 업데이트
        self.current_kics = self.kics_engine.update_and_calculate(curr_row, prev_row, hedge_ratio)
        
        # 3. Regime 탐지
        self.current_regime = self.regime_engine.predict(curr_row)
        regime_label = self.regime_engine.get_state_label(self.current_regime)
        
        # 4. 보상(Reward) 계산
        reward = 0.0
        
        # (A) K-ICS 방어 보상 (Regulation Constraint)
        if self.current_kics < 100:
            reward -= 10.0 # [Critical] 자본 부족 -> 매우 큰 벌점
        elif self.current_kics < 150:
            reward -= 1.0  # [Warning] 주의 구간 -> 벌점
        else:
            reward += 0.1  # [Good] 건전성 양호 -> 소폭 보상
            
        # (B) 패닉 시 방어 보상 (Safety Trigger)
        if regime_label == 'Panic':
            if hedge_ratio < 0.8: 
                reward -= 5.0 # 패닉인데 헤지 안하면 큰 벌점
            else:
                reward += 2.0 # 패닉 때 잘 막으면 큰 상점
                
        # 5. 다음 단계 준비
        self.current_step += 1
        done = self.current_step >= self.total_steps - 1
        
        if not done:
            next_row = self.df.iloc[self.current_step]
            next_state = self._get_observation(next_row)
        else:
            next_state = np.zeros(6) # Dummy
            
        info = {
            'Date': self.dates[self.current_step],
            'K_ICS': self.current_kics,
            'Regime': regime_label,
            'Regime_Idx': self.current_regime,
            'Hedge_Ratio': hedge_ratio
        }
        
        return next_state, reward, done, info

    def _get_observation(self, row):
        """
        AI 모델 입력용 상태 벡터 (State Vector)
        [Yield_Spread 포함 확인]
        """
        obs = [
            row.get('VIX', 15.0),
            row.get('VIX_Change', 0.0),
            row.get('FX_MA_Divergence', 0.0),
            row.get('Yield_Spread', 0.0), # CDS 대신 사용됨 (Proxy)
            self.current_kics / 100.0,    # 1.5 = 150% (Scaling)
            float(self.current_regime)    # 0, 1, 2
        ]
        return np.array(obs, dtype=np.float32)

if __name__ == "__main__":
    # 간단 테스트
    print("=== Dynamic Shield System Test (Real Data) ===")
    env = DynamicShieldSystem()
    state = env.reset()
    print(f"Start Date: {env.dates[0]}")
    print(f"Initial State: {state}")
    
    # 5일간 테스트
    for i in range(5):
        next_state, r, done, info = env.step(action=0.5)
        print(f"Step {i+1}: {info['Date'].date()} | Regime: {info['Regime']} | K-ICS: {info['K_ICS']:.1f}% | CDS_Proxy(YieldSpread): {state[3]:.2f}")