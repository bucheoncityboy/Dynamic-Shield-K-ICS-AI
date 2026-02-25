import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

class RegimeClassifier:
    """
    HMM 기반 시장 국면(Regime) 탐지기
    
    States:
    - Normal (0): 낮은 변동성, 평온한 시장
    - Transition (1): 변동성 확대 전조
    - Safe-haven (2): 안전자산 선호 국면
    - Panic (3): 극단적 위기 상황
    """
    def __init__(self, n_components=4, random_state=42):
        self.model = GaussianHMM(n_components=n_components, 
                                 covariance_type="full", 
                                 n_iter=1000, 
                                 random_state=random_state)
        self.is_fitted = False
        self.state_map = {} # {0: 'Normal', ...}

    def fit(self, df):
        """
        전체 역사적 데이터를 이용해 모델 학습
        Input Features:
        - VIX: 공포 지수
        - VIX_Change: 공포 확산 속도
        - FX_MA_Divergence: 환율 이격도
        - Yield_Spread: [중요] 한국 리스크 대리변수 (CDS Proxy)
        - Correlation: 주식-환율 상관계수 (안전자산 선호도)
        """
        features = ['VIX', 'VIX_Change', 'FX_MA_Divergence', 'Yield_Spread', 'Correlation']
        
        # 결측치는 0으로 채워서 학습
        valid_df = df[features].fillna(0)
        X = valid_df.values
        
        print(f"[-] HMM 모델 학습 시작 (Data shape: {X.shape})...")
        self.model.fit(X)
        self.is_fitted = True
        
        # 학습된 상태(0,1,2)가 무엇을 의미하는지 정의 (VIX 평균값 기준)
        hidden_states = self.model.predict(X)
        means = []
        for i in range(self.model.n_components):
            mean_vix = X[hidden_states == i, 0].mean() # VIX is col 0
            means.append((i, mean_vix))
            
        # VIX 평균이 낮은 순서대로 정렬 (Normal -> Transition -> Safe-haven -> Panic)
        means.sort(key=lambda x: x[1])
        
        self.state_map = {
            means[0][0]: 'Normal',      
            means[1][0]: 'Transition',  
            means[2][0]: 'Safe-haven',
            means[3][0]: 'Panic'        
        }
        print(f"[-] 모델 학습 완료. 상태 매핑: {self.state_map}")
        
    def predict(self, row):
        """한 시점의 데이터를 받아 상태 반환 (0, 1, 2)"""
        if not self.is_fitted:
            return 0 # Default Normal
            
        cols = ['VIX', 'VIX_Change', 'FX_MA_Divergence', 'Yield_Spread', 'Correlation']
        X = np.array([[ row.get(c, 0) for c in cols ]])
        
        state_idx = self.model.predict(X)[0]
        return state_idx
        
    def get_state_label(self, state_idx):
        """상태 인덱스를 사람이 읽을 수 있는 라벨로 변환"""
        return self.state_map.get(state_idx, 'Unknown')