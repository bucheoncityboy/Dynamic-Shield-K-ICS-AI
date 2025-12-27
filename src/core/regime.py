"""
Phase 2.3: Regime Detection (HMM)
=================================
Hidden Markov Model을 활용한 시장 국면 분류.
- Normal (평온): VIX ~ 15
- Transition (전환): VIX ~ 28
- Panic (공포): VIX ~ 50

핵심 철학: Capital Optimization, not Prediction
(환율 예측이 아닌 자본 최적화)
"""

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import pandas as pd

import warnings
warnings.filterwarnings("ignore")


class MarketRegimeDetector:
    """HMM 기반 시장 국면 분류기"""
    def __init__(self, n_components=3, n_iter=1000, random_state=42):
        self.n_components = n_components
        self.model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=n_iter, random_state=random_state)
        self.trained = False
        self.regime_map = {}

    def generate_synthetic_data(self, n_days=1000):
        """테스트용 가상 시장 데이터 생성"""
        np.random.seed(42)
        
        # Regime Transition Matrix
        states = [0] * n_days
        current_state = 0
        trans_matrix = [
            [0.95, 0.04, 0.01], # Normal
            [0.20, 0.70, 0.10], # Transition
            [0.10, 0.30, 0.60]  # Panic
        ]
        
        for i in range(1, n_days):
            current_state = np.random.choice([0, 1, 2], p=trans_matrix[current_state])
            states[i] = current_state
            
        states = np.array(states)
        
        # Generate Observations
        vix = np.zeros(n_days)
        fx_rate = np.zeros(n_days)
        fx_rate[0] = 1200.0
        
        for t in range(n_days):
            s = states[t]
            if s == 0: # Normal
                vix[t] = np.random.normal(15, 2)
                fx_change = np.random.normal(0, 2.0)
            elif s == 1: # Transition
                vix[t] = np.random.normal(28, 3)
                fx_change = np.random.normal(3.0, 6.0)
            else: # Panic
                vix[t] = np.random.normal(50, 8)
                fx_change = np.random.normal(12.0, 20.0)
            
            if t > 0:
                fx_rate[t] = fx_rate[t-1] + fx_change
                
        df = pd.DataFrame({'VIX': vix, 'FX': fx_rate, 'Regime_True': states})
        df['FX_MA20'] = df['FX'].rolling(window=20).mean()
        df['FX_MA_Div'] = (df['FX'] - df['FX_MA20']) / df['FX_MA20'] * 100
        df = df.dropna()
        
        return df

    def train(self, df):
        """HMM 학습"""
        X = df[['VIX', 'FX_MA_Div']].values
        
        print(f"[-] Training HMM on {len(X)} data points...")
        self.model.fit(X)
        self.trained = True
        
        # Map Hidden States to Logical Names
        print("[-] Mapping Hidden States...")
        means = self.model.means_
        vix_means = means[:, 0]
        
        sorted_indices = np.argsort(vix_means)
        self.regime_map = {
            sorted_indices[0]: 'Normal',
            sorted_indices[1]: 'Transition',
            sorted_indices[2]: 'Panic'
        }
        
        for i in range(self.n_components):
            print(f"    State {i}: VIX Mean = {vix_means[i]:.2f} -> {self.regime_map[i]}")

    def predict(self, df):
        """국면 예측"""
        if not self.trained:
            raise Exception("Model not trained yet!")
            
        X = df[['VIX', 'FX_MA_Div']].values
        hidden_states = self.model.predict(X)
        
        logical_states = [self.regime_map[s] for s in hidden_states]
        df['Regime_Id'] = hidden_states
        df['Regime_Pred'] = logical_states
        
        return df

    def verify_plot(self, df):
        """시각화"""
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(df.index, df['VIX'], color='black', alpha=0.6, label='VIX')
        
        colors = {'Normal': 'green', 'Transition': 'orange', 'Panic': 'red'}
        for label, color in colors.items():
            mask = df['Regime_Pred'] == label
            plt.scatter(df[mask].index, df[mask]['VIX'], color=color, label=label, s=10)
            
        plt.title("Phase 2-3: Market Regime Detection (HMM)")
        plt.ylabel("VIX Index")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(df.index, df['FX'], label='KRW/USD FX Rate')
        plt.title("FX Rate Movement")
        plt.ylabel("KRW/USD")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    print("=== Phase 2-3: Regime Detection ===")
    detector = MarketRegimeDetector()
    
    print("[-] Generating Synthetic Data...")
    market_data = detector.generate_synthetic_data(n_days=1000)
    
    detector.train(market_data)
    
    results = detector.predict(market_data)
    
    print("\n[-] Verifying outputs...")
    print(results[['VIX', 'FX_MA_Div', 'Regime_True', 'Regime_Pred']].tail(10))
    
    detector.verify_plot(results)
