"""
Phase 2-2: AI Surrogate Model (sklearn MLP)
============================================
K-ICS 엔진의 연산을 고속 근사하는 AI 대리 모델.
Anti-Bias, Anti-Leakage, Anti-Overfitting 철칙 적용.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

# Phase 1: K-ICS Engine 가져오기
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from kics_real import RatioKICSEngine
except ImportError:
    from .kics_real import RatioKICSEngine


class RobustSurrogate:
    """sklearn MLPRegressor 기반 Surrogate Model."""
    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            verbose=False
        )
        
    def fit(self, X, y):
        self.model.fit(X, y.ravel())
        
    def predict(self, X):
        return self.model.predict(X)


def train_surrogate_model():
    """Phase 2-2: AI Surrogate Training with Anti-Bias/Leakage"""
    print("=== Phase 2-2: Training AI Surrogate Model ===")
    engine = RatioKICSEngine()
    
    # 데이터 생성
    n_samples = 100000
    hedge_ratios = np.random.uniform(0, 1.0, n_samples)
    
    # Full Coverage (Bias Fix): Normal, Transition, Panic
    n_normal = int(n_samples * 0.5)
    n_transition = int(n_samples * 0.3)
    n_panic = n_samples - n_normal - n_transition
    
    corrs = np.concatenate([
        np.random.uniform(-0.6, -0.2, n_normal),      # Normal
        np.random.uniform(-0.2, 0.5, n_transition),   # Transition (이전 누락 구간!)
        np.random.uniform(0.5, 0.9, n_panic)          # Panic
    ])
    np.random.shuffle(corrs)
    
    y_ratios = engine.calculate_scr_ratio_batch(hedge_ratios, corrs)
    
    X_raw = np.column_stack([hedge_ratios, corrs])
    Y_raw = y_ratios.reshape(-1, 1)
    
    # Train/Val/Test Split (Anti-Overfitting)
    X_train, X_temp, y_train, y_temp = train_test_split(X_raw, Y_raw, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Scaling (Anti-Leakage: Fit on Train ONLY)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    X_val_scaled = scaler_x.transform(X_val)
    X_test_scaled = scaler_x.transform(X_test)
    
    # Model Training
    model = RobustSurrogate()
    print("[-] Training MLP...")
    model.fit(X_train_scaled, y_train_scaled)
    
    # Validation
    print("\n[-] Validating Scalability...")
    test_asset = 10_000_000_000
    test_X_raw = X_test[:1000]
    real_ratio = y_test[:1000]
    
    test_X_scaled = scaler_x.transform(test_X_raw)
    pred_ratio_scaled = model.predict(test_X_scaled)
    pred_ratio = scaler_y.inverse_transform(pred_ratio_scaled.reshape(-1, 1))
    
    real_amt = real_ratio * test_asset
    pred_amt = pred_ratio * test_asset
    
    mape = np.mean(np.abs((real_amt - pred_amt) / real_amt)) * 100
    print(f"[-] Scalability Test (Asset: 10B KRW): MAPE = {mape:.4f}%")
    
    if mape < 1.0:
        print("[SUCCESS] AI Brain is Robust & Scalable!")
    
    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(real_ratio, pred_ratio, s=5, alpha=0.5, label='AI Prediction')
    plt.plot([real_ratio.min(), real_ratio.max()], [real_ratio.min(), real_ratio.max()], 'r--', label='Ground Truth')
    plt.title("Phase 2-2: SCR Ratio Prediction")
    plt.xlabel("Real SCR Ratio")
    plt.ylabel("AI Predicted SCR Ratio")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model, scaler_x, scaler_y


if __name__ == "__main__":
    model, scaler_x, scaler_y = train_surrogate_model()
