import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import Regime Detector
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from regime import MarketRegimeDetector
except ImportError:
    from .regime import MarketRegimeDetector

# ==========================================
# 1. K-ICS Engine (Ground Truth)
# ==========================================
class RatioKICSEngine:
    """
    Ratio-based K-ICS Calculator.
    Input: Hedge Ratio, Correlation
    Output: SCR Ratio
    """
    def __init__(self):
        self.shock_equity = 0.35
        self.shock_fx = 0.20
        self.ir_risk_rate = 0.05
        self.base_corr = np.array([
            [ 1.00, -0.40,  0.15],
            [-0.40,  1.00,  0.10],
            [ 0.15,  0.10,  1.00]
        ])

    def calculate_scr_ratio_batch(self, hedge_ratios, correlations):
        ones = np.ones_like(hedge_ratios)
        risk_equity = ones * self.shock_equity
        risk_fx = (ones * (1 - hedge_ratios)) * self.shock_fx
        risk_ir = ones * self.ir_risk_rate
        risks = np.column_stack((risk_equity, risk_fx, risk_ir))
        
        batch_size = len(hedge_ratios)
        corr_tensor = np.tile(self.base_corr, (batch_size, 1, 1))
        corr_tensor[:, 0, 1] = correlations
        corr_tensor[:, 1, 0] = correlations
        
        temp = np.einsum('ni,nij->nj', risks, corr_tensor)
        total_risk_squared = np.einsum('nj,nj->n', temp, risks)
        return np.sqrt(total_risk_squared)

# ==========================================
# 2. sklearn-based AI Surrogate Model
# ==========================================
def train_surrogate_model():
    """
    Trains an MLP Regressor to approximate K-ICS calculations.
    Returns: Trained model, Scalers (X, Y)
    """
    print("[-] Training AI Surrogate Model (sklearn MLP)...")
    engine = RatioKICSEngine()
    
    # Generate Training Data
    n_samples = 100000
    hedge_ratios = np.random.uniform(0, 1.0, n_samples)
    
    # Full Coverage (Fixed Bias): Normal, Transition, Panic
    n_normal = int(n_samples * 0.5)
    n_transition = int(n_samples * 0.3)
    n_panic = n_samples - n_normal - n_transition
    
    corrs = np.concatenate([
        np.random.uniform(-0.6, -0.2, n_normal),
        np.random.uniform(-0.2, 0.5, n_transition),
        np.random.uniform(0.5, 0.9, n_panic)
    ])
    np.random.shuffle(corrs)
    
    y_ratios = engine.calculate_scr_ratio_batch(hedge_ratios, corrs)
    
    X_raw = np.column_stack([hedge_ratios, corrs])
    Y_raw = y_ratios.reshape(-1, 1)
    
    # Train/Test Split (Anti-Leakage)
    X_train, X_test, y_train, y_test = train_test_split(X_raw, Y_raw, test_size=0.2, random_state=42)
    
    # Scaling (Fit on Train ONLY)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train).ravel()
    
    # Train MLP
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        verbose=False
    )
    model.fit(X_train_scaled, y_train_scaled)
    
    # Validation
    X_test_scaled = scaler_x.transform(X_test)
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print(f"[-] Validation MAPE: {mape:.4f}%")
    
    if mape < 1.0:
        print("[SUCCESS] AI Brain is Robust!")
    
    return model, scaler_x, scaler_y

# ==========================================
# 3. Dynamic Shield System
# ==========================================
class DynamicShieldSystem:
    def __init__(self):
        print("=== Initializing Dynamic Shield v3.0 System ===")
        
        # 1. Train/Load AI Model & Scalers
        self.model, self.scaler_x, self.scaler_y = train_surrogate_model()
        
        # 2. Initialize Regime Detector
        print("[-] Initializing Regime Detector...")
        self.regime_detector = MarketRegimeDetector()
        history_data = self.regime_detector.generate_synthetic_data(n_days=1000)
        self.regime_detector.train(history_data)
        
        # 3. K-ICS Engine (Ground Truth)
        self.engine = RatioKICSEngine()
        
        # System State
        self.current_hedge_ratio = 0.5
        self.logs = []

    def run_realtime_simulation(self, n_ticks=200):
        print(f"\n=== Starting Real-time Simulation ({n_ticks} Ticks) ===")
        print("[-] Mode: Walk-Forward Validation (Future Data)")
        
        # Scenario: Normal -> Transition -> Panic -> Normal
        vix_series = []
        vix_series.extend(np.random.normal(15, 2, 50))       # Normal
        vix_series.extend(np.linspace(15, 28, 50) + np.random.normal(0, 2, 50)) # Transition
        vix_series.extend(np.random.normal(45, 5, 30))       # Panic
        vix_series.extend(np.random.normal(16, 2, 70))       # Recovery
        
        fx_start = 1200
        fx_series = [fx_start]
        for v in vix_series[:-1]:
            change = np.random.normal(0, v/5)
            if v > 30: change += 5
            fx_series.append(fx_series[-1] + change)
            
        simulation_data = pd.DataFrame({'VIX': vix_series, 'FX': fx_series})
        simulation_data['FX_MA20'] = simulation_data['FX'].rolling(window=20).min()
        simulation_data['FX_MA20'] = simulation_data['FX_MA20'].fillna(method='bfill')
        simulation_data['FX_MA_Div'] = (simulation_data['FX'] - simulation_data['FX_MA20']) / simulation_data['FX_MA20'] * 100

        for t in range(n_ticks):
            current_data = simulation_data.iloc[t]
            vix = current_data['VIX']
            
            # Regime Detection
            if 20 <= vix < 30:
                regime_name = 'Transition'
            elif vix >= 30:
                regime_name = 'Panic'
            else:
                regime_name = 'Normal'
            
            # Correlation Estimation from Regime
            if regime_name == 'Normal':
                estimated_corr = -0.4
            elif regime_name == 'Transition':
                estimated_corr = 0.0
            else:
                estimated_corr = 0.8
            
            test_hedge_ratio = self.current_hedge_ratio
            
            # AI Inference (Anti-Leakage: transform only)
            input_raw = np.array([[test_hedge_ratio, estimated_corr]])
            input_scaled = self.scaler_x.transform(input_raw)
            pred_scr_scaled = self.model.predict(input_scaled)
            pred_scr = self.scaler_y.inverse_transform(pred_scr_scaled.reshape(-1, 1))[0][0]
            
            # Safety Layer
            if regime_name == 'Panic':
                if self.current_hedge_ratio < 1.0:
                    self.current_hedge_ratio = min(self.current_hedge_ratio + 0.1, 1.0)
            elif regime_name == 'Normal':
                if self.current_hedge_ratio > 0.4:
                    self.current_hedge_ratio = max(self.current_hedge_ratio - 0.05, 0.0)

            # Ground Truth
            real_scr = self.engine.calculate_scr_ratio_batch(
                np.array([test_hedge_ratio]), 
                np.array([estimated_corr])
            )[0]
            
            error = abs(pred_scr - real_scr)
            self.logs.append({
                'Tick': t, 'VIX': vix, 'Regime': regime_name,
                'Hedge_Ratio': test_hedge_ratio, 'Corr_Est': estimated_corr,
                'AI_SCR': pred_scr, 'Real_SCR': real_scr, 'Error': error
            })

        self.report_results()

    def report_results(self):
        df = pd.DataFrame(self.logs)
        print("\n=== Simulation Results ===")
        print(f"Total Ticks: {len(df)}")
        
        print("\n[1] Regime Coverage:")
        print(df['Regime'].value_counts())
        
        mean_error = df['Error'].mean()
        max_error = df['Error'].max()
        print(f"\n[2] AI Accuracy:")
        print(f"[-] Mean Absolute Error: {mean_error:.6f}")
        print(f"[-] Max Error: {max_error:.6f}")
        
        if mean_error < 0.01:
            print("[PASS] AI is Highly Accurate & Robust.")
        else:
            print("[WARNING] Precision degradation detected.")

        panic_data = df[df['Regime'] == 'Panic']
        if not panic_data.empty:
            avg_hedge_panic = panic_data['Hedge_Ratio'].mean()
            print(f"\n[3] Safety Layer Check:")
            print(f"[-] Avg Hedge Ratio during Panic: {avg_hedge_panic:.2f}")
            if avg_hedge_panic > 0.6:
                print("[PASS] System successfully increased hedge ratio during Panic.")
            else:
                print("[FAIL] System failed to de-risk.")

        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(df['Tick'], df['VIX'], label='VIX')
        plt.axhline(20, color='orange', linestyle='--', label='Transition Thr')
        plt.axhline(30, color='red', linestyle='--', label='Panic Thr')
        plt.title('Market Scenario (VIX)')
        plt.legend(); plt.grid()
        
        plt.subplot(3, 1, 2)
        plt.plot(df['Tick'], df['Hedge_Ratio'], color='green', label='Dynamic Hedge Ratio')
        plt.title('Safety Layer Response'); plt.ylim(0, 1.1); plt.legend(); plt.grid()
        
        plt.subplot(3, 1, 3)
        plt.plot(df['Tick'], df['Real_SCR'], label='Real K-ICS', color='black')
        plt.plot(df['Tick'], df['AI_SCR'], label='AI Prediction', color='blue', linestyle='--')
        plt.title('AI Precision (Real vs Pred)'); plt.legend(); plt.grid()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    system = DynamicShieldSystem()
    system.run_realtime_simulation()
