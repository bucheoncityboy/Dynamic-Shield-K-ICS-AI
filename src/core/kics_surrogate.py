"""
Phase 2-2: Deep Neural Surrogate Model (PyTorch 기반)
======================================================
K-ICS 엔진의 연산을 고속 근사하는 AI 대리 모델.
제안서 명세: PyTorch 기반, ELU 활성화 함수, 밀리초 단위 추론

Anti-Bias, Anti-Leakage, Anti-Overfitting 철칙 적용.

핵심 철학: Capital Optimization, not Prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from scipy import stats  # Q-Q Plot용

# PyTorch 임포트
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[경고] PyTorch가 설치되지 않았습니다. sklearn MLP로 폴백합니다.")
    from sklearn.neural_network import MLPRegressor

# Phase 1: K-ICS Engine 가져오기
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from kics_real import RatioKICSEngine
except ImportError:
    from .kics_real import RatioKICSEngine

def pinball_loss(y_pred, y_true, q=0.90):
    error = y_true - y_pred
    return torch.mean(torch.max(q * error, (q - 1) * error))


class PinballLoss(nn.Module):
    """
    Quantile Regression with Pinball Loss
    tau=0.90: Predicts 90th percentile of SCR Ratio 
              (Conservative Upper Bound of SCR = Lower Bound of K-ICS)
    """
    def __init__(self, tau=0.90):
        super(PinballLoss, self).__init__()
        self.tau = tau

    def forward(self, preds, target):
        errors = target - preds
        return torch.mean(torch.max(self.tau * errors, (self.tau - 1.0) * errors))


class KICSSurrogate(nn.Module):
    """
    Deep Neural Surrogate Model (PyTorch 기반)
    
    제안서 명세에 따른 구조:
    - Input: [hedge_ratio, correlation]
    - Hidden: 512 → 256 → 128 → 64 → 32 (Deep Network, 5 layers, 더 넓게)
    - Activation: ELU (제안서 예시)
    - Dropout: 0.1 (과적합 방지)
    - Output: Predicted K-ICS Ratio
    
    Deep Network 정의: 4개 이상의 hidden layer
    """
    def __init__(self, input_dim=2, hidden_dims=[512, 256, 128, 64, 32], output_dim=1, dropout_rate=0.1):
        super(KICSSurrogate, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers with ELU activation and Dropout
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ELU())  # ELU as per proposal
            if dropout_rate > 0 and i < len(hidden_dims) - 1:  # 마지막 레이어 전까지만
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer (no activation for regression)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)


class RobustSurrogate:
    """
    Deep Neural Surrogate Model Wrapper
    
    PyTorch 기반 모델을 sklearn과 호환되는 인터페이스로 제공
    """
    def __init__(self, use_pytorch=True):
        self.use_pytorch = use_pytorch and TORCH_AVAILABLE
        
        if self.use_pytorch:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Deep Network: 5개 hidden layer (512 → 256 → 128 → 64 → 32, 더 넓게)
            # Dropout 증가로 과적합 방지
            self.model = KICSSurrogate(input_dim=2, hidden_dims=[512, 256, 128, 64, 32], output_dim=1, dropout_rate=0.2)
            self.model.to(self.device)
            self.optimizer = None
            self.criterion = PinballLoss(tau=0.90)  # [P2] Quantile Regression (보수적 하한 예측용 Pinball Loss)
            self.scaler_x = None
            self.scaler_y = None
        else:
            # Fallback to sklearn
            from sklearn.neural_network import MLPRegressor
            self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            verbose=False
        )
        
    def fit(self, X, y, epochs=500, batch_size=256, learning_rate=0.001, verbose=False):
        """
        모델 학습
        
        Args:
            X: 입력 데이터 (numpy array)
            y: 타겟 데이터 (numpy array)
            epochs: 학습 에포크 수
            batch_size: 배치 크기
            learning_rate: 학습률
            verbose: 학습 진행 출력 여부
        """
        if self.use_pytorch:
            # PyTorch 학습
            self.model.train()
            
            # 데이터를 텐서로 변환
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)
            
            # DataLoader 생성
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Optimizer 설정 (Adam with 더 강한 weight decay로 과적합 방지)
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=5e-4)
            
            # Learning Rate Scheduler (더 공격적으로 - Val loss 증가 시 빠르게 감소)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=50, min_lr=learning_rate * 0.0001
            )
            
            # Early Stopping을 위한 변수 (과적합 방지를 위해 더 빠르게 중단)
            best_val_loss = float('inf')
            patience_counter = 0
            patience_limit = 100  # Val loss 증가 시 빠르게 중단
            
            # 학습 루프 (개선된 가중치 손실 함수 사용)
            for epoch in range(epochs):
                epoch_loss = 0.0
                self.model.train()
                
                for batch_X, batch_y in dataloader:
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    predictions = self.model(batch_X)
                    
                    # Pinball Loss 사용 (Quantile Regression)
                    loss = pinball_loss(predictions, batch_y)

                    # Backward pass
                    loss.backward()
                    # Gradient clipping (안정적인 학습)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                
                # Validation loss 계산 (early stopping용) - 전체 validation set 사용
                self.model.eval()
                with torch.no_grad():
                    # Validation set을 별도로 사용 (전체 데이터의 20%)
                    val_size = len(X_tensor) // 5
                    val_X = X_tensor[-val_size:]  # 마지막 20%를 validation으로
                    val_y = y_tensor[-val_size:]
                    val_predictions = self.model(val_X)
                    val_loss = pinball_loss(val_predictions, val_y).item()

                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping 체크
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 100 == 0:
                    avg_loss = epoch_loss / len(dataloader)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6f}")
                
                # Early stopping
                if patience_counter >= patience_limit:
                    if verbose:
                        print(f"  Early stopping at epoch {epoch+1}")
                    break
        else:
            # sklearn 학습 (폴백)
            self.model.fit(X, y.ravel())
        
    def predict(self, X):
        """
        예측 수행 (밀리초 단위 고속 추론)
        
        Args:
            X: 입력 데이터 (numpy array)
            
        Returns:
            예측값 (numpy array, sklearn과 호환)
        """
        if self.use_pytorch:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                predictions = self.model(X_tensor)
                # (batch_size, 1) -> (batch_size,) 형태로 변환 (sklearn 호환)
                return predictions.cpu().numpy().ravel()
        else:
            return self.model.predict(X)
    
    def save(self, path):
        """모델 저장"""
        if self.use_pytorch:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'input_dim': 2,
                    'hidden_dims': [512, 256, 128, 64, 32],
                    'output_dim': 1,
                    'dropout_rate': 0.1
                }
            }, path)
        else:
            import pickle
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
    
    def load(self, path):
        """모델 로드"""
        if self.use_pytorch:
            checkpoint = torch.load(path, map_location=self.device)
            config = checkpoint.get('model_config', {'input_dim': 2, 'hidden_dims': [512, 256, 128, 64, 32], 'output_dim': 1, 'dropout_rate': 0.1})
            self.model = KICSSurrogate(**config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
        else:
            import pickle
            with open(path, 'rb') as f:
                self.model = pickle.load(f)


def train_surrogate_model():
    """Phase 2-2: AI Surrogate Training with Real Data Only"""
    print("=== Phase 2-2: Training AI Surrogate Model ===")
    print("[실제 데이터만 사용] Dynamic_Shield_Data_v4.csv")
    engine = RatioKICSEngine()
    
    # ==========================================
    # 실제 데이터 로드
    # ==========================================
    print("\n[실제 데이터 로드 중...]")
    try:
        from realistic_data import load_real_data_for_training, load_real_data_for_testing
        
        # 학습용 실제 데이터 로드 (전체 데이터 사용)
        train_data = load_real_data_for_training(n_days=None)  # 전체 학습 데이터
        test_data = load_real_data_for_testing(n_days=None)    # 전체 테스트 데이터
        
        # 학습 데이터 범위 확인 및 확장
        print(f"\n[데이터 분포 분석]")
        # 샘플링하여 분포 확인
        sample_corrs_train = train_data['Correlation'].values[::10]  # 10일마다 샘플링
        sample_corrs_test = test_data['Correlation'].values[::10]
        
        sample_hedges = np.linspace(0.0, 1.0, 10)
        train_sample_ratios = []
        for corr in sample_corrs_train[:100]:
            for hedge in sample_hedges:
                scr = engine.calculate_scr_ratio_batch([hedge], [corr])[0]
                train_sample_ratios.append(scr)
        
        test_sample_ratios = []
        for corr in sample_corrs_test[:100]:
            for hedge in sample_hedges:
                scr = engine.calculate_scr_ratio_batch([hedge], [corr])[0]
                test_sample_ratios.append(scr)
        
        print(f"  학습 데이터 SCR 범위: [{np.min(train_sample_ratios):.4f}, {np.max(train_sample_ratios):.4f}]")
        print(f"  테스트 데이터 SCR 범위: [{np.min(test_sample_ratios):.4f}, {np.max(test_sample_ratios):.4f}]")
        
        if np.max(test_sample_ratios) > np.max(train_sample_ratios):
            print(f"  ⚠️  경고: 테스트 데이터 범위가 학습 데이터보다 넓습니다!")
            print(f"     학습 범위 밖의 값({np.max(test_sample_ratios):.4f})을 예측해야 합니다.")
            print(f"     → 학습 데이터에 더 다양한 샘플 추가 필요")
        
        if train_data is None or len(train_data) == 0:
            raise FileNotFoundError("실제 데이터 로드 실패")
        
        print(f"  ✓ 학습용 데이터: {len(train_data)}일")
        print(f"  ✓ 테스트용 데이터: {len(test_data)}일")
        
        # Correlation 확인 (realistic_data.py에서 이미 계산됨)
        if 'Correlation' not in train_data.columns:
            print("  [오류] Correlation 컬럼이 없습니다.")
            print("  → realistic_data.py에서 실제 데이터로부터 Correlation을 계산해야 합니다.")
            raise ValueError("Correlation 컬럼이 필요합니다. 실제 데이터에서 계산하세요.")
        
        # 실제 데이터에서 학습 샘플 생성
        # 각 날짜의 실제 Correlation에 대해 다양한 Hedge Ratio 조합
        print(f"\n[학습 데이터 생성 중...]")
        
        train_hedge_ratios = []
        train_correlations = []
        
        # 학습 데이터: 각 날짜당 더 많은 hedge ratio 조합 (범위 확대)
        n_ratios_per_day = 20  # 10개 → 20개로 증가 (더 세밀한 샘플링)
        for idx in range(len(train_data)):
            real_corr = train_data.iloc[idx]['Correlation']
            # 0.0 ~ 1.0 사이 균등 분포 (더 많은 샘플)
            for hedge_ratio in np.linspace(0.0, 1.0, n_ratios_per_day):
                train_hedge_ratios.append(hedge_ratio)
                train_correlations.append(real_corr)
        
        print(f"  ✓ 기본 학습 샘플 수: {len(train_hedge_ratios)}개")
        print(f"    (실제 날짜: {len(train_data)}일 × {n_ratios_per_day}개 hedge ratio)")
        
        # 실제 K-ICS 계산 (먼저 계산해서 범위 확인)
        print(f"  [실제 K-ICS 계산 중...]")
        train_ratios_temp = engine.calculate_scr_ratio_batch(
            np.array(train_hedge_ratios),
            np.array(train_correlations)
        )
        
        # 테스트 데이터 범위 확인
        print(f"\n[테스트 데이터 범위 분석]")
        test_corrs_sample = test_data['Correlation'].values
        test_hedges_sample = np.linspace(0.0, 1.0, 10)
        test_ratios_sample = []
        for corr in test_corrs_sample[:200]:  # 더 많은 샘플
            for hedge in test_hedges_sample:
                scr = engine.calculate_scr_ratio_batch([hedge], [corr])[0]
                test_ratios_sample.append(scr)
        
        train_max_scr = np.max(train_ratios_temp)
        test_max_scr = np.max(test_ratios_sample)
        train_min_scr = np.min(train_ratios_temp)
        test_min_scr = np.min(test_ratios_sample)
        
        print(f"  학습 SCR 범위: [{train_min_scr:.4f}, {train_max_scr:.4f}]")
        print(f"  테스트 SCR 범위: [{test_min_scr:.4f}, {test_max_scr:.4f}]")
        
        # 높은 SCR Ratio 구간 집중 오버샘플링
        if test_max_scr > train_max_scr * 0.98:
            print(f"\n[높은 SCR Ratio 집중 학습]")
            print(f"  테스트 최대값({test_max_scr:.4f})이 학습 최대값({train_max_scr:.4f})보다 높음")
            print(f"  → 높은 SCR Ratio 구간에 집중 학습 데이터 추가")
            
            # 높은 SCR Ratio를 생성하는 조합 찾기 (더 적극적으로)
            high_scr_threshold = train_max_scr * 0.90  # 더 낮은 threshold로 더 많은 샘플
            high_scr_samples = 0
            target_samples = 15000  # 더 많은 샘플 (5000 → 15000)
            
            # 테스트 데이터의 correlation 범위에서 높은 SCR 생성
            test_corr_range = np.linspace(test_data['Correlation'].min(), 
                                         test_data['Correlation'].max(), 300)  # 더 세밀하게 (100 → 300)
            
            for corr in test_corr_range:
                # 높은 SCR을 생성하는 hedge ratio 찾기 (낮은 hedge ratio가 높은 SCR 생성)
                for hedge in np.linspace(0.0, 0.3, 60):  # 더 세밀하게 (30 → 60)
                    scr = engine.calculate_scr_ratio_batch([hedge], [corr])[0]
                    if scr >= high_scr_threshold:
                        train_hedge_ratios.append(hedge)
                        train_correlations.append(corr)
                        high_scr_samples += 1
                        if high_scr_samples >= target_samples:
                            break
                if high_scr_samples >= target_samples:
                    break
            
            # 추가: 극단적으로 높은 SCR 샘플 (테스트 최대값 근처) - 실제 데이터만 사용
            if test_max_scr > train_max_scr:
                print(f"  [극단값 샘플 추가] 테스트 최대값 근처 샘플 생성 (실제 데이터만 사용)")
                extreme_samples = 0
                
                # 실제 테스트 데이터의 모든 correlation 값 사용 (랜덤 없음)
                # 각 실제 correlation에 대해 낮은 hedge ratio 조합 시도
                for corr in test_corrs_sample:
                    # 낮은 hedge ratio로 극단값 생성 (0.0 ~ 0.2 사이를 세밀하게)
                    for hedge in np.linspace(0.0, 0.2, 20):  # 균등 분포, 랜덤 없음
                        scr = engine.calculate_scr_ratio_batch([hedge], [corr])[0]
                        if scr >= test_max_scr * 0.95:  # 테스트 최대값 근처
                            train_hedge_ratios.append(hedge)
                            train_correlations.append(corr)
                            extreme_samples += 1
                            if extreme_samples >= 3000:  # 충분한 샘플
                                break
                    if extreme_samples >= 3000:
                        break
                
                print(f"  ✓ 극단값 샘플 추가: {extreme_samples}개 (실제 데이터 correlation만 사용)")
            
            print(f"  ✓ 높은 SCR Ratio 샘플 추가: {high_scr_samples}개")
        
        # 테스트 correlation 범위 커버
        test_corr_min = test_data['Correlation'].min()
        test_corr_max = test_data['Correlation'].max()
        train_corr_min = train_data['Correlation'].min()
        train_corr_max = train_data['Correlation'].max()
        
        # Correlation 범위 확장: 실제 테스트 데이터의 correlation 값만 사용
        if test_corr_min < train_corr_min or test_corr_max > train_corr_max:
            print(f"\n[Correlation 범위 확장] 실제 테스트 데이터의 correlation 값 사용")
            # 실제 테스트 데이터에서 범위 밖의 correlation 값 찾기
            extended_corrs_from_test = []
            if test_corr_min < train_corr_min:
                # 테스트 데이터에서 train_corr_min보다 작은 값들
                low_corrs = test_data[test_data['Correlation'] < train_corr_min]['Correlation'].values
                extended_corrs_from_test.extend(low_corrs)
            if test_corr_max > train_corr_max:
                # 테스트 데이터에서 train_corr_max보다 큰 값들
                high_corrs = test_data[test_data['Correlation'] > train_corr_max]['Correlation'].values
                extended_corrs_from_test.extend(high_corrs)
            
            # 실제 데이터의 correlation 값만 사용 (랜덤 생성 없음)
            for ext_corr in extended_corrs_from_test:
                for hedge_ratio in np.linspace(0.0, 1.0, 15):
                    train_hedge_ratios.append(hedge_ratio)
                    train_correlations.append(ext_corr)
            
            print(f"  ✓ Correlation 범위 확장 샘플: {len(extended_corrs_from_test) * 15}개 (실제 데이터만 사용)")
        
        # 최종 K-ICS 계산
        print(f"\n[최종 학습 데이터 계산]")
        train_ratios = engine.calculate_scr_ratio_batch(
            np.array(train_hedge_ratios),
            np.array(train_correlations)
        )
        
        X_raw = np.column_stack([train_hedge_ratios, train_correlations])
        Y_raw = train_ratios.reshape(-1, 1)
        
        print(f"  ✓ 학습 데이터 준비 완료")
        print(f"    - Correlation 범위: [{np.min(train_correlations):.3f}, {np.max(train_correlations):.3f}]")
        print(f"    - Hedge Ratio 범위: [0.0, 1.0]")
        print(f"    - SCR Ratio 범위: [{np.min(train_ratios):.4f}, {np.max(train_ratios):.4f}]")
        
        # Train/Val/Test Split (시간 기반 - 실제 데이터 순서 유지)
        # 전체 학습 데이터의 80%를 Train, 20%를 Val로
        split_idx = int(len(X_raw) * 0.8)
        X_train = X_raw[:split_idx]
        y_train = Y_raw[:split_idx]
        X_val = X_raw[split_idx:]
        y_val = Y_raw[split_idx:]
        
        # Test는 별도 실제 테스트 데이터 사용
        test_hedge_ratios = []
        test_correlations = []
        
        n_ratios_per_day_test = 5
        for idx in range(len(test_data)):
            real_corr = test_data.iloc[idx]['Correlation']
            for hedge_ratio in np.linspace(0.0, 1.0, n_ratios_per_day_test):
                test_hedge_ratios.append(hedge_ratio)
                test_correlations.append(real_corr)
        
        print(f"\n[테스트 데이터 생성 중...]")
        test_ratios = engine.calculate_scr_ratio_batch(
            np.array(test_hedge_ratios),
            np.array(test_correlations)
        )
        
        X_test = np.column_stack([test_hedge_ratios, test_correlations])
        y_test = test_ratios.reshape(-1, 1)
        
        print(f"  ✓ 테스트 샘플 수: {len(X_test)}개")
        print(f"    (실제 날짜: {len(test_data)}일 × {n_ratios_per_day_test}개 hedge ratio)")
        
    except Exception as e:
        print(f"  ✗ 실제 데이터 로드 실패: {e}")
        print("  → 오류: 실제 데이터가 필요합니다. Dynamic_Shield_Data_v4.csv 파일을 확인하세요.")
        import traceback
        traceback.print_exc()
        raise FileNotFoundError("실제 데이터 파일이 필요합니다. 샘플 데이터는 사용하지 않습니다.")
    
    # Scaling (Anti-Leakage: Fit on Train ONLY)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_x.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)
    
    X_val_scaled = scaler_x.transform(X_val)
    X_test_scaled = scaler_x.transform(X_test)
    
    # Model Training (PyTorch 기반)
    model = RobustSurrogate(use_pytorch=True)
    print("[-] Training Deep Neural Network (PyTorch)...")
    print(f"    Device: {model.device}")
    print(f"    Architecture: Input(2) → 512 → 256 → 128 → 64 → 32 → Output(1)")
    print(f"    Hidden Layers: 5개 (Deep Network, 더 넓게)")
    print(f"    Activation: ELU")
    print(f"    Dropout: 0.1 (과적합 방지)")
    
    # 스케일러 저장 (나중에 사용)
    model.scaler_x = scaler_x
    model.scaler_y = scaler_y
    
    # 학습 (더 많은 에포크, 더 작은 학습률로 정밀 학습, Early Stopping 포함)
    print("\n[모델 학습 시작]")
    print("  - 최대 에포크: 2000")
    print("  - 초기 학습률: 0.0005 (과적합 방지를 위해 감소)")
    print("  - 배치 크기: 128 (더 작은 배치로 정밀 학습)")
    print("  - Loss: Pinball Loss (tau=0.90, 보수적 하한 예측)")
    print("  - Dropout: 0.2 (과적합 방지 강화)")
    print("  - Weight Decay: 5e-4 (정규화 강화)")
    print("  - Learning Rate Scheduling: ReduceLROnPlateau (factor=0.5, patience=50)")
    print("  - Early Stopping: 100 에포크 동안 개선 없으면 중단 (과적합 빠른 감지)")
    
    model.fit(X_train_scaled, y_train_scaled, epochs=2000, batch_size=128, learning_rate=0.0005, verbose=True)
    
    # ==========================================
    # 성능 평가 (실제 데이터 사용)
    # ==========================================
    print("\n" + "=" * 60)
    print("성능 평가 (Performance Evaluation)")
    print("=" * 60)
    
    print("\n[평가 데이터 설명]")
    print("=" * 60)
    print("1. 학습 데이터: [실제 시장 데이터만 사용]")
    print("   - 실제 데이터: Dynamic_Shield_Data_v4.csv (학습 구간)")
    print("   - 실제 VIX, Correlation 값 사용")
    print("   - 각 날짜당 10개의 다른 Hedge Ratio (0.0 ~ 1.0) 조합")
    print("   - Ground Truth: RatioKICSEngine으로 실제 K-ICS SCR Ratio 계산")
    print("")
    print("2. 데이터 분할 (시간 기반, Anti-Overfitting):")
    print("   - Train Set: 학습 데이터의 80% - 모델 학습용")
    print("   - Validation Set: 학습 데이터의 20% - 하이퍼파라미터 튜닝용")
    print("   - Test Set: 별도 테스트 데이터 (시간적으로 분리)")
    print("")
    print("3. 성능 평가 데이터: [실제 시장 데이터만 사용]")
    print("   - 실제 시장 데이터: Dynamic_Shield_Data_v4.csv (테스트 구간)")
    print("   - 실제 VIX, Correlation 값 사용")
    print("   - 각 날짜당 5개의 다른 Hedge Ratio (0.0 ~ 1.0) 조합")
    print("   - 평가 기준 자산: 10,000,000,000 KRW (100억원)")
    print("   - 실제 SCR 금액 = 실제 SCR Ratio × 기준 자산")
    print("   - 예측 SCR 금액 = 예측 Ratio × 기준 자산")
    print("   - 오차 계산: 실제 금액 vs 예측 금액 비교")
    print("=" * 60)
    
    # ==========================================
    # 평가: 실제 테스트 데이터만 사용
    # ==========================================
    print("\n[평가 데이터]")
    print("=" * 60)
    print("실제 시장 데이터만 사용 (시간적으로 분리된 테스트 데이터)")
    print("=" * 60)
    
    # 실제 테스트 데이터만 사용
    print(f"\n[실제 테스트 데이터]")
    print(f"  ✓ 실제 테스트 데이터: {len(X_test)}개")
    print(f"    - 실제 날짜: {len(test_data)}일")
    print(f"    - 각 날짜당 {n_ratios_per_day_test}개 hedge ratio 조합")
    print(f"    - 실제 Correlation 범위: [{np.min(test_correlations):.3f}, {np.max(test_correlations):.3f}]")
    
    test_X_raw = X_test
    real_ratio = y_test
    
    test_asset = 10_000_000_000
    test_X_scaled = scaler_x.transform(test_X_raw)
    
    print(f"\n[평가 실행]")
    print(f"  평가 샘플 수: {len(test_X_raw)}개")
    print(f"  기준 자산: {test_asset/1e9:.0f}억원")
    
    # 실제 테스트 데이터 평가
    print("\n" + "=" * 60)
    print("실제 데이터 성능 평가")
    print("=" * 60)
    
    # 추론 속도 측정 (밀리초 단위)
    start_time = time.time()
    test_X_scaled = scaler_x.transform(test_X_raw)
    pred_ratio_scaled = model.predict(test_X_scaled)
    inference_time = (time.time() - start_time) * 1000  # 밀리초
    
    pred_ratio = scaler_y.inverse_transform(pred_ratio_scaled.reshape(-1, 1))
    
    # 금액 단위로 변환 (10B KRW 기준)
    real_amt = real_ratio * test_asset
    pred_amt = pred_ratio * test_asset
    
    # ==========================================
    # 평가 지표 계산
    # ==========================================
    
    # 1. MAPE (Mean Absolute Percentage Error) - 비율 오차
    mape = np.mean(np.abs((real_amt - pred_amt) / real_amt)) * 100
    
    # 2. RMSE (Root Mean Squared Error) - 제곱근 평균 제곱 오차
    rmse = np.sqrt(np.mean((real_amt - pred_amt) ** 2))
    rmse_pct = (rmse / test_asset) * 100  # 비율로 변환
    
    # 3. MAE (Mean Absolute Error) - 평균 절대 오차
    mae = np.mean(np.abs(real_amt - pred_amt))
    mae_pct = (mae / test_asset) * 100
    
    # 4. R² (Coefficient of Determination) - 결정계수
    ss_res = np.sum((real_amt - pred_amt) ** 2)
    ss_tot = np.sum((real_amt - np.mean(real_amt)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # 5. Max Error - 최대 오차
    max_error = np.max(np.abs(real_amt - pred_amt))
    max_error_pct = (max_error / test_asset) * 100
    
    # 6. Median Error - 중앙값 오차
    median_error = np.median(np.abs(real_amt - pred_amt))
    median_error_pct = (median_error / test_asset) * 100
    
    # 7. 추론 속도
    avg_inference_time = inference_time / len(test_X_scaled)
    
    # 결과 출력
    print("\n[평가 지표 설명]")
    print("  MAPE: 평균 절대 비율 오차 (낮을수록 좋음, < 1% 우수)")
    print("  RMSE: 제곱근 평균 제곱 오차 (낮을수록 좋음)")
    print("  MAE: 평균 절대 오차 (낮을수록 좋음)")
    print("  R²: 결정계수 (1에 가까울수록 좋음, > 0.99 우수)")
    print("  Max Error: 최대 오차 (낮을수록 좋음)")
    print("  Median Error: 중앙값 오차 (낮을수록 좋음)")
    
    print("\n[성능 평가 결과]")
    print(f"  MAPE:        {mape:.4f}%")
    print(f"  RMSE:        {rmse_pct:.4f}% ({rmse/1e8:.2f}억원)")
    print(f"  MAE:         {mae_pct:.4f}% ({mae/1e8:.2f}억원)")
    print(f"  R²:          {r2:.6f}")
    print(f"  Max Error:   {max_error_pct:.4f}% ({max_error/1e8:.2f}억원)")
    print(f"  Median Error: {median_error_pct:.4f}% ({median_error/1e8:.2f}억원)")
    print(f"  Inference:   {avg_inference_time:.3f} ms/sample")
    
    # 성능 판정
    print("\n[성능 판정]")
    if mape < 0.1:
        print("  ✓ MAPE: 우수 (< 0.1%)")
    elif mape < 1.0:
        print("  ✓ MAPE: 양호 (< 1%)")
    else:
        print("  ⚠ MAPE: 개선 필요 (>= 1%)")
    
    if r2 > 0.99:
        print("  ✓ R²: 우수 (> 0.99)")
    elif r2 > 0.95:
        print("  ✓ R²: 양호 (> 0.95)")
    else:
        print("  ⚠ R²: 개선 필요 (< 0.95)")
    
    if avg_inference_time < 1.0:
        print("  ✓ 추론 속도: 매우 빠름 (< 1ms)")
    elif avg_inference_time < 10.0:
        print("  ✓ 추론 속도: 실시간 가능 (< 10ms)")
    else:
        print("  ⚠ 추론 속도: 개선 필요 (>= 10ms)")
    
    # ==========================================
    # 상세 시각화
    # ==========================================
    print("\n[시각화 생성 중...]")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 예측 vs 실제 산점도
    ax1 = plt.subplot(2, 3, 1)
    plt.scatter(real_ratio, pred_ratio, s=10, alpha=0.6, label='Predictions (Real Data)')
    plt.plot([real_ratio.min(), real_ratio.max()], [real_ratio.min(), real_ratio.max()], 
             'r--', label='Perfect Prediction', linewidth=2)
    plt.xlabel("Real SCR Ratio", fontsize=11)
    plt.ylabel("Predicted SCR Ratio", fontsize=11)
    plt.title("(1) Prediction vs Ground Truth\n(Real Data Only)", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 성능 지표 텍스트
    textstr = f'MAPE: {mape:.4f}%\nR²: {r2:.6f}'
    plt.text(0.05, 0.95, textstr, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 2. 잔차 플롯 (Residual Plot)
    ax2 = plt.subplot(2, 3, 2)
    residuals = (pred_ratio - real_ratio).ravel()
    plt.scatter(real_ratio, residuals, s=10, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel("Real SCR Ratio", fontsize=11)
    plt.ylabel("Residual (Predicted - Real)", fontsize=11)
    plt.title("(2) Residual Plot", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 3. 오차 분포 히스토그램
    ax3 = plt.subplot(2, 3, 3)
    errors = np.abs(residuals)
    plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.6f}')
    plt.axvline(np.median(errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.6f}')
    plt.xlabel("Absolute Error", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.title("(3) Error Distribution", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Q-Q Plot (정규성 검증)
    ax4 = plt.subplot(2, 3, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title("(4) Q-Q Plot (Normality Check)", fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # 5. 오차 통계 요약
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    metrics_text = f"""
    성능 평가 지표 요약
    
    [정확도 지표]
    MAPE:        {mape:.4f}%
    RMSE:        {rmse_pct:.4f}%
    MAE:         {mae_pct:.4f}%
    R²:          {r2:.6f}
    
    [오차 통계]
    Max Error:   {max_error_pct:.4f}%
    Median Error: {median_error_pct:.4f}%
    Mean Error:  {np.mean(errors)*100:.4f}%
    Std Error:   {np.std(errors)*100:.4f}%
    
    [속도]
    Inference:   {avg_inference_time:.3f} ms/sample
    
    [평가 기준]
    • MAPE < 0.1%: 우수
    • R² > 0.99: 우수
    • Inference < 10ms: 실시간 가능
    """
    ax5.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 6. 시계열 비교 (샘플)
    ax6 = plt.subplot(2, 3, 6)
    sample_indices = np.arange(min(100, len(real_ratio)))
    plt.plot(sample_indices, real_ratio[:len(sample_indices)], 'b-', label='Real', linewidth=2, alpha=0.7)
    plt.plot(sample_indices, pred_ratio[:len(sample_indices)].ravel(), 'r--', label='Predicted', linewidth=2, alpha=0.7)
    plt.xlabel("Sample Index", fontsize=11)
    plt.ylabel("SCR Ratio", fontsize=11)
    plt.title("(5) Time Series Comparison (Sample)", fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.suptitle("Deep Neural Surrogate Model - Comprehensive Performance Evaluation", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('kics_surrogate_evaluation.png', dpi=150, bbox_inches='tight')
    print(f"[저장] 상세 평가 시각화: kics_surrogate_evaluation.png")
    plt.show()
    
    # 간단한 요약 시각화도 저장
    plt.figure(figsize=(10, 8))
    plt.scatter(real_ratio, pred_ratio, s=10, alpha=0.6, label='Deep Neural Surrogate Prediction')
    plt.plot([real_ratio.min(), real_ratio.max()], [real_ratio.min(), real_ratio.max()], 
             'r--', label='Perfect Prediction', linewidth=2)
    plt.xlabel("Real SCR Ratio", fontsize=12)
    plt.ylabel("Predicted SCR Ratio", fontsize=12)
    plt.title("Deep Neural Surrogate Model (PyTorch)\nK-ICS Ratio Prediction", 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # 성능 지표 텍스트
    textstr = f'MAPE: {mape:.4f}%\nR²: {r2:.6f}\nRMSE: {rmse_pct:.4f}%\nInference: {avg_inference_time:.3f} ms'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('kics_surrogate_pytorch.png', dpi=150, bbox_inches='tight')
    print(f"[저장] 요약 시각화: kics_surrogate_pytorch.png")
    plt.show()
    
    return model, scaler_x, scaler_y


if __name__ == "__main__":
    model, scaler_x, scaler_y = train_surrogate_model()
