"""
Dynamic Shield v3.0 - Validation Package
=========================================
Capital Optimization, not Prediction (환율 예측이 아닌 자본 최적화)

검증 및 시각화 모듈
"""

from .proof_diversification import prove_risk_paradox
from .solvency_visualizer import run_solvency_analysis
from .stress_safety import run_stress_test
from .backtest import run_full_analysis, BacktestEngine
from .advanced_viz import run_advanced_visualization

__all__ = [
    'prove_risk_paradox',
    'run_solvency_analysis',
    'run_stress_test',
    'run_full_analysis',
    'BacktestEngine',
    'run_advanced_visualization'
]
