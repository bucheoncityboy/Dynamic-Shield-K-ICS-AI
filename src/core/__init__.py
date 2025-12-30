"""
Dynamic Shield v3.0 - Source Package
=====================================
Capital Optimization, not Prediction (환율 예측이 아닌 자본 최적화)

K-ICS 연계형 AI 환헤지 솔루션 프로토타입
"""

# Core modules
from .kics_real import RatioKICSEngine
from .kics_surrogate import RobustSurrogate, train_surrogate_model
from .regime import RegimeClassifier
from .agent import DynamicShieldAgent
from .environment import KICSEnvironment
from .system import DynamicShieldSystem
from .gym_environment import KICSGymEnv

__all__ = [
    'RatioKICSEngine',
    'RobustSurrogate',
    'train_surrogate_model',
    'RegimeClassifier',
    'DynamicShieldAgent',
    'KICSEnvironment',
    'KICSGymEnv',
    'DynamicShieldSystem'
]

__version__ = '3.0.0'
