"""
독립 Risk Control 모듈 (Safety Layer)
======================================
기존 gym_environment.py와 agent.py에 산재된 Safety 로직을 통합 관리

제안서 연관:
- "VIX > 40: Emergency De-risking"
- "K-ICS < 100%: Force 100% Hedge"
- "Max Step: ±15% per period (급발진 방지)"
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional


@dataclass
class RiskConfig:
    """Risk Control 설정"""
    # VIX 임계값
    vix_panic_threshold: float = 40.0      # 패닉 (제안서 기준)
    vix_transition_threshold: float = 25.0  # 전환 구간
    vix_normal_threshold: float = 20.0      # 정상 구간
    
    # K-ICS 임계값 (3단계 Safety Layer)
    kics_critical_threshold: float = 100.0  # Level 2: 즉시 100% 헤지
    kics_level1_threshold: float = 130.0    # Level 1: 적기시정조치 예방, 점진적 상향
    kics_danger_threshold: float = 130.0    # 경고 구간 (Level 1과 동일)
    kics_safe_threshold: float = 150.0      # 안전 구간
    
    # 헤지 비율 제한
    max_hedge_change: float = 0.15          # 최대 1회 변동 (±15%)
    min_hedge_ratio: float = 0.0            # 최소 헤지 비율
    max_hedge_ratio: float = 1.0            # 최대 헤지 비율
    
    # De-risking 설정
    gradual_derisking_step: float = 0.10    # 점진적 헤지 증가 단위
    emergency_hedge_target: float = 1.0     # 긴급 시 목표 헤지 비율
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RiskConfig':
        """딕셔너리에서 설정 로드"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
    
    @classmethod
    def load_from_yaml(cls, config_path: str = None) -> 'RiskConfig':
        """YAML 설정 파일에서 로드"""
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from core.config_loader import ConfigLoader
            
            loader = ConfigLoader()
            agent_config = loader.get_agent_config()
            
            return cls(
                vix_panic_threshold=agent_config.get('vix_panic_threshold', 40.0),
                vix_transition_threshold=agent_config.get('vix_transition_threshold', 25.0),
                kics_critical_threshold=agent_config.get('kics_critical_threshold', 100.0),
                kics_danger_threshold=agent_config.get('kics_danger_threshold', 120.0),
                max_hedge_change=agent_config.get('max_hedge_change', 0.15),
            )
        except Exception as e:
            print(f"[RiskConfig] Config 로드 실패, 기본값 사용: {e}")
            return cls()


@dataclass
class RiskState:
    """현재 리스크 상태 추적"""
    is_derisking: bool = False
    derisking_target: float = 1.0
    consecutive_danger_days: int = 0
    last_action_reason: str = ""


class RiskController:
    """
    독립 Risk Control 모듈
    
    기능:
    1. Safety Rule 적용 (VIX, K-ICS 기반)
    2. Gradual De-risking 프로토콜
    3. 행동 제한 (Max Step, 범위 제한)
    """
    
    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.state = RiskState()
        
    def apply_safety_rules(
        self, 
        proposed_action: float, 
        current_hedge: float,
        vix: float,
        kics_ratio: float,
        correlation: float = 0.0
    ) -> Tuple[float, str]:
        """
        Safety Rule 적용
        
        Args:
            proposed_action: AI가 제안한 헤지 비율 변동 (-1 ~ 1)
            current_hedge: 현재 헤지 비율 (0 ~ 1)
            vix: 현재 VIX 지수
            kics_ratio: 현재 K-ICS 비율 (%)
            correlation: 주식-환율 상관계수
            
        Returns:
            (safe_hedge_ratio, reason): 안전 처리된 헤지 비율과 사유
        """
        # 1. Action을 헤지 변동으로 변환 (±15%)
        hedge_change = proposed_action * self.config.max_hedge_change
        new_hedge = current_hedge + hedge_change
        
        # === Level 2: K-ICS < 100% → 즉시 100% 헤지 (규제 위반 방지) ===
        if kics_ratio < self.config.kics_critical_threshold:
            self.state.is_derisking = True
            self.state.derisking_target = self.config.emergency_hedge_target
            self.state.last_action_reason = f"Level2: K-ICS {kics_ratio:.1f}% < 100%, FORCE 100% HEDGE"
            return 1.0, self.state.last_action_reason
        
        # === Level 1: K-ICS < 130% → 점진적 De-risking (적기시정조치 예방) ===
        if kics_ratio < self.config.kics_level1_threshold:
            self.state.consecutive_danger_days += 1
            # 위험 구간: 단계적 헤지 증가
            target = min(current_hedge + self.config.gradual_derisking_step, 1.0)
            self.state.is_derisking = True
            self.state.derisking_target = target
            self.state.last_action_reason = f"Level1: K-ICS {kics_ratio:.1f}% < 130%, Gradual De-risking"
            return target, self.state.last_action_reason
        else:
            self.state.consecutive_danger_days = 0
        
        # === Level 1 진행: Gradual De-risking ===
        if self.state.is_derisking:
            if current_hedge >= self.state.derisking_target:
                self.state.is_derisking = False
                self.state.last_action_reason = "De-risking Complete"
            else:
                new_hedge = min(current_hedge + self.config.gradual_derisking_step, 
                               self.state.derisking_target)
                self.state.last_action_reason = "Gradual De-risking in Progress"
                return self._clamp_hedge(new_hedge), self.state.last_action_reason
        
        # === Safety Layer 2 (PDF 5.1): VIX > 40 또는 유동성 위기 → 즉시 100% 헤지 ===
        if vix >= self.config.vix_panic_threshold:
            self.state.last_action_reason = f"Level2: VIX={vix:.1f}>=40, FORCE 100% HEDGE"
            return 1.0, self.state.last_action_reason
        
        elif vix >= self.config.vix_transition_threshold:
            # 전환: AI 제안을 상향 바이어스
            biased_hedge = max(new_hedge, current_hedge)
            self.state.last_action_reason = f"TRANSITION: VIX={vix:.1f}"
            return self._clamp_hedge(biased_hedge), self.state.last_action_reason
        
        else:
            # 정상: AI 제안 수용 (범위 제한만 적용)
            self.state.last_action_reason = f"NORMAL: VIX={vix:.1f}"
            return self._clamp_hedge(new_hedge), self.state.last_action_reason
    
    def _clamp_hedge(self, hedge: float) -> float:
        """헤지 비율 범위 제한"""
        return np.clip(hedge, self.config.min_hedge_ratio, self.config.max_hedge_ratio)
    
    def get_penalty(self, kics_ratio: float, vix: float) -> float:
        """
        K-ICS 위반 및 VIX 미대응에 대한 페널티 계산 (RL Reward용)
        
        Returns:
            penalty: 음수 페널티 값
        """
        penalty = 0.0
        
        # K-ICS 페널티 (Level 2: <100%, Level 1: <130%)
        if kics_ratio < self.config.kics_critical_threshold:
            penalty -= 1000.0  # Level 2 치명적
        elif kics_ratio < self.config.kics_level1_threshold:
            penalty -= (self.config.kics_level1_threshold - kics_ratio) * 5
        
        # VIX 미대응 페널티는 apply_safety_rules에서 처리
        
        return penalty
    
    def reset(self):
        """상태 초기화"""
        self.state = RiskState()
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            'is_derisking': self.state.is_derisking,
            'derisking_target': self.state.derisking_target,
            'consecutive_danger_days': self.state.consecutive_danger_days,
            'last_reason': self.state.last_action_reason
        }


# === 테스트 코드 ===
if __name__ == "__main__":
    print("=" * 60)
    print("Risk Controller 테스트")
    print("=" * 60)
    
    controller = RiskController()
    
    # 테스트 시나리오
    test_cases = [
        # (proposed_action, current_hedge, vix, kics_ratio, expected_behavior)
        (0.0, 0.5, 15, 180, "NORMAL: 유지"),
        (0.0, 0.5, 45, 180, "PANIC: 헤지 증가"),
        (0.0, 0.5, 15, 95, "CRITICAL: 100% 강제"),
        (0.0, 0.5, 15, 115, "Level1: 단계적 증가"),
        (-0.5, 0.8, 15, 200, "NORMAL: AI 제안 수용 (감소)"),
    ]
    
    print("\n[테스트 케이스]")
    print("-" * 60)
    
    for action, hedge, vix, kics, expected in test_cases:
        controller.reset()
        new_hedge, reason = controller.apply_safety_rules(action, hedge, vix, kics)
        status = "✓" if expected.split(":")[0] in reason else "?"
        print(f"{status} VIX={vix:2d}, K-ICS={kics:3d}%, Hedge {hedge:.1f}->{new_hedge:.1f} | {reason}")
    
    print("\n[설정값]")
    print(f"  VIX Panic: {controller.config.vix_panic_threshold}")
    print(f"  K-ICS Critical: {controller.config.kics_critical_threshold}%")
    print(f"  Max Step: ±{controller.config.max_hedge_change*100:.0f}%")
