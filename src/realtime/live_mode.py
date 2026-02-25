"""
실시간 운영 모드 (Live Mode)
============================
제안서 연관:
- "완전한 실시간성(Real-time Availability)"
- "비동기 아키텍처로 위기 시에도 추론 가능"

핵심 기능:
- 실시간 데이터 수신 → 추론 → 안전 검사 → 출력
- 백그라운드 학습 가능 (Fast/Slow Track)
- 지연시간 모니터링

누수/편향/오버피팅 방지:
- 실시간 모드는 학습 없이 추론만 수행
- 학습은 별도 프로세스 (async_engine의 Slow Track)
"""

import time
import signal
import sys
import os
import numpy as np
from datetime import datetime
from typing import Dict, Any, Callable, Optional

# 프로젝트 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
project_src = os.path.dirname(script_dir)
sys.path.insert(0, project_src)


class LiveTradingLoop:
    """
    실시간 운영 루프
    
    데이터 수신 → 추론 → 안전 검사 → 출력
    """
    
    def __init__(self):
        """초기화"""
        # 의존 모듈 로드
        from realtime.async_engine import AsyncEngine
        from realtime.latency import LatencyMonitor
        from realtime.intraday import IntradayEstimator
        from safety.risk_control import RiskController
        
        self.engine = AsyncEngine()
        self.risk_controller = RiskController()
        self.latency_monitor = LatencyMonitor()
        self.intraday_estimator = IntradayEstimator()
        
        # [제안서 적용] DNN Surrogate 모델 로드
        self.surrogate = None
        self.use_surrogate = True
        self.kics_engine = None  # 폴백용
        self._load_surrogate_model()
        
        # 상태
        self.current_hedge = 0.5
        self.is_running = False
        self.step_count = 0
        
        # 히스토리 (모니터링용)
        self.action_history = []
        self.max_history = 1000
    
    def _load_surrogate_model(self):
        """DNN Surrogate 모델 로드 (제안서 적용)"""
        if not self.use_surrogate:
            # 폴백용 실제 엔진만 로드
            from core.kics_real import RatioKICSEngine
            self.kics_engine = RatioKICSEngine()
            return
        
        try:
            from core.kics_surrogate import RobustSurrogate
            
            # 모델 경로 탐색
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(script_dir))
            
            model_paths = [
                os.path.join(project_root, 'models', 'surrogate', 'kics_surrogate.pth'),
                os.path.join(project_root, 'models', 'kics_surrogate.pth'),
                os.path.join(script_dir, '..', 'models', 'surrogate', 'kics_surrogate.pth'),
            ]
            
            for path in model_paths:
                if os.path.exists(path):
                    try:
                        self.surrogate = RobustSurrogate(use_pytorch=True)
                        self.surrogate.load(path)
                        # 스케일러도 로드 시도
                        scaler_x_path = path.replace('.pth', '_scaler_x.pkl')
                        scaler_y_path = path.replace('.pth', '_scaler_y.pkl')
                        if os.path.exists(scaler_x_path) and os.path.exists(scaler_y_path):
                            import pickle
                            with open(scaler_x_path, 'rb') as f:
                                self.surrogate.scaler_x = pickle.load(f)
                            with open(scaler_y_path, 'rb') as f:
                                self.surrogate.scaler_y = pickle.load(f)
                        print(f"[Live] Surrogate 모델 로드 성공: {path}")
                        # 폴백용 실제 엔진도 준비
                        from core.kics_real import RatioKICSEngine
                        self.kics_engine = RatioKICSEngine()
                        return
                    except Exception as e:
                        print(f"[Live] Surrogate 모델 로드 실패 ({path}): {e}")
                        continue
            
            print("[Live] Surrogate 모델 파일 없음. 실제 엔진 사용 (폴백)")
            from core.kics_real import RatioKICSEngine
            self.kics_engine = RatioKICSEngine()
            self.surrogate = None
        except ImportError:
            print("[Live] kics_surrogate 모듈 없음. 실제 엔진 사용 (폴백)")
            from core.kics_real import RatioKICSEngine
            self.kics_engine = RatioKICSEngine()
            self.surrogate = None
        except Exception as e:
            print(f"[Live] Surrogate 로드 오류: {e}. 실제 엔진 사용 (폴백)")
            from core.kics_real import RatioKICSEngine
            self.kics_engine = RatioKICSEngine()
            self.surrogate = None
    
    def _get_market_data(self, data_source: str = 'simulation') -> Dict[str, float]:
        """
        시장 데이터 획득
        
        Args:
            data_source: 'simulation' | 'file' | 'api'
        
        Returns:
            {'VIX': ..., 'FX': ..., 'KOSPI': ..., 'timestamp': ...}
            
        누수 없음: 현재 시점 데이터만 반환
        """
        if data_source == 'simulation':
            # 시뮬레이션 데이터 생성
            base_vix = 20 + np.random.randn() * 5
            base_fx = 1300 + np.random.randn() * 10
            base_kospi = 2500 + np.random.randn() * 20
            
            # 간헐적 스트레스 이벤트
            if np.random.random() < 0.05:
                base_vix += np.random.uniform(10, 25)  # VIX 급등
            
            return {
                'VIX': max(10, base_vix),
                'FX': max(1100, base_fx),
                'KOSPI': max(2000, base_kospi),
                'timestamp': datetime.now()
            }
        
        elif data_source == 'file':
            # 파일에서 순차 읽기 (구현 시 추가)
            raise NotImplementedError("File source not implemented")
        
        elif data_source == 'api':
            # API 호출 (구현 시 추가)
            raise NotImplementedError("API source not implemented")
        
        return {}
    
    def _process_step(self, market_data: Dict[str, float]) -> Dict[str, Any]:
        """
        한 스텝 처리
        
        Anti-Leakage: 현재 시점 데이터만 사용
        """
        result = {}
        
        with self.latency_monitor.measure_context("total_pipeline"):
            # 1. Intraday 피처 추정
            with self.latency_monitor.measure_context("intraday"):
                self.intraday_estimator.update_tick(
                    market_data['timestamp'],
                    market_data['KOSPI'],
                    market_data['FX'],
                    market_data['VIX']
                )
                features = self.intraday_estimator.estimate_daily_features()
            
            # 2. 모델 입력 생성
            obs = np.array([
                self.current_hedge,
                np.clip(features['VIX'] / 100.0, 0, 1),
                np.clip((features['Correlation'] + 1) / 2, 0, 1),
                0.35  # SCR 비율 (실제로는 계산 필요)
            ], dtype=np.float32)
            
            # 3. AI 추론 (Fast Track)
            with self.latency_monitor.measure_context("ai_predict"):
                action, is_fallback = self.engine.predict(obs)
            
            # 4. K-ICS 비율 추정 (제안서 적용: Surrogate 사용)
            with self.latency_monitor.measure_context("kics_estimate"):
                scr_ratio = self._calculate_scr_with_surrogate(
                    self.current_hedge,
                    features['Correlation']
                )
                kics_ratio = (1.0 / scr_ratio) * 100 if scr_ratio > 0 else 999
    
    def _calculate_scr_with_surrogate(self, hedge_ratio, correlation):
        """
        SCR 계산 (Surrogate 우선, 폴백: 실제 엔진)
        
        [제안서 적용] DNN Surrogate 모델 사용 (밀리초 단위 고속 추론)
        """
        if self.use_surrogate and self.surrogate is not None:
            try:
                X = np.array([[hedge_ratio, correlation]])
                
                # 스케일러가 있으면 사용
                if hasattr(self.surrogate, 'scaler_x') and self.surrogate.scaler_x is not None:
                    X_scaled = self.surrogate.scaler_x.transform(X)
                    scr_scaled = self.surrogate.predict(X_scaled)
                    if hasattr(self.surrogate, 'scaler_y') and self.surrogate.scaler_y is not None:
                        scr = self.surrogate.scaler_y.inverse_transform(scr_scaled.reshape(-1, 1))[0, 0]
                    else:
                        scr = scr_scaled[0]
                else:
                    scr = self.surrogate.predict(X)[0]
                
                return float(scr)
            except Exception as e:
                # Surrogate 실패 시 실제 엔진으로 폴백
                if self.kics_engine is None:
                    from core.kics_real import RatioKICSEngine
                    self.kics_engine = RatioKICSEngine()
                return self.kics_engine.calculate_scr_ratio_batch(
                    np.array([hedge_ratio]),
                    np.array([correlation])
                )[0]
        else:
            # 실제 엔진 사용
            if self.kics_engine is None:
                from core.kics_real import RatioKICSEngine
                self.kics_engine = RatioKICSEngine()
            return self.kics_engine.calculate_scr_ratio_batch(
                np.array([hedge_ratio]),
                np.array([correlation])
            )[0]
            
            # 5. Safety Layer 적용
            with self.latency_monitor.measure_context("safety_layer"):
                safe_hedge, reason = self.risk_controller.apply_safety_rules(
                    float(action[0]) if hasattr(action, '__len__') else float(action),
                    self.current_hedge,
                    features['VIX'],
                    kics_ratio
                )
        
        # 결과 수집
        result = {
            'step': self.step_count,
            'timestamp': market_data['timestamp'],
            'vix': features['VIX'],
            'fx': market_data['FX'],
            'correlation': features['Correlation'],
            'kics_ratio': kics_ratio,
            'current_hedge': self.current_hedge,
            'proposed_action': float(action[0]) if hasattr(action, '__len__') else float(action),
            'safe_hedge': safe_hedge,
            'reason': reason,
            'is_fallback': is_fallback
        }
        
        # 상태 업데이트
        self.current_hedge = safe_hedge
        
        # 히스토리 저장
        self.action_history.append(result)
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
        
        return result
    
    def run(
        self, 
        interval_sec: float = 5.0,
        max_steps: int = None,
        data_source: str = 'simulation',
        verbose: bool = True
    ):
        """
        실시간 루프 실행
        
        Args:
            interval_sec: 스텝 간격 (초)
            max_steps: 최대 스텝 수 (None: 무한)
            data_source: 데이터 소스
            verbose: 상세 출력
        """
        print("=" * 70)
        print("Dynamic Shield v3.0 - 실시간 운영 모드")
        print("=" * 70)
        print(f"  간격: {interval_sec}초")
        print(f"  데이터 소스: {data_source}")
        print(f"  최대 스텝: {max_steps or '무제한'}")
        print("-" * 70)
        print("Ctrl+C로 종료\n")
        
        self.is_running = True
        self.step_count = 0
        
        # 시그널 핸들러 (Ctrl+C)
        def signal_handler(sig, frame):
            print("\n")
            print("=" * 70)
            print("종료 신호 수신")
            self.is_running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # 전일 종가 설정 (시뮬레이션)
        self.intraday_estimator.set_daily_close(
            kospi=2500, fx=1300, vix=18
        )
        
        try:
            while self.is_running:
                self.step_count += 1
                
                # 최대 스텝 체크
                if max_steps and self.step_count > max_steps:
                    print(f"\n[완료] 최대 스텝 {max_steps} 도달")
                    break
                
                # 데이터 획득
                market_data = self._get_market_data(data_source)
                
                # 스텝 처리
                result = self._process_step(market_data)
                
                # 출력
                if verbose:
                    self._print_step(result)
                
                # 대기
                time.sleep(interval_sec)
                
        except Exception as e:
            print(f"\n[오류] {e}")
        finally:
            self._print_summary()
    
    def _print_step(self, result: Dict[str, Any]):
        """스텝 결과 출력"""
        ts = result['timestamp'].strftime('%H:%M:%S') if hasattr(result['timestamp'], 'strftime') else str(result['timestamp'])
        
        # 컬러 코드 (터미널 지원 시)
        if result['reason'].startswith('CRITICAL') or result['reason'].startswith('PANIC'):
            prefix = "🔴"
        elif result['reason'].startswith('DANGER') or result['reason'].startswith('TRANSITION'):
            prefix = "🟡"
        else:
            prefix = "🟢"
        
        hedge_change = result['safe_hedge'] - result['current_hedge']
        change_str = f"+{hedge_change:.1%}" if hedge_change >= 0 else f"{hedge_change:.1%}"
        
        print(f"{prefix} [{ts}] Step {result['step']:>4} | "
              f"VIX={result['vix']:>5.1f} K-ICS={result['kics_ratio']:>5.1f}% | "
              f"Hedge: {result['current_hedge']:.1%} → {result['safe_hedge']:.1%} ({change_str}) | "
              f"{result['reason']}")
    
    def _print_summary(self):
        """종료 시 요약 출력"""
        print("\n" + "=" * 70)
        print("실행 요약")
        print("=" * 70)
        
        print(f"  총 스텝: {self.step_count}")
        print(f"  최종 헤지: {self.current_hedge:.1%}")
        
        if self.action_history:
            vix_values = [r['vix'] for r in self.action_history]
            print(f"  VIX 범위: {min(vix_values):.1f} ~ {max(vix_values):.1f}")
            
            critical_count = sum(1 for r in self.action_history if 'CRITICAL' in r['reason'])
            panic_count = sum(1 for r in self.action_history if 'PANIC' in r['reason'])
            print(f"  CRITICAL 발생: {critical_count}회")
            print(f"  PANIC 발생: {panic_count}회")
        
        # 지연시간 리포트
        print("\n[지연시간 리포트]")
        self.latency_monitor.print_report()


def run_live_mode(interval: float = 5.0, max_steps: int = None):
    """실시간 모드 실행 헬퍼"""
    loop = LiveTradingLoop()
    loop.run(interval_sec=interval, max_steps=max_steps)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Dynamic Shield 실시간 모드')
    parser.add_argument('--interval', '-i', type=float, default=5.0, help='스텝 간격 (초)')
    parser.add_argument('--steps', '-n', type=int, default=None, help='최대 스텝 수')
    
    args = parser.parse_args()
    
    run_live_mode(interval=args.interval, max_steps=args.steps)
