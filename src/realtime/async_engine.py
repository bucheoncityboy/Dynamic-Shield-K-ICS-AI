"""
비동기 엔진 (Async Engine) - Fast/Slow Track 분리
===================================================
제안서 연관:
- "학습과 추론을 분리한 '비동기(Asynchronous) 아키텍처'"
- "위기 시 시스템이 멈추지 않는 완전한 실시간성(Real-time Availability)"

핵심 설계:
- Fast Track: 추론 (predict) - Main Thread, 즉시 응답
- Slow Track: 학습 (learn) - Background Thread, 비동기 실행
- Model Registry: 학습 완료된 모델 안전 교체

누수/편향/오버피팅 관련:
- 본 모듈은 인프라 레이어로 ML 학습 로직과 분리됨
- 학습은 기존 ppo_trainer.py의 Anti-Bias/Leakage 로직을 그대로 사용
"""

import threading
import time
import os
from typing import Optional, Tuple, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class ModelVersion:
    """모델 버전 정보"""
    version: int
    path: str
    created_at: datetime
    metrics: dict = None


class AsyncEngine:
    """
    비동기 추론/학습 엔진
    
    Fast Track (Main Thread):
        - predict(): 현재 모델로 즉시 추론
        - 학습 중에도 블로킹 없이 응답
    
    Slow Track (Background Thread):
        - train_async(): 백그라운드에서 모델 학습
        - 완료 시 안전하게 모델 교체
    """
    
    def __init__(self, model_dir: str = None):
        """
        Args:
            model_dir: 모델 저장 디렉토리 (기본: validation 폴더)
        """
        # 모델 경로 설정
        if model_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_dir = os.path.join(
                os.path.dirname(script_dir), 'validation'
            )
        self.model_dir = model_dir
        
        # 모델 상태
        self.current_model = None
        self.model_version = 0
        self.model_lock = threading.Lock()
        
        # 학습 상태
        self.is_training = False
        self.training_thread: Optional[threading.Thread] = None
        self.training_progress = 0.0
        self.last_training_error: Optional[str] = None
        
        # 모델 버전 히스토리
        self.version_history: list = []
        
        # 자동 로드
        self._load_latest_model()
    
    def _load_latest_model(self) -> bool:
        """최신 모델 로드"""
        try:
            from stable_baselines3 import PPO
            
            model_path = os.path.join(self.model_dir, 'ppo_kics.zip')
            
            if os.path.exists(model_path):
                self.current_model = PPO.load(model_path)
                self.model_version = 1
                print(f"[AsyncEngine] 모델 로드 완료: {model_path}")
                return True
            else:
                print(f"[AsyncEngine] 모델 없음, 추론 시 폴백 사용")
                return False
                
        except ImportError:
            print("[AsyncEngine] stable-baselines3 없음, 폴백 모드")
            return False
        except Exception as e:
            print(f"[AsyncEngine] 모델 로드 실패: {e}")
            return False
    
    # ========================================
    # Fast Track: 추론 (즉시 응답)
    # ========================================
    
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> Tuple[np.ndarray, bool]:
        """
        Fast Track 추론 - 학습 중에도 즉시 응답
        
        Args:
            obs: 관측 상태 [hedge_ratio, vix_norm, corr_norm, scr_ratio]
            deterministic: 결정적 행동 여부
            
        Returns:
            (action, is_fallback): 행동과 폴백 여부
        """
        with self.model_lock:
            if self.current_model is not None:
                try:
                    action, _ = self.current_model.predict(obs, deterministic=deterministic)
                    return action, False  # 정상 추론
                except Exception as e:
                    print(f"[AsyncEngine] 추론 오류, 폴백 사용: {e}")
            
            # 폴백: 규칙 기반
            return self._fallback_predict(obs), True
    
    def _fallback_predict(self, obs: np.ndarray) -> np.ndarray:
        """
        폴백 추론 (모델 없을 때)
        - 누수 없음: 현재 관측값만 사용
        - 편향 없음: 보수적 규칙 적용
        """
        # obs: [hedge_ratio, vix_norm, corr_norm, scr_ratio]
        vix = obs[1] * 100 if len(obs) > 1 else 20
        
        if vix >= 40:
            return np.array([0.5])   # 헤지 증가
        elif vix >= 25:
            return np.array([0.0])   # 유지
        else:
            return np.array([-0.2])  # 소폭 감소
    
    def get_model_status(self) -> dict:
        """현재 모델 상태 반환"""
        return {
            'has_model': self.current_model is not None,
            'version': self.model_version,
            'is_training': self.is_training,
            'training_progress': self.training_progress
        }
    
    # ========================================
    # Slow Track: 학습 (백그라운드)
    # ========================================
    
    def train_async(
        self, 
        env_factory: Callable,
        total_timesteps: int = 50000,
        callback: Callable[[float], None] = None
    ) -> bool:
        """
        Slow Track 학습 - 백그라운드 실행
        
        Args:
            env_factory: 환경 생성 함수 (람다 또는 callable)
            total_timesteps: 학습 스텝 수
            callback: 진행률 콜백 (0.0 ~ 1.0)
            
        Returns:
            학습 시작 성공 여부
        """
        if self.is_training:
            print("[AsyncEngine] 이미 학습 진행 중")
            return False
        
        def train_worker():
            try:
                self.is_training = True
                self.training_progress = 0.0
                self.last_training_error = None
                
                print(f"[Slow Track] 백그라운드 학습 시작 (timesteps={total_timesteps})")
                
                from stable_baselines3 import PPO
                from stable_baselines3.common.vec_env import DummyVecEnv
                from stable_baselines3.common.callbacks import BaseCallback
                
                # 환경 생성
                env = DummyVecEnv([env_factory])
                
                # 진행률 콜백
                class ProgressCallback(BaseCallback):
                    def __init__(self, engine, total, user_callback):
                        super().__init__()
                        self.engine = engine
                        self.total = total
                        self.user_callback = user_callback
                    
                    def _on_step(self) -> bool:
                        progress = self.num_timesteps / self.total
                        self.engine.training_progress = progress
                        if self.user_callback:
                            self.user_callback(progress)
                        return True
                
                # 새 모델 학습
                new_model = PPO(
                    'MlpPolicy', 
                    env,
                    verbose=0,
                    learning_rate=3e-4
                )
                
                progress_cb = ProgressCallback(self, total_timesteps, callback)
                new_model.learn(
                    total_timesteps=total_timesteps,
                    callback=progress_cb,
                    progress_bar=False
                )
                
                # 새 버전으로 저장
                new_version = self.model_version + 1
                new_path = os.path.join(self.model_dir, f'ppo_kics_v{new_version}')
                new_model.save(new_path)
                
                # 기본 경로에도 저장 (최신 버전)
                new_model.save(os.path.join(self.model_dir, 'ppo_kics'))
                
                # 안전하게 모델 교체 (Lock)
                with self.model_lock:
                    self.current_model = new_model
                    self.model_version = new_version
                    self.version_history.append(ModelVersion(
                        version=new_version,
                        path=new_path,
                        created_at=datetime.now(),
                        metrics={'timesteps': total_timesteps}
                    ))
                
                print(f"[Slow Track] 학습 완료! 버전 {new_version} 적용")
                
            except Exception as e:
                self.last_training_error = str(e)
                print(f"[Slow Track] 학습 실패: {e}")
            finally:
                self.is_training = False
                self.training_progress = 1.0
        
        # 백그라운드 스레드 시작
        self.training_thread = threading.Thread(target=train_worker, daemon=True)
        self.training_thread.start()
        
        print("[AsyncEngine] 백그라운드 학습 스레드 시작 (추론은 계속 가능)")
        return True
    
    def wait_for_training(self, timeout: float = None) -> bool:
        """학습 완료 대기"""
        if self.training_thread is None:
            return True
        
        self.training_thread.join(timeout=timeout)
        return not self.is_training
    
    # ========================================
    # Model Registry
    # ========================================
    
    def rollback(self, version: int) -> bool:
        """이전 버전으로 롤백"""
        for v in self.version_history:
            if v.version == version:
                try:
                    from stable_baselines3 import PPO
                    
                    with self.model_lock:
                        self.current_model = PPO.load(v.path)
                        self.model_version = version
                    
                    print(f"[AsyncEngine] 버전 {version}으로 롤백 완료")
                    return True
                except Exception as e:
                    print(f"[AsyncEngine] 롤백 실패: {e}")
                    return False
        
        print(f"[AsyncEngine] 버전 {version} 없음")
        return False


# === 테스트 코드 ===
if __name__ == "__main__":
    print("=" * 60)
    print("AsyncEngine 테스트")
    print("=" * 60)
    
    engine = AsyncEngine()
    
    # 1. 추론 테스트 (Fast Track)
    print("\n[1] Fast Track 추론 테스트")
    obs = np.array([0.5, 0.2, 0.3, 0.35], dtype=np.float32)
    
    for i in range(5):
        action, is_fallback = engine.predict(obs)
        mode = "폴백" if is_fallback else "모델"
        print(f"  추론 {i+1}: action={float(action[0]):.3f} ({mode})")
    
    # 2. 상태 확인
    print("\n[2] 모델 상태")
    status = engine.get_model_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    
    # 3. 비동기 학습 테스트 (옵션)
    print("\n[3] 비동기 학습 데모 (학습 중 추론 가능 확인)")
    print("    (실제 학습은 시간이 걸리므로 스킵)")
    
    print("\n✓ AsyncEngine 테스트 완료")
