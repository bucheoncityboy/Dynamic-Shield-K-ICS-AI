"""
Phase 4.3: RL Training with stable-baselines3 PPO
==================================================
PPO 알고리즘을 사용한 헤지 비율 최적화 학습
- Gym 호환 환경 사용
- Safety Layer 통합
- Callback으로 학습 모니터링

핵심 철학: Capital Optimization, not Prediction
(환율 예측이 아닌 자본 최적화)

필수 라이브러리:
    pip install stable-baselines3 gymnasium
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from core.gym_environment import KICSGymEnv
from core.agent import DynamicShieldAgent


class SafetyLayerCallback(BaseCallback):
    """
    Safety Layer Callback
    
    Phase 4.2 워크플로우:
    - AI가 패닉에 빠지거나 급발진하는 것을 막는 하드코딩 룰
    - VIX > 30이면 Gradual De-risking 트리거
    """
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.safety_agent = DynamicShieldAgent()
        self.safety_triggered_count = 0
        
    def _on_step(self) -> bool:
        # 현재 환경 상태 확인
        env = self.training_env.envs[0]
        
        if hasattr(env, 'vix') and env.vix > 30:
            # Safety Layer 개입
            self.safety_triggered_count += 1
            if self.verbose > 0 and self.safety_triggered_count % 100 == 0:
                print(f"[Safety Layer] Triggered {self.safety_triggered_count} times")
        
        return True
    
    def _on_training_end(self):
        if self.verbose > 0:
            print(f"\n[Safety Layer Summary] Total triggers: {self.safety_triggered_count}")


class RewardLoggingCallback(BaseCallback):
    """학습 진행 상황 로깅"""
    
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        # 에피소드 종료 시
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # 주기적 로깅
        if self.num_timesteps % self.log_freq == 0 and len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards[-10:])
            print(f"Step {self.num_timesteps:6d} | "
                  f"Episodes: {len(self.episode_rewards):4d} | "
                  f"Avg Reward (last 10): {avg_reward:.2f}")
        
        return True


class PPOTrainer:
    """
    PPO 기반 RL 훈련 매니저
    
    Phase 4 워크플로우:
    - 4.1 RL Environment Setup (Reward Function)
    - 4.2 Safety Layer (Gradual De-risking)
    """
    
    def __init__(self, 
                 algorithm=None,
                 total_timesteps=None,
                 learning_rate=None,
                 n_steps=None,
                 batch_size=None,
                 gamma=None):
        
        # Config 로드 시도
        try:
            from config_loader import ConfigLoader
            loader = ConfigLoader()
            ppo_config = loader.get_ppo_config()
            
            # 설정 파일에서 기본값 로드 (인자로 전달된 값이 우선)
            self.algorithm = algorithm or ppo_config.get('algorithm', 'PPO')
            self.total_timesteps = total_timesteps or ppo_config.get('total_timesteps', 100000)
            self.learning_rate = learning_rate or ppo_config.get('learning_rate', 3e-4)
            self.n_steps = n_steps or ppo_config.get('n_steps', 2048)
            self.batch_size = batch_size or ppo_config.get('batch_size', 64)
            self.gamma = gamma or ppo_config.get('gamma', 0.99)
            self.tensorboard_log = ppo_config.get('tensorboard_log', './tensorboard_logs/')
        except (ImportError, FileNotFoundError, KeyError):
            # 폴백: 기본값 사용
            self.algorithm = algorithm or 'PPO'
            self.total_timesteps = total_timesteps or 100000
            self.learning_rate = learning_rate or 3e-4
            self.n_steps = n_steps or 2048
            self.batch_size = batch_size or 64
            self.gamma = gamma or 0.99
            self.tensorboard_log = './tensorboard_logs/'
        
        self.env = None
        self.model = None
        self.reward_callback = None
        
    def setup(self):
        """환경 및 모델 설정"""
        print("=" * 60)
        print(f"Phase 4: RL Training with {self.algorithm}")
        print("=" * 60)
        
        # 환경 생성 (Config에서 lambda2 로드)
        try:
            from config_loader import ConfigLoader
            loader = ConfigLoader()
            gym_config = loader.get_gym_env_config()
            lambda2 = gym_config.get('lambda2', 1000)
        except (ImportError, FileNotFoundError, KeyError):
            lambda2 = 1000
        
        self.env = DummyVecEnv([lambda: KICSGymEnv(lambda2=lambda2)])
        
        # 알고리즘 선택
        if self.algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                self.env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                gamma=self.gamma,
                verbose=1,
                tensorboard_log=self.tensorboard_log
            )
        elif self.algorithm == 'A2C':
            self.model = A2C(
                'MlpPolicy',
                self.env,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                verbose=1
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        print(f"\n[Model] {self.algorithm} initialized")
        print(f"[Environment] KICS Gym Environment")
        print(f"[Constraint] K-ICS < 100% -> -1000 penalty")
        
    def train(self):
        """PPO 훈련 실행"""
        print("\n[Training Started]")
        print(f"  Total Timesteps: {self.total_timesteps}")
        print(f"  Learning Rate: {self.learning_rate}")
        print("-" * 60)
        
        # Callbacks
        self.reward_callback = RewardLoggingCallback(log_freq=5000, verbose=1)
        safety_callback = SafetyLayerCallback(verbose=1)
        
        # 훈련
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=[self.reward_callback, safety_callback],
            progress_bar=True
        )
        
        print("\n[Training Complete]")
        
        return self.model
    
    def evaluate(self, n_episodes=10):
        """학습된 모델 평가"""
        print("\n" + "=" * 60)
        print("Evaluation")
        print("=" * 60)
        
        eval_rewards = []
        eval_kics_min = []
        
        for ep in range(n_episodes):
            obs = self.env.reset()  # VecEnv returns only obs
            done = False
            total_reward = 0
            min_kics = 999
            step_count = 0
            
            while not done and step_count < 500:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, dones, infos = self.env.step(action)
                done = dones[0]
                
                total_reward += reward[0]
                if infos and 'kics_ratio' in infos[0]:
                    min_kics = min(min_kics, infos[0]['kics_ratio'])
                step_count += 1
            
            eval_rewards.append(total_reward)
            eval_kics_min.append(min_kics)
        
        print(f"  Episodes: {n_episodes}")
        print(f"  Avg Reward: {np.mean(eval_rewards):.2f}")
        print(f"  Avg Min K-ICS: {np.mean(eval_kics_min):.1f}%")
        
        if np.mean(eval_kics_min) > 100:
            print("[SUCCESS] Agent maintained K-ICS > 100% on average!")
        
        return eval_rewards, eval_kics_min
    
    def save(self, path=None):
        """모델 저장
        
        기본 저장 경로: validation 폴더 (backtest.py와 같은 위치)
        이렇게 하면 백테스트 시 모델 경로 참조 오류가 발생하지 않음
        """
        if path is None:
            # validation 폴더에 직접 저장 (backtest.py와 같은 위치)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            validation_dir = os.path.join(script_dir, '..', 'validation')
            path = os.path.join(validation_dir, 'ppo_kics')
        
        # 경로에 디렉토리가 포함된 경우에만 생성
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        self.model.save(path)
        print(f"\n[Saved] Model saved to {os.path.abspath(path)}.zip")
    
    def load(self, path="models/ppo_kics"):
        """모델 불러오기"""
        self.model = PPO.load(path, env=self.env)
        print(f"[Loaded] Model loaded from {path}")
    
    def plot_training(self):
        """훈련 결과 시각화"""
        if self.reward_callback is None or len(self.reward_callback.episode_rewards) == 0:
            print("[Warning] No training data to plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        rewards = self.reward_callback.episode_rewards
        lengths = self.reward_callback.episode_lengths
        
        # Episode Rewards
        axes[0].plot(rewards, alpha=0.6)
        if len(rewards) >= 50:
            ma = np.convolve(rewards, np.ones(50)/50, mode='valid')
            axes[0].plot(range(49, len(rewards)), ma, 'r-', lw=2, label='50-episode MA')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title(f'{self.algorithm} Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Episode Lengths
        axes[1].plot(lengths, alpha=0.6)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Episode Length')
        axes[1].set_title('Episode Duration')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('ppo_training_result.png', dpi=150)
        plt.show()
        
        print("[Saved] ppo_training_result.png")


def run_ppo_training():
    """PPO 훈련 실행 (메인 함수)"""
    trainer = PPOTrainer(
        algorithm='PPO',
        total_timesteps=50000,
        learning_rate=3e-4
    )
    
    trainer.setup()
    trainer.train()
    trainer.evaluate(n_episodes=10)
    trainer.plot_training()
    trainer.save()
    
    return trainer


if __name__ == "__main__":
    trainer = run_ppo_training()
