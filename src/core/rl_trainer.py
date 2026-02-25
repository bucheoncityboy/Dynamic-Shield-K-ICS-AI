"""
Phase 4.3: RL Training (강화학습 훈련)
======================================
Simple Q-Learning을 사용한 헤지 비율 최적화 학습
- State: [Hedge_Ratio, VIX, Correlation, SCR_Ratio]
- Action: 0=감소, 1=유지, 2=증가, 3=대폭 증가
- Safety Layer가 최종 행동을 오버라이드
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os

# Import path setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from environment import KICSEnvironment
    from agent import DynamicShieldAgent
    from validation.backtest import generate_market_scenario
except ImportError:
    from .environment import KICSEnvironment
    from .agent import DynamicShieldAgent
    from validation.backtest import generate_market_scenario


class QLearningAgent:
    """
    Simple Q-Learning Agent
    - 테이블 기반 Q-Learning
    - 연속 상태를 이산화(Discretization)하여 학습
    """
    def __init__(self, 
                 n_actions=4,
                 learning_rate=0.1,
                 discount_factor=0.95,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01):
        
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-Table: state -> action values
        self.q_table = defaultdict(lambda: np.zeros(n_actions))
        
        # Safety Layer
        self.safety_layer = DynamicShieldAgent()
        
    def discretize_state(self, state):
        """
        연속 상태를 이산 상태로 변환
        state: [hedge_ratio, vix_norm, corr_norm, scr_ratio]
        """
        hedge_bin = int(state[0] * 10)  # 0-10 (11 bins)
        vix_bin = int(state[1] * 10)    # 0-10 (11 bins)
        corr_bin = int(state[2] * 5)    # 0-5 (6 bins)
        scr_bin = int(state[3] * 10)    # 0-10 (11 bins)
        
        return (hedge_bin, vix_bin, corr_bin, scr_bin)
    
    def get_action(self, state, env, training=True):
        """
        Epsilon-greedy 정책으로 행동 선택
        Safety Layer가 최종 행동을 오버라이드할 수 있음
        """
        discrete_state = self.discretize_state(state)
        
        # 1. Epsilon-greedy exploration
        if training and np.random.random() < self.epsilon:
            proposed_action = np.random.randint(self.n_actions)
        else:
            proposed_action = np.argmax(self.q_table[discrete_state])
        
        # 2. Safety Layer Check (오버라이드 가능)
        vix = state[1] * 100
        kics_ratio = env.get_kics_ratio()
        
        # 위험 상황이면 Safety Layer가 개입
        if kics_ratio < 120 or vix >= 30:
            safety_action, info = self.safety_layer.get_action(state, env)
            return safety_action, {'source': 'safety', 'reason': info['reason']}
        
        return proposed_action, {'source': 'rl', 'reason': 'Q-Learning policy'}
    
    def update(self, state, action, reward, next_state, done):
        """Q-Table 업데이트"""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Q-Learning update
        current_q = self.q_table[discrete_state][action]
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[discrete_next_state])
        
        self.q_table[discrete_state][action] += self.lr * (target - current_q)
    
    def decay_epsilon(self):
        """Epsilon 감소"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


class RLTrainer:
    """RL 훈련 매니저"""
    
    def __init__(self):
        self.env = KICSEnvironment(lambda2=1000)
        self.agent = QLearningAgent()
        self.episode_rewards = []
        self.episode_lengths = []
        
    def train(self, n_episodes=500, verbose=True):
        """Q-Learning 훈련"""
        print("=" * 60)
        print("Phase 4.3: RL Training (Q-Learning)")
        print("=" * 60)
        
        # 다양한 시나리오 준비
        scenarios = ['normal', '2008_crisis', '2020_pandemic']
        
        for episode in range(n_episodes):
            # 랜덤 시나리오 선택
            scenario = np.random.choice(scenarios)
            market_data = generate_market_scenario(200, scenario)
            
            # 환경 초기화
            state = self.env.reset(
                initial_vix=market_data['VIX'].iloc[0],
                initial_corr=market_data['Correlation'].iloc[0]
            )
            
            total_reward = 0
            steps = 0
            
            for i in range(len(market_data) - 1):
                # 행동 선택
                action, info = self.agent.get_action(state, self.env, training=True)
                
                # 다음 시장 데이터
                next_vix = market_data['VIX'].iloc[i + 1]
                next_corr = market_data['Correlation'].iloc[i + 1]
                
                # 환경 스텝
                next_state, reward, done, step_info = self.env.step(action, next_vix, next_corr)
                
                # Q-Table 업데이트
                self.agent.update(state, action, reward, next_state, done)
                
                total_reward += reward
                steps += 1
                state = next_state
                
                if done:
                    break
            
            # Epsilon 감소
            self.agent.decay_epsilon()
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            
            if verbose and (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                print(f"Episode {episode+1:4d} | Avg Reward: {avg_reward:8.2f} | Epsilon: {self.agent.epsilon:.4f}")
        
        print("\n[Training Complete]")
        print(f"  Total Episodes: {n_episodes}")
        print(f"  Final Avg Reward: {np.mean(self.episode_rewards[-50:]):.2f}")
        print(f"  Q-Table Size: {len(self.agent.q_table)} states")
        
        return self.agent
    
    def evaluate(self, n_episodes=10):
        """학습된 에이전트 평가"""
        print("\n" + "=" * 60)
        print("Evaluation (No Exploration)")
        print("=" * 60)
        
        eval_rewards = []
        eval_kics = []
        
        for episode in range(n_episodes):
            market_data = generate_market_scenario(200, '2008_crisis')
            
            state = self.env.reset(
                initial_vix=market_data['VIX'].iloc[0],
                initial_corr=market_data['Correlation'].iloc[0]
            )
            
            total_reward = 0
            min_kics = 999
            
            for i in range(len(market_data) - 1):
                action, info = self.agent.get_action(state, self.env, training=False)
                
                next_vix = market_data['VIX'].iloc[i + 1]
                next_corr = market_data['Correlation'].iloc[i + 1]
                
                state, reward, done, step_info = self.env.step(action, next_vix, next_corr)
                
                total_reward += reward
                min_kics = min(min_kics, self.env.get_kics_ratio())
                
                if done:
                    break
            
            eval_rewards.append(total_reward)
            eval_kics.append(min_kics)
        
        print(f"  Avg Reward: {np.mean(eval_rewards):.2f}")
        print(f"  Avg Min K-ICS: {np.mean(eval_kics):.1f}%")
        
        if np.mean(eval_kics) > 100:
            print("[SUCCESS] Agent maintained K-ICS > 100% during crisis!")
        
        return eval_rewards, eval_kics
    
    def plot_training(self):
        """훈련 결과 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Episode Rewards
        axes[0].plot(self.episode_rewards, alpha=0.6)
        # Moving average
        window = 50
        if len(self.episode_rewards) >= window:
            ma = pd.Series(self.episode_rewards).rolling(window).mean()
            axes[0].plot(ma, 'r-', lw=2, label=f'{window}-episode MA')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Training Progress')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Episode Lengths
        axes[1].plot(self.episode_lengths, alpha=0.6)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Episode Length')
        axes[1].set_title('Episode Duration')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('rl_training_result.png', dpi=150)
        plt.show()
        
        print("[Saved] rl_training_result.png")


def run_rl_training():
    """RL 훈련 실행"""
    trainer = RLTrainer()
    
    # 훈련
    trained_agent = trainer.train(n_episodes=300)
    
    # 평가
    trainer.evaluate(n_episodes=10)
    
    # 시각화
    trainer.plot_training()
    
    return trainer


if __name__ == "__main__":
    trainer = run_rl_training()
