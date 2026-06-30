"""
SC2026 蛋白设计赛道 - 双头PPO + 优先经验回放
============================================

本模块实现：
1. 双头网络（Dual-Head PPO）: 共享底层特征提取器，分别输出策略与价值
2. 优先经验回放（Prioritized Experience Replay, PER）: 按TD误差加权采样高价值样本

核心设计:
- 共享编码器: ESM嵌入 + 突变掩码 共同编码
- 策略头: 输出动作概率分布
- 价值头: 输出状态价值估计
- 优先回放: P(i) ∝ |TD_error(i)|^α, 重要性采样权重补偿偏差
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Dict, Optional
from collections import deque
import random


# =============================================================================
# 双头网络：共享特征 + 策略/价值分离头
# =============================================================================

class DualHeadNetwork(nn.Module):
    """
    双头 Actor-Critic 网络

    结构:
        Encoder(共享): state -> hidden_feature
        PolicyHead(策略): hidden_feature -> action_probs
        ValueHead(价值): hidden_feature -> state_value
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        """
        初始化双头网络

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 共享层隐藏维度
        """
        super(DualHeadNetwork, self).__init__()

        # 共享编码器
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )

        # 策略头（Actor）
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )

        # 价值头（Critic）
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.softmax = nn.Softmax(dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """共享特征提取"""
        return self.encoder(x)

    def forward_policy(self, features: torch.Tensor) -> torch.Tensor:
        """策略前向"""
        logits = self.policy_head(features)
        return self.softmax(logits)

    def forward_value(self, features: torch.Tensor) -> torch.Tensor:
        """价值前向"""
        return self.value_head(features)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        完整前向

        返回:
            (action_probs, state_value)
        """
        features = self.encode(x)
        probs = self.forward_policy(features)
        value = self.forward_value(features)
        return probs, value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作

        返回:
            (action, log_prob, value)
        """
        probs, value = self.forward(state)
        probs = probs.clamp(min=1e-8)

        if deterministic:
            action = probs.argmax(dim=-1)
            log_prob = (probs.log() * nn.functional.one_hot(action, probs.size(-1)).float()).sum(dim=-1)
        else:
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action, log_prob, value


# =============================================================================
# 优先经验回放缓冲区
# =============================================================================

class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区

    采样概率: P(i) = p_i^α / Σ p_j^α
    其中 p_i = |TD_error(i)| + ε

    重要性采样权重: w_i = (N * P(i))^(-β)，用于修正偏差
    """

    def __init__(
        self,
        max_size: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6
    ):
        """
        初始化优先回放缓冲区

        参数:
            max_size: 最大容量
            alpha: 优先级指数（0=均匀采样，1=完全优先）
            beta: 重要性采样指数（用于偏差补偿）
            beta_increment: beta每步的增量
            epsilon: 优先级的小常数（避免零优先级）
        """
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # 经验存储
        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.log_probs = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)
        self.dones = deque(maxlen=max_size)

        # 优先级（与经验一一对应）
        self.priorities = deque(maxlen=max_size)

        # 最大优先级（用于新样本）
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool,
        td_error: Optional[float] = None
    ) -> None:
        """
        推入新经验

        参数:
            state: 状态
            action: 动作
            reward: 奖励
            log_prob: 对数概率
            value: 价值估计
            done: 终止标志
            td_error: TD误差（用于初始化优先级，None则使用max_priority）
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

        # 初始化优先级
        if td_error is None:
            priority = self.max_priority
        else:
            priority = abs(td_error) + self.epsilon
            self.max_priority = max(self.max_priority, priority)

        self.priorities.append(priority)

    def sample(
        self,
        batch_size: int
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        优先采样一批经验

        参数:
            batch_size: 批大小

        返回:
            (data_dict, indices, importance_weights)
        """
        n = len(self.states)
        if n == 0:
            raise ValueError("缓冲区为空")

        # 计算采样概率
        priorities = np.array(self.priorities, dtype=np.float64)
        probs = priorities ** self.alpha
        probs = probs / probs.sum()

        # 采样索引
        indices = np.random.choice(n, size=min(batch_size, n), replace=True, p=probs)

        # 重要性采样权重
        weights = (n * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # 归一化

        # 提取经验
        sampled_states = np.array([self.states[i] for i in indices])
        sampled_actions = np.array([self.actions[i] for i in indices])
        sampled_rewards = np.array([self.rewards[i] for i in indices])
        sampled_log_probs = np.array([self.log_probs[i] for i in indices])
        sampled_values = np.array([self.values[i] for i in indices])
        sampled_dones = np.array([self.dones[i] for i in indices])

        data = {
            'states': torch.tensor(sampled_states, dtype=torch.float32),
            'actions': torch.tensor(sampled_actions, dtype=torch.int64),
            'rewards': torch.tensor(sampled_rewards, dtype=torch.float32),
            'log_probs': torch.tensor(sampled_log_probs, dtype=torch.float32),
            'values': torch.tensor(sampled_values, dtype=torch.float32),
            'dones': torch.tensor(sampled_dones, dtype=torch.float32)
        }

        # 增量更新 beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return data, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(
        self,
        indices: np.ndarray,
        td_errors: np.ndarray
    ) -> None:
        """
        更新样本的优先级

        参数:
            indices: 样本索引
            td_errors: 对应的TD误差
        """
        for idx, td in zip(indices, td_errors):
            priority = abs(td) + self.epsilon
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return len(self.states)


# =============================================================================
# 双头PPO代理
# =============================================================================

class PPOAgent:
    """
    双头PPO + 优先经验回放代理

    1. 共享编码器 + 策略/价值双头: 参数高效、协同优化
    2. 优先经验回放: 重点学习高TD误差样本，提升样本效率
    3. 重要性采样权重: 修正优先采样引入的分布偏差
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_epsilon: float = 1e-6,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        random_seed: Optional[int] = None
    ):
        """
        初始化双头PPO代理

        参数:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE系数
            clip_epsilon: PPO裁剪系数
            entropy_coef: 熵正则化系数
            value_coef: 价值损失系数
            max_grad_norm: 梯度裁剪
            per_alpha: 优先回放指数
            per_beta: 重要性采样指数
            per_epsilon: 优先级小常数
            device: 设备
            random_seed: 随机种子（None表示不固定）
        """
        self.device = device

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if device == 'cuda':
                torch.cuda.manual_seed_all(random_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        # 双头网络
        self.network = DualHeadNetwork(state_dim, action_dim).to(device)

        # 优化器
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # 超参数
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # 优先回放缓冲区
        self.buffer = PrioritizedReplayBuffer(
            max_size=10000,
            alpha=per_alpha,
            beta=per_beta,
            epsilon=per_epsilon
        )

    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        获取动作

        参数:
            state: 状态向量（numpy数组）
            deterministic: 是否确定性选择

        返回:
            (动作索引, 对数概率, 价值估计)
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.network.get_action(state_tensor, deterministic)

        return (
            action.item(),
            log_prob.item(),
            value.item()
        )

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算GAE优势估计

        参数:
            rewards: 奖励序列
            values: 价值估计序列
            dones: 终止标志序列
            last_value: 最后状态的价值

        返回:
            (advantages, returns)
        """
        n_steps = len(rewards)
        advantages = torch.zeros(n_steps).to(self.device)
        returns = torch.zeros(n_steps).to(self.device)

        advantage = 0.0
        next_value = last_value

        for t in reversed(range(n_steps)):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
            advantages[t] = advantage
            returns[t] = advantage + values[t]
            next_value = values[t]

        # 标准化
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool
    ) -> None:
        """
        推入转移（带TD误差估计）

        简化版本：使用|r + γ*V(s') - V(s)|的近似作为TD误差
        """
        # 新样本使用最大优先级（在 push 中处理）
        self.buffer.push(state, action, reward, log_prob, value, done, td_error=None)

    def update(
        self,
        batch_size: int = 64,
        n_epochs: int = 10
    ) -> Dict[str, float]:
        """
        使用优先经验回放更新双头网络

        参数:
            batch_size: 批次大小
            n_epochs: 每批迭代次数

        返回:
            损失统计
        """
        if len(self.buffer) < batch_size:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}

        policy_losses = []
        value_losses = []
        entropy_losses = []

        for _ in range(n_epochs):
            data, indices, weights = self.buffer.sample(batch_size)

            states = data['states'].to(self.device)
            actions = data['actions'].to(self.device)
            old_log_probs = data['log_probs'].to(self.device)
            old_values = data['values'].to(self.device)
            rewards = data['rewards'].to(self.device)
            dones = data['dones'].to(self.device)
            weights = weights.to(self.device)

            probs, new_values = self.network(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            new_values_flat = new_values.view(-1)

            advantages, returns = self.compute_gae(rewards, old_values, dones, last_value=0.0)

            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

            policy_loss_1 = ratio * advantages
            policy_loss_2 = clipped_ratio * advantages
            policy_loss = -torch.min(policy_loss_1, policy_loss_2)
            policy_loss = (policy_loss * weights).mean()

            value_loss = nn.MSELoss(reduction='none')(new_values_flat, returns)
            value_loss = (value_loss * weights).mean()

            with torch.no_grad():
                td_errors = (rewards + self.gamma * new_values_flat * (1 - dones) - old_values).cpu().numpy()
                self.buffer.update_priorities(indices, td_errors)

            total_loss = (
                policy_loss +
                self.value_coef * value_loss -
                self.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(),
                self.max_grad_norm
            )
            self.optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy.item())

        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropy_losses)
        }

    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


def train_ppo(
    env,
    agent: PPOAgent,
    n_episodes: int = 100,
    max_steps: int = 10,
    batch_size: int = 64,
    n_epochs: int = 10,
    save_path: str = 'ppo_protein_model.pth',
    early_stopping_patience: int = 50
) -> Dict[str, List[float]]:
    """
    训练双头PPO代理

    参数:
        env: 环境
        agent: 双头PPO代理
        n_episodes: 训练轮数
        max_steps: 每轮最大步数
        batch_size: 批大小
        n_epochs: 更新轮数
        save_path: 模型保存路径
        early_stopping_patience: 早停耐心

    返回:
        训练统计
    """
    stats = {
        'episode_rewards': [],
        'policy_loss': [],
        'value_loss': [],
        'entropy': []
    }

    best_avg_reward = float('-inf')
    patience_counter = 0

    for episode in range(n_episodes):
        state = env.reset()
        state_vec = env.get_state_vector(state)
        episode_reward = 0.0

        for step in range(max_steps):
            action_idx, log_prob, value = agent.get_action(state_vec)
            action = env.index_to_action(action_idx)
            next_state, reward, done, _ = env.step(action)
            next_state_vec = env.get_state_vector(next_state)

            # 推入优先回放缓冲区
            agent.push_transition(state_vec, action_idx, reward, log_prob, value, done)

            episode_reward += reward
            state_vec = next_state_vec

            if done:
                break

        # 更新策略
        losses = agent.update(batch_size, n_epochs)
        stats['policy_loss'].append(losses['policy_loss'])
        stats['value_loss'].append(losses['value_loss'])
        stats['entropy'].append(losses['entropy'])

        stats['episode_rewards'].append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(stats['episode_rewards'][-10:])

            if avg_reward > best_avg_reward + 1e-3:
                best_avg_reward = avg_reward
                patience_counter = 0
                agent.save(save_path.replace('.pth', '_best.pth'))
            else:
                patience_counter += 1

            loss_str = f"{losses['policy_loss']:.3f}" if losses['policy_loss'] > 0 else "N/A"
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Policy Loss: {loss_str} | "
                  f"Patience: {patience_counter}/{early_stopping_patience}")

            if patience_counter >= early_stopping_patience:
                print(f"\n早停触发！连续 {early_stopping_patience} 轮奖励未提升")
                break

    agent.save(save_path)
    return stats
