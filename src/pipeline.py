"""
SC2026 蛋白设计赛道 - 主流程管线
===================================
整合所有模块，实现完整的蛋白设计流程:

Phase 1: 数据加载与预处理
  - 加载GFP亮度数据
  - 解析突变并构建序列
  - 提取ESM-2嵌入 (可选)

Phase 2: 预测模型
  - 亮度预测 (ESM-2 + MLP)

Phase 3: 序列生成
  - RL强化学习生成

Phase 4: 多级筛选
  - Stage 1: 硬约束过滤 (含排除列表检查)
  - Stage 2: 亮度预测
  - Stage 3: 帕累托前沿+多样性

输出: 最终6条候选序列

依赖安装:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install fair-esm transformers biopython pandas numpy scikit-learn h5py tqdm

运行方式:
python pipeline.py --input_dir ./data --output_dir ./output
"""

import os
import sys
import argparse
import logging
import subprocess
from typing import List, Dict, Optional, Tuple
import json
import random
import numpy as np
import pandas as pd

# 本地模块
from config import (
    DEFAULT_CONFIG, PathConfig, SC2026Config,
    WT_SEQUENCES
)
from seq_builder import SeqBuilder, SeqValidator, get_mutations_string
from exclusion_checker import ExclusionListChecker
from filters import (
    HardConstraintFilter, PredictionFilter,
    ParetoDiversityFilter, ScreeningPipeline, SequenceRecord
)
from brightness_model import BrightnessModelESM


# =============================================================================
# 日志配置
# =============================================================================

def install_dependencies():
    """安装依赖包"""
    print("=" * 60)
    print("安装依赖包...")
    print("=" * 60)

    base_packages = [
        "pandas",
        "numpy",
        "scikit-learn",
        "h5py",
        "tqdm",
        "biopython",
    ]

    pytorch_install = [
        "torch",
        "torchvision",
        "torchaudio",
        "--index-url",
        "https://download.pytorch.org/whl/cu118"
    ]

    esm_packages = ["fair-esm"]

    print("\n[1/3] 安装基础依赖...")
    for pkg in base_packages:
        print(f"  安装 {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

    print("\n[2/3] 安装PyTorch (GPU版本)...")
    print("  注意: 如果没有GPU，可以改为CPU版本")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + pytorch_install)
    except subprocess.CalledProcessError:
        print("  PyTorch安装失败，尝试CPU版本...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "torch", "torchvision", "torchaudio", "-q"
        ])

    print("\n[3/3] 安装ESM...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fair-esm", "-q"])
    except subprocess.CalledProcessError:
        print("  fair-esm安装可能需要特殊处理，跳过...")

    print("\n" + "=" * 60)
    print("依赖安装完成!")
    print("=" * 60)


def check_gpu():
    """检查GPU是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\nGPU信息:")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            print(f"  当前GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("\n警告: 未检测到GPU，将使用CPU运行")
            return False
    except ImportError:
        print("\n警告: PyTorch未安装，将无法使用GPU加速")
        return False


def setup_logging(log_dir: str, log_file: str = "pipeline.log") -> logging.Logger:
    """配置日志"""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("SC2026_Pipeline")
    logger.setLevel(logging.INFO)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 文件handler
    fh = logging.FileHandler(os.path.join(log_dir, log_file))
    fh.setLevel(logging.INFO)

    # 控制台handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 格式化
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# =============================================================================
# 数据加载
# =============================================================================

class DataLoader:
    """
    数据加载器

    负责加载:
    - GFP亮度数据 (Excel)
    - 排除列表 (CSV)
    - WT序列
    """

    def __init__(self, data_dir: str, exclusion_list_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.data_dir = data_dir
        self.exclusion_list_path = exclusion_list_path
        self.logger = logger or logging.getLogger(__name__)

    def load_exclusion_list(self) -> ExclusionListChecker:
        """加载排除列表"""
        if self.exclusion_list_path and os.path.exists(self.exclusion_list_path):
            exclusion_path = self.exclusion_list_path
        else:
            exclusion_path = os.path.join(self.data_dir, "Exclusion_List.csv")

        if not os.path.exists(exclusion_path):
            raise FileNotFoundError(f"排除列表文件不存在: {exclusion_path}")

        checker = ExclusionListChecker()
        checker.load_from_csv(exclusion_path)
        self.logger.info(f"加载了 {checker.get_size()} 条排除序列")
        return checker

    def load_gfp_data(self) -> pd.DataFrame:
        """
        加载GFP亮度数据

        期望格式:
        - seq_id: 序列标识
        - sequence: 氨基酸序列
        - brightness: 亮度值
        - gfp_type: GFP类型

        如果Excel中没有序列列，则需要从aaMutations列构建
        """
        gfp_data_path = os.path.join(self.data_dir, "GFP_data.xlsx")

        if not os.path.exists(gfp_data_path):
            raise FileNotFoundError(f"GFP数据文件不存在: {gfp_data_path}")

        try:
            df = pd.read_excel(gfp_data_path, sheet_name='brightness')
            self.logger.info(f"从 {gfp_data_path} 加载了 {len(df)} 条数据")
            return df
        except Exception as e:
            raise RuntimeError(f"无法读取brightness sheet: {e}")

    def load_starting_sequences(self) -> List[str]:
        """
        加载起始序列

        从 SC突变起始序列.txt 加载
        """
        starting_seq_path = os.path.join(self.data_dir, "SC突变起始序列.txt")

        if not os.path.exists(starting_seq_path):
            raise FileNotFoundError(f"起始序列文件不存在: {starting_seq_path}")

        sequences = []
        with open(starting_seq_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('WT') and not line.startswith('PIII') and not line.startswith('FPbase'):
                    if line.startswith('M') and len(line) > 200:
                        sequences.append(line)

        if not sequences:
            raise ValueError("起始序列文件中没有有效序列")

        self.logger.info(f"加载了 {len(sequences)} 条起始序列")
        return sequences


# =============================================================================
# 候选序列生成器（基于双头PPO + 优先经验回放）
# =============================================================================

class CandidateGenerator:
    """
    候选序列生成器

    使用双头PPO + 优先经验回放 探索蛋白质突变空间。
    强化学习代理通过与环境交互学习高奖励（亮度）的突变模式。
    """

    def __init__(
        self,
        wt_sequence: str,
        max_mutations: int = 8,
        logger: Optional[logging.Logger] = None,
        output_dir: str = None,
        seq_id: str = None
    ):
        """
        初始化RL候选生成器

        参数:
            wt_sequence: 野生型序列
            max_mutations: 最大突变数
            logger: 日志记录器
            output_dir: 输出目录（用于保存RL结果）
            seq_id: 序列标识，用于创建子目录保存结果
        """
        self.wt_sequence = wt_sequence
        self.max_mutations = max_mutations
        self.logger = logger or logging.getLogger(__name__)
        self.output_dir = output_dir or 'output'
        self.seq_id = seq_id or f"seq_{hash(wt_sequence[:20])}"

    def generate_rl_candidates(
        self,
        n_candidates: int,
        rl_episodes: int = 1000,
        max_mutations: int = 5
    ) -> List[str]:
        """
        使用双头PPO强化学习生成候选序列

        参数:
            n_candidates: 候选数量
            rl_episodes: RL训练轮数
            max_mutations: 最大突变数

        返回:
            候选序列列表
        """
        from rl_env import ProteinDesignEnv
        from ppo_agent import PPOAgent
        import pandas as pd
        import json
        import os
        import matplotlib.pyplot as plt

        self.logger.info(f"使用RL生成 {n_candidates} 条候选 (训练{rl_episodes}轮)")

        try:
            # 动态获取ESM提取器的实际嵌入维度（避免配置值与实际不一致）
            extractor = ProteinDesignEnv._get_esm_extractor()
            actual_embedding_dim = extractor.embedding_dim
            
            env = ProteinDesignEnv(
                wt_sequence=self.wt_sequence,
                max_mutations=max_mutations,
                embedding_dim=actual_embedding_dim
            )
            state_dim = env.state_dim
            action_dim = env.n_actions

            agent = PPOAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                lr=3e-4,
                gamma=0.99,
                gae_lambda=0.95,
                clip_epsilon=0.2
            )

            training_stats = {
                'episodes': [],
                'rewards': [],
                'n_mutations': [],
                'brightness': [],
                'stop_rate': []
            }

            update_interval = 10
            stop_count = 0

            for episode in range(rl_episodes):
                state = env.reset()
                state_vec = env.get_state_vector(state)
                episode_reward = 0.0
                episode_mutations = 0
                stopped_early = False

                for step in range(max_mutations):
                    action_idx, log_prob, value = agent.get_action(state_vec)
                    action = env.index_to_action(action_idx)
                    next_state, reward, done, info = env.step(action)
                    next_state_vec = env.get_state_vector(next_state)

                    agent.push_transition(state_vec, action_idx, reward, log_prob, value, done)

                    episode_reward += reward
                    episode_mutations += 1
                    state_vec = next_state_vec

                    if info.get('stopped_early', False):
                        stopped_early = True
                    if done:
                        break

                if stopped_early:
                    stop_count += 1

                if (episode + 1) % update_interval == 0:
                    agent.update(batch_size=64, n_epochs=5)

                training_stats['episodes'].append(episode)
                training_stats['rewards'].append(episode_reward)
                training_stats['n_mutations'].append(episode_mutations)
                if 'brightness' in info:
                    training_stats['brightness'].append(info['brightness'])
                else:
                    training_stats['brightness'].append(0.0)
                training_stats['stop_rate'].append(stop_count / (episode + 1))

                if (episode + 1) % 100 == 0:
                    avg_reward = sum(training_stats['rewards'][-100:]) / 100
                    avg_brightness = sum(training_stats['brightness'][-100:]) / 100
                    avg_mutations = sum(training_stats['n_mutations'][-100:]) / 100
                    stop_rate = training_stats['stop_rate'][-1]
                    print(
                        f"  [RL] 进度: {episode+1}/{rl_episodes} | "
                        f"avg_reward={avg_reward:.2f} | "
                        f"avg_brightness={avg_brightness:.3f} | "
                        f"avg_mutations={avg_mutations:.1f} | "
                        f"stop_rate={stop_rate:.2%}"
                    )

            candidates = []
            seen = set()
            candidate_info = []
            max_attempts = n_candidates * 20

            for i in range(max_attempts):
                if len(candidates) >= n_candidates:
                    break

                state = env.reset()
                state_vec = env.get_state_vector(state)

                for _ in range(max_mutations):
                    deterministic = i >= max_attempts * 0.3
                    action_idx, _, _ = agent.get_action(state_vec, deterministic=deterministic)
                    action = env.index_to_action(action_idx)
                    state, reward, done, info = env.step(action)
                    state_vec = env.get_state_vector(state)
                    if done:
                        break

                seq = state.sequence
                if seq not in seen and seq != self.wt_sequence:
                    seen.add(seq)
                    candidates.append(seq)
                    candidate_info.append({
                        'sequence': seq,
                        'n_mutations': info.get('n_mutations', 0),
                        'brightness': info.get('brightness', 0.0),
                        'reward': reward,
                        'strategy': 'deterministic' if i >= max_attempts * 0.3 else 'stochastic'
                    })

            if len(candidates) < n_candidates:
                self.logger.warning(f"RL仅生成了 {len(candidates)}/{n_candidates} 条去重候选，将补充随机生成")
                fallback = self._fallback_random(n_candidates - len(candidates), max_mutations)
                for seq in fallback:
                    if seq not in seen:
                        seen.add(seq)
                        candidates.append(seq)
                        candidate_info.append({
                            'sequence': seq,
                            'n_mutations': seq.count('.') if '.' in seq else -1,
                            'brightness': 0.0,
                            'reward': 0.0,
                            'strategy': 'fallback_random'
                        })

            self.logger.info(f"RL最终生成 {len(candidates)} 条候选 (尝试了 {i+1} 次)")

            output_dir = os.path.join(self.output_dir, 'rl_results', self.seq_id)
            os.makedirs(output_dir, exist_ok=True)

            df_candidates = pd.DataFrame({'sequence': candidates})
            df_candidates.to_csv(os.path.join(output_dir, 'rl_candidates.csv'), index=False)

            df_with_scores = pd.DataFrame(candidate_info)
            df_with_scores.to_csv(os.path.join(output_dir, 'rl_candidates_with_scores.csv'), index=False)

            with open(os.path.join(output_dir, 'rl_training_stats.json'), 'w', encoding='utf-8') as f:
                json.dump(training_stats, f, ensure_ascii=False, indent=2)

            plt.figure(figsize=(10, 6))
            plt.plot(training_stats['episodes'], training_stats['rewards'], label='Episode Reward', color='blue', alpha=0.6)
            
            window_size = 50
            if len(training_stats['rewards']) >= window_size:
                rolling_avg = []
                for i in range(len(training_stats['rewards']) - window_size + 1):
                    rolling_avg.append(sum(training_stats['rewards'][i:i+window_size]) / window_size)
                plt.plot(range(window_size-1, len(training_stats['rewards'])), rolling_avg, label=f'{window_size}-episode Rolling Avg', color='red', linewidth=2)
            
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.title('RL Training Reward Curve')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'rl_reward_curve.png'), dpi=150, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(training_stats['episodes'], training_stats['brightness'], label='Brightness', color='green', alpha=0.6)
            plt.xlabel('Episode')
            plt.ylabel('Normalized Brightness')
            plt.title('RL Training Brightness Trend')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'rl_brightness_trend.png'), dpi=150, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.plot(training_stats['episodes'], training_stats['stop_rate'], label='Stop Action Rate', color='purple', linewidth=2)
            plt.xlabel('Episode')
            plt.ylabel('Stop Action Rate')
            plt.title('RL Stop Action Usage Trend')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.savefig(os.path.join(output_dir, 'rl_stop_rate.png'), dpi=150, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(10, 6))
            plt.hist(training_stats['n_mutations'][-500:], bins=range(0, max_mutations + 2), 
                     edgecolor='black', alpha=0.7, color='steelblue')
            plt.xlabel('Number of Mutations')
            plt.ylabel('Frequency')
            plt.title('Distribution of Mutations per Episode (Last 500)')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'rl_mutation_distribution.png'), dpi=150, bbox_inches='tight')
            plt.close()

            self.logger.info(f"RL结果已保存到: {output_dir}")

            return candidates

        except Exception as e:
            self.logger.warning(f"RL生成失败: {e}, 降级到随机生成")
            return self._fallback_random(n_candidates, max_mutations)

    def _fallback_random(
        self,
        n_candidates: int,
        max_mutations: int = 5
    ) -> List[str]:
        """
        降级方案：随机突变生成

        参数:
            n_candidates: 候选数量
            max_mutations: 最大突变数

        返回:
            随机候选序列
        """
        self.logger.info(f"使用随机突变生成 {n_candidates} 条候选")
        candidates = []
        valid_aa = 'ACDEFGHIKLMNPQRSTVWY'

        for _ in range(n_candidates):
            n_muts = random.randint(1, max_mutations)
            length = len(self.wt_sequence)
            positions = random.sample(range(10, length - 10), n_muts)

            wt_list = list(self.wt_sequence)
            for pos in positions:
                wt_aa = wt_list[pos]
                mut_options = [aa for aa in valid_aa if aa != wt_aa]
                if mut_options:
                    wt_list[pos] = random.choice(mut_options)

            candidates.append(''.join(wt_list))

        return candidates

    def generate_variant_candidates(
        self,
        n_candidates: int,
        max_mutations: int = 5
    ) -> List[str]:
        """
        生成随机突变变体（兼容接口，内部使用RL）

        参数:
            n_candidates: 候选数量
            max_mutations: 最大突变数

        返回:
            候选序列列表
        """
        return self.generate_rl_candidates(n_candidates, max_mutations=max_mutations)


# =============================================================================
# 主流程管线
# =============================================================================

class SC2026Pipeline:
    """
    SC2026 蛋白设计主流程管线

    整合所有阶段:
    1. 数据加载
    2. 候选生成
    3. 多级筛选
    4. 输出结果
    """

    def __init__(
        self,
        config: SC2026Config,
        output_dir: str,
        logger: Optional[logging.Logger] = None,
        brightness_model_path: Optional[str] = None
    ):
        self.config = config
        self.output_dir = output_dir
        self.logger = logger or logging.getLogger(__name__)
        self.brightness_model_path = brightness_model_path  # 保存自定义模型路径

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 初始化组件
        self.data_loader = DataLoader(
            config.paths.data_dir,
            exclusion_list_path=config.paths.exclusion_list_file,
            logger=self.logger
        )
        self.exclusion_checker = None
        self.pipeline = None

    def run(
        self,
        n_candidates: int = 1000,
        use_variants: bool = True,
        rl_episodes: int = None
    ) -> Dict:
        """
        运行完整管线（完整评估模式）

        参数:
            n_candidates: 生成候选序列总数
            use_variants: 是否使用RL生成变体
            rl_episodes: RL训练轮数，None则使用配置文件中的值

        返回:
            结果字典
        """
        self.logger.info("=" * 60)
        self.logger.info("SC2026 蛋白设计管线启动")
        self.logger.info("运行模式: full (完整评估)")
        self.logger.info("=" * 60)

        results = {
            'stage1_input': 0,
            'stage1_passed': 0,
            'stage2_passed': 0,
            'stage3_passed': 0,
            'final_candidates': []
        }

        # =================================================================
        # Phase 1: 数据加载
        # =================================================================
        self.logger.info("\n[Phase 1] 数据加载...")

        # 加载排除列表
        self.exclusion_checker = self.data_loader.load_exclusion_list()

        # 加载起始序列
        starting_sequences = self.data_loader.load_starting_sequences()
        sfGFP = starting_sequences[0]
        self.logger.info(f"使用sfGFP序列，长度: {len(sfGFP)}")

        # =================================================================
        # Phase 2: 候选序列生成（使用双头PPO + 优先经验回放）
        # =================================================================
        self.logger.info("\n[Phase 2] 候选序列生成（双头PPO + 优先经验回放）...")

        all_candidates = []
        seq_ids = []
        candidate_index = 0

        # 加载亮度模型（供RL环境使用）
        # 优先级：自定义路径 > models_dir > output/brightness_model
        if self.brightness_model_path:
            model_path = self.brightness_model_path
            if not os.path.exists(model_path):
                self.logger.warning(f"  警告: 自定义亮度模型不存在: {model_path}")
                model_path = None
        else:
            model_path = os.path.join(self.config.paths.models_dir, 'brightness_model.pth')
            if not os.path.exists(model_path):
                model_path = os.path.join(self.config.paths.output_dir, 'brightness_model', 'brightness_model.pth')
        
        if model_path and os.path.exists(model_path):
            self.logger.info(f"  加载亮度模型: {model_path}")
            from rl_env import ProteinDesignEnv
            ProteinDesignEnv.load_brightness_model(model_path)
        else:
            self.logger.warning("亮度模型不存在，RL奖励将使用零样本相似度方法")

        if use_variants:
            # RL生成：从每个起始序列生成更多候选，然后通过筛选选出最佳的
            n_starting = len(starting_sequences)
            # 每条起始序列最终需要贡献的候选数
            variant_n_per_seq = max(1, (n_candidates - len(all_candidates)) // n_starting)
            # 每条起始序列生成的候选数（放大候选池，便于后续筛选）
            # 默认每条序列生成 variant_n_per_seq × 10 条候选，至少50条
            rl_candidates_per_seq = max(variant_n_per_seq * 10, 50)
            
            # 记录每个候选来自哪个起始序列（用于分组筛选）
            origin_groups = []
            
            for seq_idx, wt_seq in enumerate(starting_sequences):
                self.logger.info(f"  从起始序列 {seq_idx+1}/{n_starting} 使用RL生成变体...")
                
                seq_id = f"start_seq_{seq_idx+1}"
                generator = CandidateGenerator(wt_seq, output_dir=self.output_dir, seq_id=seq_id)
                # 使用RL生成更多候选（放大候选池）
                actual_rl_episodes = rl_episodes if rl_episodes is not None else self.config.sequence_generation.rl_episodes
                variants = generator.generate_rl_candidates(
                    n_candidates=rl_candidates_per_seq,
                    rl_episodes=actual_rl_episodes,
                    max_mutations=self.config.sequence_generation.rl_max_mutations
                )
                all_candidates.extend(variants)
                seq_ids.extend([f"cand_{candidate_index + i}" for i in range(len(variants))])
                # 记录该批次的所有候选都属于组seq_idx
                origin_groups.extend([seq_idx] * len(variants))
                candidate_index += len(variants)

        # 去重（同时保持ID和origin_group对应）
        seen = set()
        unique_candidates = []
        unique_seq_ids = []
        unique_origin_groups = []
        for seq, seq_id, group_id in zip(all_candidates, seq_ids, origin_groups):
            if seq not in seen:
                seen.add(seq)
                unique_candidates.append(seq)
                unique_seq_ids.append(seq_id)
                unique_origin_groups.append(group_id)
        
        all_candidates = unique_candidates
        seq_ids = unique_seq_ids
        origin_groups = unique_origin_groups
        
        # 计算每个候选的突变数（相对于WT）
        n_mutations_dict = {}
        for seq, seq_id in zip(all_candidates, seq_ids):
            n_mutations = sum(a != b for a, b in zip(seq, sfGFP))
            n_mutations_dict[seq_id] = n_mutations
        
        results['stage1_input'] = len(all_candidates)
        self.logger.info(f"生成了 {len(all_candidates)} 条候选 (去重后)")

        # =================================================================
        # Phase 3: 多级筛选
        # =================================================================
        self.logger.info("\n[Phase 3] 多级筛选...")

        # 初始化筛选管线
        self.pipeline = ScreeningPipeline(
            wt_sequence=sfGFP,
            exclusion_list_path=self.config.paths.exclusion_list_file,
            B_threshold=self.config.screening.B_threshold,
            max_similarity=self.config.screening.diversity_max_similarity,
            n_final=self.config.screening.n_final,
            min_length=self.config.screening.min_length,
            max_length=self.config.screening.max_length
        )

        # 准备评估数据（seq_ids已在候选生成时创建）
        B_dict = {}

        self.logger.info("评估候选序列...")

        # 使用完整评估（默认使用ESM-2）
        self.logger.info("  加载ESM-2模型...")
        from extract_esm2 import ESM2Extractor
        esm_extractor = ESM2Extractor()

        self.logger.info("  提取ESM-2嵌入...")
        embeddings = esm_extractor.extract_batch(all_candidates)

        # 使用训练好的亮度预测模型进行预测
        # 优先级：自定义路径 > models_dir > output/brightness_model
        if self.brightness_model_path:
            model_path = self.brightness_model_path
        else:
            model_path = os.path.join(self.config.paths.models_dir, 'brightness_model.pth')
            if not os.path.exists(model_path):
                model_path = os.path.join(self.config.paths.output_dir, 'brightness_model', 'brightness_model.pth')
        
        if os.path.exists(model_path):
            self.logger.info(f"  加载训练好的亮度模型: {model_path}")
            from brightness_model import load_brightness_model_from_checkpoint
            import torch
            
            # 使用工具函数加载模型（自动处理架构参数和归一化）
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model, norm_info = load_brightness_model_from_checkpoint(model_path, device=device)
            
            norm_method = norm_info['method']
            use_log1p = norm_info.get('use_log1p', False)
            
            if norm_method == 'zscore':
                norm_mean = norm_info['mean']
                norm_std = norm_info['std']
                data_min = norm_info.get('min', norm_mean - 3 * norm_std)
                data_max = norm_info.get('max', norm_mean + 3 * norm_std)
            else:
                # 兼容旧模型（Min-Max）
                data_min = norm_info['min']
                data_max = norm_info['max']
                norm_mean = (data_min + data_max) / 2
                norm_std = (data_max - data_min) / 2
            
            data_range = data_max - data_min
            if data_range < 1e-8:
                data_range = 1.0
            
            # 模型预测（返回原始Z-score/对数空间值）
            predictions_norm = model.predict(embeddings, return_raw=True)
            
            # 完整反归一化到原始亮度尺度
            # 第一步：反Z-score
            predictions_raw = predictions_norm * norm_std + norm_mean
            # 第二步：反log1p（如果使用了）
            if use_log1p:
                predictions_raw = np.expm1(predictions_raw)
                predictions_raw = np.maximum(predictions_raw, 0.0)
            
            # 进一步归一化到 [0, 1]（用于筛选，与 B_threshold 匹配）
            predictions_01 = (predictions_raw - data_min) / data_range
            predictions_01 = np.clip(predictions_01, 0.0, 1.0)
            
            # B_dict 存储 [0,1] 归一化值，供筛选使用
            for i, seq_id in enumerate(seq_ids):
                B_dict[seq_id] = float(predictions_01[i])
            
            # 保存原始亮度值，供最终输出使用
            self._raw_brightness = {}
            for i, seq_id in enumerate(seq_ids):
                self._raw_brightness[seq_id] = float(predictions_raw[i])
        else:
            raise FileNotFoundError(f"训练好的亮度模型不存在: {model_path}\n请先运行训练脚本: python train_brightness_model.py")

        # 运行筛选（按组分别筛选）
        screening_results = self.pipeline.run(
            sequences=all_candidates,
            seq_ids=seq_ids,
            B_dict=B_dict,
            origin_groups=origin_groups,
            n_mutations_dict=n_mutations_dict
        )

        # 调试信息
        if screening_results['stage1_passed'] == 0 and len(all_candidates) > 0:
            self.logger.warning("Stage 1全部失败! 检查前5个序列的失败原因:")
            records = screening_results['records']
            for i, record in enumerate(records[:5]):
                if not record.stage1_pass:
                    self.logger.warning(f"  序列{i}: {record.fail_reasons}")

        results['stage1_passed'] = screening_results['stage1_passed']
        results['stage2_passed'] = screening_results['stage2_passed']
        results['stage3_passed'] = screening_results['stage3_passed']
        results['final_candidates'] = [
            {
                'seq_id': r.seq_id,
                'sequence': r.sequence,
                'mutations': get_mutations_string(r.sequence, 'sfGFP'),
                'B_hat': self._raw_brightness.get(r.seq_id, r.B_hat) if hasattr(self, '_raw_brightness') else r.B_hat,
                'f_score': r.f_score,
                'origin_group': origin_groups[seq_ids.index(r.seq_id)] if r.seq_id in seq_ids else -1
            }
            for r in screening_results['final_sequences']
        ]
        
        # 按组统计最终候选
        group_counts = {}
        for candidate in results['final_candidates']:
            g = candidate['origin_group']
            group_counts[g] = group_counts.get(g, 0) + 1
        results['group_distribution'] = group_counts

        # =================================================================
        # Phase 4: 输出结果
        # =================================================================
        self.logger.info("\n[Phase 4] 输出结果...")

        self.save_results(results)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("管线完成!")
        self.logger.info(f"最终候选: {len(results['final_candidates'])} 条")
        if results.get('group_distribution'):
            for group_id, count in sorted(results['group_distribution'].items()):
                wt_name = starting_sequences[group_id][:20] if group_id < len(starting_sequences) else "Unknown"
                self.logger.info(f"  - 起始序列{group_id+1} ({wt_name}...): {count} 条")
        self.logger.info("=" * 60)

        return results

    def save_results(self, results: Dict):
        """保存结果到文件"""
        # 保存最终序列
        final_df = pd.DataFrame(results['final_candidates'])
        final_path = os.path.join(self.output_dir, "final_candidates.csv")
        final_df.to_csv(final_path, index=False)
        self.logger.info(f"最终候选已保存到: {final_path}")

        # 保存筛选报告
        report_path = os.path.join(self.output_dir, "screening_report.csv")

        # 创建报告数据
        report_data = {
            'stage': ['input', 'stage1', 'stage2', 'stage3', 'final'],
            'count': [
                results['stage1_input'],
                results['stage1_passed'],
                results['stage2_passed'],
                results['stage3_passed'],
                len(results['final_candidates'])
            ]
        }
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(report_path, index=False)
        self.logger.info(f"筛选报告已保存到: {report_path}")

        # 保存为JSON (包含完整信息)
        json_path = os.path.join(self.output_dir, "results.json")
        
        # 将numpy类型转换为Python原生类型
        def convert_to_native(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            else:
                return obj
        
        results_native = convert_to_native(results)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_native, f, indent=2, ensure_ascii=False)
        self.logger.info(f"JSON结果已保存到: {json_path}")


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="SC2026 蛋白设计赛道 - 主流程管线"
    )

    _default_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser.add_argument(
        '--install',
        action='store_true',
        help='安装依赖包'
    )

    parser.add_argument(
        '--check',
        action='store_true',
        help='检查环境（GPU等）'
    )

    parser.add_argument(
        '--input_dir',
        type=str,
        default=_default_root,
        help='输入数据目录'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(_default_root, "output"),
        help='输出目录'
    )

    parser.add_argument(
        '--n_candidates',
        type=int,
        default=1000,
        help='生成候选序列数量'
    )

    parser.add_argument(
        '--rl_episodes',
        type=int,
        default=None,
        help='RL训练轮数（不指定则使用配置文件中的默认值）'
    )

    parser.add_argument(
        '--brightness_model',
        type=str,
        default=None,
        help='亮度模型路径，例如: ../output/pretrained/brightness_model.pth'
    )

    parser.add_argument(
        '--use_variants',
        action='store_true',
        default=True,
        help='是否使用RL生成变体（默认True）'
    )

    parser.add_argument(
        '--no_variants',
        action='store_true',
        help='禁用RL变体生成'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    if args.install:
        install_dependencies()
        return
    elif args.check:
        check_gpu()
        print("\n检查完成!")
        return

    # 处理 use_variants 参数
    use_variants = not args.no_variants

    # 设置日志
    logger = setup_logging(args.output_dir)

    # 创建配置
    config = SC2026Config()
    config.paths.root_dir = args.input_dir
    config.paths.data_dir = args.input_dir
    config.paths.output_dir = args.output_dir

    # 创建并运行管线
    pipeline = SC2026Pipeline(
        config=config,
        output_dir=args.output_dir,
        logger=logger,
        brightness_model_path=args.brightness_model
    )

    results = pipeline.run(
        n_candidates=args.n_candidates,
        rl_episodes=args.rl_episodes,
        use_variants=use_variants
    )

    # 打印最终结果
    print("\n" + "=" * 60)
    print("最终候选序列:")
    print("=" * 60)
    for i, cand in enumerate(results['final_candidates']):
        print(f"\n候选 {i+1}:")
        print(f"  ID: {cand['seq_id']}")
        print(f"  序列: {cand['sequence'][:60]}...")
        print(f"  突变位点: {cand['mutations']}")
        print(f"  B̂: {cand['B_hat']:.3f}")
        print(f"  f_score: {cand['f_score']:.3f}")


if __name__ == "__main__":
    main()
