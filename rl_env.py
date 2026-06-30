"""
SC2026 蛋白设计赛道 - 强化学习环境
===================================

本模块实现蛋白质设计的强化学习环境，使用ESM嵌入作为状态表示。

环境设计：
- 状态: ESM-2嵌入向量 + 突变掩码 + 突变氨基酸特征 + 突变数
- 动作: (位置, 氨基酸) 二元组
- 奖励函数:
  - 基础亮度奖励: (brightness + 2.0) * 3.0（偏移量确保初始奖励为正）
  - 亮度提升奖励: improvement * 2.0（鼓励相对于WT的提升）
  - 突变惩罚: n_mutations * 0.5（限制突变数量）
  - 终止奖励: 1-5突变时分别给予30/20/12/5奖励（鼓励少突变提前终止）
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from config import STANDARD_AA, AA_TO_IDX

# 全局嵌入缓存（所有环境实例共享）
_GLOBAL_EMBEDDING_CACHE = {}


@dataclass
class RLState:
    """强化学习状态"""
    sequence: str
    embedding: np.ndarray  # ESM嵌入（完整序列，用于奖励计算）
    mutation_mask: np.ndarray  # 已突变位置掩码 [seq_length]
    mutation_aas: np.ndarray  # 突变位置的氨基酸one-hot编码 [seq_length, 20]
    n_mutations: int  # 当前突变数


class ProteinDesignEnv:
    """
    蛋白质设计强化学习环境
    
    状态空间:
    - ESM-2序列嵌入
    - 突变掩码
    - 当前突变数
    
    动作空间:
    - 离散动作: 位置 × 氨基酸类型
    
    奖励函数:
    - reward = brightness * 10 - mutation_penalty
    """
    
    def __init__(
        self,
        wt_sequence: str,
        max_mutations: int = 8,
        mutation_penalty: float = 0.5,
        embedding_dim: int = 1280,
        reference_length: Optional[int] = None
    ):
        """
        初始化环境
        
        参数:
            wt_sequence: 野生型序列
            max_mutations: 最大突变数
            mutation_penalty: 每次突变的惩罚系数
            embedding_dim: ESM嵌入维度
            reference_length: 参考序列长度（用于统一状态维度，None表示使用当前序列长度）
        """
        self.wt_sequence = wt_sequence
        self.max_mutations = max_mutations
        self.mutation_penalty = mutation_penalty
        self.embedding_dim = embedding_dim
        
        # 实际序列长度
        self.seq_length = len(wt_sequence)
        
        # 参考长度（用于统一状态维度）
        self.reference_length = reference_length if reference_length is not None else self.seq_length
        
        # 动作空间大小: 可突变位置数 × 氨基酸种类 + 1个停止动作
        self.n_actions = self.seq_length * len(STANDARD_AA) + 1
        self.stop_action_idx = self.seq_length * len(STANDARD_AA)
        
        # 状态空间大小（使用参考长度统一维度）
        # embedding_dim: ESM-2多层特征融合后的维度（6层×1280=7680，3层×1280=3840）
        # reference_length: 统一不同长度序列的状态维度，避免PPO网络输入维度不一致
        # 状态向量构成:
        #   - ESM嵌入: [embedding_dim]
        #   - 突变掩码: [reference_length] (0=未突变, 1=已突变)
        #   - 突变氨基酸特征: [reference_length, 20] (one-hot编码)
        #   - 突变数: [1] (标量)
        self.state_dim = embedding_dim + self.reference_length + self.reference_length * len(STANDARD_AA) + 1
        
        # 当前状态
        self._current_state = None
        
        # WT序列的亮度值（用于奖励计算）
        self._wt_brightness = None
    
    def _calculate_wt_brightness(self):
        """计算并缓存WT序列的亮度值"""
        if self._wt_brightness is None:
            try:
                wt_embedding = self._embed_sequence(self.wt_sequence)
                if ProteinDesignEnv._brightness_model is not None:
                    embedding_input = wt_embedding.reshape(1, -1)
                    self._wt_brightness = float(ProteinDesignEnv._brightness_model.predict(
                        embedding_input,
                        normalization=ProteinDesignEnv._brightness_normalization,
                        return_raw=True
                    )[0])
                    print(f"  [RL] WT亮度(Z-score): {self._wt_brightness:.3f}")
                else:
                    self._wt_brightness = 0.0
            except Exception as e:
                print(f"  [RL] WT亮度计算失败: {e}")
                self._wt_brightness = 0.0
        return self._wt_brightness
    
    def _calculate_reward(self, sequence: str, n_mutations: int, is_terminal: bool = False) -> Tuple[float, float]:
        """
        计算奖励
        
        参数:
            sequence: 当前序列
            n_mutations: 已突变数量
            is_terminal: 是否是终止状态（用于额外奖励）
        
        返回:
            (奖励值, 亮度值)
        """
        try:
            current_embedding = self._embed_sequence(sequence)
            
            if ProteinDesignEnv._brightness_model is not None:
                embedding_input = current_embedding.reshape(1, -1)
                brightness_zscore = float(ProteinDesignEnv._brightness_model.predict(
                    embedding_input,
                    normalization=ProteinDesignEnv._brightness_normalization,
                    return_raw=True
                )[0])
                
                brightness = brightness_zscore
            else:
                raise RuntimeError("亮度模型未加载，请先调用 ProteinDesignEnv.load_brightness_model()")
        except Exception as e:
            print(f"  [RL] 亮度计算失败: {e}")
            brightness = 0.0
        
        # 获取WT亮度（首次调用时计算并缓存）
        wt_brightness = self._calculate_wt_brightness()
        
        # 奖励函数设计：
        # 1. 亮度基础奖励：添加偏移量，确保初始奖励为正
        # 2. 亮度提升奖励：降低门槛，任何提升都给奖励
        # 3. 突变惩罚：增加到0.5，限制突变数量
        # 4. 终止奖励：增加奖励，鼓励提前终止
        
        # 基础亮度奖励（添加偏移量+权重3）
        # 偏移量=2.0确保初始奖励为正（WT亮度范围约-2.1到-1.2）
        brightness_reward = (brightness + 2.0) * 3.0
        
        # 亮度提升奖励（相对于WT，无论正负都给奖励）
        brightness_improvement = brightness - wt_brightness
        improvement_reward = brightness_improvement * 2.0
        
        # 突变惩罚（增加到0.5，限制突变数量）
        mutation_penalty = n_mutations * 0.5
        
        # 终止奖励（大幅提升少突变奖励，强力鼓励提前终止）
        if is_terminal and brightness > wt_brightness:
            if n_mutations == 1:
                terminal_bonus = 30.0
            elif n_mutations == 2:
                terminal_bonus = 20.0
            elif n_mutations <= 3:
                terminal_bonus = 12.0
            else:
                terminal_bonus = 5.0
        else:
            terminal_bonus = 0.0
        
        # 总奖励
        reward = brightness_reward + improvement_reward - mutation_penalty + terminal_bonus
        
        return reward, brightness
    
    # 类级别共享资源（所有环境实例共享）
    _shared_esm_extractor = None          # ESM提取器
    _brightness_model = None              # 亮度模型（需调用load_brightness_model加载）
    _norm_method = None                   # 归一化方法（Z-score或Min-Max）
    _norm_mean = None                     # 归一化均值
    _norm_std = None                      # 归一化标准差
    
    @classmethod
    def load_brightness_model(cls, model_path: str):
        """
        加载训练好的亮度模型（类级别共享）
        
        注意: 这是全局单例，所有ProteinDesignEnv实例共享同一个模型。
              如果需要使用不同模型，需要先调用此方法重新加载。
        
        参数:
            model_path: 模型权重文件路径
        """
        import torch
        from brightness_model import load_brightness_model_from_checkpoint
        
        print(f"  [RL] 警告: 亮度模型使用类级别全局状态，所有环境实例共享")
        
        # 使用工具函数加载模型（自动处理架构参数和归一化）
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model, norm_info = load_brightness_model_from_checkpoint(model_path, device=device)
        cls._brightness_model = model
        
        # 存储归一化参数
        cls._brightness_normalization = norm_info
        cls._norm_method = norm_info['method']
        if cls._norm_method == 'zscore':
            cls._norm_mean = norm_info['mean']
            cls._norm_std = norm_info['std']
            print(f"  [RL] 已加载亮度模型: {model_path}")
            print(f"  [RL] 归一化方法: Z-score, mean={cls._norm_mean:.4f}, std={cls._norm_std:.4f}")
            if norm_info.get('use_log1p', False):
                print(f"  [RL] 使用 log1p 变换")
        else:
            cls._norm_mean = norm_info.get('mean', 0.0)
            cls._norm_std = norm_info.get('std', 1.0)
            print(f"  [RL] 已加载亮度模型: {model_path}")
            print(f"  [RL] 归一化方法: {cls._norm_method}")
    
    @classmethod
    def _get_esm_extractor(cls):
        """
        获取ESM提取器（类级别单例，所有环境实例共享）
        
        返回:
            ESM2Extractor实例
        """
        if ProteinDesignEnv._shared_esm_extractor is None:
            print("加载ESM-2模型...", end="", flush=True)
            from extract_esm2 import ESM2Extractor
            ProteinDesignEnv._shared_esm_extractor = ESM2Extractor()
            print("完成")
        return ProteinDesignEnv._shared_esm_extractor
    
    def _embed_sequence(self, sequence: str) -> np.ndarray:
        """
        获取序列的ESM嵌入（使用全局缓存优化）
        
        参数:
            sequence: 氨基酸序列
        
        返回:
            ESM嵌入向量
        """
        # 使用全局缓存，所有环境实例共享
        global _GLOBAL_EMBEDDING_CACHE
        
        # 检查全局缓存
        if sequence in _GLOBAL_EMBEDDING_CACHE:
            return _GLOBAL_EMBEDDING_CACHE[sequence]
        
        # 获取提取器（单例）
        extractor = self._get_esm_extractor()
        embedding = extractor.extract(sequence)
        
        # 缓存结果到全局缓存
        _GLOBAL_EMBEDDING_CACHE[sequence] = embedding
        
        return embedding
    
    def reset(self) -> RLState:
        """
        重置环境到初始状态
        
        返回:
            初始状态
        """
        # 初始序列为野生型
        sequence = self.wt_sequence
        
        # 获取WT嵌入（只在第一次时计算，之后使用全局缓存）
        embedding = self._embed_sequence(sequence)
        
        # 初始突变掩码（全0）
        mutation_mask = np.zeros(self.seq_length, dtype=np.float32)
        
        # 初始突变氨基酸特征（全0）
        mutation_aas = np.zeros((self.seq_length, len(STANDARD_AA)), dtype=np.float32)
        
        # 初始突变数
        n_mutations = 0
        
        self._current_state = RLState(
            sequence=sequence,
            embedding=embedding,
            mutation_mask=mutation_mask,
            mutation_aas=mutation_aas,
            n_mutations=n_mutations
        )
        
        return self._current_state
    
    def step(self, action) -> Tuple[RLState, float, bool, Dict]:
        """
        执行一步动作
        
        参数:
            action: 可以是以下三种形式：
                - 整数动作索引：PPO网络输出的动作索引
                - (位置, 氨基酸) 二元组：显式指定突变
                - (-1, None)：停止信号
        
        返回:
            (新状态, 奖励, 是否结束, 额外信息)
        """
        # 处理整数动作索引（PPO网络输出）
        if isinstance(action, int):
            if action == self.stop_action_idx:
                return self._do_stop()
            # 将整数索引转换为 (位置, 氨基酸) 二元组
            pos = action // len(STANDARD_AA)
            aa_idx = action % len(STANDARD_AA)
            aa = STANDARD_AA[aa_idx]
            action = (pos, aa)
        
        # 处理停止信号 (-1, None)
        if isinstance(action, tuple) and len(action) == 2 and action[0] == -1:
            return self._do_stop()
        
        # 此时action应该是 (位置, 氨基酸) 二元组
        pos, aa = action
        
        # 检查位置是否有效
        if pos < 0 or pos >= self.seq_length:
            raise ValueError(f"位置 {pos} 超出范围")
        
        # 检查氨基酸是否有效
        if aa not in STANDARD_AA:
            raise ValueError(f"无效氨基酸: {aa}")
        
        # 获取当前状态
        current_seq = list(self._current_state.sequence)
        current_n_mutations = self._current_state.n_mutations
        
        # 执行突变（如果位置已有不同氨基酸，不算新突变）
        is_new_mutation = current_seq[pos] != aa
        if is_new_mutation:
            current_seq[pos] = aa
            current_n_mutations += 1
        
        new_sequence = ''.join(current_seq)
        
        # 更新嵌入（使用全局缓存，避免重复计算）
        new_embedding = self._embed_sequence(new_sequence)
        
        # 更新突变掩码
        new_mutation_mask = self._current_state.mutation_mask.copy()
        new_mutation_mask[pos] = 1.0
        
        # 更新突变氨基酸特征
        new_mutation_aas = self._current_state.mutation_aas.copy()
        if is_new_mutation:
            aa_idx = AA_TO_IDX[aa]
            new_mutation_aas[pos, :] = 0.0
            new_mutation_aas[pos, aa_idx] = 1.0
        
        # 创建新状态
        new_state = RLState(
            sequence=new_sequence,
            embedding=new_embedding,
            mutation_mask=new_mutation_mask,
            mutation_aas=new_mutation_aas,
            n_mutations=current_n_mutations
        )
        
        # 检查是否结束
        done = current_n_mutations >= self.max_mutations
        
        # 计算奖励（如果达到最大突变数，视为终止状态）
        reward, brightness = self._calculate_reward(new_sequence, current_n_mutations, is_terminal=done)
        
        # 更新当前状态
        self._current_state = new_state
        
        return new_state, reward, done, {
            "n_mutations": current_n_mutations,
            "position": pos,
            "amino_acid": aa,
            "brightness": brightness
        }
    
    def _do_stop(self) -> Tuple[RLState, float, bool, Dict]:
        """
        执行停止动作，返回当前状态和最终奖励
        
        返回:
            (当前状态, 最终奖励, 结束标志, 额外信息)
        """
        current_seq = self._current_state.sequence
        current_n_mutations = self._current_state.n_mutations
        
        # 停止动作是终止状态，传递 is_terminal=True
        final_reward, brightness = self._calculate_reward(current_seq, current_n_mutations, is_terminal=True)
        
        return self._current_state, final_reward, True, {
            "n_mutations": current_n_mutations,
            "position": -1,
            "amino_acid": "STOP",
            "brightness": brightness,
            "stopped_early": True
        }
    
    def get_state_vector(self, state: RLState) -> np.ndarray:
        """
        将状态转换为向量表示
        
        参数:
            state: RL状态
        
        返回:
            状态向量（用于输入神经网络）
        """
        # 使用原始嵌入（与亮度模型训练时一致，不进行L2归一化）
        # 亮度模型训练时使用原始ESM-2嵌入，这里保持一致
        raw_embedding = state.embedding
        
        # 归一化突变数
        normalized_n_mutations = state.n_mutations / self.max_mutations
        
        # 对突变掩码进行padding到参考长度
        mutation_mask = state.mutation_mask
        if len(mutation_mask) < self.reference_length:
            padded_mask = np.zeros(self.reference_length, dtype=np.float32)
            padded_mask[:len(mutation_mask)] = mutation_mask
            mutation_mask = padded_mask
        
        # 对突变氨基酸特征进行padding到参考长度
        mutation_aas = state.mutation_aas.flatten()
        expected_aas_len = self.reference_length * len(STANDARD_AA)
        if len(mutation_aas) < expected_aas_len:
            padded_aas = np.zeros(expected_aas_len, dtype=np.float32)
            padded_aas[:len(mutation_aas)] = mutation_aas
            mutation_aas = padded_aas
        
        # 拼接所有特征: 嵌入 + 突变掩码 + 突变氨基酸特征 + 突变数
        state_vector = np.concatenate([
            raw_embedding,
            mutation_mask,
            mutation_aas,
            [normalized_n_mutations]
        ])
        
        return state_vector.astype(np.float32)
    
    def action_to_index(self, action: Tuple[int, str]) -> int:
        """
        将动作转换为索引
        
        参数:
            action: (位置, 氨基酸) 二元组
        
        返回:
            动作索引
        """
        pos, aa = action
        aa_idx = AA_TO_IDX[aa]
        return pos * len(STANDARD_AA) + aa_idx
    
    def index_to_action(self, idx: int):
        """
        将索引转换为动作
        
        参数:
            idx: 动作索引
        
        返回:
            (位置, 氨基酸) 二元组，或 (-1, "STOP") 表示停止动作
        """
        if idx == self.stop_action_idx:
            return (-1, "STOP")
        
        n_aa = len(STANDARD_AA)
        pos = idx // n_aa
        aa_idx = idx % n_aa
        
        if pos >= self.seq_length:
            pos = pos % self.seq_length
        
        return pos, STANDARD_AA[aa_idx]