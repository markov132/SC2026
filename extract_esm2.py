"""
SC2026 蛋白设计赛道 - ESM-2嵌入提取器
===================================
本模块实现从蛋白质序列提取ESM-2嵌入的功能。

ESM-2 (Evolutionary Scale Modeling):
- 基于Transformer的蛋白质语言模型
- 学习蛋白质序列的进化表示

嵌入提取流程:
1. 加载预训练的ESM-2模型
2. 将氨基酸序列token化
3. 通过模型前向传播获取每残基嵌入
4. Mean Pooling汇聚为固定维度向量 [1280]
"""

import torch
import esm
from typing import List, Dict, Optional, Tuple
import numpy as np
import h5py
import os
import time
from tqdm import tqdm


# =============================================================================
# ESM-2嵌入提取器
# =============================================================================

class ESM2Extractor:
    """
    ESM-2嵌入提取器

    支持的模型:
    - esm2_t30_150M_UR50D: 150M参数，嵌入维度768
    - esm2_t33_650M_UR50D: 650M参数，嵌入维度1280
    - esm2_t36_3B_UR50D: 3B参数，嵌入维度2560

    使用示例:
    >>> extractor = ESM2Extractor(model_name="esm2_t33_650M_UR50D")
    >>> embedding = extractor.extract("MSKGEELFTGVVPILVELDGD...")
    >>> print(embedding.shape)  # (1280,)
    """

    # 模型名称到实际模型名称的映射
    MODEL_ALIASES = {
        'esm2_t30_150M': 'esm2_t30_150M_UR50D',
        'esm2_t33_650M': 'esm2_t33_650M_UR50D',
        'esm2_t36_3B': 'esm2_t36_3B_UR50D',
    }

    # 各模型的嵌入维度（单一层）
    EMBEDDING_DIMS = {
        'esm2_t30_150M_UR50D': 768,
        'esm2_t33_650M_UR50D': 1280,
        'esm2_t36_3B_UR50D': 2560,
    }

    # 默认提取的层（用于多层特征融合）
    DEFAULT_LAYERS = [23, 28, 33]

    def __init__(
        self,
        model_name: str = 'esm2_t33_650M_UR50D',
        device: Optional[str] = None,
        batch_size: int = 4,
        layers: Optional[List[int]] = None
    ):
        """
        初始化ESM-2提取器

        参数:
            model_name: 模型名称或别名
            device: 计算设备 (cuda/cpu)，None则自动选择
            batch_size: 批处理大小
            layers: 要提取的层列表，None则使用默认层 [23, 28, 33]
        """
        # 解析模型别名
        self.model_name = self.MODEL_ALIASES.get(model_name, model_name)

        # 自动选择设备
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.batch_size = batch_size
        self.model = None
        self.alphabet = None
        self.batch_converter = None

        # 设置提取层
        self.layers = layers if layers is not None else self.DEFAULT_LAYERS

        # 获取单一层的嵌入维度
        self.single_layer_dim = self.EMBEDDING_DIMS.get(
            self.model_name,
            1280  # 默认值
        )

        # 计算融合后的总嵌入维度
        self.embedding_dim = self.single_layer_dim * len(self.layers)

    def load_model(self):
        """加载预训练的ESM-2模型"""
        print(f"正在加载ESM-2模型: {self.model_name}")

        # 加载模型和alphabet
        # esm.pretrained.esm2_t33_650M_UR50D() 返回 (model, alphabet)
        if self.model_name == 'esm2_t33_650M_UR50D':
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        elif self.model_name == 'esm2_t30_150M_UR50D':
            self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        elif self.model_name == 'esm2_t36_3B_UR50D':
            self.model, self.alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        else:
            raise ValueError(f"不支持的模型: {self.model_name}")

        # 移动到设备
        self.model = self.model.to(self.device)
        self.model.eval()

        # 创建batch converter
        self.batch_converter = self.alphabet.get_batch_converter()

        print(f"模型加载完成，设备: {self.device}")
        print(f"嵌入维度: {self.embedding_dim}")

    def extract(self, sequence: str, truncate_len: int = 1022, verbose: bool = False) -> np.ndarray:
        """
        从单条序列提取嵌入（多层特征融合）

        参数:
            sequence: 氨基酸序列
            truncate_len: 最大序列长度 (ESM-2限制)
            verbose: 是否打印详细进度

        返回:
            嵌入向量 [embedding_dim]，多层特征拼接后的向量
        """
        if self.model is None:
            self.load_model()

        original_len = len(sequence)
        
        # 截断过长序列
        if original_len > truncate_len:
            print(f"警告: 序列长度 {original_len} 超过限制 {truncate_len}，将被截断")
            sequence = sequence[:truncate_len]

        if verbose:
            print(f"  提取嵌入: 序列长度={len(sequence)}, 设备={self.device}")

        # 准备数据
        data = [('seq1', sequence)]

        # Batched forward
        with torch.no_grad():
            # Convert to tokens
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens = batch_tokens.to(self.device)
            
            if verbose:
                print(f"  Token化完成，开始前向传播...")
                torch.cuda.synchronize() if self.device == 'cuda' else None

            # Forward pass - 提取指定的多个层
            results = self.model(
                batch_tokens,
                repr_layers=self.layers,
                return_contacts=False
            )
            
            if verbose:
                print(f"  前向传播完成，处理结果...")

            # 收集所有层的表示并拼接
            all_layer_embeddings = []
            for layer in self.layers:
                # 获取该层的表示
                token_representations = results['representations'][layer]

                # 排除特殊token (<cls>, <eos>)，计算mean pooling
                # token_representations shape: [1, seq_len+2, embedding_dim]
                aa_repr = token_representations[0, 1:-1, :]  # [seq_len, embedding_dim]

                # Mean pooling
                layer_embedding = aa_repr.mean(dim=0).cpu().numpy()
                all_layer_embeddings.append(layer_embedding)

            # 拼接所有层的嵌入向量
            # 输出维度: single_layer_dim * n_layers
            embedding = np.concatenate(all_layer_embeddings)
            
            if verbose:
                print(f"  嵌入提取完成，维度={embedding.shape}")

        return embedding

    def extract_batch(
        self,
        sequences: List[str],
        truncate_len: int = 1022,
        show_progress: bool = True,
        checkpoint_path: Optional[str] = None,
        checkpoint_interval: int = 1000
    ) -> np.ndarray:
        """
        批量提取嵌入（支持检查点恢复）

        参数:
            sequences: 序列列表
            truncate_len: 最大序列长度
            show_progress: 是否显示进度条
            checkpoint_path: 检查点文件路径（None表示不使用检查点）
            checkpoint_interval: 检查点保存间隔（每处理多少条序列保存一次）

        返回:
            嵌入矩阵 [N, embedding_dim]
        """
        if self.model is None:
            self.load_model()

        n_sequences = len(sequences)
        all_embeddings = []
        start_idx = 0

        # 尝试从检查点恢复
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = np.load(checkpoint_path, allow_pickle=True)
                if 'embeddings' in checkpoint and 'index' in checkpoint:
                    loaded_embeddings = checkpoint['embeddings']
                    start_idx = int(checkpoint['index'])
                    all_embeddings = loaded_embeddings.tolist()
                    print(f"从检查点恢复: 已处理 {start_idx}/{n_sequences} 条序列")
            except Exception as e:
                print(f"检查点文件损坏，将从头开始: {e}")

        # 处理剩余序列
        iterator = range(start_idx, n_sequences)
        if show_progress:
            iterator = tqdm(iterator, desc="提取嵌入", initial=start_idx, total=n_sequences)

        for i in iterator:
            seq = sequences[i]
            emb = self.extract(seq, truncate_len)
            all_embeddings.append(emb)

            # 保存检查点
            if checkpoint_path and (i + 1) % checkpoint_interval == 0:
                np.savez(checkpoint_path, embeddings=np.array(all_embeddings), index=i + 1)

        # 清理检查点（处理完成后删除）
        if checkpoint_path and os.path.exists(checkpoint_path):
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    os.remove(checkpoint_path)
                    break
                except OSError as e:
                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                    else:
                        print(f"  警告: 无法删除检查点文件: {e}")
                        print(f"        文件将保留: {checkpoint_path}")

        return np.array(all_embeddings)

    def extract_to_h5(
        self,
        sequences: List[str],
        sequence_ids: List[str],
        output_path: str,
        truncate_len: int = 1022
    ):
        """
        批量提取嵌入并保存到HDF5文件

        HDF5文件结构:
        - /embeddings: [N, embedding_dim] 嵌入矩阵
        - /sequence_ids: [N] 序列ID列表
        - /sequences: [N] 序列列表

        参数:
            sequences: 序列列表
            sequence_ids: 序列ID列表
            output_path: 输出HDF5文件路径
            truncate_len: 最大序列长度
        """
        if self.model is None:
            self.load_model()

        # 批量提取
        all_embeddings = []
        iterator = tqdm(zip(sequence_ids, sequences), desc="提取嵌入")

        for seq_id, seq in iterator:
            emb = self.extract(seq, truncate_len)
            all_embeddings.append(emb)

        # 转换为数组
        all_embeddings = np.array(all_embeddings)

        # 保存到HDF5
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('embeddings', data=all_embeddings)
            f.create_dataset('sequence_ids', data=[s.encode() for s in sequence_ids])
            f.create_dataset('sequences', data=[s.encode() for s in sequences])

            # 保存元数据
            f.attrs['model_name'] = self.model_name
            f.attrs['embedding_dim'] = self.embedding_dim
            f.attrs['n_sequences'] = len(sequences)

        print(f"嵌入已保存到: {output_path}")
        print(f"形状: {all_embeddings.shape}")


# =============================================================================
# 便捷函数
# =============================================================================

def load_embeddings_from_h5(h5_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    从HDF5文件加载嵌入

    参数:
        h5_path: HDF5文件路径

    返回:
        (embeddings, sequence_ids, sequences) 元组
    """
    with h5py.File(h5_path, 'r') as f:
        embeddings = f['embeddings'][:]

        # 解码bytes
        sequence_ids = [
            s.decode() if isinstance(s, bytes) else s
            for s in f['sequence_ids'][:]
        ]
        sequences = [
            s.decode() if isinstance(s, bytes) else s
            for s in f['sequences'][:]
        ]

    return embeddings, sequence_ids, sequences


def extract_esm2_embeddings(
    sequences: List[str],
    model_name: str = 'esm2_t33_650M_UR50D',
    batch_size: int = 4,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    便捷函数：提取ESM-2嵌入

    参数:
        sequences: 序列列表
        model_name: 模型名称
        batch_size: 批处理大小
        output_path: 可选的输出路径

    返回:
        嵌入矩阵
    """
    extractor = ESM2Extractor(model_name=model_name, batch_size=batch_size)

    embeddings = extractor.extract_batch(sequences)

    if output_path:
        # 生成序列ID
        sequence_ids = [f"seq_{i}" for i in range(len(sequences))]
        extractor.extract_to_h5(
            sequences,
            sequence_ids,
            output_path
        )

    return embeddings
