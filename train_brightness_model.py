"""
SC2026 蛋白设计赛道 - 亮度模型训练
===================================

本模块实现亮度预测模型的完整训练流程：
1. 加载和筛选GFP数据
2. 提取ESM-2序列嵌入
3. 监督学习训练MLP预测头
4. 模型评估和测试

训练流程:
GFP_data.xlsx → 数据筛选 → ESM-2嵌入提取 → MLP训练 → 模型评估
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
import time
import argparse
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr, kendalltau

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import GFPDataLoader
from brightness_model import BrightnessModelESM, BrightnessTrainer, BrightnessDataset
from extract_esm2 import ESM2Extractor


# =============================================================================
# 数据筛选模块
# =============================================================================

def filter_gfp_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    完整的GFP数据筛选流程
    
    参数:
        df: 原始DataFrame（列名: aaMutations, GFP type, Brightness）
    
    返回:
        筛选后的DataFrame
    """
    original_len = len(df)
    
    # 1. 去除空值
    df = df.dropna(subset=['aaMutations', 'Brightness'])
    print(f"  去除空值: {original_len} → {len(df)}")
    
    # 2. 去除重复突变
    df = df.drop_duplicates(subset='aaMutations')
    print(f"  去除重复: {len(df)}")
    
    print(f"  最终样本数: {len(df)}")
    return df


# =============================================================================
# 嵌入提取模块
# =============================================================================

class EmbeddingExtractor:
    """
    ESM-2嵌入提取器，支持批量处理、缓存和检查点
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.extractor = None
        self.cache = {}
    
    def _get_extractor(self):
        if self.extractor is None:
            print("加载ESM-2模型...")
            self.extractor = ESM2Extractor()
            print("ESM-2模型加载完成")
        return self.extractor
    
    def extract_batch(self, sequences: list, batch_size: int = 8, 
                      checkpoint_path: str = None, checkpoint_interval: int = 1000) -> np.ndarray:
        """
        批量提取序列嵌入（支持检查点恢复）
        
        参数:
            sequences: 序列列表
            batch_size: 批处理大小
            checkpoint_path: 检查点文件路径（None表示不使用检查点）
            checkpoint_interval: 检查点保存间隔
            
        返回:
            嵌入数组 [N, 3840]（多层特征融合，1280 * 3层）
        """
        extractor = self._get_extractor()
        all_embeddings = []
        n_sequences = len(sequences)
        start_idx = 0

        # 尝试从检查点恢复
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint = np.load(checkpoint_path, allow_pickle=True)
                if 'embeddings' in checkpoint and 'index' in checkpoint:
                    loaded_embeddings = checkpoint['embeddings']
                    start_idx = int(checkpoint['index'])
                    all_embeddings = loaded_embeddings.tolist()
                    print(f"  从检查点恢复: 已处理 {start_idx}/{n_sequences} 条序列")
            except Exception as e:
                print(f"  检查点文件损坏，将从头开始: {e}")

        n_batches = (n_sequences + batch_size - 1) // batch_size
        
        for i in range(start_idx, n_sequences, batch_size):
            end = min(i + batch_size, n_sequences)
            batch = sequences[i:end]
            batch_num = i // batch_size + 1
            
            if batch_num % 10 == 0 or batch_num == 1:
                print(f"  提取嵌入进度: {batch_num}/{n_batches}")
            
            # 检查缓存
            batch_embeddings = []
            for seq in batch:
                if seq in self.cache:
                    batch_embeddings.append(self.cache[seq])
                else:
                    emb = extractor.extract(seq)
                    self.cache[seq] = emb
                    batch_embeddings.append(emb)
            
            all_embeddings.extend(batch_embeddings)

            # 保存检查点
            if checkpoint_path and end % checkpoint_interval == 0:
                np.savez(checkpoint_path, embeddings=np.array(all_embeddings), index=end)

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


# =============================================================================
# 模型评估模块
# =============================================================================

def evaluate_model(model, X_test, y_test, device='cuda'):
    """
    全面评估模型性能（包含NaN处理）
    
    参数:
        model: 训练好的模型
        X_test: 测试集嵌入
        y_test: 测试集标签
        device: 计算设备
    
    返回:
        评估指标字典
    """
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(X_tensor).cpu().numpy()
    
    # 处理NaN值
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 计算各种评估指标
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    
    # R²分数（可能为NaN当方差为零时）
    try:
        r2 = r2_score(y_test, predictions)
        if np.isnan(r2):
            r2 = 0.0
    except Exception:
        r2 = 0.0
    
    # Pearson相关系数（可能为NaN当标准差为零时）
    try:
        if np.std(y_test) == 0 or np.std(predictions) == 0:
            pearson_corr = 0.0
            pearson_p = 1.0
        else:
            pearson_corr, pearson_p = pearsonr(y_test, predictions)
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
                pearson_p = 1.0
    except Exception:
        pearson_corr = 0.0
        pearson_p = 1.0
    
    # Spearman相关系数（当存在大量重复值时可能为NaN，使用Kendall's tau作为替代）
    try:
        spearman_corr, spearman_p = spearmanr(y_test, predictions)
        if np.isnan(spearman_corr):
            print("    [警告] Spearman系数因大量重复值无法计算，使用Kendall's tau替代")
            tau_corr, tau_p = kendalltau(y_test, predictions)
            spearman_corr = tau_corr
            spearman_p = tau_p
    except Exception as e:
        print(f"    [警告] Spearman系数计算失败: {e}，使用Kendall's tau替代")
        try:
            tau_corr, tau_p = kendalltau(y_test, predictions)
            spearman_corr = tau_corr
            spearman_p = tau_p
        except Exception:
            spearman_corr = 0.0
            spearman_p = 1.0
    
    # 预测偏差分析
    residuals = y_test - predictions
    mean_bias = np.mean(residuals)
    std_bias = np.std(residuals)
    
    metrics = {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'Pearson_r': float(pearson_corr),
        'Pearson_p': float(pearson_p),
        'Spearman_r': float(spearman_corr),
        'Spearman_p': float(spearman_p),
        'Mean_Bias': float(mean_bias),
        'Std_Bias': float(std_bias),
        'Min_Prediction': float(predictions.min()),
        'Max_Prediction': float(predictions.max()),
        'Min_Actual': float(y_test.min()),
        'Max_Actual': float(y_test.max())
    }
    
    return metrics, predictions


def print_evaluation_results(metrics, split_name='Test'):
    """
    打印评估结果
    """
    print(f"\n{'='*60}")
    print(f"{split_name} Set Evaluation Results")
    print(f"{'='*60}")
    print(f"MSE (均方误差):           {metrics['MSE']:.4f}")
    print(f"RMSE (均方根误差):        {metrics['RMSE']:.4f}")
    print(f"MAE (平均绝对误差):       {metrics['MAE']:.4f}")
    print(f"R² (决定系数):            {metrics['R2']:.4f}")
    print(f"Pearson相关系数:          {metrics['Pearson_r']:.4f} (p={metrics['Pearson_p']:.2e})")
    print(f"Spearman相关系数:         {metrics['Spearman_r']:.4f} (p={metrics['Spearman_p']:.2e})")
    print(f"预测偏差 (Mean±Std):      {metrics['Mean_Bias']:.4f} ± {metrics['Std_Bias']:.4f}")
    print(f"预测范围:                 [{metrics['Min_Prediction']:.2f}, {metrics['Max_Prediction']:.2f}]")
    print(f"实际范围:                 [{metrics['Min_Actual']:.2f}, {metrics['Max_Actual']:.2f}]")
    print(f"{'='*60}")


# =============================================================================
# 主训练流程
# =============================================================================

def train_brightness_model(
    data_path: str,
    output_dir: str,
    embed_cache_path: str = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-4,
    hidden_dims: list = None,
    early_stopping_patience: int = 15,
    random_seed: int = 42,
    sample_size: int = None,
    resume_from: str = None,
    freeze_layers: bool = False,
    pretrain_normalization: dict = None,
    use_pretrain_norm: bool = False,
    use_log1p: bool = False,
    huber_delta: float = 1.0
):
    """
    完整的亮度模型训练流程
    
    参数:
        data_path: GFP数据Excel文件路径
        output_dir: 输出目录
        embed_cache_path: 嵌入缓存文件路径（用于加速）
        test_size: 测试集比例
        val_size: 验证集比例（相对于非测试集）
        batch_size: 批处理大小
        epochs: 最大训练轮数
        learning_rate: 学习率
        hidden_dims: MLP隐藏层维度
        early_stopping_patience: 早停耐心值
        random_seed: 随机种子
        resume_from: 从checkpoint恢复训练的路径
        sample_size: 随机采样数量（默认None表示使用全部数据）
        freeze_layers: 是否冻结底层隐藏层（仅微调顶层）
        pretrain_normalization: 预训练阶段的归一化参数（用于微调时保持一致）
        use_pretrain_norm: 是否使用预训练的归一化参数（与pretrain_normalization配合使用）
        use_log1p: 是否对目标亮度值进行log1p对数变换（长尾分布数据推荐）
        huber_delta: Huber损失的delta参数（默认1.0）
    """
    
    # 设置随机种子（确保可复现性）
    import random
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'embeddings'), exist_ok=True)
    
    print("=" * 60)
    print("亮度模型训练")
    print("=" * 60)
    print(f"数据路径: {data_path}")
    print(f"输出目录: {output_dir}")
    print(f"测试集比例: {test_size}")
    print(f"验证集比例: {val_size}")
    print(f"批处理大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"隐藏层维度: {hidden_dims or [512, 128]}")
    
    # 1. 加载数据
    print("\n" + "=" * 60)
    print("Step 1: 加载GFP数据")
    print("=" * 60)
    
    loader = GFPDataLoader()
    # 尝试自动检测sheet名
    try:
        df = loader.load_from_excel(data_path, sheet_name=0)  # 尝试第一个sheet
    except Exception as e:
        print(f"  警告: 自动检测sheet失败 ({e})，尝试 'brightness'...")
        try:
            df = loader.load_from_excel(data_path, sheet_name='brightness')
        except Exception as e2:
            raise RuntimeError(f"无法加载数据文件: {e2}")
    print(f"原始数据: {len(df)} 条")
    
    # 2. 数据筛选
    print("\n" + "=" * 60)
    print("Step 2: 数据筛选")
    print("=" * 60)
    
    df = filter_gfp_data(df)
    
    if len(df) < 50:
        raise ValueError(f"筛选后数据量太少 ({len(df)}), 无法训练")
    
    # 随机采样
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_seed)
        print(f"  随机采样: {sample_size} 条")
    
    # 3. 构建完整序列并划分数据集
    print("\n" + "=" * 60)
    print("Step 3: 构建序列并划分")
    print("=" * 60)
    
    # 构建完整序列
    print("  正在构建完整序列...")
    sequences = []
    brightness_values = []
    
    from seq_builder import WT_SEQUENCES, SeqBuilder
    
    seq_builder = SeqBuilder()
    
    success_count = 0
    fail_count = 0
    
    gfp_type_col = 'GFP type' if 'GFP type' in df.columns else 'gfp_type' if 'gfp_type' in df.columns else None

    for _, row in df.iterrows():
        mutation_str = row['aaMutations']
        try:
            # 根据GFP类型获取对应的WT序列
            gfp_type = row.get(gfp_type_col, 'sfGFP')
            wt_seq = WT_SEQUENCES.get(gfp_type, WT_SEQUENCES['sfGFP'])
            
            # 解析突变并构建序列
            if mutation_str == 'WT':
                seq = wt_seq
            else:
                seq = seq_builder.build_from_mutations(
                    gfp_type=gfp_type,
                    mutation_str=str(mutation_str),
                    validate=False
                )
            
            # 验证序列长度
            if len(seq) == len(wt_seq):
                sequences.append(seq)
                brightness_values.append(row['Brightness'])
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            fail_count += 1
    
    print(f"  成功构建 {success_count} 条序列，失败 {fail_count} 条")
    
    if len(sequences) < 50:
        raise ValueError(f"成功构建的序列太少 ({len(sequences)}), 无法训练")
    
    brightness = np.array(brightness_values)
    
    # 先划分测试集
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        sequences, brightness, 
        test_size=test_size, 
        random_state=random_seed
    )
    
    # 再划分训练集和验证集
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio,
        random_state=random_seed
    )
    
    print(f"训练集: {len(X_train)} 条")
    print(f"验证集: {len(X_val)} 条")
    print(f"测试集: {len(X_test)} 条")
    
    # ========== 目标值变换（log1p + Z-score） ==========
    print(f"\n原始亮度范围: [{brightness.min():.4f}, {brightness.max():.4f}]")
    
    # Step 1: log1p 变换（可选，用于长尾分布）
    y_train_transformed = y_train.copy()
    y_val_transformed = y_val.copy()
    y_test_transformed = y_test.copy()
    
    if use_log1p:
        print(f"\n对目标亮度值进行 log1p 变换...")
        # 确保非负
        y_train_transformed = np.log1p(np.maximum(y_train, 0.0))
        y_val_transformed = np.log1p(np.maximum(y_val, 0.0))
        y_test_transformed = np.log1p(np.maximum(y_test, 0.0))
        print(f"  log1p后范围: [{y_train_transformed.min():.4f}, {y_train_transformed.max():.4f}]")
    
    # Step 2: 使用训练集计算 Z-score 参数
    brightness_mean = float(np.mean(y_train_transformed))
    brightness_std = float(np.std(y_train_transformed))
    brightness_min = float(np.min(y_train_transformed))
    brightness_max = float(np.max(y_train_transformed))
    
    # 防止标准差为零
    if brightness_std < 1e-8:
        brightness_std = 1.0
        print(f"警告: 训练集亮度标准差过小，已设置为 1.0")
    
    # Step 3: Z-score 归一化
    y_train_norm = (y_train_transformed - brightness_mean) / brightness_std
    y_val_norm = (y_val_transformed - brightness_mean) / brightness_std
    y_test_norm = (y_test_transformed - brightness_mean) / brightness_std
    
    print(f"Z-score 归一化后范围: [{y_train_norm.min():.4f}, {y_train_norm.max():.4f}]")
    print(f"归一化参数: mean={brightness_mean:.4f}, std={brightness_std:.4f}")
    print(f"变换后范围: min={brightness_min:.4f}, max={brightness_max:.4f}")
    print(f"use_log1p: {use_log1p}")
    
    # 初始化 data_info 字典
    data_info = {}
    
    # 保存归一化参数
    data_info['normalization'] = {
        'mean': brightness_mean,
        'std': brightness_std,
        'min': brightness_min,
        'max': brightness_max,
        'method': 'zscore',
        'use_log1p': use_log1p
    }
    data_info['test_ratio'] = test_size
    data_info['val_ratio'] = val_size
    data_info['brightness_stats'] = {
        'train_mean': float(np.mean(y_train)),
        'train_std': float(np.std(y_train)),
        'test_mean': float(np.mean(y_test)),
        'test_std': float(np.std(y_test))
    }
    data_info['brightness_range'] = {
        'original_min': float(brightness.min()),
        'original_max': float(brightness.max())
    }
    
    # 4. 提取嵌入
    print("\n" + "=" * 60)
    print("Step 4: 提取ESM-2嵌入")
    print("=" * 60)
    
    # 计算序列哈希用于缓存校验
    import hashlib
    
    def compute_sequences_hash(sequences):
        """计算序列列表的哈希值"""
        combined = ''.join(sequences)
        return hashlib.md5(combined.encode()).hexdigest()
    
    train_hash = compute_sequences_hash(X_train)
    val_hash = compute_sequences_hash(X_val)
    test_hash = compute_sequences_hash(X_test)
    
    # 检查缓存
    cache_file = embed_cache_path or os.path.join(output_dir, 'embeddings', 'esm2_embeddings.npz')
    
    use_cache = False
    if os.path.exists(cache_file):
        try:
            data = np.load(cache_file)
            # 检查缓存数据量是否匹配
            if (len(data['train']) == len(X_train) and 
                len(data['val']) == len(X_val) and 
                len(data['test']) == len(X_test)):
                # 检查序列哈希是否匹配
                cached_train_hash = data.get('train_hash', '')
                cached_val_hash = data.get('val_hash', '')
                cached_test_hash = data.get('test_hash', '')
                
                if (cached_train_hash == train_hash and 
                    cached_val_hash == val_hash and 
                    cached_test_hash == test_hash):
                    print(f"从缓存加载嵌入（序列哈希验证通过）: {cache_file}")
                    train_embeddings = data['train']
                    val_embeddings = data['val']
                    test_embeddings = data['test']
                    use_cache = True
                else:
                    print(f"缓存序列哈希不匹配，重新提取嵌入")
            else:
                print(f"缓存数据量不匹配（缓存: train={len(data['train'])}, val={len(data['val'])}, test={len(data['test'])}），重新提取嵌入")
        except Exception as e:
            print(f"缓存文件读取失败: {e}，重新提取嵌入")
    
    if not use_cache:
        print("提取训练集嵌入...")
        extractor = EmbeddingExtractor()
        
        # 使用检查点提取训练集嵌入
        train_checkpoint = os.path.join(output_dir, 'embeddings', 'train_checkpoint.npz')
        train_embeddings = extractor.extract_batch(X_train, checkpoint_path=train_checkpoint)
        
        print("提取验证集嵌入...")
        val_checkpoint = os.path.join(output_dir, 'embeddings', 'val_checkpoint.npz')
        val_embeddings = extractor.extract_batch(X_val, checkpoint_path=val_checkpoint)
        
        print("提取测试集嵌入...")
        test_checkpoint = os.path.join(output_dir, 'embeddings', 'test_checkpoint.npz')
        test_embeddings = extractor.extract_batch(X_test, checkpoint_path=test_checkpoint)
        
        # 保存缓存（包含序列哈希用于校验）
        np.savez(cache_file, 
                 train=train_embeddings, 
                 val=val_embeddings, 
                 test=test_embeddings,
                 train_hash=train_hash,
                 val_hash=val_hash,
                 test_hash=test_hash)
        print(f"嵌入已保存到: {cache_file}")
    
    print(f"嵌入维度: {train_embeddings.shape[1]}")
    
    # 5. 创建数据集（使用归一化后的目标值）
    print("\n" + "=" * 60)
    print("Step 5: 创建数据集")
    print("=" * 60)
    
    train_dataset = BrightnessDataset(train_embeddings, y_train_norm)
    val_dataset = BrightnessDataset(val_embeddings, y_val_norm)
    test_dataset = BrightnessDataset(test_embeddings, y_test_norm)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    # 6. 创建模型
    print("\n" + "=" * 60)
    print("Step 6: 创建模型")
    print("=" * 60)
    
    if hidden_dims is None:
        hidden_dims = [512, 128]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if resume_from and os.path.exists(resume_from):
        print(f"  从预训练模型加载: {resume_from}")
        checkpoint = torch.load(resume_from, map_location='cpu')
        model = BrightnessModelESM(
            input_dim=checkpoint['config']['input_dim'],
            hidden_dims=checkpoint['config']['hidden_dims'],
            dropout=0.3
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if freeze_layers:
            print("  冻结底层隐藏层，仅训练顶层...")
            for name, param in model.named_parameters():
                if 'fc' in name and name != 'fc_final':
                    param.requires_grad = False
        
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  加载后可训练参数量: {trainable_params:,}")
        
        if pretrain_normalization:
            if 'mean' in pretrain_normalization and 'std' in pretrain_normalization:
                print(f"  使用预训练归一化参数: mean={pretrain_normalization['mean']:.4f}, std={pretrain_normalization['std']:.4f}")
                brightness_mean = pretrain_normalization['mean']
                brightness_std = pretrain_normalization['std']
            else:
                print(f"  警告: 预训练归一化参数缺少mean或std字段 ({list(pretrain_normalization.keys())})，使用当前数据的归一化参数")
                # 保持使用当前数据计算的 brightness_mean 和 brightness_std
        elif use_pretrain_norm:
            print(f"  警告: 指定了 --use_pretrain_norm 但未提供 pretrain_normalization，使用当前数据的归一化参数")
    else:
        model = BrightnessModelESM(
            input_dim=train_embeddings.shape[1],
            hidden_dims=hidden_dims,
            dropout=0.3
        ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 7. 训练模型
    print("\n" + "=" * 60)
    print("Step 7: 训练模型")
    print("=" * 60)
    
    trainer = BrightnessTrainer(
        model=model,
        learning_rate=learning_rate,
        device=device,
        huber_delta=huber_delta,
        monitor_metric='val_r2'
    )
    
    model_path = os.path.join(output_dir, 'brightness_model.pth')
    
    # 准备归一化和配置参数（Z-score + 可选 log1p）
    normalization = {
        'mean': brightness_mean,
        'std': brightness_std,
        'min': brightness_min,
        'max': brightness_max,
        'method': 'zscore',
        'use_log1p': use_log1p
    }
    model_config = {
        'input_dim': train_embeddings.shape[1],
        'hidden_dims': hidden_dims,
        'dropout': 0.3
    }
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        save_path=model_path,
        resume_from=resume_from if not freeze_layers else None,
        normalization=normalization,
        model_config=model_config
    )
    
    # 8. 评估模型
    print("\n" + "=" * 60)
    print("Step 8: 模型评估")
    print("=" * 60)
    
    # 定义完整反归一化函数（Z-score + 可选的 log1p 逆变换）
    def denormalize_full(y_norm):
        # 第一步：反Z-score
        y_transformed = y_norm * brightness_std + brightness_mean
        # 第二步：反log1p（如果使用了）
        if use_log1p:
            y_raw = np.expm1(y_transformed)
            y_raw = np.maximum(y_raw, 0.0)  # 确保非负
            return y_raw
        return y_transformed
    
    # 训练集评估（完整反归一化到原始亮度空间）
    train_metrics, train_pred_norm = evaluate_model(model, train_embeddings, y_train_norm, device)
    train_pred = denormalize_full(train_pred_norm)
    train_metrics['Min_Prediction'] = float(train_pred.min())
    train_metrics['Max_Prediction'] = float(train_pred.max())
    train_metrics['Min_Actual'] = float(y_train.min())
    train_metrics['Max_Actual'] = float(y_train.max())
    train_metrics['MSE'] = float(np.mean((y_train - train_pred) ** 2))
    train_metrics['RMSE'] = float(np.sqrt(train_metrics['MSE']))
    train_metrics['MAE'] = float(np.mean(np.abs(y_train - train_pred)))
    train_metrics['Mean_Bias'] = float(np.mean(y_train - train_pred))
    train_metrics['Std_Bias'] = float(np.std(y_train - train_pred))
    from sklearn.metrics import r2_score
    train_metrics['R2'] = float(r2_score(y_train, train_pred))
    print_evaluation_results(train_metrics, 'Train')
    
    # 验证集评估（完整反归一化到原始亮度空间）
    val_metrics, val_pred_norm = evaluate_model(model, val_embeddings, y_val_norm, device)
    val_pred = denormalize_full(val_pred_norm)
    val_metrics['Min_Prediction'] = float(val_pred.min())
    val_metrics['Max_Prediction'] = float(val_pred.max())
    val_metrics['Min_Actual'] = float(y_val.min())
    val_metrics['Max_Actual'] = float(y_val.max())
    val_metrics['MSE'] = float(np.mean((y_val - val_pred) ** 2))
    val_metrics['RMSE'] = float(np.sqrt(val_metrics['MSE']))
    val_metrics['MAE'] = float(np.mean(np.abs(y_val - val_pred)))
    val_metrics['Mean_Bias'] = float(np.mean(y_val - val_pred))
    val_metrics['Std_Bias'] = float(np.std(y_val - val_pred))
    val_metrics['R2'] = float(r2_score(y_val, val_pred))
    print_evaluation_results(val_metrics, 'Validation')
    
    # 测试集评估（完整反归一化到原始亮度空间）
    test_metrics, test_pred_norm = evaluate_model(model, test_embeddings, y_test_norm, device)
    test_pred = denormalize_full(test_pred_norm)
    test_metrics['Min_Prediction'] = float(test_pred.min())
    test_metrics['Max_Prediction'] = float(test_pred.max())
    test_metrics['Min_Actual'] = float(y_test.min())
    test_metrics['Max_Actual'] = float(y_test.max())
    test_metrics['MSE'] = float(np.mean((y_test - test_pred) ** 2))
    test_metrics['RMSE'] = float(np.sqrt(test_metrics['MSE']))
    test_metrics['MAE'] = float(np.mean(np.abs(y_test - test_pred)))
    test_metrics['Mean_Bias'] = float(np.mean(y_test - test_pred))
    test_metrics['Std_Bias'] = float(np.std(y_test - test_pred))
    test_metrics['R2'] = float(r2_score(y_test, test_pred))
    print_evaluation_results(test_metrics, 'Test')
    
    # 9. 保存结果
    print("\n" + "=" * 60)
    print("Step 9: 保存结果")
    print("=" * 60)
    
    # 保存带有归一化参数的模型
    model_checkpoint = {
        'model_state_dict': model.state_dict(),
        'normalization': {
            'mean': brightness_mean,
            'std': brightness_std,
            'min': brightness_min,
            'max': brightness_max,
            'method': 'zscore',
            'use_log1p': use_log1p
        },
        'config': {
            'input_dim': train_embeddings.shape[1],
            'hidden_dims': hidden_dims,
            'dropout': 0.3
        }
    }
    model_path_full = os.path.join(output_dir, 'brightness_model.pth')
    torch.save(model_checkpoint, model_path_full)
    print(f"模型（含归一化参数）已保存到: {model_path_full}")
    
    # 保存评估指标
    results = {
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'data_info': data_info,
        'training_config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'hidden_dims': hidden_dims,
            'early_stopping_patience': early_stopping_patience,
            'random_seed': random_seed
        },
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"评估结果已保存到: {results_path}")
    
    # 保存预测结果
    predictions_df = pd.DataFrame({
        'sequence': X_test,
        'actual_brightness': y_test,
        'predicted_brightness': test_pred,
        'residual': y_test - test_pred
    })
    predictions_path = os.path.join(output_dir, 'test_predictions.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"测试集预测结果已保存到: {predictions_path}")
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    
    return model, results


# =============================================================================
# 命令行入口
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='训练亮度预测模型')
    
    parser.add_argument('--data', type=str, default=None,
                        help='GFP数据Excel文件路径')
    parser.add_argument('--output', type=str, default='output/brightness_model',
                        help='输出目录')
    parser.add_argument('--embed_cache', type=str, default=None,
                        help='嵌入缓存文件路径')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--epochs', type=int, default=100,
                        help='最大训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--hidden_dims', type=str, default='512,128',
                        help='MLP隐藏层维度，用逗号分隔')
    parser.add_argument('--patience', type=int, default=15,
                        help='早停耐心值')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--sample', type=int, default=None,
                        help='随机采样的样本数量 (默认: 全部使用)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从checkpoint恢复训练的路径（迁移学习预训练模型）')
    parser.add_argument('--freeze', action='store_true',
                        help='冻结底层隐藏层，仅微调顶层（迁移学习时使用）。推荐用于微调阶段，防止过拟合')
    parser.add_argument('--use_pretrain_norm', action='store_true',
                        help='使用预训练模型的归一化参数（mean/std）。用于迁移学习时保持亮度尺度一致，避免因数据分布不同导致的预测偏差')
    parser.add_argument('--use_log1p', action='store_true',
                        help='对目标亮度值进行log1p对数变换。用于长尾分布数据，减少极端值影响，使分布更接近正态')
    parser.add_argument('--huber_delta', type=float, default=1.0,
                        help='Huber损失的delta参数（默认1.0，Z-score空间中约68%数据在内部）')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # 默认数据路径
    if args.data is None:
        # 尝试多个可能的数据路径
        possible_paths = [
            'GFP_data.xlsx',
            'data/GFP_data.xlsx',
            '../GFP_data.xlsx',
            '../data/GFP_data.xlsx'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                args.data = path
                break
        
        if args.data is None:
            print("错误: 未找到GFP数据文件")
            print("请使用 --data 参数指定数据路径")
            sys.exit(1)
    
    # 解析隐藏层维度
    hidden_dims = [int(x) for x in args.hidden_dims.split(',')]
    
    # 获取预训练归一化参数
    pretrain_normalization = None
    if args.resume and args.use_pretrain_norm and os.path.exists(args.resume):
        try:
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'normalization' in checkpoint:
                pretrain_normalization = checkpoint['normalization']
                print(f"使用预训练归一化参数: {pretrain_normalization}")
        except Exception as e:
            print(f"加载预训练归一化参数失败: {e}")
    
    # 运行训练
    train_brightness_model(
        data_path=args.data,
        output_dir=args.output,
        embed_cache_path=args.embed_cache,
        test_size=args.test_size,
        val_size=args.val_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dims=hidden_dims,
        early_stopping_patience=args.patience,
        random_seed=args.seed,
        sample_size=args.sample,
        resume_from=args.resume,
        freeze_layers=args.freeze,
        pretrain_normalization=pretrain_normalization,
        use_pretrain_norm=args.use_pretrain_norm,
        use_log1p=args.use_log1p,
        huber_delta=args.huber_delta
    )
