"""
SC2026 蛋白设计赛道 - 亮度预测模型
===================================
本模块实现基于ESM-2嵌入的亮度预测

模型架构:
- ESM-2 (Evolutionary Scale Modeling) 蛋白质语言模型
- 输入: 蛋白质序列 → ESM-2嵌入 [seq_len, 1280]
- Mean Pooling: 汇聚为固定维度向量 [1280]
- MLP预测头: [1280] → [512] → [128] → [1]
- 输出: 归一化亮度预测值 [0, 1]

训练损失: HuberLoss（对异常值鲁棒，平衡MSE和MAE）
优化器: AdamW
学习率: 1e-4
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np
import os

try:
    import scipy.stats as stats
except ImportError:
    stats = None

from config import format_time


# =============================================================================
# MLP预测头
# =============================================================================

class MLPHead(nn.Module):
    """
    多层感知机预测头

    架构: Input → Linear → GELU → Dropout → Linear → ... → Output

    GELU激活函数:
    - GELU(x) = x * Φ(x)，其中Φ是标准正态分布的CDF
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        dropout: float = 0.3
    ):
        """
        初始化MLP头

        参数:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度
            dropout: Dropout概率
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        # 输入端LayerNorm：稳定来自大模型的高维嵌入向量
        layers.append(nn.LayerNorm(input_dim))

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.mlp(x)


# =============================================================================
# Attention Pooling (替代Mean Pooling，保留位置信息)
# =============================================================================

class AttentionPooling(nn.Module):
    """
    注意力加权池化

    为每个序列位置学习一个权重，让模型决定哪些位置对预测更重要。
    相比Mean Pooling，可以保留关键位置的信息。
    """

    def __init__(self, dim: int = 1280):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: [batch, seq_len, dim]
        返回:
            [batch, dim]
        """
        w = torch.softmax(self.attn(x), dim=1)  # [batch, seq_len, 1]
        return (w * x).sum(dim=1)               # [batch, dim]


# =============================================================================
# ESM-2 亮度预测模型
# =============================================================================

class BrightnessModelESM(nn.Module):
    """
    基于ESM-2的亮度预测模型

    输入: ESM-2嵌入向量 [batch, 1280] (已池化) 或 [batch, seq_len, 1280] (未池化)
    输出: 预测亮度 [batch]

    模型流程:
    1. ESM-2编码器生成序列嵌入
    2. (可选) Attention Pooling / Mean Pooling汇聚为固定维度
    3. MLP预测头输出亮度值
    """

    def __init__(
        self,
        input_dim: int = 3840,
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        use_attention_pooling: bool = False
    ):
        """
        初始化亮度模型

        参数:
            input_dim: 嵌入维度 (ESM-2 650M 3层特征融合 = 1280 * 3 = 3840)
            hidden_dims: MLP隐藏层维度
            dropout: Dropout概率
            use_attention_pooling: 是否使用Attention Pooling（需要输入未池化的token级嵌入）
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 128]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_attention_pooling = use_attention_pooling

        # 可选的Attention Pooling
        if use_attention_pooling:
            self.attention_pool = AttentionPooling(dim=input_dim)

        # MLP预测头
        self.mlp = MLPHead(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: ESM-2嵌入 [batch, input_dim] 或 [batch, seq_len, input_dim]

        返回:
            亮度预测值 [batch]
        """
        # 如果使用Attention Pooling且输入是3D的 [batch, seq_len, dim]
        if self.use_attention_pooling and x.dim() == 3:
            x = self.attention_pool(x)  # [batch, dim]
        # 如果输入是3D但未开启attention pooling，做mean pooling
        elif x.dim() == 3:
            x = x.mean(dim=1)  # [batch, dim]

        # MLP前向传播（线性输出，无激活函数）
        # 注意：Z-score归一化后目标值范围无界，因此输出层不使用Sigmoid
        output = self.mlp(x).squeeze(-1)  # [batch]

        return output

    def predict(
        self,
        embeddings: np.ndarray,
        normalization: Optional[Dict] = None,
        return_raw: bool = False
    ) -> np.ndarray:
        """
        使用训练好的模型进行预测

        参数:
            embeddings: ESM-2嵌入数组 [batch, input_dim]
            normalization: 归一化参数字典（用于反归一化）
                - method: 归一化方法 ('zscore' 或 'minmax')
                - mean, std: Z-score参数
                - min, max: Min-Max参数
                - use_log1p: 是否使用了log1p变换
            return_raw: 是否返回原始网络输出（Z-score/对数空间），True时忽略normalization

        返回:
            预测亮度值数组 [batch]
            - return_raw=True: 返回网络原始输出（Z-score/对数空间）
            - return_raw=False: 返回反归一化后的原始亮度值
        """
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            x = torch.tensor(embeddings, dtype=torch.float32).to(device)
            predictions = self.forward(x).cpu().numpy()
        
        if return_raw or normalization is None:
            return predictions
        
        # 反归一化：从网络输出空间还原到原始亮度空间
        method = normalization.get('method', 'zscore')
        use_log1p = normalization.get('use_log1p', False)
        
        # 第一步：反Z-score归一化
        if method == 'zscore':
            mean = normalization['mean']
            std = normalization['std']
            predictions = predictions * std + mean
        else:  # minmax
            dmin = normalization['min']
            dmax = normalization['max']
            predictions = predictions * (dmax - dmin) + dmin
        
        # 第二步：反log1p变换（如果使用了）
        if use_log1p:
            predictions = np.expm1(predictions)
            # 确保非负
            predictions = np.maximum(predictions, 0.0)
        
        return predictions


# =============================================================================
# 模型训练器
# =============================================================================

class BrightnessTrainer:
    """
    亮度模型训练器

    功能:
    - 批量训练
    - 验证集评估
    - Early Stopping
    - 学习率调度
    """

    def __init__(
        self,
        model: BrightnessModelESM,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        huber_delta: float = 1.0,
        monitor_metric: str = 'val_r2'
    ):
        """
        初始化训练器

        参数:
            model: 亮度模型
            learning_rate: 学习率
            weight_decay: 权重衰减
            device: 计算设备
            huber_delta: Huber损失的delta参数（默认1.0，Z-score空间中约68%数据在内部）
            monitor_metric: 早停监控指标 ('val_r2' 或 'val_loss')
        """
        self.model = model
        self.device = device
        model.to(device)
        self.monitor_metric = monitor_metric

        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 损失函数: Huber Loss（兼顾MSE的平滑性和MAE的鲁棒性）
        # delta=1.0在Z-score空间中：残差<1时像MSE，残差>1时像MAE
        self.criterion = nn.HuberLoss(delta=huber_delta)

        # 学习率调度器（根据监控指标决定mode）
        scheduler_mode = 'max' if monitor_metric == 'val_r2' else 'min'
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=scheduler_mode,
            factor=0.5,
            patience=5,
            verbose=True
        )

        self.train_losses = []
        self.val_losses = []
        self.val_r2s = []

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader
    ) -> float:
        """
        训练一个epoch

        参数:
            train_loader: 训练数据加载器

        返回:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            embeddings, targets = batch
            embeddings = embeddings.to(self.device)
            targets = targets.to(self.device)

            # 前向传播
            self.optimizer.zero_grad()
            predictions = self.model(embeddings)

            # 计算损失
            loss = self.criterion(predictions, targets)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self, val_loader: torch.utils.data.DataLoader, normalization: Optional[Dict] = None) -> Dict[str, float]:
        """
        在验证集上评估模型
        
        参数:
            val_loader: 验证数据加载器
            normalization: 归一化参数，用于计算原始空间的R²
        
        返回:
            {'val_loss', 'val_r2'}
        """
        self.model.eval()
        
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                embeddings, targets = batch
                embeddings = embeddings.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(embeddings)
                loss = self.criterion(predictions, targets)

                total_loss += loss.item()
                n_batches += 1
                
                all_preds.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_loss = total_loss / n_batches
        self.val_losses.append(avg_loss)
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # 如果提供了归一化参数，计算原始空间的R²
        if normalization is not None:
            # 反Z-score
            preds_transformed = all_preds * normalization['std'] + normalization['mean']
            targets_transformed = all_targets * normalization['std'] + normalization['mean']
            
            # 反log1p（如果使用了）
            if normalization.get('use_log1p', False):
                preds_raw = np.expm1(preds_transformed)
                targets_raw = np.expm1(targets_transformed)
                preds_raw = np.maximum(preds_raw, 0.0)
                targets_raw = np.maximum(targets_raw, 0.0)
            else:
                preds_raw = preds_transformed
                targets_raw = targets_transformed
            
            # 在原始空间计算R²
            ss_res = np.sum((targets_raw - preds_raw) ** 2)
            ss_tot = np.sum((targets_raw - np.mean(targets_raw)) ** 2)
            val_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        else:
            # 在变换后空间计算R²（与训练目标一致）
            ss_res = np.sum((all_targets - all_preds) ** 2)
            ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
            val_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        self.val_r2s.append(val_r2)
        
        return {'val_loss': avg_loss, 'val_r2': val_r2}

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 50,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None,
        resume_from: Optional[str] = None,
        normalization: Optional[Dict] = None,
        model_config: Optional[Dict] = None
    ) -> Dict:
        """
        完整训练流程

        参数:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 最大epoch数
            early_stopping_patience: 早停patience
            save_path: 模型保存路径
            resume_from: 从checkpoint恢复训练的路径
            normalization: 归一化参数 {'min': float, 'max': float}
            model_config: 模型配置 {'input_dim': int, 'hidden_dims': list}

        返回:
            训练历史字典
        """
        import time
        
        start_epoch = 0
        # 根据监控指标设置最佳值初始状态
        if self.monitor_metric == 'val_r2':
            best_metric = float('-inf')
        else:
            best_metric = float('inf')
        patience_counter = 0
        
        # 从checkpoint恢复
        if resume_from and os.path.exists(resume_from):
            print(f"从checkpoint恢复: {resume_from}")
            checkpoint = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            # 恢复最佳指标值
            if self.monitor_metric == 'val_r2':
                best_metric = checkpoint.get('val_r2', float('-inf'))
            else:
                best_metric = checkpoint.get('val_loss', float('inf'))
            # 恢复归一化参数（如果存在）
            if 'normalization' in checkpoint:
                normalization = checkpoint['normalization']
            if 'config' in checkpoint:
                model_config = checkpoint['config']
            print(f"  从epoch {start_epoch} 继续训练")
        
        # 进度计时
        training_start_time = time.time()
        last_report_time = training_start_time
        last_report_epoch = start_epoch
        
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(train_loader)

            # 验证（返回字典：{'val_loss', 'val_r2'}）
            val_results = self.validate(val_loader, normalization)
            val_loss = val_results['val_loss']
            val_r2 = val_results['val_r2']
            
            # 计算Pearson相关系数
            pearson_r = None
            if stats is not None:
                all_preds, all_targets = [], []
                self.model.eval()
                with torch.no_grad():
                    for emb, tgt in val_loader:
                        pred = self.model(emb.to(self.device)).cpu().numpy()
                        all_preds.extend(pred)
                        all_targets.extend(tgt.numpy())
                pearson_r, _ = stats.pearsonr(all_targets, all_preds)
            
            # 学习率调度（根据监控指标传入对应值）
            if self.monitor_metric == 'val_r2':
                self.scheduler.step(val_r2)
            else:
                self.scheduler.step(val_loss)
            
            # 进度计时和报告
            current_time = time.time()
            elapsed = current_time - training_start_time
            epochs_done = epoch + 1 - last_report_epoch
            time_since_last = current_time - last_report_time
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val R²: {val_r2:.4f}")
            if pearson_r is not None:
                print(f"  Val Pearson r: {pearson_r:.4f}")
            
            # 每5个epoch或训练结束时报告进度
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                if epochs_done > 0 and time_since_last > 0:
                    speed = epochs_done / time_since_last  # epochs/second
                    remaining_epochs = epochs - (epoch + 1)
                    remaining_time = remaining_epochs / speed if speed > 0 else 0
                    
                    print(f"  进度: {(epoch+1)/epochs*100:.1f}% | "
                          f"已用: {format_time(elapsed)} | "
                          f"剩余: ~{format_time(remaining_time)} | "
                          f"速度: {speed:.2f} epoch/秒")
                    
                    last_report_time = current_time
                    last_report_epoch = epoch + 1

            # 早停检查（根据监控指标判断是否改善）
            current_metric = val_r2 if self.monitor_metric == 'val_r2' else val_loss
            is_better = (val_r2 > best_metric) if self.monitor_metric == 'val_r2' else (val_loss < best_metric)
            
            if is_better:
                best_metric = current_metric
                patience_counter = 0

                # 保存最佳模型（包含归一化参数）
                if save_path:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'val_loss': val_loss,
                        'val_r2': val_r2,
                    }
                    # 添加归一化参数（如果提供）
                    if normalization is not None:
                        checkpoint['normalization'] = normalization
                    if model_config is not None:
                        checkpoint['config'] = model_config
                    
                    torch.save(checkpoint, save_path)
                    print(f"  Model saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 训练完成总时间
        total_time = time.time() - training_start_time
        
        print(f"\n训练完成！总用时: {format_time(total_time)}")
        
        # 收集最佳指标
        if self.monitor_metric == 'val_r2':
            best_val_loss = min(self.val_losses) if self.val_losses else float('inf')
        else:
            best_val_loss = best_metric
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': best_val_loss
        }


# =============================================================================
# 数据集类
# =============================================================================

class BrightnessDataset(torch.utils.data.Dataset):
    """
    亮度预测数据集

    从HDF5文件加载预计算的ESM-2嵌入
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        brightness: np.ndarray
    ):
        """
        初始化数据集

        参数:
            embeddings: ESM-2嵌入数组 [N, 1280]
            brightness: 归一化亮度值数组 [N]
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.brightness = torch.tensor(brightness, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.brightness)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.brightness[idx]


# =============================================================================
# 工具函数
# =============================================================================

def load_brightness_model_from_checkpoint(
    model_path: str,
    device: str = 'cpu'
) -> Tuple[BrightnessModelESM, Dict]:
    """
    从checkpoint加载亮度模型和归一化参数

    参数:
        model_path: 模型checkpoint文件路径
        device: 计算设备 ('cpu' 或 'cuda')

    返回:
        (model, norm_info_dict):
            model: 加载好的BrightnessModelESM模型（eval模式）
            norm_info_dict: 归一化参数字典，包含:
                - method: 归一化方法 ('zscore' 或 'minmax')
                - mean: 均值 (zscore方法)
                - std: 标准差 (zscore方法)
                - min: 最小值 (minmax方法)
                - max: 最大值 (minmax方法)
                - use_log1p: 是否使用了log1p对数变换
    """
    checkpoint = torch.load(model_path, map_location=device)

    input_dim = checkpoint['config']['input_dim']
    hidden_dims = checkpoint['config'].get('hidden_dims', [512, 128])
    dropout = checkpoint['config'].get('dropout', 0.3)
    
    # 检查checkpoint是否为旧模型（没有输入端LayerNorm）
    # 旧模型: mlp.mlp.0 是 Linear，权重形状为 [hidden_dims[0], input_dim]
    # 新模型: mlp.mlp.0 是 LayerNorm，权重形状为 [input_dim]
    has_input_layernorm = False
    state_dict = checkpoint['model_state_dict']
    
    if 'mlp.mlp.0.weight' in state_dict:
        first_weight_shape = state_dict['mlp.mlp.0.weight'].shape
        if len(first_weight_shape) == 2 and first_weight_shape[0] == hidden_dims[0]:
            has_input_layernorm = False
        else:
            has_input_layernorm = True
    
    model = BrightnessModelESM(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    )
    
    # 如果是旧模型，需要调整state_dict的键名
    if not has_input_layernorm:
        print(f"  [加载] 检测到旧模型格式（无输入端LayerNorm），正在转换...")
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('mlp.mlp.'):
                parts = key.split('.')
                idx = int(parts[2])
                new_idx = idx + 1
                new_key = '.'.join(parts[:2] + [str(new_idx)] + parts[3:])
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        # 添加新的输入端LayerNorm参数（初始化为单位变换）
        # LayerNorm权重初始化为1，偏置初始化为0，这样不会改变原有计算结果
        new_state_dict['mlp.mlp.0.weight'] = torch.ones(input_dim)
        new_state_dict['mlp.mlp.0.bias'] = torch.zeros(input_dim)
        
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    norm_info_dict = {}
    normalization = checkpoint.get('normalization', {})
    method = normalization.get('method', 'zscore')
    norm_info_dict['method'] = method
    norm_info_dict['use_log1p'] = normalization.get('use_log1p', False)

    if method == 'zscore':
        norm_info_dict['mean'] = normalization['mean']
        norm_info_dict['std'] = normalization['std']
        norm_info_dict['min'] = normalization.get('min')
        norm_info_dict['max'] = normalization.get('max')
    else:
        norm_info_dict['min'] = normalization['min']
        norm_info_dict['max'] = normalization['max']
        norm_info_dict['mean'] = None
        norm_info_dict['std'] = None

    return model, norm_info_dict
