"""
SC2026 蛋白设计赛道 - 配置文件
===================================
本文件包含所有可调参数的定义。

主要参数类别:
1. 数据预处理参数 (Data Preprocessing)
2. 嵌入提取参数 (Embedding Extraction)
3. 亮度预测模型参数 (Brightness Prediction Model)
4. 序列生成参数 (Sequence Generation)
5. 筛选参数 (Screening)
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


# =============================================================================
# 全局常量
# =============================================================================

STANDARD_AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(STANDARD_AA)}
IDX_TO_AA = {i: aa for i, aa in enumerate(STANDARD_AA)}


# =============================================================================
# 0. 路径配置
# =============================================================================

@dataclass
class PathConfig:
    """项目路径配置"""
    # 根目录（基于本文件位置自动推断项目根）
    root_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 数据目录
    data_dir: str = os.path.join(root_dir, "data")
    processed_dir: str = os.path.join(data_dir, "processed")

    # GFP数据文件
    gfp_data_file: str = os.path.join(data_dir, "GFP_data.xlsx")
    aa_seqs_file: str = os.path.join(data_dir, "AAseqs of 5 GFP proteins.txt")
    mutation起始_file: str = os.path.join(data_dir, "SC突变起始序列.txt")
    # 排除列表文件: 优先检查根目录，然后检查data目录
    exclusion_list_file: str = os.path.join(root_dir, "Exclusion_List.csv")
    submission_template_file: str = os.path.join(data_dir, "submission_template.csv")

    # 模型输出目录
    models_dir: str = os.path.join(root_dir, "src", "models")
    output_dir: str = os.path.join(root_dir, "output")
    logs_dir: str = os.path.join(root_dir, "logs")

    # 预训练模型缓存目录
    model_cache_dir: str = os.path.join(root_dir, "model_cache")


# =============================================================================
# 1. 数据预处理参数
# =============================================================================

@dataclass
class DataPreprocessingConfig:
    """
    数据预处理参数

    生物学背景:
    - GFP亮度数据通常呈现右偏分布(少数高亮度变体)
    - log变换可以使数据更接近正态分布
    - min-max归一化将数据缩放到[0,1]便于模型训练
    """
    # 亮度变换方式: 'log1p' | 'log10' | 'none'
    # log1p = log(1 + x), 适合处理包含0的值
    log_transform: str = 'log1p'

    # 归一化方式: 'minmax' | 'standard' | 'robust'
    # minmax: (x - min) / (max - min)
    # standard: (x - mean) / std
    # robust: 使用中位数和IQR，适合异常值较多的情况
    normalization: str = 'minmax'

    # 最小亮度阈值，过滤极低亮度噪音样本
    min_brightness_threshold: float = 0.01

    # 训练集比例
    train_split: float = 0.80

    # 分层划分键: 'n_mutations' | 'gfp_type'
    # 确保高阶突变在训练集和验证集都有分布
    stratify_by: str = 'n_mutations'


# =============================================================================
# 2. 嵌入提取参数
# =============================================================================

@dataclass
class EmbeddingConfig:
    """
    嵌入提取参数

    ESM-2 (Evolutionary Scale Modeling):
    - 基于Transformer的蛋白质语言模型
    - 650M参数版本在性能和计算成本间取得平衡
    - Mean pooling对序列长度变化具有鲁棒性
    """
    # ESM-2模型选择: 'esm2_t30_150M' | 'esm2_t33_650M' | 'esm2_t36_3B'
    esm_model_name: str = 'esm2_t33_650M_UR50D'

    # 提取的层（用于多层特征融合）
    esm_layers: List[int] = field(default_factory=lambda: [23, 28, 33])

    # 嵌入维度 (根据模型和层数动态计算)
    esm_embedding_dim: int = 3840

    # 池化方式: 'mean'
    pooling: str = 'mean'

    # GPU批次大小，根据显存调整
    batch_size: int = 4

    # 最大序列长度
    max_seq_len: int = 260

    def __post_init__(self):
        """根据模型名称和层数自动计算嵌入维度"""
        base_dims = {
            'esm2_t30_150M_UR50D': 768,
            'esm2_t33_650M_UR50D': 1280,
            'esm2_t36_3B_UR50D': 2560,
        }
        base_dim = base_dims.get(self.esm_model_name, 1280)
        self.esm_embedding_dim = base_dim * len(self.esm_layers)


# =============================================================================
# 3. 亮度预测模型参数
# =============================================================================

@dataclass
class BrightnessModelConfig:
    """
    亮度预测模型参数

    模型架构: ESM-2嵌入 → MLP → 亮度预测

    MLP结构:
    - 输入层: 嵌入维度
    - 隐藏层: [512, 128]
    - 输出层: 1 (标量亮度值)
    - 激活函数: GELU
    - Dropout: 0.3 (防止过拟合)
    """
    # MLP隐藏层维度
    hidden_dims: List[int] = field(default_factory=lambda: [512, 128])

    # Dropout概率
    dropout: float = 0.30

    # 优化器参数
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5

    # 训练参数
    batch_size: int = 64
    epochs: int = 50
    early_stopping_patience: int = 10


# =============================================================================
# 4. 序列生成参数
# =============================================================================

@dataclass
class SequenceGenerationConfig:
    """
    序列生成参数

    生成路径:
    A. RL强化学习生成: 使用双头PPO + 优先经验回放

    候选池目标: ≥ 10,000 条序列以确保多样性
    """
    # --- RL生成参数 ---
    rl_episodes: int = 1000  # RL训练轮数
    rl_max_mutations: int = 5  # 最大突变数

    # --- 公共参数 ---
    total_candidate_pool_target: int = 10000  # 候选池目标大小


# =============================================================================
# 6. 筛选参数
# =============================================================================

@dataclass
class ScreeningConfig:
    """
    筛选参数

    三级筛选漏斗:
    Stage 1: 硬约束过滤 - 长度、起始密码子、标准AA、排除列表
    Stage 2: 亮度预测 - B̂ > 0.5
    Stage 3: 帕累托前沿 + 多样性 - Top 6

    序列约束:
    - 长度: 220-250 aa
    - 起始: M
    """
    # Stage 1: 硬约束
    min_length: int = 220
    max_length: int = 250
    valid_aa_set: str = 'ACDEFGHIKLMNPQRSTVWY'

    # Stage 2: 预测阈值
    B_threshold: float = 0.5  # 亮度预测阈值（归一化到[0,1]后的值，基于训练数据min-max）

    # Stage 3: 帕累托筛选
    diversity_max_similarity: float = 0.90  # 最大序列相似度
    n_final: int = 6  # 最终提交序列数


# =============================================================================
# 完整配置类
# =============================================================================

@dataclass
class SC2026Config:
    """完整配置类，整合所有参数模块"""
    paths: PathConfig = field(default_factory=PathConfig)
    data_preprocessing: DataPreprocessingConfig = field(default_factory=DataPreprocessingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    brightness_model: BrightnessModelConfig = field(default_factory=BrightnessModelConfig)
    sequence_generation: SequenceGenerationConfig = field(default_factory=SequenceGenerationConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str):
        """从YAML文件加载配置"""
        # TODO: 实现YAML加载
        pass

    def to_yaml(self, yaml_path: str):
        """保存配置到YAML文件"""
        # TODO: 实现YAML保存
        pass


# =============================================================================
# 默认配置实例
# =============================================================================

# 全局默认配置
DEFAULT_CONFIG = SC2026Config()


# =============================================================================
# WT序列定义
# =============================================================================

WT_SEQUENCES = {
    'avGFP': 'MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
    'sfGFP': 'MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNIVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
    'amacGFP': 'MSKGEELFTGIVPVLIELDGDVHGHKFSVRGEGEGDADYGKLEIKFICTTGKLPVPWPTLVTTLSYGILCFARYPEHMKMNDFFKSAMPEGYIQERTIFFQDDGKYKTRGEVKFEGDTLVNRIELKGMDFKEDGNILGHKLEYNFNSHNVYIMPDKANNGLKVNFKIRHNIEGGGVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK',
    'cgreGFP': 'MTALTEGAKLFEKEIPYITELEGDVEGMKFIIKGEGTGDATTGTIKAKYICTTGDLPVPWATILSSLSYGVFCFAKYPRHIADFFKSTQPDGYSQDRIISFDNDGQYDVKAKVTYENGTLYNRVTVKGTGFKSNGNILGMRVLYHSPPHAVYILPDRKNGGMKIEYNKAFDVMGGGHQMARHAQFNKPLGAWEEDYPLYHHLTVWTSFGKDPDDDETDHLTIVEVIKAVDLETYR',
    'ppluGFP': 'MPAMKIECRITGTLNGVEFELVGGGEGTPEQGRMTNKMKSTKGALTFSPYLLSHVMGYGFYHFGTYPSGYENPFLHAINNGGYTNTRIEKYEDGGVLHVSFSYRYEAGRVIGDFKVVGTGFPEDSVIFTDKIIRSNATVEHLHPMGDNVLVGSFARTFSLRDGGYYSFVVDSHMHFKSAIHPSILQNGGPMFAFRRVEELHSNTELGIVEYQHAFKTPIAFA',
}


# =============================================================================
# 工具函数
# =============================================================================


def format_time(seconds: float) -> str:
    """
    将秒数格式化为可读的时间字符串

    参数:
        seconds: 时间（秒）

    返回:
        格式化的时间字符串，如 "2小时30分15秒"
    """
    if seconds < 60:
        return f"{int(seconds)}秒"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}分{secs}秒"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}小时{minutes}分{secs}秒"

