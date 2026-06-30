"""
SC2026 蛋白设计赛道 - 数据加载与预处理模块
===================================
本模块负责加载和预处理GFP亮度数据。

功能:
1. 从Excel文件加载GFP突变数据
2. 解析突变字符串并构建完整序列
3. 数据清洗和归一化
4. 训练/验证集划分
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import os

from seq_builder import SeqBuilder, WT_SEQUENCES
from config import DataPreprocessingConfig


# =============================================================================
# GFP数据加载器
# =============================================================================

class GFPDataLoader:
    """
    GFP数据加载器

    从Excel文件加载GFP突变-亮度数据

    期望的Excel格式:
    - Sheet 'brightness': 包含突变数据和亮度值
    - 列: seq_id, sequence, aaMutations, brightness, gfp_type等

    数据预处理步骤:
    1. 解析aaMutations列，应用突变构建完整序列
    2. log1p变换亮度值
    3. min-max归一化
    4. 去除极低亮度样本
    5. 训练/验证划分
    """

    def __init__(
        self,
        config: Optional[DataPreprocessingConfig] = None
    ):
        self.config = config or DataPreprocessingConfig()
        self.seq_builder = SeqBuilder()

    def load_from_excel(
        self,
        excel_path: str,
        sheet_name: str = 'brightness',
        auto_convert: bool = True
    ) -> pd.DataFrame:
        """
        从Excel文件加载数据，自动识别并转换不同的数据格式

        支持的数据格式:
        1. 标准格式: ['aaMutations', 'GFP type', 'Brightness'] - 用冒号':'分隔多点突变
        2. 实验格式: ['样品名', 'Fold-xxx Flu/OD', 'aa_changes'] - 用分号';'分隔多点突变，自动识别亮度列

        参数:
            excel_path: Excel文件路径
            sheet_name: 工作表名称，默认为'brightness'
            auto_convert: 是否自动转换列名

        返回:
            标准化后的DataFrame（列名: aaMutations, gfp_type, Brightness）
        """
        if not os.path.exists(excel_path):
            raise FileNotFoundError(f"文件不存在: {excel_path}")

        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            print(f"从 {excel_path} 加载了 {len(df)} 条数据")

            if auto_convert:
                df = self._auto_convert_format(df)

            return df
        except Exception as e:
            print(f"加载失败: {e}")
            raise

    def _auto_convert_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        自动检测并转换不同的数据格式为标准格式

        标准格式列:
        - aaMutations: 突变字符串
        - gfp_type: GFP类型（默认'sfGFP'）
        - Brightness: 亮度值
        """
        df = df.copy()
        original_cols = df.columns.tolist()

        # 统一GFP类型列名
        if 'GFP type' in df.columns and 'gfp_type' not in df.columns:
            df = df.rename(columns={'GFP type': 'gfp_type'})

        return df

    def _validate_mutations(
        self,
        mutation_str: str,
        gfp_type: str
    ) -> Tuple[bool, List[str]]:
        """
        验证突变字符串是否与WT序列一致

        参数:
            mutation_str: 突变字符串，如 "A109D:N145D"
            gfp_type: GFP类型

        返回:
            (是否有效, 错误信息列表)
        """
        errors = []
        gfp_type = gfp_type if gfp_type in WT_SEQUENCES else 'sfGFP'
        wt_seq = WT_SEQUENCES[gfp_type]

        try:
            mutations = self.seq_builder.parser.parse(mutation_str)
            for pos, wt_aa, _ in mutations:
                if pos >= len(wt_seq):
                    errors.append(f"位置{pos+1}超出序列长度{len(wt_seq)}")
                elif wt_seq[pos] != wt_aa:
                    errors.append(f"位置{pos+1}预期WT为{wt_aa}，实际为{wt_seq[pos]}")
        except Exception as e:
            errors.append(f"突变解析失败: {e}")

        return len(errors) == 0, errors

    def parse_mutations_and_build_sequences(
        self,
        df: pd.DataFrame,
        mutation_col: str = 'aaMutations',
        gfp_type_col: str = 'gfp_type',
        strict_validation: bool = False
    ) -> pd.DataFrame:
        """
        解析突变列，构建完整序列

        参数:
            df: 包含突变数据的DataFrame
            mutation_col: 突变列名
            gfp_type_col: GFP类型列名
            strict_validation: 是否启用严格验证（True时验证失败将跳过该样本）

        返回:
            添加了sequence列的DataFrame（如果strict_validation=True，
            验证失败的样本将被标记在mutation_valid列中）
        """
        sequences = []
        build_errors = 0
        validation_warnings = 0
        mutation_valid_flags = []

        for idx, row in df.iterrows():
            try:
                gfp_type = row.get(gfp_type_col, 'sfGFP')
                mutations = row.get(mutation_col, '')

                if pd.isna(mutations) or mutations == 'WT':
                    # WT序列
                    if gfp_type in WT_SEQUENCES:
                        seq = WT_SEQUENCES[gfp_type]
                    else:
                        seq = WT_SEQUENCES['sfGFP']
                    mutation_valid_flags.append(True)  # WT不算突变，默认有效
                else:
                    # 先验证突变是否与WT一致
                    is_valid, errors = self._validate_mutations(str(mutations), gfp_type)

                    if not is_valid:
                        validation_warnings += 1
                        print(f"  [警告] 行{idx} 突变验证失败: {errors[0]}")
                        if strict_validation:
                            # 严格模式：跳过验证失败的样本
                            sequences.append(None)
                            mutation_valid_flags.append(False)
                            continue

                    # 构建突变体（严格验证时validate=True，否则validate=False）
                    seq = self.seq_builder.build_from_mutations(
                        gfp_type=gfp_type,
                        mutation_str=str(mutations),
                        validate=strict_validation
                    )
                    mutation_valid_flags.append(is_valid)

                sequences.append(seq)

            except Exception as e:
                # 如果构建失败，尝试使用已有序列
                if 'sequence' in row and not pd.isna(row['sequence']):
                    sequences.append(row['sequence'])
                    mutation_valid_flags.append(False)  # 回退使用的序列
                else:
                    sequences.append(None)
                    mutation_valid_flags.append(False)
                    build_errors += 1

        df['sequence'] = sequences
        df['mutation_valid'] = mutation_valid_flags

        if build_errors > 0:
            print(f"警告: {build_errors} 条序列构建失败")
        if validation_warnings > 0:
            print(f"警告: {validation_warnings} 条突变记录与WT不一致（已启用宽松模式构建）")

        return df

    def preprocess(
        self,
        df: pd.DataFrame,
        brightness_col: str = 'brightness'
    ) -> pd.DataFrame:
        """
        数据预处理（基础清洗，不包含归一化）

        注意：亮度归一化应在训练流程中统一处理，
        确保与train_brightness_model.py中的逻辑一致。

        步骤:
        1. 去除空值
        2. 过滤极低亮度
        3. 计算突变数量

        参数:
            df: DataFrame
            brightness_col: 亮度列名

        返回:
            预处理后的DataFrame
        """
        # 复制
        df = df.copy()

        # 1. 去除空值
        df = df.dropna(subset=[brightness_col, 'sequence'])
        df = df[df['sequence'].notna()]

        # 2. 过滤极低亮度
        min_threshold = self.config.min_brightness_threshold
        df = df[df[brightness_col] >= min_threshold]

        # 3. 计算突变数量
        def count_mutations(row):
            gfp_type = row.get('gfp_type', 'sfGFP')
            if gfp_type not in WT_SEQUENCES:
                gfp_type = 'sfGFP'
            wt_seq = WT_SEQUENCES[gfp_type]
            seq = row['sequence']
            if len(seq) != len(wt_seq):
                return -1  # 标记长度不一致
            return sum(a != b for a, b in zip(wt_seq, seq))

        df['n_mutations'] = df.apply(count_mutations, axis=1)

        print(f"预处理完成: {len(df)} 条数据")
        print(f"  亮度范围: [{df[brightness_col].min():.4f}, {df[brightness_col].max():.4f}]")

        return df

    def train_val_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        stratify_by: Optional[str] = 'n_mutations'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        训练/验证集划分

        参数:
            df: DataFrame
            test_size: 验证集比例
            stratify_by: 分层键 ('n_mutations' 或 'gfp_type')

        返回:
            (train_df, val_df)
        """
        from sklearn.model_selection import train_test_split

        if stratify_by and stratify_by in df.columns:
            stratify_col = df[stratify_by]
        else:
            stratify_col = None

        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=stratify_col
        )

        print(f"划分完成:")
        print(f"  训练集: {len(train_df)} 条")
        print(f"  验证集: {len(val_df)} 条")

        return train_df, val_df

    def load_and_preprocess(
        self,
        excel_path: str,
        sheet_name: str = 'brightness'
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        完整加载和预处理流程

        参数:
            excel_path: Excel文件路径
            sheet_name: 工作表名称

        返回:
            (train_df, val_df)
        """
        # 加载
        df = self.load_from_excel(excel_path, sheet_name)

        # 构建序列
        df = self.parse_mutations_and_build_sequences(df)

        # 预处理
        df = self.preprocess(df)

        # 划分
        train_df, val_df = self.train_val_split(
            df,
            test_size=1 - self.config.train_split,
            stratify_by=self.config.stratify_by
        )

        return train_df, val_df


# =============================================================================
# 数据集导出
# =============================================================================

class DatasetExporter:
    """
    数据集导出器

    将处理后的数据导出为:
    1. CSV格式 (train.csv, val.csv)
    2. HDF5格式 (嵌入向量)
    """

    @staticmethod
    def export_to_csv(
        df: pd.DataFrame,
        output_path: str,
        columns: Optional[List[str]] = None
    ):
        """
        导出为CSV

        参数:
            df: DataFrame
            output_path: 输出路径
            columns: 要导出的列
        """
        if columns:
            df = df[columns]

        df.to_csv(output_path, index=False)
        print(f"已导出到: {output_path}")

    @staticmethod
    def export_fasta(
        sequences: List[str],
        ids: List[str],
        output_path: str
    ):
        """
        导出为FASTA格式

        参数:
            sequences: 序列列表
            ids: ID列表
            output_path: 输出路径
        """
        with open(output_path, 'w') as f:
            for seq_id, seq in zip(ids, sequences):
                f.write(f">{seq_id}\n")
                # 每行60个氨基酸
                for i in range(0, len(seq), 60):
                    f.write(f"{seq[i:i+60]}\n")

        print(f"已导出 {len(sequences)} 条序列到: {output_path}")


# =============================================================================
# 便捷函数
# =============================================================================

def load_gfp_data(
    excel_path: str,
    output_dir: str,
    sheet_name: str = 'brightness'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    便捷函数：加载和预处理GFP数据

    参数:
        excel_path: Excel文件路径
        output_dir: 输出目录
        sheet_name: 工作表名称

    返回:
        (train_df, val_df)
    """
    os.makedirs(output_dir, exist_ok=True)

    loader = GFPDataLoader()
    train_df, val_df = loader.load_and_preprocess(excel_path, sheet_name)

    # 导出
    exporter = DatasetExporter()
    exporter.export_to_csv(
        train_df,
        os.path.join(output_dir, "train.csv")
    )
    exporter.export_to_csv(
        val_df,
        os.path.join(output_dir, "val.csv")
    )

    # 导出FASTA
    exporter.export_fasta(
        sequences=train_df['sequence'].tolist(),
        ids=train_df['seq_id'].tolist() if 'seq_id' in train_df.columns
             else [f"train_{i}" for i in range(len(train_df))],
        output_path=os.path.join(output_dir, "train.fasta")
    )

    return train_df, val_df
