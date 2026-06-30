"""
SC2026 蛋白设计赛道 - 排除列表检查器
===================================
本模块负责检查序列是否在排除列表(Exclusion_List)中。
"""

import csv
import os
from typing import List, Set, Tuple


# =============================================================================
# 排除列表加载器
# =============================================================================

class ExclusionListChecker:
    """
    排除列表检查器

    功能:
    1. 从CSV文件加载排除列表
    2. 快速检查序列是否在排除列表中
    3. 支持批量检查

    数据结构:
    - 使用set存储排除序列，实现O(1)查找
    - 序列存储为字符串，大写形式

    使用示例:
    >>> checker = ExclusionListChecker()
    >>> checker.load_from_csv("Exclusion_List.csv")
    >>> is_excluded = checker.is_excluded("MSKGEELFT...")
    >>> excluded_seqs = checker.filter_excluded(["seq1", "seq2", "seq3"])
    """

    def __init__(self):
        self.exclusion_set: Set[str] = set()
        self.source_file: str = ""

    def load_from_csv(self, csv_path: str) -> int:
        """
        从CSV文件加载排除列表

        CSV格式要求:
        - 必须有'Sequence'列
        - 第一行是表头
        - 后续每行是一条序列

        参数:
            csv_path: CSV文件路径

        返回:
            加载的序列数量
        """
        if not os.path.exists(csv_path):
            # 优雅降级：排除列表缺失时不抛错，仅记录警告
            print(f"[WARNING] 排除列表文件不存在: {csv_path}（继续运行，禁用排除检查）")
            return 0

        self.exclusion_set.clear()
        self.source_file = csv_path

        loaded_count = 0

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'Sequence' not in row:
                    raise ValueError("CSV文件必须包含'Sequence'列")

                sequence = row['Sequence'].strip().upper()

                # 跳过空序列
                if sequence:
                    self.exclusion_set.add(sequence)
                    loaded_count += 1

        print(f"[排除列表] 从 {csv_path} 加载了 {loaded_count} 条排除序列")
        return loaded_count

    def load_from_set(self, sequences: List[str]) -> int:
        """
        从序列列表加载排除列表

        参数:
            sequences: 序列列表

        返回:
            加载的序列数量
        """
        self.exclusion_set.clear()
        for seq in sequences:
            seq_clean = seq.strip().upper()
            if seq_clean:
                self.exclusion_set.add(seq_clean)

        return len(self.exclusion_set)

    def is_excluded(self, sequence: str) -> bool:
        """
        检查序列是否在排除列表中

        参数:
            sequence: 待检查的氨基酸序列

        返回:
            True如果序列在排除列表中，False otherwise
        """
        # 标准化: 去除空白，转大写
        seq_normalized = sequence.strip().upper()
        return seq_normalized in self.exclusion_set

    def filter_excluded(self, sequences: List[str]) -> List[Tuple[str, bool]]:
        """
        批量检查序列是否在排除列表中

        参数:
            sequences: 序列列表

        返回:
            List of (sequence, is_excluded) tuples
        """
        return [(seq, self.is_excluded(seq)) for seq in sequences]

    def get_excluded_sequences(self, sequences: List[str]) -> List[str]:
        """
        获取列表中被排除的序列

        参数:
            sequences: 序列列表

       返回:
            被排除的序列列表
        """
        return [seq for seq in sequences if self.is_excluded(seq)]

    def get_allowed_sequences(self, sequences: List[str]) -> List[str]:
        """
        获取列表中未被排除的序列

        参数:
            sequences: 序列列表

        返回:
            未被排除的序列列表
        """
        return [seq for seq in sequences if not self.is_excluded(seq)]

    def add_sequence(self, sequence: str) -> None:
        """
        添加序列到排除列表

        参数:
            sequence: 要添加的序列
        """
        seq_normalized = sequence.strip().upper()
        self.exclusion_set.add(seq_normalized)

    def remove_sequence(self, sequence: str) -> bool:
        """
        从排除列表移除序列

        参数:
            sequence: 要移除的序列

        返回:
            True如果序列被移除，False如果序列不在列表中
        """
        seq_normalized = sequence.strip().upper()
        if seq_normalized in self.exclusion_set:
            self.exclusion_set.remove(seq_normalized)
            return True
        return False

    def get_size(self) -> int:
        """
        获取排除列表大小

        返回:
            排除列表中的序列数量
        """
        return len(self.exclusion_set)

    def save_to_csv(self, csv_path: str) -> int:
        """
        保存排除列表到CSV文件

        参数:
            csv_path: 输出文件路径

        返回:
            保存的序列数量
        """
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Sequence'])  # 表头
            for seq in sorted(self.exclusion_set):
                writer.writerow([seq])

        return len(self.exclusion_set)


# =============================================================================
# 便捷函数
# =============================================================================

def check_exclusion(sequence: str, exclusion_list_path: str) -> Tuple[bool, str]:
    """
    便捷函数：检查单个序列是否在排除列表中

    参数:
        sequence: 待检查的序列
        exclusion_list_path: 排除列表CSV文件路径

    返回:
        (is_excluded, reason) 元组
    """
    checker = ExclusionListChecker()
    checker.load_from_csv(exclusion_list_path)

    is_excluded = checker.is_excluded(sequence)

    if is_excluded:
        return True, "序列存在于排除列表中"
    else:
        return False, "OK"


def filter_by_exclusion_list(
    sequences: List[str],
    exclusion_list_path: str
) -> List[str]:
    """
    便捷函数：从序列列表中过滤掉排除列表中的序列

    参数:
        sequences: 待过滤的序列列表
        exclusion_list_path: 排除列表CSV文件路径

    返回:
        过滤后的序列列表
    """
    checker = ExclusionListChecker()
    checker.load_from_csv(exclusion_list_path)
    return checker.get_allowed_sequences(sequences)
