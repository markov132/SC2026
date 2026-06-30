"""
SC2026 蛋白设计赛道 - 序列构建器
===================================
本模块负责将突变字符串转换为完整的氨基酸序列。

功能:
1. 解析突变字符串 (如 "A109D:N145D")
2. 从WT序列应用突变生成突变体序列
3. 验证序列合法性
4. 比较序列与WT，生成突变位点列表
"""

from typing import Dict, List, Optional, Tuple
import re
import argparse
import sys
from config import WT_SEQUENCES, STANDARD_AA


# =============================================================================
# 突变解析
# =============================================================================

class MutationParser:
    """
    突变字符串解析器

    支持的突变格式:
    - 单点突变: "A109D"
    - 多点突变: "A109D:N145D"
    - 范围突变: "A109D:L110E:N145D"

    位置系统:
    - 输入使用1-based索引 (生物学惯例)
    - 内部转换为0-based索引 (Python惯例)

    示例:
    >>> parser = MutationParser()
    >>> parser.parse("A109D")
    [(108, 'A', 'D')]  # (position, WT_AA, Mut_AA)
    """

    def __init__(self):
        # 正则表达式匹配突变格式: WT_AA + position + Mut_AA
        # 例如: A109D -> 野生型A, 位置109, 突变型D
        self.mutation_pattern = re.compile(r'^([A-Z])(\d+)([A-Z])$')

    def parse(self, mutation_str: str) -> List[Tuple[int, str, str]]:
        """
        解析突变字符串

        参数:
            mutation_str: 突变字符串，如 "A109D:N145D"

        返回:
            List of (position, wt_aa, mut_aa)，position为0-based索引

        示例:
            >>> parser.parse("A109D:N145D")
            [(108, 'A', 'D'), (144, 'N', 'D')]
        """
        mutations = []

        # 支持冒号 ':' 和分号 ';' 两种分隔符
        if ':' in mutation_str:
            parts = mutation_str.split(':')
        elif ';' in mutation_str:
            parts = mutation_str.split(';')
        else:
            parts = [mutation_str]

        for mut in parts:
            mut = mut.strip()
            if not mut:
                continue

            match = self.mutation_pattern.match(mut)
            if not match:
                raise ValueError(f"无效的突变格式: {mut}，期望格式如 A109D")

            wt_aa, pos_str, mut_aa = match.groups()
            position = int(pos_str) - 1  # 转换为0-based索引

            # 验证是否为标准氨基酸
            if wt_aa not in STANDARD_AA:
                raise ValueError(f"无效的野生型氨基酸: {wt_aa}")
            if mut_aa not in STANDARD_AA:
                raise ValueError(f"无效的突变型氨基酸: {mut_aa}")

            mutations.append((position, wt_aa, mut_aa))

        return mutations

    def validate_mutation(self, sequence: str, position: int, expected_wt_aa: str) -> bool:
        """
        验证序列在指定位置是否为预期的野生型氨基酸

        参数:
            sequence: 完整序列
            position: 位置 (0-based)
            expected_wt_aa: 预期的野生型氨基酸

        返回:
            True如果匹配，False否则
        """
        if position < 0 or position >= len(sequence):
            return False
        return sequence[position] == expected_wt_aa


# =============================================================================
# 序列构建器
# =============================================================================

class SeqBuilder:
    """
    序列构建器

    给定WT序列和突变列表，生成突变体序列。

    用法:
    >>> builder = SeqBuilder()
    >>> seq = builder.build_from_mutations('sfGFP', 'A109D:N145D')
    >>> print(seq)  # sfGFP序列，第109位A→D，第145位N→D
    """

    def __init__(self, wt_sequences: Optional[Dict[str, str]] = None):
        """
        初始化序列构建器

        参数:
            wt_sequences: 野生型序列字典，键为GFP类型，值为序列
        """
        self.wt_sequences = wt_sequences or WT_SEQUENCES
        self.parser = MutationParser()

    def build_from_mutations(
        self,
        gfp_type: str,
        mutation_str: str,
        validate: bool = True
    ) -> str:
        """
        从突变字符串构建突变体序列

        参数:
            gfp_type: GFP类型，如 'sfGFP', 'avGFP'
            mutation_str: 突变字符串，如 "A109D:N145D"
            validate: 是否验证突变位置的野生型氨基酸

        返回:
            突变体序列

        示例:
            >>> builder = SeqBuilder()
            >>> seq = builder.build_from_mutations('sfGFP', 'A109D')
            >>> print(len(seq))  # 230 (与WT长度相同)
        """
        if gfp_type not in self.wt_sequences:
            raise ValueError(f"未知的GFP类型: {gfp_type}，可用类型: {list(self.wt_sequences.keys())}")

        # 获取WT序列
        wt_seq = self.wt_sequences[gfp_type]

        # 解析突变
        mutations = self.parser.parse(mutation_str)

        # 应用突变
        seq_list = list(wt_seq)  # 转换为列表以便修改

        for pos, wt_aa, mut_aa in mutations:
            # 验证
            if validate and not self.parser.validate_mutation(wt_seq, pos, wt_aa):
                raise ValueError(
                    f"突变验证失败: 位置{pos+1}预期为{wt_aa}，"
                    f"实际为{wt_seq[pos]}"
                )

            # 应用突变
            seq_list[pos] = mut_aa

        return ''.join(seq_list)

    def build_from_mutation_list(
        self,
        gfp_type: str,
        mutations: List[Tuple[int, str, str]],
        validate: bool = True
    ) -> str:
        """
        从突变列表构建突变体序列

        参数:
            gfp_type: GFP类型
            mutations: 突变列表，每项为 (position_0based, wt_aa, mut_aa)
            validate: 是否验证

        返回:
            突变体序列
        """
        if gfp_type not in self.wt_sequences:
            raise ValueError(f"未知的GFP类型: {gfp_type}")

        wt_seq = self.wt_sequences[gfp_type]
        seq_list = list(wt_seq)

        for pos, wt_aa, mut_aa in mutations:
            if validate and not self.parser.validate_mutation(wt_seq, pos, wt_aa):
                raise ValueError(
                    f"突变验证失败: 位置{pos+1}预期为{wt_aa}，实际为{wt_seq[pos]}"
                )
            seq_list[pos] = mut_aa

        return ''.join(seq_list)

    def get_mutation_positions(self, sequence: str, gfp_type: str) -> List[Tuple[int, str, str]]:
        """
        比较序列与WT，获取突变位置列表

        参数:
            sequence: 待检测序列
            gfp_type: GFP类型

        返回:
            突变列表，每项为 (position_0based, wt_aa, mut_aa)
        """
        if gfp_type not in self.wt_sequences:
            raise ValueError(f"未知的GFP类型: {gfp_type}")

        wt_seq = self.wt_sequences[gfp_type]

        if len(sequence) != len(wt_seq):
            raise ValueError(
                f"序列长度不匹配: 期望{len(wt_seq)}，实际{len(sequence)}"
            )

        mutations = []
        for i, (wt_aa, seq_aa) in enumerate(zip(wt_seq, sequence)):
            if wt_aa != seq_aa:
                mutations.append((i, wt_aa, seq_aa))

        return mutations


# =============================================================================
# 序列验证
# =============================================================================

class SeqValidator:
    """
    序列验证器

    验证序列是否符合设计约束:
    1. 长度: 220-250 aa
    2. 起始密码子: M
    3. 仅含标准氨基酸
    4. 无终止密码子(*)
    """

    def __init__(self, min_length: int = 220, max_length: int = 250):
        self.min_length = min_length
        self.max_length = max_length
        self.valid_aa_set = set(STANDARD_AA)

    def validate_length(self, sequence: str) -> Tuple[bool, str]:
        """验证序列长度"""
        length = len(sequence)
        if length < self.min_length or length > self.max_length:
            return False, f"长度{length}不在范围内[{self.min_length}, {self.max_length}]"
        return True, "OK"

    def validate_start(self, sequence: str) -> Tuple[bool, str]:
        """验证起始密码子"""
        if not sequence.startswith('M'):
            return False, "序列必须以M开头"
        return True, "OK"

    def validate_standard_aa(self, sequence: str) -> Tuple[bool, str]:
        """验证仅含标准氨基酸"""
        invalid_aa = [aa for aa in sequence if aa not in self.valid_aa_set]
        if invalid_aa:
            return False, f"包含非标准氨基酸: {set(invalid_aa)}"
        return True, "OK"

    def validate_no_stop(self, sequence: str) -> Tuple[bool, str]:
        """验证无终止密码子"""
        if '*' in sequence:
            return False, "序列包含终止密码子*"
        return True, "OK"

    def validate_all(self, sequence: str) -> Tuple[bool, List[str]]:
        """
        执行所有验证

        返回:
            (是否通过, 失败原因列表)
        """
        checks = [
            self.validate_length,
            self.validate_start,
            self.validate_standard_aa,
            self.validate_no_stop,
        ]

        failures = []
        for check in checks:
            passed, msg = check(sequence)
            if not passed:
                failures.append(msg)

        return len(failures) == 0, failures


# =============================================================================
# 导出函数
# =============================================================================

def build_sequence(gfp_type: str, mutation_str: str) -> str:
    """
    便捷函数：从突变字符串构建序列

    参数:
        gfp_type: GFP类型
        mutation_str: 突变字符串

    返回:
        突变体序列
    """
    builder = SeqBuilder()
    return builder.build_from_mutations(gfp_type, mutation_str)


def count_mutations(sequence: str, gfp_type: str) -> int:
    """
    便捷函数：计算序列与WT的突变数

    参数:
        sequence: 待检测序列
        gfp_type: GFP类型

    返回:
        突变数量
    """
    builder = SeqBuilder()
    return len(builder.get_mutation_positions(sequence, gfp_type))


def compare_sequences(wt: str, mutant: str) -> List[str]:
    """
    比较WT和突变体序列，返回突变位点列表（字符串格式）

    参数:
        wt: 野生型序列
        mutant: 突变体序列

    返回:
        突变位点列表，如 ["A109D", "N145D"]
    """
    wt = wt.upper().strip()
    mutant = mutant.upper().strip()

    if len(wt) != len(mutant):
        raise ValueError(f"序列长度不一致: WT={len(wt)}, Mutant={len(mutant)}")

    mutations = []
    for i, (wt_aa, mut_aa) in enumerate(zip(wt, mutant)):
        if wt_aa != mut_aa:
            position = i + 1  # 1-based indexing
            mutations.append(f"{wt_aa}{position}{mut_aa}")

    return mutations


def mutations_to_string(mutations: List[str], sep: str = ", ") -> str:
    """
    将突变位点列表转换为字符串

    参数:
        mutations: 突变位点列表
        sep: 分隔符

    返回:
        突变位点字符串，如 "A109D, N145D"
    """
    return sep.join(mutations)


def get_mutations_string(sequence: str, gfp_type: str = 'sfGFP', sep: str = ", ") -> str:
    """
    获取序列相对于WT的突变位点字符串

    参数:
        sequence: 突变体序列
        gfp_type: GFP类型
        sep: 分隔符

    返回:
        突变位点字符串
    """
    builder = SeqBuilder()
    mutations = builder.get_mutation_positions(sequence, gfp_type)
    mutation_strings = [f"{wt_aa}{pos+1}{mut_aa}" for pos, wt_aa, mut_aa in mutations]
    return sep.join(mutation_strings)


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='突变体序列 -> 突变位点转换工具')
    parser.add_argument('--mutant', type=str, help='突变体序列')
    parser.add_argument('--source', type=str, 
                       choices=list(WT_SEQUENCES.keys()),
                       help=f'参考序列来源 ({", ".join(WT_SEQUENCES.keys())})')
    parser.add_argument('--wt', type=str, help='自定义WT序列 (覆盖 --source)')
    parser.add_argument('--sep', type=str, default=', ', help='突变位点分隔符')
    parser.add_argument('--test', action='store_true', help='运行测试')
    
    args = parser.parse_args()

    if args.test:
        print("=" * 60)
        print("序列构建器测试")
        print("=" * 60)

        # 测试突变解析
        parser = MutationParser()
        print("\n1. 突变解析测试:")
        mutations = parser.parse("A109D:N145D")
        print(f"   'A109D:N145D' -> {mutations}")

        # 测试序列构建
        print("\n2. 序列构建测试:")
        builder = SeqBuilder()
        seq = builder.build_from_mutations('sfGFP', 'R109D')
        print(f"   sfGFP + R109D 的第110位: {seq[109]}")

        # 测试突变计数
        print("\n3. 突变计数测试:")
        n_muts = count_mutations(seq, 'sfGFP')
        print(f"   序列与sfGFP的突变数: {n_muts}")

        # 测试序列验证
        print("\n4. 序列验证测试:")
        validator = SeqValidator()

        # 测试sfGFP WT
        is_valid, failures = validator.validate_all(WT_SEQUENCES['sfGFP'])
        print(f"   sfGFP WT验证: {'通过' if is_valid else '失败'}")
        if not is_valid:
            print(f"   失败原因: {failures}")

        # 测试突变位点转换
        print("\n5. 突变位点转换测试:")
        mut_str = get_mutations_string(seq, 'sfGFP')
        print(f"   突变位点: {mut_str}")

        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
    
    elif args.mutant:
        if args.wt:
            wt_seq = args.wt
            source_name = "自定义序列"
        elif args.source:
            wt_seq = WT_SEQUENCES.get(args.source)
            source_name = args.source
        else:
            wt_seq = WT_SEQUENCES.get('sfGFP')
            source_name = "sfGFP (默认)"

        try:
            mutations = compare_sequences(wt_seq, args.mutant)

            print("=" * 60)
            print("突变位点分析结果")
            print("=" * 60)
            print(f"参考序列: {source_name}")
            print(f"突变数量: {len(mutations)}")
            print(f"突变位点: {args.sep.join(mutations)}")

        except ValueError as e:
            print(f"错误: {e}")
            sys.exit(1)
    else:
        parser.print_help()
