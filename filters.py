"""
SC2026 蛋白设计赛道 - 筛选器模块
===================================
本模块实现三级筛选漏斗:

Stage 1: 硬约束过滤
  - 长度检查 (220-250 aa)
  - 起始密码子 (M)
  - 仅含标准氨基酸
  - 无终止密码子
  - 不在排除列表中

Stage 2: 亮度预测
  - B̂(s) > 0.5 (归一化亮度)

Stage 3: 帕累托前沿 + 多样性筛选
  - Pareto最优
  - 序列相似度 < 0.90
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass

from seq_builder import SeqValidator
from exclusion_checker import ExclusionListChecker


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class SequenceRecord:
    """序列记录，包含序列及其各种评估分数"""
    seq_id: str
    sequence: str
    # 筛选状态
    stage1_pass: bool = False
    stage2_pass: bool = False
    stage3_pass: bool = False
    # 评估指标
    B_hat_esm: Optional[float] = None
    B_hat: Optional[float] = None
    f_score: Optional[float] = None
    n_mutations: Optional[int] = None
    # 失败原因
    fail_reasons: List[str] = None

    def __post_init__(self):
        if self.fail_reasons is None:
            self.fail_reasons = []


# =============================================================================
# Stage 1: 硬约束过滤器
# =============================================================================

class HardConstraintFilter:
    """
    硬约束过滤器 - Stage 1

    检查项:
    1. 长度: 220-250 aa
    2. 起始密码子: M
    3. 仅含20种标准氨基酸
    4. 无终止密码子(*)
    5. 不在排除列表中

    生物学意义:
    - 排除列表确保不重复提交已知序列
    """

    def __init__(
        self,
        wt_sequence: Optional[str] = None,
        min_length: int = 220,
        max_length: int = 250,
        exclusion_list_path: Optional[str] = None
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.valid_aa_set = set('ACDEFGHIKLMNPQRSTVWY')

        # 初始化序列验证器
        self.validator = SeqValidator(min_length, max_length)

        # 初始化排除列表检查器
        self.exclusion_checker = ExclusionListChecker()
        if exclusion_list_path:
            self.exclusion_checker.load_from_csv(exclusion_list_path)

    def load_exclusion_list(self, path: str) -> None:
        """加载排除列表"""
        self.exclusion_checker.load_from_csv(path)

    def filter(self, sequence: str) -> Tuple[bool, List[str]]:
        """
        对序列进行硬约束过滤

        参数:
            sequence: 待过滤的氨基酸序列

        返回:
            (是否通过, 失败原因列表)
        """
        failures = []

        # 1-4. 使用 SeqValidator 进行序列验证
        passed_len, _ = self.validator.validate_length(sequence)
        if not passed_len:
            failures.append(f"length_{len(sequence)}")

        passed_start, _ = self.validator.validate_start(sequence)
        if not passed_start:
            failures.append("no_start_M")

        passed_aa, msg_aa = self.validator.validate_standard_aa(sequence)
        if not passed_aa:
            invalid_aa_str = msg_aa.split(": ", 1)[1]
            failures.append(f"non_standard_AA:{invalid_aa_str}")

        passed_stop, _ = self.validator.validate_no_stop(sequence)
        if not passed_stop:
            failures.append("has_stop_codon")

        # 5. 排除列表检查
        if self.exclusion_checker.get_size() > 0:
            if self.exclusion_checker.is_excluded(sequence):
                failures.append("in_exclusion_list")

        return len(failures) == 0, failures

    def filter_batch(
        self,
        sequences: List[str],
        seq_ids: Optional[List[str]] = None
    ) -> List[SequenceRecord]:
        """
        批量过滤序列

        参数:
            sequences: 序列列表
            seq_ids: 可选的ID列表，如果为None则使用索引作为ID

        返回:
            SequenceRecord列表
        """
        if seq_ids is None:
            seq_ids = [f"seq_{i}" for i in range(len(sequences))]

        records = []
        for seq_id, seq in zip(seq_ids, sequences):
            passed, failures = self.filter(seq)
            record = SequenceRecord(
                seq_id=seq_id,
                sequence=seq,
                stage1_pass=passed,
                fail_reasons=failures
            )
            records.append(record)

        return records

    def get_pass_rate(self, records: List[SequenceRecord]) -> float:
        """计算通过率"""
        if not records:
            return 0.0
        passed = sum(1 for r in records if r.stage1_pass)
        return passed / len(records)


# =============================================================================
# Stage 2: 预测过滤器
# =============================================================================

class PredictionFilter:
    """
    预测过滤器 - Stage 2

    检查项:
    1. B̂(s) > 0.5: 归一化亮度预测值

    生物学意义:
    - B̂ > 0.5 确保初始亮度足够高
    """

    def __init__(
        self,
        B_threshold: float = 0.5
    ):
        self.B_threshold = B_threshold

    def check(
        self,
        B_hat: float
    ) -> Tuple[bool, List[str]]:
        """
        检查预测值是否满足阈值

        参数:
            B_hat: 归一化亮度预测值

        返回:
            (是否通过, 失败原因列表)
        """
        failures = []

        if B_hat < self.B_threshold:
            failures.append(f"B_too_low:{B_hat:.3f}")

        return len(failures) == 0, failures

    def update_records(self, records: List[SequenceRecord]) -> None:
        """更新SequenceRecord列表的Stage 2信息"""
        for record in records:
            if record.B_hat is not None:
                passed, failures = self.check(record.B_hat)
                record.stage2_pass = passed
                record.fail_reasons.extend(failures)
            elif record.stage1_pass:
                record.fail_reasons.append("missing_prediction_data")


# =============================================================================
# Stage 3: 帕累托前沿 + 多样性筛选
# =============================================================================

class ParetoDiversityFilter:
    """
    帕累托前沿 + 多样性筛选器 - Stage 3

    帕累托最优:
    - 序列s1支配s2当且仅当:
      B(s1) ≥ B(s2)
      且至少有一个严格不等式
    - 帕累托前沿 = 所有不被支配的序列

    多样性筛选:
    - 从帕累托前沿中选择相似度 < 0.90 的序列
    - 确保最终提交的6条序列具有足够多样性
    """

    def __init__(
        self,
        max_similarity: float = 0.90,
        n_final: int = 6
    ):
        self.max_similarity = max_similarity
        self.n_final = n_final

    def is_pareto_dominated(
        self,
        scores: List[Tuple[int, float, float]]
    ) -> List[bool]:
        """
        判断每个序列是否被帕累托支配

        参数:
            scores: List of (index, B_score, diversity_score)

        返回:
            List of 是否被支配
        """
        n = len(scores)
        if n == 0:
            return []

        is_dominated = [True] * n

        sorted_indices = sorted(range(n), key=lambda i: (-scores[i][1], -scores[i][2]))

        max_diversity = -float('inf')
        last_pareto_B = None

        for idx in sorted_indices:
            _, B, D = scores[idx]

            if D > max_diversity:
                is_dominated[idx] = False
                max_diversity = D
                last_pareto_B = B
            elif D == max_diversity and B == last_pareto_B:
                is_dominated[idx] = False

        return is_dominated

    def get_pareto_frontier(
        self,
        records: List[SequenceRecord]
    ) -> List[SequenceRecord]:
        """
        获取帕累托前沿序列（二维：亮度 + 突变多样性）

        参数:
            records: SequenceRecord列表

        返回:
            帕累托前沿序列列表（至少返回n_final条，如果可能）
        """
        # 筛选Stage 1-2都通过的记录
        candidates = [r for r in records if r.stage1_pass and r.stage2_pass]

        if not candidates:
            return []

        # 如果候选序列很少，直接返回所有
        if len(candidates) <= self.n_final:
            for r in candidates:
                r.stage3_pass = True
            return candidates

        # 计算每个候选的多样性分数（基于突变数，突变越多越多样）
        # 使用相对于WT的差异比例作为多样性分数
        diversity_scores = []
        for r in candidates:
            if r.n_mutations is not None:
                diversity_scores.append(r.n_mutations)
            else:
                # 如果没有突变数，计算与第一条序列的差异
                seq1 = r.sequence
                seq_ref = candidates[0].sequence
                n_diff = sum(a != b for a, b in zip(seq1, seq_ref))
                diversity_scores.append(n_diff)
        
        # 归一化多样性分数到0-1范围
        max_div = max(diversity_scores) if diversity_scores else 1
        if max_div == 0:
            max_div = 1
        diversity_scores_norm = [d / max_div for d in diversity_scores]

        # 构建分数列表: (index, B_score, diversity_score)
        scores = [
            (i, r.B_hat if r.B_hat else 0, diversity_scores_norm[i])
            for i, r in enumerate(candidates)
        ]

        # 找到帕累托前沿
        is_dominated = self.is_pareto_dominated(scores)

        pareto_records = []
        for i, dominated in enumerate(is_dominated):
            if not dominated:
                candidates[i].stage3_pass = True
                pareto_records.append(candidates[i])

        # 如果帕累托前沿序列不够，从候选中补充（按f_score排序）
        if len(pareto_records) < self.n_final:
            # 按f_score排序（综合亮度和多样性）
            for i, r in enumerate(candidates):
                if r.f_score is None:
                    b_score = r.B_hat if r.B_hat else 0
                    r.f_score = b_score * (1.0 + 0.3 * diversity_scores_norm[i])
            candidates_sorted = sorted(candidates, key=lambda r: (r.f_score or 0), reverse=True)
            
            for r in candidates_sorted:
                if len(pareto_records) >= self.n_final:
                    break
                if r not in pareto_records:
                    r.stage3_pass = True
                    pareto_records.append(r)

        return pareto_records

    def calculate_similarity(self, seq1: str, seq2: str) -> float:
        """
        计算两条序列的相似度 (位置对齐)

        参数:
            seq1, seq2: 两条等长序列

        返回:
            相似度 (0-1)
        """
        if len(seq1) != len(seq2):
            # 如果长度不同，使用较短的长度
            min_len = min(len(seq1), len(seq2))
            matching = sum(a == b for a, b in zip(seq1[:min_len], seq2[:min_len]))
            return matching / max(len(seq1), len(seq2))
        else:
            matching = sum(a == b for a, b in zip(seq1, seq2))
            return matching / len(seq1)

    def select_diverse_sequences(
        self,
        pareto_records: List[SequenceRecord]
    ) -> List[SequenceRecord]:
        """
        从帕累托前沿中选择多样化的序列

        参数:
            pareto_records: 帕累托前沿序列列表

        返回:
            选中的序列列表 (最多n_final条)
        """
        if not pareto_records:
            return []

        # 如果帕累托前沿序列少于等于n_final，直接返回所有
        if len(pareto_records) <= self.n_final:
            return pareto_records

        # 尝试选择多样化的序列
        selected = [pareto_records[0]]

        for record in pareto_records[1:]:
            if len(selected) >= self.n_final:
                break

            # 计算与已选序列的最大相似度
            max_sim = max(
                self.calculate_similarity(record.sequence, s.sequence)
                for s in selected
            )

            # 如果最大相似度低于阈值，选择该序列
            if max_sim < self.max_similarity:
                selected.append(record)
        
        # 如果仍然不够，直接选择剩余的
        if len(selected) < self.n_final:
            for record in pareto_records:
                if len(selected) >= self.n_final:
                    break
                if record not in selected:
                    selected.append(record)
        
        return selected


# =============================================================================
# 综合筛选管线
# =============================================================================

class ScreeningPipeline:
    """
    综合筛选管线

    整合三个阶段的筛选:
    1. HardConstraintFilter - 硬约束过滤
    2. PredictionFilter - 预测值过滤
    3. ParetoDiversityFilter - 帕累托前沿+多样性

    使用示例:
    >>> pipeline = ScreeningPipeline(exclusion_list_path="Exclusion_List.csv")
    >>> results = pipeline.run(candidate_sequences)
    >>> final_seqs = results['final_sequences']
    """

    def __init__(
        self,
        wt_sequence: Optional[str] = None,
        exclusion_list_path: Optional[str] = None,
        B_threshold: float = 0.5,
        max_similarity: float = 0.90,
        n_final: int = 6,
        min_length: int = 220,
        max_length: int = 250
    ):
        # Stage 1: 硬约束
        self.hard_filter = HardConstraintFilter(
            wt_sequence=wt_sequence,
            min_length=min_length,
            max_length=max_length,
            exclusion_list_path=exclusion_list_path
        )

        # Stage 2: 预测过滤
        self.prediction_filter = PredictionFilter(
            B_threshold=B_threshold
        )

        # Stage 3: 帕累托+多样性
        self.pareto_filter = ParetoDiversityFilter(
            max_similarity=max_similarity,
            n_final=n_final
        )

    def load_exclusion_list(self, path: str) -> None:
        """加载排除列表"""
        self.hard_filter.load_exclusion_list(path)

    def run(
        self,
        sequences: List[str],
        seq_ids: Optional[List[str]] = None,
        B_dict: Optional[Dict[str, float]] = None,
        origin_groups: Optional[List[int]] = None,
        n_mutations_dict: Optional[Dict[str, int]] = None
    ) -> Dict:
        """
        运行完整筛选管线

        参数:
            sequences: 候选序列列表
            seq_ids: 序列ID列表
            B_dict: seq_id -> B_hat的字典
            origin_groups: 每个序列的来源组ID列表（可选）
            n_mutations_dict: seq_id -> n_mutations的字典（可选，用于多样性计算）
                          如果提供，则按组分别筛选，每组选出n_final条
                          例如: [0, 0, 0, 1, 1, 1, 2, 2, 2] 表示3组，每组3条

        返回:
            包含各阶段结果的字典
        """
        if seq_ids is None:
            seq_ids = [f"seq_{i}" for i in range(len(sequences))]
        
        # 如果没有提供origin_groups，默认所有序列为一组
        if origin_groups is None:
            origin_groups = [0] * len(sequences)
        
        # Stage 1: 硬约束过滤
        print("[Stage 1] 硬约束过滤...")
        records = self.hard_filter.filter_batch(sequences, seq_ids)
        stage1_passed = sum(1 for r in records if r.stage1_pass)
        print(f"       通过: {stage1_passed}/{len(records)}")

        # 填充突变数信息（用于多样性计算）
        if n_mutations_dict:
            for record in records:
                if record.seq_id in n_mutations_dict:
                    record.n_mutations = n_mutations_dict[record.seq_id]

        # Stage 2: 亮度预测
        print("[Stage 2] 亮度预测...")
        if B_dict:
            for record in records:
                if record.seq_id in B_dict:
                    record.B_hat = B_dict[record.seq_id]
                if record.B_hat is not None:
                    # f_score = B_hat * (1 + 0.3 * diversity_norm) - 综合亮度和突变多样性
                    # 多样性分数基于突变数，突变越多越多样
                    if record.n_mutations is not None and record.n_mutations > 0:
                        diversity_norm = min(record.n_mutations / 8.0, 1.0)  # 归一化到0-1
                        record.f_score = record.B_hat * (1.0 + 0.3 * diversity_norm)
                    else:
                        record.f_score = record.B_hat

            self.prediction_filter.update_records(records)

        stage2_passed = sum(1 for r in records if r.stage2_pass)
        print(f"       通过: {stage2_passed}/{len(records)}")

        # 检查是否需要分组筛选
        unique_groups = set(origin_groups)
        if len(unique_groups) > 1:
            # 分组筛选：每组分别选出n_final条
            print(f"[Stage 3] 帕累托前沿 + 多样性筛选（按{len(unique_groups)}组分选）...")
            final_records = []
            pareto_records = []
            
            for group_id in sorted(unique_groups):
                group_indices = [i for i, g in enumerate(origin_groups) if g == group_id]
                group_records = [records[i] for i in group_indices]
                
                # 帕累托筛选
                group_pareto = self.pareto_filter.get_pareto_frontier(group_records)
                print(f"       组{group_id}: 帕累托前沿 {len(group_pareto)} 条", end="")
                pareto_records.extend(group_pareto)
                
                # 多样性选择
                group_final = self.pareto_filter.select_diverse_sequences(group_pareto)
                final_records.extend(group_final)
                print(f" → 最终 {len(group_final)} 条")
            
            print(f"       最终选择: {len(final_records)} 条（来自{len(unique_groups)}组）")
        else:
            # 统一筛选（原有逻辑）
            print("[Stage 3] 帕累托前沿 + 多样性筛选...")
            pareto_records = self.pareto_filter.get_pareto_frontier(records)
            print(f"       帕累托前沿: {len(pareto_records)} 条")

            final_records = self.pareto_filter.select_diverse_sequences(pareto_records)
            print(f"       最终选择: {len(final_records)} 条")

        return {
            'records': records,
            'stage1_passed': stage1_passed,
            'stage2_passed': stage2_passed,
            'stage3_passed': len(final_records),
            'pareto_frontier': pareto_records,
            'final_sequences': final_records
        }
