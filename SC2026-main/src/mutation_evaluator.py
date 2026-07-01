"""
SC2026 蛋白设计赛道 - 突变结果评估指标
===================================

本模块提供全面的突变结果评估指标，包括：
1. 性能指标：亮度
2. 多样性指标：序列多样性、突变位置分布
3. 质量指标：突变合理性、氨基酸替换
4. 统计指标：平均值、标准差、最佳/最差值

评估维度:
├── 性能评估
│   └── 亮度得分(B̂)
├── 多样性评估
│   ├── 序列相似度矩阵
│   ├── 平均成对距离
│   └── 突变位置多样性
└── 质量评估
    ├── 突变数量分布
    └── 氨基酸替换合理性
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import json
from collections import Counter
from scipy.spatial.distance import pdist, squareform


# =============================================================================
# 常量定义
# =============================================================================

# 氨基酸分类
AA_GROUPS = {
    'hydrophobic': set('AILMFPWV'),
    'polar': set('STNQ'),
    'positive': set('KRH'),
    'negative': set('DE'),
    'aromatic': set('FYW'),
    'special': set('CG')
}


# =============================================================================
# 性能评估指标
# =============================================================================

class PerformanceMetrics:
    """性能评估指标"""
    
    @staticmethod
    def mean_brightness(brightness_scores: List[float]) -> float:
        """计算平均亮度"""
        return np.mean(brightness_scores)
    
    @staticmethod
    def best_brightness(brightness_scores: List[float]) -> Tuple[float, int]:
        """获取最佳亮度及其索引"""
        best_idx = np.argmax(brightness_scores)
        return brightness_scores[best_idx], best_idx
    
    @staticmethod
    def worst_brightness(brightness_scores: List[float]) -> Tuple[float, int]:
        """获取最差亮度及其索引"""
        worst_idx = np.argmin(brightness_scores)
        return brightness_scores[worst_idx], worst_idx
    
    @staticmethod
    def std_brightness(brightness_scores: List[float]) -> float:
        """计算亮度标准差"""
        return np.std(brightness_scores)
    
    @staticmethod
    def performance_stats(
        brightness: List[float]
    ) -> Dict[str, float]:
        """计算所有性能指标"""
        return {
            'mean_brightness': float(np.mean(brightness)),
            'std_brightness': float(np.std(brightness)),
            'min_brightness': float(np.min(brightness)),
            'max_brightness': float(np.max(brightness)),
            'median_brightness': float(np.median(brightness)),
            'iqr_brightness': float(np.percentile(brightness, 75) - np.percentile(brightness, 25))
        }


# =============================================================================
# 多样性评估指标
# =============================================================================

class DiversityMetrics:
    """多样性评估指标"""
    
    @staticmethod
    def hamming_distance(seq1: str, seq2: str) -> int:
        """计算两条序列的Hamming距离"""
        return sum(a != b for a, b in zip(seq1, seq2))
    
    @staticmethod
    def similarity_matrix(sequences: List[str]) -> np.ndarray:
        """计算序列相似度矩阵（基于Hamming距离）"""
        if not sequences:
            return np.zeros((0, 0))
        
        n = len(sequences)
        if n == 1:
            return np.ones((1, 1))
        
        lengths = [len(s) for s in sequences]
        max_len = max(lengths)
        
        if len(set(lengths)) == 1:
            all_aa = sorted(set(''.join(sequences)))
            aa_to_int = {aa: idx for idx, aa in enumerate(all_aa)}
            
            encoded = np.zeros((n, max_len), dtype=np.int16)
            for i, seq in enumerate(sequences):
                for j, aa in enumerate(seq):
                    encoded[i, j] = aa_to_int[aa]
            
            hamming_proportions = pdist(encoded, metric='hamming')
            dist_vector = hamming_proportions * max_len
            dist_matrix = squareform(dist_vector)
            
            sim_matrix = 1 - (dist_matrix / max_len)
        else:
            dist_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    dist_matrix[i, j] = DiversityMetrics.hamming_distance(sequences[i], sequences[j])
            
            sim_matrix = 1 - (dist_matrix / max_len)
        
        return sim_matrix
    
    @staticmethod
    def mean_pairwise_distance(sequences: List[str]) -> float:
        """计算平均成对距离"""
        n = len(sequences)
        total_dist = 0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_dist += DiversityMetrics.hamming_distance(sequences[i], sequences[j])
                count += 1
        
        return total_dist / count if count > 0 else 0
    
    @staticmethod
    def sequence_diversity(sequences: List[str]) -> float:
        """计算序列多样性分数（基于成对距离的变异系数）"""
        if len(sequences) < 2:
            return 0.0
        
        dists = []
        n = len(sequences)
        
        for i in range(n):
            for j in range(i + 1, n):
                dists.append(DiversityMetrics.hamming_distance(sequences[i], sequences[j]))
        
        mean_dist = np.mean(dists)
        std_dist = np.std(dists)
        
        return std_dist / mean_dist if mean_dist > 0 else 0.0
    
    @staticmethod
    def mutation_position_diversity(mutations_list: List[List[int]]) -> Dict[str, float]:
        """
        计算突变位置多样性
        
        参数:
            mutations_list: 每个序列的突变位置列表
        
        返回:
            多样性指标字典
        """
        all_positions = []
        for mutations in mutations_list:
            all_positions.extend(mutations)
        
        if not all_positions:
            return {'unique_positions': 0, 'position_entropy': 0.0}
        
        # 统计位置分布
        position_counts = Counter(all_positions)
        unique_positions = len(position_counts)
        
        # 计算位置熵（衡量分布均匀性）
        total = sum(position_counts.values())
        probs = [count / total for count in position_counts.values()]
        entropy = -sum(p * np.log(p) for p in probs if p > 0)
        
        return {
            'unique_positions': unique_positions,
            'total_mutations': len(all_positions),
            'mean_mutations_per_position': len(all_positions) / unique_positions,
            'position_entropy': float(entropy),
            'max_position_usage': max(position_counts.values()),
            'min_position_usage': min(position_counts.values())
        }
    
    @staticmethod
    def diversity_stats(sequences: List[str], mutations_list: Optional[List[List[int]]] = None) -> Dict:
        """计算所有多样性指标"""
        return {
            'mean_pairwise_distance': float(DiversityMetrics.mean_pairwise_distance(sequences)),
            'sequence_diversity': float(DiversityMetrics.sequence_diversity(sequences)),
            'num_sequences': len(sequences),
            'unique_positions': DiversityMetrics.mutation_position_diversity(mutations_list or [])
        }


# =============================================================================
# 质量评估指标
# =============================================================================

class QualityMetrics:
    """质量评估指标"""
    
    @staticmethod
    def count_mutations(wt_seq: str, mutant_seq: str) -> int:
        """计算突变数量"""
        return sum(a != b for a, b in zip(wt_seq, mutant_seq))
    
    @staticmethod
    def mutation_positions(wt_seq: str, mutant_seq: str) -> List[int]:
        """获取所有突变位置（0-based）"""
        return [i for i, (a, b) in enumerate(zip(wt_seq, mutant_seq)) if a != b]
    
    @staticmethod
    def amino_acid_substitution_score(wt_aa: str, mutant_aa: str) -> float:
        """
        评估氨基酸替换的合理性
        
        评分标准:
        - 同组替换: 1.0
        - 相近组替换: 0.5-0.7
        - 跨组替换: 0.1-0.3
        """
        wt_aa = wt_aa.upper()
        mutant_aa = mutant_aa.upper()
        
        if wt_aa == mutant_aa:
            return 1.0
        
        # 找到氨基酸所属的组
        wt_groups = [group for group, aas in AA_GROUPS.items() if wt_aa in aas]
        mut_groups = [group for group, aas in AA_GROUPS.items() if mutant_aa in aas]
        
        # 同组替换
        if wt_groups and mut_groups and wt_groups[0] == mut_groups[0]:
            return 1.0
        
        # 相近组替换
        similar_groups = [
            ('hydrophobic', 'aromatic'),
            ('polar', 'special'),
            ('positive', 'polar'),
            ('negative', 'polar')
        ]
        
        for g1, g2 in similar_groups:
            if (g1 in wt_groups and g2 in mut_groups) or (g2 in wt_groups and g1 in mut_groups):
                return 0.6
        
        return 0.2
    
    @staticmethod
    def sequence_substitution_score(wt_seq: str, mutant_seq: str) -> float:
        """计算整条序列的替换合理性得分"""
        scores = []
        for wt_aa, mut_aa in zip(wt_seq, mutant_seq):
            if wt_aa != mut_aa:
                scores.append(QualityMetrics.amino_acid_substitution_score(wt_aa, mut_aa))
        
        return np.mean(scores) if scores else 1.0
    
    @staticmethod
    def quality_stats(sequences: List[str], wt_seq: str) -> Dict:
        """计算所有质量指标"""
        mutation_counts = [QualityMetrics.count_mutations(wt_seq, seq) for seq in sequences]
        substitution_scores = [QualityMetrics.sequence_substitution_score(wt_seq, seq) for seq in sequences]
        
        return {
            'mean_mutations': float(np.mean(mutation_counts)),
            'std_mutations': float(np.std(mutation_counts)),
            'min_mutations': int(np.min(mutation_counts)),
            'max_mutations': int(np.max(mutation_counts)),
            'mean_substitution_score': float(np.mean(substitution_scores)),
            'std_substitution_score': float(np.std(substitution_scores))
        }


# =============================================================================
# 综合评估器
# =============================================================================

class MutationEvaluator:
    """
    突变结果综合评估器
    
    提供全面的突变评估功能
    """
    
    def __init__(self, wt_sequence: str):
        """
        初始化评估器
        
        参数:
            wt_sequence: 野生型序列（用于对比）
        """
        self.wt_sequence = wt_sequence
    
    def evaluate(
        self,
        sequences: List[str],
        brightness_scores: Optional[List[float]] = None
    ) -> Dict[str, Dict]:
        """
        综合评估突变结果
        
        参数:
            sequences: 候选序列列表
            brightness_scores: 亮度得分列表（可选）
        
        返回:
            综合评估结果字典
        """
        # 计算各维度指标
        results = {
            'performance': PerformanceMetrics.performance_stats(
                brightness_scores or [0.0] * len(sequences)
            ),
            'diversity': DiversityMetrics.diversity_stats(sequences),
            'quality': QualityMetrics.quality_stats(sequences, self.wt_sequence),
            'summary': self._generate_summary(sequences, brightness_scores)
        }
        
        return results
    
    def _generate_summary(
        self,
        sequences: List[str],
        brightness: Optional[List[float]]
    ) -> Dict:
        """生成评估摘要"""
        if not sequences:
            return {}
        
        # 最佳候选（基于亮度）
        if brightness:
            best_idx = np.argmax(brightness)
            best_seq = sequences[best_idx]
        else:
            best_idx = 0
            best_seq = sequences[0]
        
        # 统计信息
        mutation_counts = [QualityMetrics.count_mutations(self.wt_sequence, seq) for seq in sequences]
        
        return {
            'total_candidates': len(sequences),
            'best_candidate_index': int(best_idx),
            'best_brightness': float(brightness[best_idx]) if brightness else None,
            'best_mutation_count': int(mutation_counts[best_idx]),
            'mean_brightness': float(np.mean(brightness)) if brightness else 0.0,
            'top_5_brightness_mean': float(np.mean(sorted(brightness, reverse=True)[:5])) if brightness else 0.0,
            'pass_rate': float(sum(1 for b in brightness if b >= 3.0) / len(brightness)) if brightness else 0.0
        }
    
    def evaluate_single(
        self,
        sequence: str,
        brightness: float = None
    ) -> Dict:
        """
        评估单个序列
        
        参数:
            sequence: 待评估序列
            brightness: 亮度得分
        
        返回:
            单个序列的评估结果
        """
        return {
            'mutation_count': QualityMetrics.count_mutations(self.wt_sequence, sequence),
            'mutation_positions': QualityMetrics.mutation_positions(self.wt_sequence, sequence),
            'substitution_score': float(QualityMetrics.sequence_substitution_score(self.wt_sequence, sequence)),
            'brightness': brightness,
            'sequence_length': len(sequence)
        }
    
    def print_report(self, results: Dict) -> None:
        """打印评估报告"""
        print("\n" + "=" * 70)
        print("                    突变结果评估报告")
        print("=" * 70)
        
        # 性能指标
        print("\n【性能指标】")
        perf = results['performance']
        print(f"  平均亮度:        {perf['mean_brightness']:.3f} ± {perf['std_brightness']:.3f}")
        print(f"  最佳亮度:        {perf['max_brightness']:.3f}")
        print(f"  最差亮度:        {perf['min_brightness']:.3f}")
        print(f"  中位数亮度:      {perf['median_brightness']:.3f}")
        
        # 多样性指标
        print("\n【多样性指标】")
        div = results['diversity']
        print(f"  候选数量:        {div['num_sequences']}")
        print(f"  平均成对距离:    {div['mean_pairwise_distance']:.2f}")
        print(f"  序列多样性:      {div['sequence_diversity']:.3f}")
        
        # 质量指标
        print("\n【质量指标】")
        qual = results['quality']
        print(f"  平均突变数:      {qual['mean_mutations']:.2f} ± {qual['std_mutations']:.2f}")
        print(f"  替换合理性得分:  {qual['mean_substitution_score']:.3f}")
        
        # 摘要
        print("\n【评估摘要】")
        summ = results['summary']
        print(f"  最佳候选索引:    {summ['best_candidate_index'] + 1}")
        print(f"  最佳亮度:        {summ['best_brightness']:.3f}")
        print(f"  前5候选亮度均值: {summ['top_5_brightness_mean']:.3f}")
        print(f"  达标率(≥3.0):    {summ['pass_rate']:.1%}")
        
        print("\n" + "=" * 70)
    
    def save_report(self, results: Dict, file_path: str) -> None:
        """保存评估报告到文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"评估报告已保存到: {file_path}")


# =============================================================================
# 命令行工具
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='突变结果评估工具')
    parser.add_argument('--wt_seq', type=str, required=True, help='野生型序列')
    parser.add_argument('--sequences', type=str, required=True, help='候选序列文件（每行一条）')
    parser.add_argument('--brightness', type=str, help='亮度得分文件（每行一个值）')
    parser.add_argument('--output', type=str, default='mutation_evaluation_report.json', help='输出文件')
    
    args = parser.parse_args()
    
    # 加载序列
    with open(args.sequences, 'r') as f:
        sequences = [line.strip() for line in f if line.strip()]
    
    # 加载得分
    brightness = []
    if args.brightness:
        with open(args.brightness, 'r') as f:
            brightness = [float(line.strip()) for line in f if line.strip()]
    
    # 创建评估器
    evaluator = MutationEvaluator(args.wt_seq)
    
    # 评估
    results = evaluator.evaluate(sequences, brightness)
    
    # 打印报告
    evaluator.print_report(results)
    
    # 保存报告
    evaluator.save_report(results, args.output)


if __name__ == '__main__':
    main()
