import pandas as pd
import json
import argparse
import os


def analyze_candidates(csv_path: str):
    """分析RL候选序列统计信息"""
    df = pd.read_csv(csv_path)
    
    print(f'\n{"="*60}')
    print('RL候选序列统计分析')
    print(f'{"="*60}')
    print(f'候选总数: {len(df)}')
    print()
    
    if 'strategy' in df.columns:
        print(df.groupby('strategy').agg({
            'n_mutations': 'mean',
            'brightness': 'mean',
            'reward': 'mean',
            'sequence': 'count'
        }))
        print()
    
    print(f'最高亮度: {df["brightness"].max():.4f}')
    print(f'最低亮度: {df["brightness"].min():.4f}')
    print(f'平均亮度: {df["brightness"].mean():.4f}')
    print(f'平均突变数: {df["n_mutations"].mean():.1f}')
    print(f'突变数范围: {df["n_mutations"].min()} - {df["n_mutations"].max()}')


def analyze_training_stats(json_path: str):
    """分析RL训练统计信息"""
    data = json.load(open(json_path))
    
    print(f'\n{"="*60}')
    print('RL训练统计分析')
    print(f'{"="*60}')
    print(f'总轮数: {len(data["episodes"])}')
    print(f'前10轮奖励: {data["rewards"][:10]}')
    print(f'最后10轮奖励: {data["rewards"][-10:]}')
    print(f'最高奖励: {max(data["rewards"]):.2f}')
    print(f'最低奖励: {min(data["rewards"]):.2f}')
    print(f'平均奖励: {sum(data["rewards"])/len(data["rewards"]):.2f}')
    
    if len(data["rewards"]) >= 50:
        print(f'前50轮平均: {sum(data["rewards"][:50])/50:.2f}')
        print(f'后50轮平均: {sum(data["rewards"][-50:])/50:.2f}')
    
    print(f'突变数: {set(data["n_mutations"])}')
    print(f'最高亮度: {max(data["brightness"]):.4f}')
    print(f'最低亮度: {min(data["brightness"]):.4f}')
    print(f'平均亮度: {sum(data["brightness"])/len(data["brightness"]):.4f}')


def main():
    parser = argparse.ArgumentParser(description='RL结果分析工具')
    parser.add_argument('--mode', type=str, choices=['candidates', 'stats', 'all'], 
                       default='all', help='分析模式')
    parser.add_argument('--candidates', type=str, 
                       default='output/rl_results/rl_candidates_with_scores.csv',
                       help='候选序列CSV文件路径')
    parser.add_argument('--stats', type=str, 
                       default='output/rl_results/rl_training_stats.json',
                       help='训练统计JSON文件路径')
    
    args = parser.parse_args()
    
    if args.mode in ['candidates', 'all']:
        if os.path.exists(args.candidates):
            analyze_candidates(args.candidates)
        else:
            print(f'警告: 候选序列文件不存在: {args.candidates}')
    
    if args.mode in ['stats', 'all']:
        if os.path.exists(args.stats):
            analyze_training_stats(args.stats)
        else:
            print(f'警告: 训练统计文件不存在: {args.stats}')


if __name__ == '__main__':
    main()