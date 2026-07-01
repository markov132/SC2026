"""
SC2026 蛋白设计赛道 - RL训练脚本
==================================

使用PPO强化学习探索蛋白质突变空间

运行方式:
python run_rl.py --rl_episodes 1000 --max_mutations 8

输出:
- ppo_protein_model.pth: 训练好的PPO模型
- rl_results.json: 训练结果
- rl_candidates.csv: 生成的候选序列

奖励函数:
详细定义见 rl_env.py 的 _compute_reward 方法
- 基础亮度奖励: (brightness + 2.0) * 3.0
- 亮度提升奖励: improvement * 2.0
- 突变惩罚: n_mutations * 0.5
- 终止奖励: 根据突变数量给予5-30奖励
"""

import argparse
import json
import numpy as np
import torch
from mutation_evaluator import MutationEvaluator
import pandas as pd
import os
import sys
from typing import List, Optional, Any

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rl_env import ProteinDesignEnv
from ppo_agent import PPOAgent, train_ppo
from seq_builder import compare_sequences
from config import format_time


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="PPO蛋白质设计训练")
    
    parser.add_argument(
        '--rl_episodes', '--n_episodes',
        type=int,
        default=500,
        dest='rl_episodes',
        help='RL训练轮数（建议至少500轮）'
    )
    
    parser.add_argument(
        '--max_mutations',
        type=int,
        default=8,
        help='最大突变数'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='批次大小'
    )
    
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=10,
        help='每轮更新次数'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=3e-4,
        help='学习率'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output',
        help='输出目录'
    )
    
    parser.add_argument(
        '--exclusion_list',
        type=str,
        default=None,
        help='排除列表CSV文件路径'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='ppo_protein_model.pth',
        help='模型保存路径'
    )

    parser.add_argument(
        '--brightness_model',
        type=str,
        default='output/brightness_model/brightness_model.pth',
        help='亮度模型路径（不指定则使用启发式评分）'
    )

    parser.add_argument(
        '--start_seqs',
        type=str,
        default=None,
        help='指定起始序列名称，逗号分隔（如 WT,PIII-4），不指定则使用全部'
    )

    parser.add_argument(
        '--custom_seq',
        type=str,
        default=None,
        help='直接提供自定义起始序列（如 "MSKGEELFT..."）'
    )

    parser.add_argument(
        '--start_file',
        type=str,
        default=None,
        help='自定义起始序列文件路径'
    )

    parser.add_argument(
        '--n_candidates',
        type=int,
        default=6,
        help='每个起始序列生成的候选序列数量（默认6）'
    )

    return parser.parse_args()


def generate_candidates(
    env: ProteinDesignEnv,
    agent: PPOAgent,
    n_candidates: int = 6,
    exclusion_checker=None
) -> List[dict]:
    """
    使用训练好的策略生成候选序列（带多样性保证和排除列表检查）
    
    参数:
        env: 环境
        agent: PPO代理
        n_candidates: 生成候选数量
        exclusion_checker: ExclusionListChecker实例，可选
    
    返回:
        候选序列列表，每个元素包含 sequence 和 brightness
    """
    candidates = []
    seen_sequences = set()
    excluded_count = 0
    duplicate_count = 0
    
    # 大幅增加最大尝试次数，确保即使排除很多序列也能生成足够的候选
    max_attempts = n_candidates * 200
    attempt = 0
    
    while attempt < max_attempts:
        if len(candidates) >= n_candidates:
            break
        
        state = env.reset()
        state_vec = env.get_state_vector(state)
        
        for step in range(env.max_mutations):
            # 提高随机策略比例以增加多样性（每2次尝试中1次使用随机策略）
            use_stochastic = (attempt % 2 == 0)
            action_idx, _, _ = agent.get_action(state_vec, deterministic=not use_stochastic)
            action = env.index_to_action(action_idx)
            state, _, done, info = env.step(action)
            state_vec = env.get_state_vector(state)
            
            if done:
                break
        
        seq = state.sequence
        
        # 检查是否在排除列表中
        is_excluded = False
        if exclusion_checker is not None:
            is_excluded = exclusion_checker.is_excluded(seq)
        
        # 确保序列唯一且不在排除列表中
        if seq not in seen_sequences and not is_excluded:
            seen_sequences.add(seq)
            candidates.append({
                'sequence': seq,
                'n_mutations': state.n_mutations,
                'diversity_seed': attempt
            })
        elif is_excluded:
            excluded_count += 1
        else:
            duplicate_count += 1
        
        attempt += 1
        
        # 如果快用完了还不够，自动扩展尝试次数
        if len(candidates) < n_candidates and attempt == max_attempts - 1:
            max_attempts += n_candidates * 50
            print(f"  [扩展] 候选不足（当前{len(candidates)}/{n_candidates}），重复{duplicate_count}条，排除{excluded_count}条，继续尝试...")
    
    # 输出统计信息
    print(f"  生成完成: {len(candidates)} 条候选（尝试 {attempt} 次，重复 {duplicate_count} 次，排除 {excluded_count} 次）")
    
    return candidates


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 读取起始序列
    start_sequences = []
    start_names = []
    
    # 优先级：自定义序列 > 自定义文件 > 默认文件
    if args.custom_seq:
        # 使用命令行提供的自定义序列
        custom_seq = args.custom_seq.strip().replace(" ", "")
        if custom_seq:
            start_sequences = [custom_seq]
            start_names = ["custom"]
        else:
            print("错误：自定义序列为空")
            sys.exit(1)
    else:
        # 确定起始序列文件路径
        start_file = args.start_file if args.start_file else \
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "SC突变起始序列.txt")
        
        if os.path.exists(start_file):
            with open(start_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            current_name = None
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("WT"):
                    current_name = "WT"
                elif line.startswith("PIII"):
                    current_name = "PIII-4"
                elif line.startswith("FPbase"):
                    current_name = "FPbase"
                elif line.startswith("M") and current_name:
                    seq = line.replace(" ", "")
                    start_sequences.append(seq)
                    start_names.append(current_name)
                    current_name = None
        else:
            print("警告：未找到起始序列文件，使用默认sfGFP序列")
            start_sequences = ["MSKGEELFTGVVPILVELDGDVNGHKFSVRGEGEGDATNGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTISFKDDGTYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNFNSHNVYITADKQKNGIKANFKIRHNIVEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSVLSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"]
            start_names = ["sfGFP"]
    
    # 如果指定了要选择的序列名称，进行过滤
    if args.start_seqs:
        selected_names = [name.strip() for name in args.start_seqs.split(",")]
        filtered_names = []
        filtered_seqs = []
        for name, seq in zip(start_names, start_sequences):
            if name in selected_names:
                filtered_names.append(name)
                filtered_seqs.append(seq)
        
        if filtered_names:
            start_names = filtered_names
            start_sequences = filtered_seqs
        else:
            print(f"错误：未找到指定的起始序列: {args.start_seqs}")
            print(f"可用序列: {start_names}")
            sys.exit(1)
    
    print("=" * 60)
    print(f"检测到 {len(start_sequences)} 条起始序列")
    print("=" * 60)
    
    # 过滤长度超过240 aa的序列
    max_seq_length = 240
    filtered_names = []
    filtered_seqs = []
    for i, (name, seq) in enumerate(zip(start_names, start_sequences)):
        if len(seq) <= max_seq_length:
            filtered_names.append(name)
            filtered_seqs.append(seq)
            print(f"  {i+1}. {name}: {len(seq)} aa")
        else:
            print(f"  {i+1}. {name}: {len(seq)} aa (超过限制 {max_seq_length} aa，已跳过)")
    
    if filtered_seqs:
        start_names = filtered_names
        start_sequences = filtered_seqs
    
    # 加载排除列表
    exclusion_checker = None
    if args.exclusion_list and os.path.exists(args.exclusion_list):
        from exclusion_checker import ExclusionListChecker
        exclusion_checker = ExclusionListChecker()
        exclusion_checker.load_from_csv(args.exclusion_list)
        print(f"\n排除列表已加载，共 {exclusion_checker.get_size()} 条序列")
    elif args.exclusion_list:
        print(f"\n警告：排除列表文件不存在: {args.exclusion_list}")
    else:
        print("  错误: 没有符合长度限制的序列")
        sys.exit(1)
    
    # 获取最大序列长度作为统一状态维度
    max_seq_length = max(len(seq) for seq in start_sequences)
    print(f"  最大序列长度: {max_seq_length} aa")

    # 存储所有候选序列
    all_results = []

    # 加载亮度模型（全局单例，与pipeline.py保持一致）
    brightness_model_path = args.brightness_model
    if brightness_model_path and brightness_model_path.lower() != 'none':
        if os.path.exists(brightness_model_path):
            print(f"加载亮度模型: {brightness_model_path}")
            ProteinDesignEnv.load_brightness_model(brightness_model_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"亮度模型加载成功，设备: {device}")
        else:
            print(f"警告: 未找到亮度模型 ({brightness_model_path})")
            print("错误: 必须提供有效的亮度模型路径")
            sys.exit(1)
    else:
        print("错误: 必须提供亮度模型路径")
        sys.exit(1)
    
    # 创建第一个环境来获取状态和动作维度
    # 通过共享ESM提取器获取实际嵌入维度（单例机制，不会重复创建）
    extractor = ProteinDesignEnv._get_esm_extractor()
    actual_embedding_dim = extractor.embedding_dim
    
    first_env = ProteinDesignEnv(
        wt_sequence=start_sequences[0],
        max_mutations=args.max_mutations,
        reference_length=max_seq_length,
        embedding_dim=actual_embedding_dim
    )
    state_dim = first_env.state_dim
    
    # 计算最大动作空间（考虑所有序列）
    max_actions = max(len(seq) * 20 for seq in start_sequences)
    action_dim = max_actions
    
    # 创建共享的PPO代理（所有起始序列共用一个策略网络）
    print("\n创建共享PPO代理...")
    shared_agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr
    )
    
    # 确定PIII-4的权重（偏向于PIII-4）
    # PIII-4是性能最好的起始序列（亮度最高），给予更高训练权重：
    # 1. PIII-4的突变策略更有价值，能生成更高亮度的突变体
    # 2. 更多训练样本可以学到更好的突变模式
    # 3. 权重=2.0意味着约2/3的训练轮次使用PIII-4
    piii4_weight = 2.0  # PIII-4的训练权重是其他序列的2倍
    total_weight = len(start_sequences) + (piii4_weight - 1)
    
    # 共享训练：轮流在不同起始序列上训练共享策略
    print("\n" + "=" * 60)
    print(f"开始共享策略训练 ({args.rl_episodes} 轮)")
    print(f"序列权重: PIII-4 x{piii4_weight}, 其他 x1.0")
    print("=" * 60)
    
    # 进度计时
    import time
    start_time = time.time()
    last_report_time = start_time
    last_report_episode = 0
    episode_rewards = []
    
    for episode in range(args.rl_episodes):
        # 根据权重选择训练序列
        if episode % int(total_weight) < piii4_weight:
            seq_idx = start_names.index('PIII-4') if 'PIII-4' in start_names else np.random.randint(len(start_sequences))
        else:
            # 从非PIII-4序列中选择
            non_piii4_indices = [i for i, name in enumerate(start_names) if name != 'PIII-4']
            if non_piii4_indices:
                seq_idx = non_piii4_indices[episode % len(non_piii4_indices)]
            else:
                seq_idx = 0
        
        start_name = start_names[seq_idx]
        start_seq = start_sequences[seq_idx]
        
        # 创建环境（训练循环中）
        env = ProteinDesignEnv(
            wt_sequence=start_seq,
            max_mutations=args.max_mutations,
            reference_length=max_seq_length,
            embedding_dim=actual_embedding_dim
        )
        
        # 收集轨迹并更新策略
        state = env.reset()
        episode_reward = 0.0
        
        for _ in range(args.max_mutations):
            # 将RLState转换为状态向量
            state_vec = env.get_state_vector(state)
            action_idx, log_prob, value = shared_agent.get_action(state_vec)
            action = env.index_to_action(action_idx)
            next_state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            
            # 将经验推入缓冲区（使用封装方法，包含TD误差估计）
            shared_agent.push_transition(state_vec, action_idx, reward, log_prob, value, done)
            
            state = next_state
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        # 更新共享策略
        shared_agent.update(batch_size=args.batch_size)
        
        # 进度计时和报告（每10个episode报告一次）
        current_time = time.time()
        if (episode + 1) % 10 == 0 or episode == 0:
            elapsed = current_time - start_time
            episodes_done = episode + 1 - last_report_episode
            time_since_last = current_time - last_report_time
            
            if episodes_done > 0 and time_since_last > 0:
                speed = episodes_done / time_since_last  # episodes/second
                remaining_episodes = args.rl_episodes - (episode + 1)
                remaining_time = remaining_episodes / speed if speed > 0 else 0
                
                print(f"\nEpisode {episode+1}/{args.rl_episodes} | Seq: {start_name}")
                print(f"  进度: {(episode+1)/args.rl_episodes*100:.1f}% | "
                      f"已用: {format_time(elapsed)} | "
                      f"剩余: ~{format_time(remaining_time)} | "
                      f"速度: {speed:.1f} episode/秒")
                
                # 显示奖励统计
                if len(episode_rewards) >= 10:
                    recent_rewards = episode_rewards[-10:]
                    avg_reward = np.mean(recent_rewards)
                    max_reward = max(recent_rewards)
                    min_reward = min(recent_rewards)
                    print(f"  奖励: 平均={avg_reward:.2f}, 最大={max_reward:.2f}, 最小={min_reward:.2f}")
                elif episode_rewards:
                    print(f"  奖励: 当前={episode_rewards[-1]:.2f}")
                
                last_report_time = current_time
                last_report_episode = episode + 1
    
    # 训练完成总时间
    total_time = time.time() - start_time
    
    print(f"\n训练完成！总用时: {format_time(total_time)}")
    
    # 保存共享模型
    shared_model_path = os.path.join(args.output_dir, 'ppo_model_shared.pth')
    shared_agent.save(shared_model_path)
    print(f"\n共享模型已保存到: {shared_model_path}")
    
    # 使用共享策略为每条起始序列生成候选
    for seq_idx, (start_name, start_seq) in enumerate(zip(start_names, start_sequences)):
        print("\n" + "=" * 60)
        print(f"处理起始序列 {seq_idx+1}/{len(start_sequences)}: {start_name}")
        print("=" * 60)
        
        # 创建环境（候选生成）
        env = ProteinDesignEnv(
            wt_sequence=start_seq,
            max_mutations=args.max_mutations,
            reference_length=max_seq_length,
            embedding_dim=actual_embedding_dim
        )
        
        print(f"序列长度: {env.seq_length}")
        print(f"动作空间大小: {env.n_actions}")
        print(f"状态空间大小: {env.state_dim}")
        
        # 生成候选序列（使用共享代理）
        print(f"\n生成候选序列...")
        candidates = generate_candidates(env, shared_agent, n_candidates=args.n_candidates, exclusion_checker=exclusion_checker)
        
        # 评估候选序列
        for i, cand in enumerate(candidates):
            seq = cand['sequence']
            
            try:
                # 使用环境中的亮度模型评估（Z-score归一化后的值）
                brightness_zscore = float(ProteinDesignEnv._brightness_model.predict(
                    ProteinDesignEnv._shared_esm_extractor.extract(seq).reshape(1, -1),
                    normalization=ProteinDesignEnv._brightness_normalization,
                    return_raw=True
                )[0])
                
                # 完整反归一化到原始尺度
                norm_stats = ProteinDesignEnv._brightness_normalization
                brightness_raw = brightness_zscore * norm_stats['std'] + norm_stats['mean']
                
                # 如果使用了log1p变换，进行逆变换
                if norm_stats.get('use_log1p', False):
                    brightness_raw = np.expm1(brightness_raw)
                    brightness_raw = max(brightness_raw, 0.0)
                
                brightness = float(brightness_raw)
            except Exception as e:
                brightness = 3.0  # 默认中等亮度
            
            # 分析突变位点
            mutations = compare_sequences(start_seq, seq)
            mutation_sites = ', '.join(mutations)
            
            all_results.append({
                'seq_id': f'{start_name}_candidate_{i+1}',
                'start_sequence': start_name,
                'sequence': seq,
                'n_mutations': cand['n_mutations'],
                'mutation_sites': mutation_sites,
                'brightness': brightness
            })
            
            print(f"  {start_name}_candidate_{i+1}: B={brightness:.3f}, mutations={mutation_sites}")
    
    # 保存所有候选序列
    candidates_df = pd.DataFrame(all_results)
    candidates_path = os.path.join(args.output_dir, 'rl_candidates_all.csv')
    candidates_df.to_csv(candidates_path, index=False)
    print(f"\n所有候选序列已保存到: {candidates_path}")
    
    # 使用评估器评估结果
    print("\n" + "=" * 60)
    print("突变结果综合评估")
    print("=" * 60)
    
    # 获取野生型序列（第一条起始序列）
    wt_sequence = start_sequences[0]
    
    evaluator = MutationEvaluator(wt_sequence)
    
    # 提取数据
    sequences = [r['sequence'] for r in all_results]
    brightness_scores = [r['brightness'] for r in all_results]
    
    # 评估
    evaluation_results = evaluator.evaluate(sequences, brightness_scores)
    
    # 打印报告
    evaluator.print_report(evaluation_results)
    
    # 保存评估报告
    report_path = os.path.join(args.output_dir, 'mutation_evaluation_report.json')
    evaluator.save_report(evaluation_results, report_path)
    
    # 打印最佳候选
    best_candidate = max(all_results, key=lambda x: x['brightness'])
    print("\n" + "=" * 60)
    print("最佳候选序列")
    print("=" * 60)
    print(f"ID: {best_candidate['seq_id']}")
    print(f"起始序列: {best_candidate['start_sequence']}")
    print(f"突变数: {best_candidate['n_mutations']}")
    print(f"亮度: {best_candidate['brightness']:.3f}")


if __name__ == "__main__":
    main()