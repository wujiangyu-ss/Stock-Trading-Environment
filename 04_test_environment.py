"""
配对交易环境测试脚本
验证PairsTradingEnv能否正常运行
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.PairsTradingEnv import PairsTradingEnv


def test_environment_basic():
    """测试环境的基本功能"""
    print("=" * 70)
    print("配对交易环境测试")
    print("=" * 70)
    
    # 加载配对数据
    print("\n【第一步】加载配对数据...")
    try:
        df = pd.read_csv('./data/pair_NINGDE_BYD.csv')
        print(f"✓ 加载成功: {len(df)} 行数据")
        print(f"  数据列: {list(df.columns)}")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return False
    
    # 创建环境
    print("\n【第二步】创建环境...")
    try:
        env = PairsTradingEnv(
            df=df,
            initial_balance=10000.0,
            window_size=10
        )
        print(f"✓ 环境创建成功")
        print(f"  动作空间: {env.action_space}")
        print(f"  观察空间: {env.observation_space}")
    except Exception as e:
        print(f"✗ 创建失败: {e}")
        return False
    
    # 重置环境
    print("\n【第三步】重置环境...")
    try:
        obs, info = env.reset(seed=42)
        print(f"✓ 重置成功")
        print(f"  初始观察形状: {obs.shape}")
        print(f"  初始净资产: ¥{env.net_worth:.2f}")
    except Exception as e:
        print(f"✗ 重置失败: {e}")
        return False
    
    # 运行几步
    print("\n【第四步】运行几步交互...")
    try:
        action_names = {0: "做多价差", 1: "平仓", 2: "做空价差"}
        
        total_reward = 0
        for step in range(5):
            # 随机选择动作
            action = np.random.randint(0, 3)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            print(f"\n  步骤 {step + 1}:")
            print(f"    动作: {action_names[action]}")
            print(f"    奖励: {reward:.6f}")
            print(f"    净资产: ¥{info['net_worth']:.2f}")
            print(f"    股票1头寸: {info['stock1_shares']} 股")
            print(f"    股票2头寸: {info['stock2_shares']} 股")
            
            if terminated or truncated:
                print(f"\n  环境结束 (terminated={terminated}, truncated={truncated})")
                break
        
        print(f"\n✓ 交互成功，总累计奖励: {total_reward:.6f}")
        
    except Exception as e:
        print(f"✗ 交互失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_full_episode():
    """运行一个完整的回合"""
    print("\n" + "=" * 70)
    print("运行完整回合测试（200步）")
    print("=" * 70)
    
    # 加载数据和创建环境
    df = pd.read_csv('./data/pair_NINGDE_BYD.csv')
    env = PairsTradingEnv(df=df, initial_balance=10000.0, window_size=10)
    
    # 重置
    obs, _ = env.reset(seed=42)
    
    # 记录指标
    episode_reward = 0
    episode_trades = 0
    max_net_worth = env.net_worth
    min_net_worth = env.net_worth
    
    # 运行回合
    for step in range(200):
        # 简单的随机策略（50%概率平仓，25%做多，25%做空）
        rand = np.random.random()
        if rand < 0.5:
            action = 1  # 平仓
        elif rand < 0.75:
            action = 0  # 做多
        else:
            action = 2  # 做空
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        episode_reward += reward
        episode_trades += 1
        max_net_worth = max(max_net_worth, info['net_worth'])
        min_net_worth = min(min_net_worth, info['net_worth'])
        
        if (step + 1) % 50 == 0:
            print(f"\n  步骤 {step + 1}/200:")
            print(f"    净资产: ¥{info['net_worth']:.2f}")
            print(f"    累计奖励: {episode_reward:.6f}")
        
        if terminated or truncated:
            print(f"\n  环境在步骤 {step + 1} 结束")
            break
    
    # 汇总
    print("\n" + "-" * 70)
    print("回合统计:")
    print(f"  起始净资产: ¥{env.initial_balance:.2f}")
    print(f"  最终净资产: ¥{info['net_worth']:.2f}")
    print(f"  最大净资产: ¥{max_net_worth:.2f}")
    print(f"  最小净资产: ¥{min_net_worth:.2f}")
    print(f"  总收益率: {(info['net_worth'] - env.initial_balance) / env.initial_balance * 100:.2f}%")
    print(f"  总步数: {episode_trades}")
    print(f"  累计奖励: {episode_reward:.6f}")
    print(f"  平均步奖励: {episode_reward / episode_trades:.6f}")
    print(f"  交易次数: {info['stock1_shares']} (股票1)")
    
    return True


def test_action_validation():
    """测试动作的合理性"""
    print("\n" + "=" * 70)
    print("动作逻辑验证测试")
    print("=" * 70)
    
    df = pd.read_csv('./data/pair_NINGDE_BYD.csv')
    env = PairsTradingEnv(df=df, initial_balance=10000.0, window_size=10)
    
    obs, _ = env.reset(seed=42)
    
    print("\n初始状态:")
    print(f"  余额: ¥{env.balance:.2f}")
    print(f"  股票1头寸: {env.stock1_shares}")
    print(f"  股票2头寸: {env.stock2_shares}")
    print(f"  净资产: ¥{env.net_worth:.2f}")
    
    # 测试做多
    print("\n【做多价差】执行动作0...")
    obs, reward, _, _, info = env.step(0)
    print(f"  执行后:")
    print(f"    股票1头寸: {info['stock1_shares']} (多头)")
    print(f"    股票2头寸: {info['stock2_shares']} (空头)")
    print(f"    余额: ¥{info['balance']:.2f}")
    print(f"    净资产: ¥{info['net_worth']:.2f}")
    
    # 保存做多时的净资产
    long_net_worth = info['net_worth']
    
    # 测试平仓
    print("\n【平仓】执行动作1...")
    obs, reward, _, _, info = env.step(1)
    print(f"  执行后:")
    print(f"    股票1头寸: {info['stock1_shares']}")
    print(f"    股票2头寸: {info['stock2_shares']}")
    print(f"    余额: ¥{info['balance']:.2f}")
    print(f"    净资产: ¥{info['net_worth']:.2f}")
    
    # 测试做空
    print("\n【做空价差】执行动作2...")
    obs, reward, _, _, info = env.step(2)
    print(f"  执行后:")
    print(f"    股票1头寸: {info['stock1_shares']} (空头)")
    print(f"    股票2头寸: {info['stock2_shares']} (多头)")
    print(f"    余额: ¥{info['balance']:.2f}")
    print(f"    净资产: ¥{info['net_worth']:.2f}")
    
    # 再次平仓
    print("\n【再次平仓】执行动作1...")
    obs, reward, _, _, info = env.step(1)
    print(f"  执行后:")
    print(f"    股票1头寸: {info['stock1_shares']}")
    print(f"    股票2头寸: {info['stock2_shares']}")
    print(f"    余额: ¥{info['balance']:.2f}")
    print(f"    净资产: ¥{info['net_worth']:.2f}")
    
    print("\n✓ 动作逻辑验证完成")
    return True


def main():
    """主函数"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " 配对交易强化学习环境测试套件 ".center(68) + "║")
    print("╚" + "=" * 68 + "╝")
    
    # 运行测试
    success = True
    
    if not test_environment_basic():
        success = False
    
    if success and not test_action_validation():
        success = False
    
    if success and not test_full_episode():
        success = False
    
    # 总结
    print("\n" + "=" * 70)
    if success:
        print("✓ 所有测试通过！")
        print("\n配对交易环境可以使用。接下来可以：")
        print("1. 集成到强化学习训练脚本中")
        print("2. 使用PPO等算法训练交易策略")
        print("3. 评估和回测策略表现")
    else:
        print("✗ 部分测试失败，请检查错误信息")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
