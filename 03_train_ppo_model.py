"""
配对交易PPO模型训练脚本
使用稳定基线3（Stable-Baselines3）库训练配对交易策略
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# 导入必要的库
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

# 导入自定义环境
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env.PairsTradingEnv import PairsTradingEnv


class TrainingConfig:
    """训练配置类"""
    
    def __init__(self, **kwargs):
        # 文件路径
        self.data_path = kwargs.get('data_path', './data/pair_NINGDE_BYD.csv')
        self.models_dir = kwargs.get('models_dir', './models')
        self.logs_dir = kwargs.get('logs_dir', './logs/tb_logs')
        self.results_dir = kwargs.get('results_dir', './results')
        
        # 训练参数
        self.total_timesteps = kwargs.get('total_timesteps', 100000)
        self.learning_rate = kwargs.get('learning_rate', 0.0001)
        self.batch_size = kwargs.get('batch_size', 64)
        self.n_steps = kwargs.get('n_steps', 2048)
        self.n_epochs = kwargs.get('n_epochs', 10)
        self.gamma = kwargs.get('gamma', 0.99)
        self.gae_lambda = kwargs.get('gae_lambda', 0.95)
        self.clip_range = kwargs.get('clip_range', 0.2)
        self.ent_coef = kwargs.get('ent_coef', 0.0)
        
        # 数据参数
        self.train_test_split = kwargs.get('train_test_split', 0.8)
        self.initial_balance = kwargs.get('initial_balance', 10000.0)
        self.window_size = kwargs.get('window_size', 10)
        
        # 检查点
        self.checkpoint_interval = kwargs.get('checkpoint_interval', 10000)
        self.test_interval = kwargs.get('test_interval', 50000)
        
        # 其他
        self.seed = kwargs.get('seed', 42)
        self.verbose = kwargs.get('verbose', 1)
    
    def __str__(self):
        return json.dumps(self.__dict__, indent=2)


def create_directories(config):
    """创建必要的目录"""
    for directory in [config.models_dir, config.logs_dir, config.results_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print(f"✓ 创建目录完成")
    print(f"  模型目录: {config.models_dir}")
    print(f"  日志目录: {config.logs_dir}")
    print(f"  结果目录: {config.results_dir}")


def load_and_split_data(config):
    """
    加载数据并按时间顺序划分训练集和测试集
    
    Args:
        config: 训练配置
        
    Returns:
        train_df, test_df: 训练集和测试集DataFrame
    """
    print("\n" + "=" * 70)
    print("数据加载与划分")
    print("=" * 70)
    
    # 加载数据
    try:
        df = pd.read_csv(config.data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        print(f"✓ 加载数据成功: {len(df)} 行")
        print(f"  日期范围: {df['Date'].min().date()} 到 {df['Date'].max().date()}")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        return None, None
    
    # 按时间顺序划分
    split_idx = int(len(df) * config.train_test_split)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"\n✓ 数据划分完成:")
    print(f"  训练集: {len(train_df)} 行 ({config.train_test_split*100:.0f}%)")
    print(f"    日期: {train_df['Date'].min().date()} 到 {train_df['Date'].max().date()}")
    print(f"  测试集: {len(test_df)} 行 ({(1-config.train_test_split)*100:.0f}%)")
    print(f"    日期: {test_df['Date'].min().date()} 到 {test_df['Date'].max().date()}")
    
    return train_df, test_df


def create_env(df, config, env_id='train'):
    """
    创建环境
    
    Args:
        df: 数据集
        config: 训练配置
        env_id: 环境标识（用于区分训练和测试）
        
    Returns:
        vectorized_env: 向量化的环境
    """
    def _make_env():
        env = PairsTradingEnv(
            df=df,
            initial_balance=config.initial_balance,
            window_size=config.window_size
        )
        # 添加Monitor用于记录统计信息
        env = Monitor(env)
        return env
    
    # 创建向量化环境
    vec_env = DummyVecEnv([_make_env])
    
    print(f"✓ 创建{env_id}环境完成")
    
    return vec_env


def train_model(train_env, config):
    """
    训练PPO模型
    
    Args:
        train_env: 训练环境
        config: 训练配置
        
    Returns:
        model: 训练好的模型
    """
    print("\n" + "=" * 70)
    print("模型训练")
    print("=" * 70)
    
    print(f"\n训练配置:")
    print(f"  总步数: {config.total_timesteps:,}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  批大小: {config.batch_size}")
    print(f"  N步: {config.n_steps}")
    print(f"  Gamma: {config.gamma}")
    print(f"  GAE Lambda: {config.gae_lambda}")
    
    # 创建PPO模型
    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        n_steps=config.n_steps,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        tensorboard_log=config.logs_dir,
        verbose=config.verbose,
        seed=config.seed,
    )
    
    print(f"\n✓ 模型创建完成")
    print(f"  策略: MlpPolicy (多层感知机)")
    print(f"  参数数量: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # 设置检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_interval,
        save_path=os.path.join(config.models_dir, 'checkpoints'),
        name_prefix='ppo_pairs_ckpt',
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # 开始训练
    print(f"\n{'=' * 70}")
    print("开始训练...")
    print(f"{'=' * 70}\n")
    
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        print(f"\n✓ 训练完成！")
        return model
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_model(model, config):
    """
    保存模型
    
    Args:
        model: 训练好的模型
        config: 训练配置
    """
    print("\n" + "=" * 70)
    print("保存模型")
    print("=" * 70)
    
    model_path = os.path.join(config.models_dir, 'ppo_pairs_trading')
    
    try:
        model.save(model_path)
        print(f"✓ 模型已保存: {model_path}.zip")
    except Exception as e:
        print(f"✗ 保存失败: {e}")


def test_model(model, test_df, config):
    """
    在测试集上测试模型
    
    Args:
        model: 训练好的模型
        test_df: 测试集
        config: 训练配置
        
    Returns:
        dict: 测试结果指标
    """
    print("\n" + "=" * 70)
    print("模型测试")
    print("=" * 70)
    
    # 创建测试环境
    test_env = PairsTradingEnv(
        df=test_df,
        initial_balance=config.initial_balance,
        window_size=config.window_size
    )
    
    # 运行测试
    obs, _ = test_env.reset(seed=config.seed)
    
    total_reward = 0
    episode_rewards = []
    net_worths = []
    actions_taken = {0: 0, 1: 0, 2: 0}
    
    done = False
    step = 0
    max_steps = len(test_df) - config.window_size
    
    while step < max_steps and not done:
        # 使用模型预测动作
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        actions_taken[action] += 1
        
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        net_worths.append(info['net_worth'])
        
        done = terminated or truncated
        step += 1
    
    # 计算指标
    initial_balance = config.initial_balance
    final_net_worth = test_env.net_worth
    total_return = (final_net_worth - initial_balance) / initial_balance
    
    # 简单的夏普比率（基于每日收益率）
    returns = np.diff(net_worths) / np.array(net_worths[:-1])
    sharpe_ratio = (returns.mean() / (returns.std() + 1e-6)) * np.sqrt(252) if len(returns) > 1 else 0
    
    # 最大回撤
    cummax = np.maximum.accumulate(net_worths)
    drawdown = (cummax - net_worths) / cummax
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
    
    results = {
        'initial_balance': initial_balance,
        'final_net_worth': final_net_worth,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'total_reward': total_reward,
        'steps': step,
        'max_net_worth': max(net_worths),
        'min_net_worth': min(net_worths),
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'action_distribution': {
            '做多价差': actions_taken[0],
            '平仓': actions_taken[1],
            '做空价差': actions_taken[2]
        }
    }
    
    # 打印结果
    print(f"\n测试结果:")
    print(f"  初始资金: ¥{results['initial_balance']:.2f}")
    print(f"  最终净资产: ¥{results['final_net_worth']:.2f}")
    print(f"  总收益率: {results['total_return_pct']:.2f}%")
    print(f"  总累计奖励: {results['total_reward']:.6f}")
    print(f"  运行步数: {results['steps']}")
    print(f"  最大净资产: ¥{results['max_net_worth']:.2f}")
    print(f"  最小净资产: ¥{results['min_net_worth']:.2f}")
    print(f"  夏普比率: {results['sharpe_ratio']:.4f}")
    print(f"  最大回撤: {results['max_drawdown']:.2f}%")
    print(f"\n动作分布:")
    for action_name, count in results['action_distribution'].items():
        pct = count / results['steps'] * 100 if results['steps'] > 0 else 0
        print(f"  {action_name}: {count} ({pct:.1f}%)")
    
    return results


def plot_results(test_df, results, config):
    """
    绘制测试结果
    
    Args:
        test_df: 测试集
        results: 测试结果
        config: 训练配置
    """
    print("\n" + "=" * 70)
    print("生成结果可视化")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 第一个图：净资产变化
    ax1 = axes[0]
    dates = test_df['Date'].values[:results['steps']]
    
    ax1.axhline(y=config.initial_balance, color='gray', linestyle='--', 
               linewidth=1, alpha=0.7, label='初始资金')
    ax1.set_ylabel('净资产（元）', fontsize=11)
    ax1.set_title('测试集 - 净资产变化', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # 第二个图：Z-Score和动作
    ax2 = axes[1]
    z_scores = test_df['zscore'].values[:results['steps']]
    ax2.plot(range(len(z_scores)), z_scores, color='blue', linewidth=1, alpha=0.7)
    ax2.axhline(y=2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='超买/超卖阈值')
    ax2.axhline(y=-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.set_xlabel('时间步', fontsize=11)
    ax2.set_ylabel('Z-Score', fontsize=11)
    ax2.set_title('Z-Score序列（标准化价差）', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    plt.tight_layout()
    
    output_path = os.path.join(config.results_dir, 'test_results.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ 结果图表已保存: {output_path}")
    
    plt.close()


def save_results(results, config):
    """
    保存测试结果为JSON
    
    Args:
        results: 测试结果
        config: 训练配置
    """
    output_path = os.path.join(config.results_dir, 'test_results.json')
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✓ 结果已保存: {output_path}")
    except Exception as e:
        print(f"✗ 保存失败: {e}")


def main():
    """主函数"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='配对交易PPO模型训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train_pairs.py --total-timesteps 100000 --learning-rate 0.0001
  python train_pairs.py --data-path ./data/pair_data.csv --seed 123
        """
    )
    
    # 文件路径参数
    parser.add_argument('--data-path', type=str, default='./data/pair_NINGDE_BYD.csv',
                       help='配对数据文件路径')
    parser.add_argument('--models-dir', type=str, default='./models',
                       help='模型保存目录')
    parser.add_argument('--logs-dir', type=str, default='./logs/tb_logs',
                       help='TensorBoard日志目录')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='结果保存目录')
    
    # 训练参数
    parser.add_argument('--total-timesteps', type=int, default=100000,
                       help='总训练步数')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='学习率')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='批大小')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='N步数')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='折扣因子')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                       help='GAE Lambda')
    
    # 数据参数
    parser.add_argument('--train-test-split', type=float, default=0.8,
                       help='训练集比例')
    parser.add_argument('--initial-balance', type=float, default=10000.0,
                       help='初始账户余额')
    parser.add_argument('--window-size', type=int, default=10,
                       help='观察窗口大小')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--verbose', type=int, default=1,
                       help='日志详细程度')
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainingConfig(**vars(args))
    
    # 打印配置
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " 配对交易PPO模型训练 ".center(68) + "║")
    print("╚" + "=" * 68 + "╝")
    print("\n训练配置:")
    print(config)
    
    # 创建目录
    create_directories(config)
    
    # 加载和划分数据
    train_df, test_df = load_and_split_data(config)
    if train_df is None or test_df is None:
        return
    
    # 创建训练环境
    train_env = create_env(train_df, config, env_id='训练')
    
    # 训练模型
    model = train_model(train_env, config)
    if model is None:
        return
    
    # 保存模型
    save_model(model, config)
    
    # 在测试集上测试
    results = test_model(model, test_df, config)
    
    # 绘制结果
    plot_results(test_df, results, config)
    
    # 保存结果
    save_results(results, config)
    
    # 总结
    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)
    print(f"\n📁 输出文件:")
    print(f"  模型: {os.path.join(config.models_dir, 'ppo_pairs_trading.zip')}")
    print(f"  结果: {os.path.join(config.results_dir, 'test_results.json')}")
    print(f"  图表: {os.path.join(config.results_dir, 'test_results.png')}")
    print(f"  日志: {config.logs_dir}")
    print(f"\n📊 查看TensorBoard日志:")
    print(f"  tensorboard --logdir {config.logs_dir}")
    print()


if __name__ == "__main__":
    main()
