"""
基于分区数据的训练脚本（严格避免数据泄露）
- 输入：train_csv, val_csv（可选，用于监控/早停）, test_csv, train_stats.json
- 在训练时仅使用训练集数据计算模型和归一化参数
- 评估时在测试集上使用训练集统计量进行归一化
"""
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.PairsTradingEnv import PairsTradingEnv

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback


def make_vec_env_from_csv(train_csv, train_stats, initial_balance=10000.0, window_size=10):
    train_df = pd.read_csv(train_csv)
    train_df['Date'] = pd.to_datetime(train_df['Date'])

    def _make_env():
        env = PairsTradingEnv(df=train_df, initial_balance=initial_balance, window_size=window_size, train_stats=train_stats)
        env = Monitor(env)
        return env
    return DummyVecEnv([_make_env])


def evaluate_model(model, test_csv, train_stats, initial_balance=10000.0, window_size=10):
    test_df = pd.read_csv(test_csv)
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    env = PairsTradingEnv(df=test_df, initial_balance=initial_balance, window_size=window_size, train_stats=train_stats)

    obs, _ = env.reset(seed=42)
    net_worths = []
    steps = 0
    done = False
    max_steps = len(test_df) - window_size
    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        net_worths.append(info['net_worth'])
        steps += 1
        done = terminated or truncated

    if len(net_worths) < 2:
        return None

    initial_balance = initial_balance
    final_net = env.net_worth
    returns = np.diff(net_worths) / np.array(net_worths[:-1])
    sharpe = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if len(returns) > 1 else 0
    return {
        'initial_balance': initial_balance,
        'final_net_worth': final_net,
        'total_return_pct': (final_net - initial_balance) / initial_balance * 100,
        'sharpe': sharpe,
        'steps': steps
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--val-csv', required=False)
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--train-stats', required=True)
    parser.add_argument('--models-dir', default='./models/partitioned')
    parser.add_argument('--logs-dir', default='./logs/tb_logs/partitioned')
    parser.add_argument('--results-dir', default='./results/partitioned')
    parser.add_argument('--total-timesteps', type=int, default=100000)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--window-size', type=int, default=10)
    parser.add_argument('--initial-balance', type=float, default=10000.0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    with open(args.train_stats, 'r', encoding='utf-8') as f:
        train_stats = json.load(f)

    # 创建训练环境（仅基于train_csv）
    vec_env = make_vec_env_from_csv(args.train_csv, train_stats, initial_balance=args.initial_balance, window_size=args.window_size)

    model = PPO('MlpPolicy', vec_env, learning_rate=args.learning_rate, tensorboard_log=args.logs_dir, verbose=1, seed=args.seed)

    checkpoint_cb = CheckpointCallback(save_freq=10000, save_path=os.path.join(args.models_dir,'checkpoints'), name_prefix='ppo_part')

    model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_cb)

    model_path = os.path.join(args.models_dir, 'ppo_partitioned')
    model.save(model_path)
    print('Saved model to', model_path + '.zip')

    # 评估
    results = evaluate_model(model, args.test_csv, train_stats, initial_balance=args.initial_balance, window_size=args.window_size)
    print('Evaluation on test set:', results)

    # 保存结果
    res_path = os.path.join(args.results_dir, 'eval_results.json')
    with open(res_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('Saved results to', res_path)

if __name__ == '__main__':
    main()
