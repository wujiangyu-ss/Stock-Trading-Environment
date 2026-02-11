import os
import json
import time
import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from env.PairsTradingEnv import PairsTradingEnv

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
CONFIG_DIR = os.path.join(ROOT, 'config')
MODELS_DIR = os.path.join(ROOT, 'models')
LOGS_DIR = os.path.join(ROOT, 'logs', 'retrain_tb')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, 'train_dataset.csv')
VAL_CSV = os.path.join(DATA_DIR, 'val_dataset.csv')
TEST_CSV = os.path.join(DATA_DIR, 'test_dataset.csv')
STATS_JSON = os.path.join(CONFIG_DIR, 'train_stats.json')

TOTAL_TIMESTEPS = 100_000
EVAL_INTERVAL = 10_000


class SaveBestCallback(BaseCallback):
    def __init__(self, val_env, save_path, verbose=1):
        super().__init__(verbose)
        self.val_env = val_env
        self.save_path = save_path
        self.best_score = -np.inf

    def _on_step(self) -> bool:
        return True

    def evaluate_and_maybe_save(self, model):
        # 运行一次完整的验证集评估（从start_index=0开始），返回最终 net_worth
        env = self.val_env
        obs, info = env.reset(options={'start_index': 0})
        done = False
        truncated = False
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        final_net = info.get('net_worth', None)
        score = float(final_net) if final_net is not None else -np.inf
        if score > self.best_score:
            self.best_score = score
            model.save(self.save_path)
            if self.verbose:
                print(f"Saved new best model with val net_worth={score:.2f} to {self.save_path}")
        return score


def make_env(df, mode, frozen_stats):
    return PairsTradingEnv(df.copy().reset_index(drop=True), initial_balance=10000.0, window_size=10,
                           use_features=['zscore', 'spread_rm', 'spread_rstd'], train_stats=None,
                           mode=mode, frozen_stats=frozen_stats)


def load_data():
    train_df = pd.read_csv(TRAIN_CSV, parse_dates=['Date'])
    val_df = pd.read_csv(VAL_CSV, parse_dates=['Date'])
    test_df = pd.read_csv(TEST_CSV, parse_dates=['Date'])
    with open(STATS_JSON, 'r', encoding='utf-8') as f:
        frozen_stats = json.load(f)
    return train_df, val_df, test_df, frozen_stats


def evaluate_on_env(model, env):
    obs, info = env.reset(options={'start_index': 0})
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    return info


def main():
    train_df, val_df, test_df, frozen_stats = load_data()

    train_env = DummyVecEnv([lambda: make_env(train_df, mode='train', frozen_stats=frozen_stats)])
    val_env = make_env(val_df, mode='val', frozen_stats=frozen_stats)
    test_env = make_env(test_df, mode='test', frozen_stats=frozen_stats)

    model = PPO('MlpPolicy', train_env, verbose=1, tensorboard_log=LOGS_DIR, learning_rate=1e-4)

    saver = SaveBestCallback(val_env, os.path.join(MODELS_DIR, 'retrained_clean_model'))

    timesteps_done = 0
    while timesteps_done < TOTAL_TIMESTEPS:
        to_learn = min(EVAL_INTERVAL, TOTAL_TIMESTEPS - timesteps_done)
        model.learn(total_timesteps=to_learn)
        timesteps_done += to_learn
        # Evaluate on val_env
        score = saver.evaluate_and_maybe_save(model)
        print(f"After {timesteps_done} timesteps, val final net_worth = {score}")

    # Load best model if exists
    best_path = os.path.join(MODELS_DIR, 'retrained_clean_model.zip')
    best_base = os.path.join(MODELS_DIR, 'retrained_clean_model')
    if os.path.exists(best_base + '.zip'):
        print('Loading best model from', best_base)
        model = PPO.load(best_base, env=train_env)
        model.save(os.path.join(MODELS_DIR, 'retrained_clean_model'))
    else:
        # save current model
        model.save(os.path.join(MODELS_DIR, 'retrained_clean_model'))

    # Final evaluation on test (no learning)
    final_info = evaluate_on_env(model, test_env)
    print('Final evaluation on test set:')
    print(final_info)
    print('Training complete. Total timesteps:', timesteps_done)


if __name__ == '__main__':
    main()
