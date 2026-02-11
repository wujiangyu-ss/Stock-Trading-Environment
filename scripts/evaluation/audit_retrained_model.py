import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from env.PairsTradingEnv import PairsTradingEnv

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
CONFIG_DIR = os.path.join(ROOT, 'config')
MODELS_DIR = os.path.join(ROOT, 'models')
RESULTS_DIR = os.path.join(ROOT, 'results', 'audit')
os.makedirs(RESULTS_DIR, exist_ok=True)

TEST_CSV = os.path.join(DATA_DIR, 'test_dataset.csv')
STATS_JSON = os.path.join(CONFIG_DIR, 'train_stats.json')
MODEL_PATH = os.path.join(MODELS_DIR, 'retrained_clean_model')

# action meaning
ACTION_MEANING = {0: '做多价差', 1: '平仓', 2: '做空价差'}


def load_resources():
    test_df = pd.read_csv(TEST_CSV, parse_dates=['Date'])
    with open(STATS_JSON, 'r', encoding='utf-8') as f:
        frozen_stats = json.load(f)
    # load model (without env)
    model = PPO.load(MODEL_PATH)
    return test_df, frozen_stats, model


def run_audit():
    test_df, frozen_stats, model = load_resources()

    env = PairsTradingEnv(test_df.copy().reset_index(drop=True), initial_balance=10000.0,
                          window_size=10, use_features=['zscore', 'spread_rm', 'spread_rstd'],
                          train_stats=None, mode='test', frozen_stats=frozen_stats)

    obs, info = env.reset(options={'start_index': 0})

    records = []
    step = 0
    terminated = False
    truncated = False

    while True:
        # record pre-action positions
        pos1_before = env.stock1_shares
        pos2_before = env.stock2_shares

        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        # compute spread and zscore using frozen stats
        row = env.df.iloc[env.current_step]
        date = row['Date'] if 'Date' in row.index else None
        ningde_close = float(row.get('NINGDE_Close', np.nan))
        byd_close = float(row.get('BYD_Close', np.nan))
        price_spread = ningde_close - byd_close
        spread_mean = float(frozen_stats.get('spread_mean', 0.0))
        spread_std = float(frozen_stats.get('spread_std', 1.0))
        if spread_std <= 1e-8:
            spread_std = 1.0
        z_score = (price_spread - spread_mean) / spread_std

        # apply action
        obs, reward, terminated, truncated, info = env.step(action)

        pos1_after = env.stock1_shares
        pos2_after = env.stock2_shares

        record = {
            'step': step,
            'date': pd.to_datetime(date).strftime('%Y-%m-%d') if pd.notna(date) else None,
            'ningde_close': ningde_close,
            'byd_close': byd_close,
            'price_spread': price_spread,
            'z_score': z_score,
            'action': action,
            'action_meaning': ACTION_MEANING.get(action, ''),
            'position_ningde_before': pos1_before,
            'position_byd_before': pos2_before,
            'position_ningde_after': pos1_after,
            'position_byd_after': pos2_after,
            'reward': reward,
            'net_worth': info.get('net_worth', np.nan)
        }
        records.append(record)
        step += 1

        if terminated or truncated:
            break

    df_logs = pd.DataFrame(records)
    detailed_path = os.path.join(RESULTS_DIR, 'detailed_trade_log.csv')
    df_logs.to_csv(detailed_path, index=False, float_format='%.6f')

    # Action statistics
    total_steps = len(df_logs)
    counts = df_logs['action'].value_counts().to_dict()
    c0 = counts.get(0, 0)
    c1 = counts.get(1, 0)
    c2 = counts.get(2, 0)
    pct0 = c0 / total_steps * 100
    pct1 = c1 / total_steps * 100
    pct2 = c2 / total_steps * 100

    # total trade events: count steps where positions changed or action==1 (liquidation)
    trade_flags = ((df_logs['position_ningde_before'] != df_logs['position_ningde_after']) |
                   (df_logs['position_byd_before'] != df_logs['position_byd_after']) |
                   (df_logs['action'] == 1))
    total_trades = int(trade_flags.sum())

    # first liquidation occurrence
    if c1 > 0:
        first_liq = df_logs[df_logs['action'] == 1].iloc[0]
        first_liq_step = int(first_liq['step'])
        first_liq_date = first_liq['date']
    else:
        first_liq_step = None
        first_liq_date = None

    stats_text = (
        f"动作0 (做多) 次数: {c0}, 占比: {pct0:.1f}%\n"
        f"动作1 (平仓) 次数: {c1}, 占比: {pct1:.1f}%\n"
        f"动作2 (做空) 次数: {c2}, 占比: {pct2:.1f}%\n"
        f"总交易次数 (开仓+平仓): {total_trades}\n"
        f"首次出现平仓的步数/日期: {first_liq_step}/{first_liq_date if first_liq_date is not None else '无'}\n"
    )

    stats_path = os.path.join(RESULTS_DIR, 'action_statistics.txt')
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(stats_text)

    print(stats_text)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Z-score curve
    dates = pd.to_datetime(df_logs['date'])
    ax1.plot(dates, df_logs['z_score'], label='Z-Score', color='tab:orange')
    ax1.axhline(0, color='gray', linestyle='--')
    ax1.axhline(2, color='gray', linestyle=':')
    ax1.axhline(-2, color='gray', linestyle=':')

    # markers
    long_idx = df_logs[df_logs['action'] == 0].index
    flat_idx = df_logs[df_logs['action'] == 1].index
    short_idx = df_logs[df_logs['action'] == 2].index

    ax1.scatter(dates.iloc[long_idx], df_logs.loc[long_idx, 'z_score'], marker='^', color='red', label='做多开仓')
    ax1.scatter(dates.iloc[flat_idx], df_logs.loc[flat_idx, 'z_score'], marker='o', color='black', label='平仓')
    ax1.scatter(dates.iloc[short_idx], df_logs.loc[short_idx, 'z_score'], marker='v', color='blue', label='做空开仓')

    ax1.set_ylabel('Z-Score')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # positions and net worth
    ax2.plot(dates, df_logs['position_ningde_after'], label='宁德头寸', color='tab:green')
    ax2.plot(dates, df_logs['position_byd_after'], label='比亚迪头寸', color='tab:purple')
    ax2.set_ylabel('持仓数量')

    ax2b = ax2.twinx()
    ax2b.plot(dates, df_logs['net_worth'], label='净值', color='tab:red', linewidth=1.5)
    ax2b.set_ylabel('净值 (¥)')

    ax2.legend(loc='upper left')
    ax2b.legend(loc='upper right')

    ax2.set_xlabel('Date')
    plt.suptitle('配对交易模型终极行为审计')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_path = os.path.join(RESULTS_DIR, 'figure_trading_audit.png')
    fig.savefig(fig_path, dpi=300)

    print('Saved files to', RESULTS_DIR)


if __name__ == '__main__':
    run_audit()
