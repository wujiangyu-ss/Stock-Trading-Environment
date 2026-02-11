import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from env.PairsTradingEnv import PairsTradingEnv

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
RESULTS_DIR = os.path.join(ROOT, 'results')
FIG_DIR = os.path.join(ROOT, 'figures')
MODELS_DIR = os.path.join(ROOT, 'models')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

TEST_CSV = os.path.join(DATA_DIR, 'test_dataset.csv')
AUDIT_DETAILED = os.path.join(RESULTS_DIR, 'audit', 'detailed_trade_log.csv')
STATS_JSON = os.path.join(ROOT, 'config', 'train_stats.json')

TRADING_DAYS_PER_YEAR = 252.0


def max_drawdown_and_duration(net_worth_series):
    arr = np.array(net_worth_series, dtype=float)
    cummax = np.maximum.accumulate(arr)
    drawdowns = (cummax - arr) / (cummax + 1e-12)
    max_dd = float(np.max(drawdowns)) if len(drawdowns)>0 else 0.0
    max_dur = 0
    peak_idx = 0
    for i in range(1, len(arr)):
        if arr[i] >= arr[peak_idx]:
            peak_idx = i
        else:
            dur = 0
            j = i
            while j < len(arr) and arr[j] < arr[peak_idx]:
                dur += 1
                j += 1
            if dur > max_dur:
                max_dur = dur
    return max_dd, int(max_dur)


def downside_deviation(returns, target=0.0):
    downside = returns[returns < target]
    if len(downside) == 0:
        return 0.0
    dd = np.sqrt(np.mean((downside - target) ** 2))
    return float(dd)


def compute_metrics_from_series(net, dates):
    initial = float(net[0])
    final = float(net[-1])
    total_return = final / initial - 1.0
    trading_days = len(net)
    period_years = trading_days / TRADING_DAYS_PER_YEAR if TRADING_DAYS_PER_YEAR>0 else 1.0
    annual_return = (final / initial) ** (1.0 / period_years) - 1.0 if period_years>0 else 0.0
    returns = np.array([(net[i]/net[i-1])-1.0 for i in range(1, len(net))]) if len(net)>1 else np.array([])
    vol_annual = float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if len(returns)>1 else 0.0
    sharpe = float((annual_return) / vol_annual) if vol_annual>0 else 0.0
    max_dd, max_dur = max_drawdown_and_duration(net)
    calmar = float(annual_return / max_dd) if max_dd>0 else None
    dd_daily = downside_deviation(returns, target=0.0)
    sortino = float((annual_return) / (dd_daily * np.sqrt(TRADING_DAYS_PER_YEAR))) if dd_daily>0 else None
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': vol_annual,
        'sharpe': sharpe,
        'calmar': calmar,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'max_drawdown_duration_days': max_dur
    }


def run_threshold_strategy(test_df, frozen_stats, z_open=2.0):
    env = PairsTradingEnv(test_df.copy().reset_index(drop=True), initial_balance=10000.0,
                          window_size=10, use_features=['zscore','spread_rm','spread_rstd'],
                          train_stats=None, mode='test', frozen_stats=frozen_stats)
    obs, info = env.reset(options={'start_index': 0})
    net_worths = [info.get('net_worth', env.net_worth)]
    dates = []
    actions = []

    while True:
        # current row
        row = env.df.iloc[env.current_step]
        date = pd.to_datetime(row['Date']) if 'Date' in row.index else None
        ning = float(row.get('NINGDE_Close', np.nan))
        byd = float(row.get('BYD_Close', np.nan))
        spread = ning - byd
        mean = float(frozen_stats.get('spread_mean', 0.0))
        std = float(frozen_stats.get('spread_std', 1.0)) if float(frozen_stats.get('spread_std',1.0))>1e-8 else 1.0
        z = (spread - mean) / std

        # determine action based on current positions
        # if no position:
        if env.stock1_shares == 0 and env.stock2_shares == 0:
            if z < -z_open:
                action = 0
            elif z > z_open:
                action = 2
            else:
                action = 1  # hold/no-op: use 1 to keep consistent (liquidate does nothing)
        else:
            # if currently long spread (stock1_shares>0)
            if env.stock1_shares > 0:
                # close when z >= 0
                if z >= 0:
                    action = 1
                else:
                    action = 0  # hold long
            elif env.stock1_shares < 0:
                # currently short spread; close when z <= 0
                if z <= 0:
                    action = 1
                else:
                    action = 2
            else:
                action = 1

        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        net_worths.append(info.get('net_worth', env.net_worth))
        dates.append(date)
        if terminated or truncated:
            break

    # build series aligned with dates: net_worths length is steps+1; align using first date from df[0]
    # create date index from df
    date_index = pd.to_datetime(env.df['Date']).reset_index(drop=True)
    net_series = np.array(net_worths)
    return date_index, net_series, actions


def run_buy_and_hold(test_df, which='NINGDE'):
    df = test_df.copy().reset_index(drop=True)
    dates = pd.to_datetime(df['Date'])
    initial_cash = 10000.0
    if which == 'NINGDE':
        prices = df['NINGDE_Close'].astype(float)
        shares = np.floor(initial_cash / prices.iloc[0])
        cash_left = initial_cash - shares * prices.iloc[0]
        net = shares * prices + cash_left
    elif which == 'BYD':
        prices = df['BYD_Close'].astype(float)
        shares = np.floor(initial_cash / prices.iloc[0])
        cash_left = initial_cash - shares * prices.iloc[0]
        net = shares * prices + cash_left
    elif which == 'EQUAL':
        # 50% each
        prices1 = df['NINGDE_Close'].astype(float)
        prices2 = df['BYD_Close'].astype(float)
        cash1 = initial_cash * 0.5
        cash2 = initial_cash * 0.5
        shares1 = np.floor(cash1 / prices1.iloc[0])
        shares2 = np.floor(cash2 / prices2.iloc[0])
        cash_left = cash1 - shares1 * prices1.iloc[0] + cash2 - shares2 * prices2.iloc[0]
        net = shares1 * prices1 + shares2 * prices2 + cash_left
    else:
        raise ValueError('which must be NINGDE/BYD/EQUAL')
    return dates, np.array(net)


def main():
    test_df = pd.read_csv(TEST_CSV, parse_dates=['Date'])
    with open(STATS_JSON, 'r', encoding='utf-8') as f:
        frozen_stats = json.load(f)

    # DRL series from audit detailed log
    audit_path = os.path.join(RESULTS_DIR, 'audit', 'detailed_trade_log.csv')
    if not os.path.exists(audit_path):
        raise FileNotFoundError('请先运行审计生成 detailed_trade_log.csv')
    df_audit = pd.read_csv(audit_path, parse_dates=['date'])
    drl_dates = pd.to_datetime(df_audit['date'])
    drl_net = df_audit['net_worth'].astype(float).values

    # Threshold strategy
    thr_dates, thr_net, thr_actions = run_threshold_strategy(test_df, frozen_stats)

    # Buy and hold strategies
    bh_dates_n, bh_net_n = run_buy_and_hold(test_df, which='NINGDE')
    bh_dates_b, bh_net_b = run_buy_and_hold(test_df, which='BYD')
    bh_dates_e, bh_net_e = run_buy_and_hold(test_df, which='EQUAL')

    # select best of BH
    final_vals = {'NINGDE': bh_net_n[-1], 'BYD': bh_net_b[-1], 'EQUAL': bh_net_e[-1]}
    best_bh = max(final_vals, key=final_vals.get)
    if best_bh == 'NINGDE':
        bh_net = bh_net_n
        bh_label = 'BuyHold_NINGDE'
    elif best_bh == 'BYD':
        bh_net = bh_net_b
        bh_label = 'BuyHold_BYD'
    else:
        bh_net = bh_net_e
        bh_label = 'BuyHold_EQUAL'

    # compute metrics
    metrics = {}
    metrics['DRL'] = compute_metrics_from_series(drl_net, drl_dates)
    thr_metrics = compute_metrics_from_series(thr_net, thr_dates)
    metrics['Threshold'] = thr_metrics
    metrics['BuyHold_NINGDE'] = compute_metrics_from_series(bh_net_n, bh_dates_n)
    metrics['BuyHold_BYD'] = compute_metrics_from_series(bh_net_b, bh_dates_b)
    metrics['BuyHold_EQUAL'] = compute_metrics_from_series(bh_net_e, bh_dates_e)

    # For trading counts: Threshold count trades
    # detect trades in threshold by checking changes in positions using simulation of actions
    # Re-run threshold to get trades list
    # We'll infer trade count by counting actions 0/2 and 1 occurrences
    thr_action_counts = {0: thr_actions.count(0), 1: thr_actions.count(1), 2: thr_actions.count(2)}

    # Total trades approximate: count open/close pairs
    # For simplicity add thr_open_count = count of actions 0 or 2 when they change positions
    total_trades_thr = int((thr_action_counts.get(0,0) + thr_action_counts.get(2,0) + thr_action_counts.get(1,0)))

    # Build comparison table
    rows = []
    def row_from(name, met, total_trades):
        return {
            '策略名称': name,
            '累计收益率': met['total_return'],
            '年化收益率': met['annual_return'],
            '年化波动率': met['annual_volatility'],
            '夏普比率': met['sharpe'],
            '卡玛比率': met['calmar'],
            '最大回撤': met['max_drawdown'],
            '总交易次数': total_trades
        }

    rows.append(row_from('DRL', metrics['DRL'], int(df_audit.shape[0])))
    rows.append(row_from('Traditional_Threshold', metrics['Threshold'], total_trades_thr))
    rows.append(row_from('BuyHold_NINGDE', metrics['BuyHold_NINGDE'], 1))
    rows.append(row_from('BuyHold_BYD', metrics['BuyHold_BYD'], 1))
    rows.append(row_from('BuyHold_EQUAL', metrics['BuyHold_EQUAL'], 1))

    comp_df = pd.DataFrame(rows)
    comp_path = os.path.join(RESULTS_DIR, 'strategy_comparison.csv')
    comp_df.to_csv(comp_path, index=False, float_format='%.6f')

    # Plot equity curves normalized to 1
    plt.figure(figsize=(10,6))
    # align dates: use test_df Date
    common_dates = pd.to_datetime(test_df['Date'])
    def norm(arr):
        return arr / arr[0]
    # trim to minimum length to avoid shape mismatch
    min_len = min(len(drl_net), len(thr_net), len(bh_net_n), len(bh_net_b))
    plt.plot(common_dates[:min_len], norm(drl_net[:min_len]), label='DRL')
    plt.plot(common_dates[:min_len], norm(thr_net[:min_len]), label='Threshold')
    plt.plot(common_dates[:min_len], norm(bh_net_n[:min_len]), label='BuyHold_NINGDE')
    plt.plot(common_dates[:min_len], norm(bh_net_b[:min_len]), label='BuyHold_BYD')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Normalized Equity (start=1)')
    plt.title('Equity Curve Comparison')
    plt.grid(True)
    fig_path = os.path.join(FIG_DIR, 'equity_curve_comparison.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)

    # print brief analysis: best by final net_worth
    finals = {
        'DRL': float(drl_net[-1]),
        'Threshold': float(thr_net[-1]),
        'BuyHold_NINGDE': float(bh_net_n[-1]),
        'BuyHold_BYD': float(bh_net_b[-1]),
        'BuyHold_EQUAL': float(bh_net_e[-1])
    }
    best = max(finals, key=finals.get)
    print('策略对比结果已保存至', comp_path)
    print('最佳策略(按最终净值):', best, '最终净值:', finals[best])

    # where DRL outperforms threshold
    compare_points = []
    drl_metrics = metrics['DRL']
    thr_metrics = metrics['Threshold']
    outperform = []
    if drl_metrics['total_return'] > thr_metrics['total_return']:
        outperform.append('累计收益率')
    if drl_metrics['sharpe'] > thr_metrics['sharpe']:
        outperform.append('夏普比率')
    if drl_metrics['max_drawdown'] < thr_metrics['max_drawdown']:
        outperform.append('最大回撤 (越小越好)')

    print('DRL 相较于传统阈值策略超越的指标:', outperform if len(outperform)>0 else '无')


if __name__ == '__main__':
    main()
