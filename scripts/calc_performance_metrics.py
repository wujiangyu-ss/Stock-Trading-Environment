import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_AUDIT = os.path.join(ROOT, 'results', 'audit')
OUT_DIR = os.path.join(ROOT, 'results')
os.makedirs(OUT_DIR, exist_ok=True)

DETAILED_CSV = os.path.join(RESULTS_AUDIT, 'detailed_trade_log.csv')
OUT_JSON = os.path.join(OUT_DIR, 'performance_metrics.json')

TRADING_DAYS_PER_YEAR = 252.0
RISK_FREE_RATE = 0.0


def max_drawdown_and_duration(net_worth_series):
    arr = np.array(net_worth_series, dtype=float)
    cummax = np.maximum.accumulate(arr)
    drawdowns = (cummax - arr) / (cummax + 1e-12)
    max_dd = float(np.max(drawdowns)) if len(drawdowns)>0 else 0.0
    # duration: longest period between a peak and a new peak
    max_dur = 0
    peak_idx = 0
    for i in range(1, len(arr)):
        if arr[i] >= arr[peak_idx]:
            peak_idx = i
        else:
            dur = 0
            j = i
            # find next index >= peak value
            while j < len(arr) and arr[j] < arr[peak_idx]:
                dur += 1
                j += 1
            if dur > max_dur:
                max_dur = dur
    return max_dd, int(max_dur)


def downside_deviation(returns, target=0.0):
    # returns is array of periodic returns (e.g., daily)
    downside = returns[returns < target]
    if len(downside) == 0:
        return 0.0
    dd = np.sqrt(np.mean((downside - target) ** 2))
    return float(dd)


def main():
    if not os.path.exists(DETAILED_CSV):
        raise FileNotFoundError(f"找不到 {DETAILED_CSV}，请先生成审计明细表。")

    df = pd.read_csv(DETAILED_CSV, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Use net_worth series
    net = df['net_worth'].astype(float).values
    dates = pd.to_datetime(df['date']).values

    initial_net = float(net[0])
    final_net = float(net[-1])
    total_return = final_net / initial_net - 1.0

    # Determine period in trading steps; assume steps ~ trading days
    trading_days = len(net)
    period_years = trading_days / TRADING_DAYS_PER_YEAR if TRADING_DAYS_PER_YEAR>0 else 1.0

    if period_years <= 0:
        annual_return = 0.0
    else:
        annual_return = (final_net / initial_net) ** (1.0 / period_years) - 1.0

    # periodic returns (simple) per step
    returns = np.zeros(len(net)-1)
    for i in range(1, len(net)):
        returns[i-1] = (net[i] / net[i-1]) - 1.0

    # annualized volatility
    if len(returns) > 1:
        vol_annual = float(np.std(returns, ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    else:
        vol_annual = 0.0

    # Sharpe (assume rf=0)
    sharpe = float((annual_return - RISK_FREE_RATE) / vol_annual) if vol_annual > 0 else 0.0

    # Max drawdown and duration
    max_dd, max_dd_dur = max_drawdown_and_duration(net)

    # Calmar ratio: annual_return / max_drawdown
    calmar = float(annual_return / max_dd) if max_dd > 0 else None

    # Sortino ratio
    dd_daily = downside_deviation(returns, target=0.0)
    sortino = None
    if dd_daily > 0:
        sortino = float((annual_return - RISK_FREE_RATE) / (dd_daily * np.sqrt(TRADING_DAYS_PER_YEAR)))

    # Trading behavior metrics
    # Identify trades by open (action 0 or 2) and their subsequent close (action 1)
    trades = []
    open_idx = None
    open_step = None
    open_net = None
    open_date = None
    open_action = None

    for idx, row in df.iterrows():
        act = int(row['action'])
        nw = float(row['net_worth'])
        date = row['date']
        if act in (0, 2):
            # open if currently no open trade
            if open_idx is None:
                open_idx = idx
                open_step = int(row['step'])
                open_net = nw
                open_date = date
                open_action = act
        elif act == 1:
            # close if we have open
            if open_idx is not None:
                close_idx = idx
                close_step = int(row['step'])
                close_net = nw
                close_date = date
                pnl = close_net - open_net
                holding_period_days = (pd.to_datetime(close_date) - pd.to_datetime(open_date)).days
                trades.append({
                    'open_idx': open_idx,
                    'close_idx': close_idx,
                    'open_step': open_step,
                    'close_step': close_step,
                    'open_date': str(open_date),
                    'close_date': str(close_date),
                    'open_net': open_net,
                    'close_net': close_net,
                    'pnl': pnl,
                    'holding_days': holding_period_days,
                    'open_action': int(open_action)
                })
                open_idx = None
                open_step = None
                open_net = None
                open_date = None
                open_action = None

    total_trades = len(trades)
    win_trades = [t for t in trades if t['pnl'] > 0]
    win_rate = float(len(win_trades) / total_trades) if total_trades > 0 else None

    avg_profit = float(np.mean([t['pnl'] for t in trades if t['pnl'] > 0])) if any(t['pnl']>0 for t in trades) else None
    avg_loss = float(-np.mean([t['pnl'] for t in trades if t['pnl'] < 0])) if any(t['pnl']<0 for t in trades) else None
    avg_profit_loss_ratio = None
    if avg_loss and avg_loss > 0:
        avg_profit_loss_ratio = float(avg_profit / avg_loss) if avg_profit is not None else None

    avg_holding = float(np.mean([t['holding_days'] for t in trades])) if len(trades)>0 else None

    metrics = {
        'total_return': total_return,
        'annualized_return': annual_return,
        'annualized_volatility': vol_annual,
        'sharpe_ratio': sharpe,
        'calmar_ratio': calmar,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'max_drawdown_duration_days': max_dd_dur,
        'volatility': vol_annual,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'average_profit_loss_ratio': avg_profit_loss_ratio,
        'average_holding_period_days': avg_holding
    }

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print('Saved performance metrics to', OUT_JSON)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
