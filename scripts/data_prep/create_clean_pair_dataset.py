import os
import json
import pandas as pd
from datetime import datetime

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# User-specified identifiers
NINGDE_KEYS = ['300750', '宁德时代', 'Ningde', 'NINGDE']
BYD_KEYS = ['002594', '比亚迪', 'BYD']

# Rolling window for features
ROLLING_WINDOW = 20


def try_load_all_stocks():
    path = os.path.join(DATA_DIR, 'all_stocks_data.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=['Date'])
        return df
    return None


def try_load_individual(code):
    # try common filenames
    candidates = [
        f"{code}.csv",
        f"{code}.SZ.csv",
        f"{code}.SH.csv",
        f"{code}.CSV",
    ]
    for c in candidates:
        p = os.path.join(DATA_DIR, c)
        if os.path.exists(p):
            df = pd.read_csv(p, parse_dates=['Date'])
            return df
    return None


def find_stock_columns(df):
    # Return mapping of found column name prefixes for Ningde and BYD
    cols = df.columns.astype(str).tolist()
    found = {'ningde': None, 'byd': None}
    for c in cols:
        for key in NINGDE_KEYS:
            if key in c:
                found['ningde'] = c
                break
        for key in BYD_KEYS:
            if key in c:
                found['byd'] = c
                break
    return found


def build_pair_from_all(df_all):
    # Try detect Close columns for each stock
    # Expect multi-column format like '300750_Close' or '宁德时代_Close'
    cols = df_all.columns.astype(str).tolist()
    ng_col = None
    byd_col = None
    for c in cols:
        for k in NINGDE_KEYS:
            if k in c and 'Close' in c:
                ng_col = c
                break
        for k in BYD_KEYS:
            if k in c and 'Close' in c:
                byd_col = c
                break
    # fallback: any column containing code even if not Close
    if ng_col is None:
        for c in cols:
            for k in NINGDE_KEYS:
                if k in c:
                    ng_col = c
                    break
    if byd_col is None:
        for c in cols:
            for k in BYD_KEYS:
                if k in c:
                    byd_col = c
                    break
    if ng_col is None or byd_col is None:
        return None
    df = df_all[['Date', ng_col, byd_col]].copy()
    # rename
    df = df.rename(columns={ng_col: 'NINGDE_Close', byd_col: 'BYD_Close'})
    return df


def build_pair_from_individuals():
    # Try to load individual files and extract Close
    # Ningde
    for key in NINGDE_KEYS:
        # if key is numeric code, try that filename
        df_ng = try_load_individual(key)
        if df_ng is not None:
            break
    else:
        df_ng = None
    for key in BYD_KEYS:
        df_byd = try_load_individual(key)
        if df_byd is not None:
            break
    else:
        df_byd = None

    if df_ng is None or df_byd is None:
        return None

    # standardize date and close column names
    df_ng = df_ng.rename(columns={c: c.strip() for c in df_ng.columns})
    df_byd = df_byd.rename(columns={c: c.strip() for c in df_byd.columns})
    # find Close col
    def find_close(df):
        for c in df.columns:
            if 'Close' == c or c.lower() == 'close' or c.lower().endswith('_close'):
                return c
        # try common names
        for c in df.columns:
            if 'close' in c.lower():
                return c
        return None
    ng_close = find_close(df_ng)
    byd_close = find_close(df_byd)
    if ng_close is None or byd_close is None:
        return None
    df_ng = df_ng[['Date', ng_close]].rename(columns={ng_close: 'NINGDE_Close'})
    df_byd = df_byd[['Date', byd_close]].rename(columns={byd_close: 'BYD_Close'})
    df_ng['Date'] = pd.to_datetime(df_ng['Date'])
    df_byd['Date'] = pd.to_datetime(df_byd['Date'])
    df = pd.merge(df_ng, df_byd, on='Date', how='outer')
    return df


def main():
    # 1. Load data
    df_all = try_load_all_stocks()
    if df_all is not None:
        # ensure Date column
        if 'Date' not in df_all.columns:
            # try index
            df_all = df_all.reset_index()
        df_all['Date'] = pd.to_datetime(df_all['Date'])
        df_all = df_all.sort_values('Date').reset_index(drop=True)
        df = build_pair_from_all(df_all)
        if df is None:
            df = build_pair_from_individuals()
    else:
        df = build_pair_from_individuals()

    if df is None:
        raise FileNotFoundError('无法找到包含宁德时代和比亚迪收盘价的输入数据，请将原始数据放到 data/all_stocks_data.csv 或 data/300750.csv + data/002594.csv')

    # normalize column names
    df = df.rename(columns={c: c.strip() for c in df.columns})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    # keep only two columns + Date
    keep_cols = [col for col in df.columns if col in ['Date', 'NINGDE_Close', 'BYD_Close']]
    df_clean = df[keep_cols].copy()

    # forward/backward fill
    df_clean = df_clean.set_index('Date').sort_index()
    df_clean = df_clean.ffill().bfill().reset_index()

    # Save cleaned full data
    cleaned_path = os.path.join(DATA_DIR, 'pair_data_cleaned.csv')
    df_clean.to_csv(cleaned_path, index=False)

    # Partition by date
    def date_in_range(d, start_s, end_s):
        return (d >= pd.to_datetime(start_s)) & (d <= pd.to_datetime(end_s))

    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    train_mask = date_in_range(df_clean['Date'], '2020-01-02', '2023-12-31')
    val_mask = date_in_range(df_clean['Date'], '2024-01-01', '2024-12-31')
    test_mask = date_in_range(df_clean['Date'], '2025-01-01', '2025-12-31')

    df_train = df_clean[train_mask].copy().reset_index(drop=True)
    df_val = df_clean[val_mask].copy().reset_index(drop=True)
    df_test = df_clean[test_mask].copy().reset_index(drop=True)

    # Ensure non-empty
    if df_train.empty:
        raise ValueError('训练集为空，请检查原始数据覆盖时间是否包含 2020-01-02 至 2023-12-31')

    # Compute spread
    df_train['spread'] = df_train['NINGDE_Close'] - df_train['BYD_Close']
    df_val['spread'] = df_val['NINGDE_Close'] - df_val['BYD_Close']
    df_test['spread'] = df_test['NINGDE_Close'] - df_test['BYD_Close']

    # Rolling features on each partition separately (causal within partition)
    df_train['spread_rm'] = df_train['spread'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    df_train['spread_rstd'] = df_train['spread'].rolling(window=ROLLING_WINDOW, min_periods=1).std().fillna(0)
    # Train overall statistics (freeze)
    spread_mean = float(df_train['spread'].mean())
    spread_std = float(df_train['spread'].std())
    if spread_std <= 1e-8:
        spread_std = 1.0

    # Z-score using frozen train overall mean/std for all partitions
    df_train['zscore'] = (df_train['spread'] - spread_mean) / spread_std
    # For val/test, compute rolling mean/std causally within partition, but zscore uses train mean/std
    df_val['spread_rm'] = df_val['spread'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    df_val['spread_rstd'] = df_val['spread'].rolling(window=ROLLING_WINDOW, min_periods=1).std().fillna(0)
    df_val['zscore'] = (df_val['spread'] - spread_mean) / spread_std

    df_test['spread_rm'] = df_test['spread'].rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    df_test['spread_rstd'] = df_test['spread'].rolling(window=ROLLING_WINDOW, min_periods=1).std().fillna(0)
    df_test['zscore'] = (df_test['spread'] - spread_mean) / spread_std

    # Optionally tag dataset
    df_train['dataset'] = 'train'
    df_val['dataset'] = 'val'
    df_test['dataset'] = 'test'

    # Save partitions
    train_path = os.path.join(DATA_DIR, 'train_dataset.csv')
    val_path = os.path.join(DATA_DIR, 'val_dataset.csv')
    test_path = os.path.join(DATA_DIR, 'test_dataset.csv')

    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)

    # Save full combined with dataset label (recommended)
    df_full_labeled = pd.concat([df_train, df_val, df_test], ignore_index=True)
    full_labeled_path = os.path.join(DATA_DIR, 'pair_data_labeled.csv')
    df_full_labeled.to_csv(full_labeled_path, index=False)

    # Save train stats
    stats = {
        'spread_mean': spread_mean,
        'spread_std': spread_std,
        'rolling_window': ROLLING_WINDOW,
        'features': ['spread', 'spread_rm', 'spread_rstd', 'zscore']
    }
    stats_path = os.path.join(CONFIG_DIR, 'train_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print('Saved:')
    print('  ', cleaned_path)
    print('  ', train_path)
    print('  ', val_path)
    print('  ', test_path)
    print('  ', stats_path)


if __name__ == '__main__':
    main()
