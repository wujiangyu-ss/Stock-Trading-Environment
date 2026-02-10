"""
严格时间切分与因果特征工程脚本
- 输入：两支股票的原始OHLCV CSV（Date,Open,High,Low,Close,Volume）
- 输出：train/val/test 三份配对CSV、训练集统计（train_stats.json）

特性：
- 按时间边界严格切分（无随机）
- 对于val/test，计算滚动特征时会使用训练集末尾的历史上下文（仅历史，不包含未来）
- 所有归一化参数仅基于训练集计算并固化
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np


def strict_time_split(df, train_end, val_end):
    """
    严格按时间划分数据（不随机）
    Args:
        df: 含Date列的DataFrame（已按Date升序）
        train_end: 训练集结束日期（包含），字符串，格式 'YYYY-MM-DD'
        val_end: 验证集结束日期（包含），字符串
    Returns:
        train_df, val_df, test_df
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)

    train_df = df[df['Date'] <= train_end_dt].reset_index(drop=True)
    val_df = df[(df['Date'] > train_end_dt) & (df['Date'] <= val_end_dt)].reset_index(drop=True)
    test_df = df[df['Date'] > val_end_dt].reset_index(drop=True)

    return train_df, val_df, test_df


def compute_pair_base(df1, df2):
    """合并两支股票并计算基础配对列（price_spread, price_ratio, volume_ratio）"""
    merged = pd.merge(
        df1[['Date', 'Close', 'Volume']].rename(columns={'Close':'Close_1','Volume':'Vol_1'}),
        df2[['Date', 'Close', 'Volume']].rename(columns={'Close':'Close_2','Volume':'Vol_2'}),
        on='Date', how='inner'
    )
    merged = merged.sort_values('Date').reset_index(drop=True)
    merged['price_spread'] = merged['Close_1'] - merged['Close_2']
    merged['price_ratio'] = merged['Close_1'] / (merged['Close_2'].replace(0, np.nan))
    merged['volume_ratio'] = merged['Vol_1'] / (merged['Vol_2'].replace(0, np.nan))
    # 若出现inf或nan值（如除0），用np.nan处理，后续填充
    merged.replace([np.inf, -np.inf], np.nan, inplace=True)
    return merged


def causal_rolling_features(partition_df, history_df=None, windows=(5,20)):
    """
    对单个分区（train/val/test）计算滚动特征，保证因果性。
    如果提供 history_df（训练集或此前数据的末尾），会在partition前面拼接历史用于滚动窗口计算历史上下文（但不会使用partition之后的数据）。
    返回仅属于partition_df长度的特征（去掉历史前缀）。
    """
    if history_df is not None and len(history_df) > 0:
        concat = pd.concat([history_df, partition_df], ignore_index=True)
        offset = len(history_df)
    else:
        concat = partition_df.copy().reset_index(drop=True)
        offset = 0

    s = concat['price_spread']
    result = pd.DataFrame({'Date': concat['Date']})

    # moving averages
    result['spread_ma5'] = s.rolling(window=5, min_periods=1).mean()
    result['spread_ma20'] = s.rolling(window=20, min_periods=1).mean()
    result['spread_std'] = s.rolling(window=20, min_periods=1).std().fillna(0)

    # bollinger
    result['bollinger_high'] = result['spread_ma20'] + 2 * result['spread_std']
    result['bollinger_low'] = result['spread_ma20'] - 2 * result['spread_std']

    # zscore: (spread - ma20)/std, protect divide by zero
    result['zscore'] = np.where(result['spread_std'] > 0, (s - result['spread_ma20']) / result['spread_std'], 0)

    # price_ratio and volume_ratio carried from concat
    result['price_ratio'] = concat['price_ratio']
    result['volume_ratio'] = concat['volume_ratio']
    result['price_spread'] = concat['price_spread']

    # slice to partition portion
    partition_result = result.iloc[offset:].reset_index(drop=True)
    return partition_result


def prepare_strict_pair(stock1_csv, stock2_csv, output_dir, train_end='2023-12-31', val_end='2024-12-31'):
    """
    主流程：加载原始CSV，合并，按时间严格划分，并为每个分区计算因果特征。
    保存：train.csv, val.csv, test.csv, train_stats.json, combined_all.csv
    """
    os.makedirs(output_dir, exist_ok=True)

    df1 = pd.read_csv(stock1_csv)
    df2 = pd.read_csv(stock2_csv)
    df1['Date'] = pd.to_datetime(df1['Date'])
    df2['Date'] = pd.to_datetime(df2['Date'])

    merged = compute_pair_base(df1, df2)

    # 划分
    train_df, val_df, test_df = strict_time_split(merged, train_end, val_end)

    # 为val和test的滚动特征提供训练集尾部历史上下文（最多 window-1 行），以保持因果性
    max_window = 20
    history_tail = train_df.tail(max_window - 1).reset_index(drop=True)

    train_feats = causal_rolling_features(train_df, history_df=None)
    val_feats = causal_rolling_features(val_df, history_df=history_tail)
    test_feats = causal_rolling_features(test_df, history_df=pd.concat([history_tail, val_feats.tail(max_window-1)], ignore_index=True))

    # 训练集统计仅基于train_feats
    feature_cols = ['price_ratio','price_spread','zscore','spread_ma5','spread_ma20','spread_std','bollinger_high','bollinger_low','volume_ratio']
    train_stats = {}
    for col in feature_cols:
        col_series = train_feats[col].astype(float)
        train_stats[col] = {
            'mean': float(col_series.mean(skipna=True)),
            'std': float(col_series.std(skipna=True)),
            'min': float(col_series.min(skipna=True)),
            'max': float(col_series.max(skipna=True))
        }

    # 保存CSV
    train_out = os.path.join(output_dir, 'pair_train.csv')
    val_out = os.path.join(output_dir, 'pair_val.csv')
    test_out = os.path.join(output_dir, 'pair_test.csv')
    all_out = os.path.join(output_dir, 'pair_all.csv')
    stats_out = os.path.join(output_dir, 'train_stats.json')

    train_feats.to_csv(train_out, index=False, encoding='utf-8')
    val_feats.to_csv(val_out, index=False, encoding='utf-8')
    test_feats.to_csv(test_out, index=False, encoding='utf-8')
    merged.to_csv(all_out, index=False, encoding='utf-8')

    with open(stats_out, 'w', encoding='utf-8') as f:
        json.dump(train_stats, f, indent=2, ensure_ascii=False)

    print('Saved:', train_out, val_out, test_out, stats_out)
    return train_out, val_out, test_out, stats_out


if __name__ == '__main__':
    # 示例调用（可按需修改路径）
    base = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
    s1 = os.path.join(base, 'data', '300750.csv')
    s2 = os.path.join(base, 'data', '002594.csv')
    outdir = os.path.join(base, 'data', 'strict')
    prepare_strict_pair(s1, s2, outdir)
