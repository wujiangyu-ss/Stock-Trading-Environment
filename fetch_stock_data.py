"""
A股新能源汽车龙头股票历史数据爬取脚本
使用AKShare库从网上抓取指定股票的日K线数据
"""

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time

# 配置
STOCKS = {
    '300750.SZ': '宁德时代',
    '002594.SZ': '比亚迪',
    '002460.SZ': '赣锋锂业'
}

DATA_DIR = './data'
START_DATE = '20200101'
END_DATE = '20251231'

# 必需列
REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']


def create_data_dir():
    """创建数据文件夹（如果不存在）"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"✓ 创建数据文件夹: {DATA_DIR}")


def fetch_stock_data(stock_code, stock_name):
    """
    获取单只股票的历史数据
    
    Args:
        stock_code: 股票代码（如'300750.SZ'）
        stock_name: 股票名称
        
    Returns:
        DataFrame或None
    """
    print(f"\n正在获取 {stock_name} ({stock_code}) 的数据...")
    
    try:
        # 使用AKShare库获取日K线数据
        df = ak.stock_zh_a_hist(
            symbol=stock_code.split('.')[0],  # 移除后缀，只保留代码
            period='daily',
            start_date=START_DATE,
            end_date=END_DATE,
            adjust='qfq'  # 使用前复权
        )
        
        if df is None or len(df) == 0:
            print(f"✗ 无法获取 {stock_name} 的数据")
            return None
        
        return df
        
    except Exception as e:
        print(f"✗ 获取 {stock_name} 数据时出错: {str(e)}")
        return None


def clean_and_validate_data(df, stock_name):
    """
    清洗和验证数据
    
    Args:
        df: 原始DataFrame
        stock_name: 股票名称
        
    Returns:
        清洗后的DataFrame或None
    """
    if df is None or len(df) == 0:
        return None
    
    print(f"  数据清洗中...")
    
    # 创建副本，避免修改原数据
    df = df.copy()
    
    # 重命名列以匹配要求（处理不同版本AKShare的列名差异）
    column_mapping = {
        '日期': 'Date',
        'date': 'Date',
        '开盘': 'Open',
        'open': 'Open',
        '最高': 'High',
        'high': 'High',
        '最低': 'Low',
        'low': 'Low',
        '收盘': 'Close',
        'close': 'Close',
        '成交量': 'Volume',
        'volume': 'Volume'
    }
    
    df = df.rename(columns=column_mapping)
    
    # 检查必需列是否存在
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            print(f"  ✗ 缺少必需列: {col}")
            return None
    
    # 选择必需列
    df = df[REQUIRED_COLUMNS].copy()
    
    # 移除重复行
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['Date'], keep='first')
    removed_duplicates = initial_rows - len(df)
    if removed_duplicates > 0:
        print(f"  ✓ 移除了 {removed_duplicates} 条重复数据")
    
    # 将Date列转换为datetime格式
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d', errors='coerce')
    except:
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        except Exception as e:
            print(f"  ✗ 日期格式转换失败: {str(e)}")
            return None
    
    # 移除日期为NaT的行
    rows_with_invalid_date = df['Date'].isna().sum()
    if rows_with_invalid_date > 0:
        print(f"  ✓ 移除了 {rows_with_invalid_date} 条日期无效的数据")
    df = df.dropna(subset=['Date'])
    
    # 按日期升序排列
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 数据类型转换
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    
    # 检查异常值（价格应为正数）
    price_cols = ['Open', 'High', 'Low', 'Close']
    initial_rows = len(df)
    
    for col in price_cols:
        df = df[df[col] > 0]
    
    removed_invalid = initial_rows - len(df)
    if removed_invalid > 0:
        print(f"  ✓ 移除了 {removed_invalid} 条异常数据（价格为非正数）")
    
    # 检查High >= Low >= Close的逻辑
    df = df[(df['High'] >= df['Low']) & (df['High'] >= df['Close']) & (df['Low'] <= df['Close'])].copy()
    
    # 重新设置索引
    df = df.reset_index(drop=True)
    
    if len(df) == 0:
        print(f"  ✗ 清洗后无有效数据")
        return None
    
    return df


def save_to_csv(df, stock_code, stock_name):
    """
    将数据保存为CSV文件
    
    Args:
        df: DataFrame
        stock_code: 股票代码
        stock_name: 股票名称
        
    Returns:
        bool: 是否保存成功
    """
    if df is None or len(df) == 0:
        print(f"  ✗ {stock_name} 无数据可保存")
        return False
    
    file_path = os.path.join(DATA_DIR, f"{stock_code.split('.')[0]}.csv")
    
    try:
        # 转换Date为字符串格式以便保存
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"  ✓ 数据已保存: {file_path}")
        return True
    except Exception as e:
        print(f"  ✗ 保存文件失败: {str(e)}")
        return False


def display_data_info(df, stock_name):
    """
    显示数据的基本信息
    
    Args:
        df: DataFrame
        stock_name: 股票名称
    """
    if df is None or len(df) == 0:
        print(f"  {stock_name}: 无数据")
        return
    
    print(f"\n  {stock_name} 数据信息:")
    print(f"    - 行数: {len(df)}")
    print(f"    - 起始日期: {df['Date'].min()}")
    print(f"    - 结束日期: {df['Date'].max()}")
    print(f"    - 收盘价范围: ¥{df['Close'].min():.2f} - ¥{df['Close'].max():.2f}")
    print(f"    - 平均成交量: {df['Volume'].mean():.0f}")


def main():
    """主函数"""
    print("=" * 60)
    print("A股新能源汽车龙头股票历史数据爬取")
    print("=" * 60)
    
    # 创建数据文件夹
    create_data_dir()
    
    # 遍历每只股票
    successful_stocks = 0
    
    for stock_code, stock_name in STOCKS.items():
        # 获取数据
        df = fetch_stock_data(stock_code, stock_name)
        
        if df is None:
            continue
        
        # 清洗数据
        df_cleaned = clean_and_validate_data(df, stock_name)
        
        if df_cleaned is None:
            continue
        
        # 显示信息
        display_data_info(df_cleaned, stock_name)
        
        # 保存为CSV
        if save_to_csv(df_cleaned, stock_code, stock_name):
            successful_stocks += 1
        
        # 礼貌地等待（避免请求过于频繁）
        time.sleep(1)
    
    # 总结
    print("\n" + "=" * 60)
    print(f"完成！成功获取 {successful_stocks}/{len(STOCKS)} 只股票的数据")
    print("=" * 60)


if __name__ == "__main__":
    main()
