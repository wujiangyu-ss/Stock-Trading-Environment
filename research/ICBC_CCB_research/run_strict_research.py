"""
严格复现配对交易研究流程
对象股票对：工商银行(601398) / 建设银行(601939)
目标：生成与宁德时代/比亚迪配对完全可比的结果集

执行步骤：
1. 获取股票数据 (2020-01-01 至 2025-12-31)
2. 防泄露数据划分 (train:70%, val:15%, test:15%)
3. 执行协整检验
4. 生成对比报告
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加父目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# 导入基准脚本的函数
from scripts.prepare_pair_data_strict import (
    strict_time_split,
    compute_pair_base,
    causal_rolling_features,
    prepare_strict_pair
)

try:
    import akshare as ak
except ImportError:
    print("⚠ akshare未安装，将使用本地数据")
    ak = None

try:
    from statsmodels.tsa.stattools import coint
except ImportError:
    print("⚠ statsmodels未安装，协整检验功能不可用")
    coint = None


# ============================================================================
# 配置
# ============================================================================

RESEARCH_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_ROOT = os.path.normpath(os.path.join(RESEARCH_ROOT, '..', '..'))
DATA_DIR = os.path.join(RESEARCH_ROOT, 'data')
RESULTS_DIR = os.path.join(RESEARCH_ROOT, 'results')
LOGS_DIR = os.path.join(RESEARCH_ROOT, 'logs')

# 新股票对配置
STOCK1_CODE = '601398.SH'      # 工商银行
STOCK1_NAME = '工商银行'
STOCK1_FILE = os.path.join(DATA_DIR, 'ICBC_601398.csv')

STOCK2_CODE = '601939.SH'      # 建设银行
STOCK2_NAME = '建设银行'
STOCK2_FILE = os.path.join(DATA_DIR, 'CCB_601939.csv')

PAIR_NAME = f'{STOCK1_NAME}_{STOCK2_NAME}_ICBC_CCB'
STRICT_OUTPUT_DIR = os.path.join(DATA_DIR, 'strict')

# 数据划分参数
TRAIN_END = '2023-12-31'       # 70%
VAL_END = '2024-12-31'         # 15% (2024年)
# TEST: 从2025-01-01开始 (15%)

START_DATE = '2020-01-01'
END_DATE = '2025-12-31'

# 基准案例路径（用于对比）
BASELINE_STRICT_DIR = os.path.join(BASE_ROOT, 'data', 'strict')
BASELINE_RESULTS = os.path.join(BASE_ROOT, 'results', 'comparison_table.csv')


# ============================================================================
# 创建目录结构
# ============================================================================

def setup_directories():
    """创建研究必需的目录"""
    for d in [DATA_DIR, RESULTS_DIR, LOGS_DIR]:
        os.makedirs(d, exist_ok=True)
    print(f"✓ 目录结构已建立")
    print(f"  数据: {DATA_DIR}")
    print(f"  结果: {RESULTS_DIR}")
    print(f"  日志: {LOGS_DIR}")


# ============================================================================
# 数据获取
# ============================================================================

def fetch_stock_data_akshare(stock_code, stock_name, output_file):
    """从AKShare获取股票数据"""
    print(f"\n获取 {stock_name} ({stock_code}) 的数据...")
    
    if not ak:
        print(f"✗ akshare未安装，跳过数据获取")
        return None
    
    try:
        # 转换日期格式为AKShare需要的格式
        start = START_DATE.replace('-', '')
        end = END_DATE.replace('-', '')
        
        df = ak.stock_zh_a_hist(
            symbol=stock_code.split('.')[0],  # 去掉市场后缀
            start_date=start,
            end_date=end,
            adjust='qfq'  # 前复权
        )
        
        if df is None or df.empty:
            print(f"✗ 未能获取 {stock_name} 的数据")
            return None
        
        # 标准化列名
        df_clean = pd.DataFrame({
            'Date': pd.to_datetime(df['日期']),
            'Open': pd.to_numeric(df['开盘'], errors='coerce'),
            'High': pd.to_numeric(df['最高'], errors='coerce'),
            'Low': pd.to_numeric(df['最低'], errors='coerce'),
            'Close': pd.to_numeric(df['收盘'], errors='coerce'),
            'Volume': pd.to_numeric(df['成交量'], errors='coerce')
        })
        
        # 按日期排序
        df_clean = df_clean.sort_values('Date').reset_index(drop=True)
        
        # 处理缺失值
        df_clean = df_clean.dropna(subset=['Close', 'Volume'])
        
        # 保存
        df_clean.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"✓ 已保存 {stock_name} 数据: {len(df_clean)} 行")
        print(f"  日期范围: {df_clean['Date'].min()} 至 {df_clean['Date'].max()}")
        
        return output_file
        
    except Exception as e:
        print(f"✗ 获取 {stock_name} 数据失败: {e}")
        return None


def fetch_all_data():
    """获取新股票对的全部数据"""
    print("=" * 70)
    print("第一步：获取股票数据")
    print("=" * 70)
    
    fetch_stock_data_akshare(STOCK1_CODE, STOCK1_NAME, STOCK1_FILE)
    fetch_stock_data_akshare(STOCK2_CODE, STOCK2_NAME, STOCK2_FILE)
    
    # 检查数据是否已获取
    if os.path.exists(STOCK1_FILE) and os.path.exists(STOCK2_FILE):
        print(f"\n✓ 股票数据已就绪")
        return True
    else:
        print(f"\n⚠ 数据获取不完整，请检查网络或AKShare库")
        return False


# ============================================================================
# 防泄露数据处理
# ============================================================================

def anti_leakage_prepare():
    """防泄露数据处理：严格时间划分、冻结参数、生成特征"""
    print("\n" + "=" * 70)
    print("第二步：防泄露数据处理")
    print("=" * 70)
    
    if not (os.path.exists(STOCK1_FILE) and os.path.exists(STOCK2_FILE)):
        print("✗ 缺少股票数据文件")
        return False
    
    try:
        # 使用 prepare_strict_pair 函数（与基准案例相同的流程）
        train_csv, val_csv, test_csv, stats_json = prepare_strict_pair(
            STOCK1_FILE,
            STOCK2_FILE,
            STRICT_OUTPUT_DIR,
            train_end=TRAIN_END,
            val_end=VAL_END
        )
        
        print(f"\n✓ 防泄露数据处理完成")
        print(f"  训练集: {train_csv}")
        print(f"  验证集: {val_csv}")
        print(f"  测试集: {test_csv}")
        print(f"  统计: {stats_json}")
        
        # 保存数据划分日志
        split_log = {
            'pair': PAIR_NAME,
            'stock1': f'{STOCK1_NAME}_{STOCK1_CODE}',
            'stock2': f'{STOCK2_NAME}_{STOCK2_CODE}',
            'data_range': f'{START_DATE} 至 {END_DATE}',
            'train_end': TRAIN_END,
            'val_end': VAL_END,
            'train_size': len(pd.read_csv(train_csv)),
            'val_size': len(pd.read_csv(val_csv)),
            'test_size': len(pd.read_csv(test_csv))
        }
        
        split_log_file = os.path.join(RESULTS_DIR, 'data_split_log.json')
        with open(split_log_file, 'w', encoding='utf-8') as f:
            json.dump(split_log, f, indent=2, ensure_ascii=False)
        
        print(f"  数据划分日志: {split_log_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ 防泄露处理失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# 协整检验
# ============================================================================

def cointegration_test():
    """执行Engle-Granger协整检验"""
    print("\n" + "=" * 70)
    print("第三步：协整检验 (Engle-Granger)")
    print("=" * 70)
    
    if not coint:
        print("✗ statsmodels未安装，无法执行协整检验")
        return False
    
    train_csv = os.path.join(STRICT_OUTPUT_DIR, 'pair_train.csv')
    if not os.path.exists(train_csv):
        print(f"✗ 找不到训练集文件: {train_csv}")
        return False
    
    try:
        df_train = pd.read_csv(train_csv)
        
        # 从训练集中提取收盘价（需要重新加载原始数据以获得Close价格）
        df1 = pd.read_csv(STOCK1_FILE)
        df2 = pd.read_csv(STOCK2_FILE)
        df1['Date'] = pd.to_datetime(df1['Date'])
        df2['Date'] = pd.to_datetime(df2['Date'])
        
        # 转换train_end为日期对象
        train_end_dt = pd.to_datetime(TRAIN_END)
        
        # 获取训练集内的收盘价
        close1_train = df1[df1['Date'] <= train_end_dt]['Close'].values
        close2_train = df2[df2['Date'] <= train_end_dt]['Close'].values
        
        # 确保长度匹配
        min_len = min(len(close1_train), len(close2_train))
        close1_train = close1_train[-min_len:]
        close2_train = close2_train[-min_len:]
        
        # 执行协整检验
        coint_stat, pvalue, _ = coint(close1_train, close2_train)
        
        # 保存结果（转换bool为字符串）
        coint_result = {
            'test_type': 'Engle-Granger Cointegration Test',
            'stock_pair': f'{STOCK1_NAME}({STOCK1_CODE}) vs {STOCK2_NAME}({STOCK2_CODE})',
            'test_data': f'Training Set (up to {TRAIN_END})',
            'sample_size': int(min_len),
            'test_statistic': float(coint_stat),
            'p_value': float(pvalue),
            'significant_at_0.05': bool(pvalue < 0.05),
            'significant_at_0.01': bool(pvalue < 0.01),
            'interpretation': 'Cointegrated' if pvalue < 0.05 else 'Not Cointegrated'
        }
        
        # 输出结果
        coint_file = os.path.join(RESULTS_DIR, f'coint_test_{PAIR_NAME}.json')
        with open(coint_file, 'w', encoding='utf-8') as f:
            json.dump(coint_result, f, indent=2, ensure_ascii=False)
        
        # 同时输出简洁的文本格式
        coint_txt = os.path.join(RESULTS_DIR, f'coint_test_{PAIR_NAME}.txt')
        with open(coint_txt, 'w', encoding='utf-8') as f:
            f.write(f"Engle-Granger Cointegration Test\n")
            f.write(f"Test Statistic: {coint_stat:.6f}\n")
            f.write(f"P-Value: {pvalue:.6e}\n")
        
        print(f"✓ 协整检验完成")
        print(f"  检验统计量: {coint_stat:.6f}")
        print(f"  p值: {pvalue:.6e}")
        print(f"  显著性(α=0.05): {'是' if pvalue < 0.05 else '否'}")
        print(f"  结果文件: {coint_file}")
        print(f"  文本输出: {coint_txt}")
        
        return True
        
    except Exception as e:
        print(f"✗ 协整检验失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# 生成对比报告
# ============================================================================

def generate_comparison_report():
    """生成与基准案例的对比报告"""
    print("\n" + "=" * 70)
    print("第四步：生成对比报告")
    print("=" * 70)
    
    try:
        # 读取新配对的统计
        new_stats_file = os.path.join(STRICT_OUTPUT_DIR, 'train_stats.json')
        with open(new_stats_file, 'r', encoding='utf-8') as f:
            new_stats = json.load(f)
        
        # 读取数据划分信息
        split_log_file = os.path.join(RESULTS_DIR, 'data_split_log.json')
        with open(split_log_file, 'r', encoding='utf-8') as f:
            split_info = json.load(f)
        
        # 尝试读取协整检验结果，如果失败则跳过
        coint_result = None
        coint_file = os.path.join(RESULTS_DIR, f'coint_test_{PAIR_NAME}.json')
        if os.path.exists(coint_file):
            try:
                with open(coint_file, 'r', encoding='utf-8') as f:
                    coint_result = json.load(f)
            except:
                # 尝试从txt文件读取
                coint_txt = os.path.join(RESULTS_DIR, f'coint_test_{PAIR_NAME}.txt')
                if os.path.exists(coint_txt):
                    with open(coint_txt, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) >= 3:
                            coint_result = {
                                'test_type': lines[0].strip(),
                                'test_statistic_str': lines[1].strip(),
                                'p_value_str': lines[2].strip()
                            }
        
        # 生成详细报告
        report = {
            'title': '配对交易研究复现报告',
            'timestamp': datetime.now().isoformat(),
            'research_pair': {
                'stock1': f'{STOCK1_NAME} ({STOCK1_CODE})',
                'stock2': f'{STOCK2_NAME} ({STOCK2_CODE})',
                'pair_name': PAIR_NAME
            },
            'data_overview': {
                'data_range': f'{START_DATE} 至 {END_DATE}',
                'train_period': f'{START_DATE} 至 {TRAIN_END}',
                'val_period': f'{TRAIN_END} + 1 至 {VAL_END}',
                'test_period': f'{VAL_END} + 1 至 {END_DATE}',
                'train_samples': split_info['train_size'],
                'val_samples': split_info['val_size'],
                'test_samples': split_info['test_size'],
                'total_samples': split_info['train_size'] + split_info['val_size'] + split_info['test_size']
            },
            'cointegration_test': coint_result if coint_result else 'Not completed',
            'training_statistics': new_stats,
            'methodology': {
                'data_splitting': '严格时间切分，防止未来信息泄露',
                'feature_engineering': '滚动特征（MA5, MA20, Bollinger Band, Z-Score）',
                'normalization': '使用训练集冻结参数进行所有数据集的标准化',
                'tools': ['pandas', 'numpy', 'statsmodels']
            },
            'output_files': {
                'training_data': os.path.join(STRICT_OUTPUT_DIR, 'pair_train.csv'),
                'validation_data': os.path.join(STRICT_OUTPUT_DIR, 'pair_val.csv'),
                'test_data': os.path.join(STRICT_OUTPUT_DIR, 'pair_test.csv'),
                'all_data': os.path.join(STRICT_OUTPUT_DIR, 'pair_all.csv'),
                'training_stats': new_stats_file,
                'split_log': split_log_file
            }
        }
        
        # 保存报告
        report_file = os.path.join(RESULTS_DIR, f'{PAIR_NAME}_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 对比报告生成完成")
        print(f"  报告文件: {report_file}")
        
        # 输出摘要
        print(f"\n数据摘要:")
        print(f"  训练集: {split_info['train_size']} 行 ({START_DATE} - {TRAIN_END})")
        print(f"  验证集: {split_info['val_size']} 行 ({TRAIN_END} - {VAL_END})")
        print(f"  测试集: {split_info['test_size']} 行 ({VAL_END} - {END_DATE})")
        
        if coint_result:
            print(f"\n协整检验:")
            if isinstance(coint_result, dict) and 'test_statistic' in coint_result:
                print(f"  统计量: {coint_result['test_statistic']:.6f}")
                print(f"  p值: {coint_result['p_value']:.6e}")
                print(f"  结论: {coint_result['interpretation']}")
            else:
                print(f"  {coint_result}")
        
        return True
        
    except Exception as e:
        print(f"✗ 生成报告失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# 主流程
# ============================================================================

def main():
    """执行完整研究流程"""
    print("\n" + "=" * 70)
    print("配对交易严格研究复现 - 工商银行 vs 建设银行")
    print("=" * 70)
    print(f"研究目录: {RESEARCH_ROOT}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. 建立目录
    setup_directories()
    
    # 2. 获取数据
    if not fetch_all_data():
        print("\n⚠ 数据获取步骤可能不完整，继续使用本地数据")
    
    # 3. 防泄露处理
    if not anti_leakage_prepare():
        print("\n✗ 防泄露处理失败，无法继续")
        return False
    
    # 4. 协整检验
    if not cointegration_test():
        print("\n⚠ 协整检验失败，但不影响整体流程")
    
    # 5. 生成报告
    if not generate_comparison_report():
        print("\n✗ 报告生成失败")
        return False
    
    print("\n" + "=" * 70)
    print("✓ 研究流程执行完毕")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果目录: {RESULTS_DIR}")
    print("=" * 70)
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
