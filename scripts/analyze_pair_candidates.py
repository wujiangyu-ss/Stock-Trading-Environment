"""
配对候选库分析与筛选工具
用于系统化地分析和筛选满足条件的股票配对

使用方式：
    python analyze_pair_candidates.py --stocks 002074 300014 002460 002466
    python analyze_pair_candidates.py --industry battery
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# 配对候选库定义
PAIR_CANDIDATES = {
    'battery': {
        '国轩高科_亿纬锂能': {
            'stock1': '002074',
            'stock2': '300014',
            'stock1_name': '国轩高科',
            'stock2_name': '亿纬锂能',
            'industry': '锂电池制造',
            'similarity': 'high',
            'reason': '同为锂电池龙头，竞争关系'
        },
        '宁德时代_比亚迪': {
            'stock1': '300750',
            'stock2': '002594',
            'stock1_name': '宁德时代',
            'stock2_name': '比亚迪',
            'industry': '新能源汽车',
            'similarity': 'very_high',
            'reason': '基准案例，极高相关性'
        }
    },
    'lithium': {
        '赣锋锂业_天齐锂业': {
            'stock1': '002460',
            'stock2': '002466',
            'stock1_name': '赣锋锂业',
            'stock2_name': '天齐锂业',
            'industry': '锂矿开采',
            'similarity': 'very_high',
            'reason': '完全相同业务，国际锂价驱动'
        }
    },
    'steel': {
        '宝钢股份_新钢股份': {
            'stock1': '600019',
            'stock2': '600782',
            'stock1_name': '宝钢股份',
            'stock2_name': '新钢股份',
            'industry': '钢铁制造',
            'similarity': 'high',
            'reason': '都是钢铁龙头，成本驱动相似'
        }
    },
    'pharma': {
        '复星医药_同仁堂': {
            'stock1': '600196',
            'stock2': '600085',
            'stock1_name': '复星医药',
            'stock2_name': '同仁堂',
            'industry': '医药生物',
            'similarity': 'medium',
            'reason': '都是医药龙头，但业务有差异'
        }
    }
}


class PairAnalyzer:
    """股票配对分析器"""
    
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        self.pairs_data = {}
        
    def load_pair_data(self, stock1_code, stock2_code, stock1_name='Stock1', stock2_name='Stock2'):
        """加载一对股票的数据"""
        try:
            file1 = os.path.join(self.data_dir, f'{stock1_code}.csv')
            file2 = os.path.join(self.data_dir, f'{stock2_code}.csv')
            
            if not os.path.exists(file1) or not os.path.exists(file2):
                print(f"❌ 数据文件不存在: {file1} 或 {file2}")
                return None
            
            # 加载数据
            df1 = pd.read_csv(file1)
            df2 = pd.read_csv(file2)
            
            # 标准化列名
            df1.columns = df1.columns.str.lower()
            df2.columns = df2.columns.str.lower()
            
            # 获取日期和收盘价
            date_col1 = 'date' if 'date' in df1.columns else df1.columns[0]
            date_col2 = 'date' if 'date' in df2.columns else df2.columns[0]
            
            price_col1 = 'close' if 'close' in df1.columns else df1.columns[-1]
            price_col2 = 'close' if 'close' in df2.columns else df2.columns[-1]
            
            df1 = df1[[date_col1, price_col1]].copy()
            df2 = df2[[date_col2, price_col2]].copy()
            
            df1.columns = ['date', 'price']
            df2.columns = ['date', 'price']
            
            # 对齐日期
            df1['date'] = pd.to_datetime(df1['date'])
            df2['date'] = pd.to_datetime(df2['date'])
            
            merged = df1.merge(df2, on='date', how='inner', suffixes=('_1', '_2'))
            
            if len(merged) == 0:
                print(f"❌ 没有共同日期数据")
                return None
            
            print(f"✓ 已加载配对数据:")
            print(f"  {stock1_name} ({stock1_code}): {len(df1)} 条")
            print(f"  {stock2_name} ({stock2_code}): {len(df2)} 条")
            print(f"  共同日期: {len(merged)} 条 ({merged['date'].min().date()} 到 {merged['date'].max().date()})")
            
            return merged
        
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return None
    
    def analyze_correlation(self, df, stock1_name='Stock1', stock2_name='Stock2'):
        """分析两只股票的相关性"""
        
        if df is None or len(df) < 30:
            print("❌ 数据不足")
            return {}
        
        # 基础统计
        price1 = df['price_1'].values
        price2 = df['price_2'].values
        
        # 计算收益率
        ret1 = np.diff(price1) / price1[:-1]
        ret2 = np.diff(price2) / price2[:-1]
        
        # 相关系数（价格级别）
        corr_price = np.corrcoef(price1, price2)[0, 1]
        
        # 相关系数（收益率）
        corr_ret = np.corrcoef(ret1, ret2)[0, 1]
        
        # 价差
        spread = price1 - price2
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        
        # Z-Score
        zscore = (spread - spread_mean) / (spread_std + 1e-8)
        zscore_extremes = np.sum(np.abs(zscore) > 2)  # 极值数量
        
        results = {
            'corr_price': corr_price,
            'corr_ret': corr_ret,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'spread_min': spread.min(),
            'spread_max': spread.max(),
            'zscore_extremes': zscore_extremes,
            'zscore_extremes_pct': zscore_extremes / len(zscore) * 100,
            'price1_mean': price1.mean(),
            'price2_mean': price2.mean(),
            'price1_volatility': np.std(ret1),
            'price2_volatility': np.std(ret2),
        }
        
        print(f"\n📊 {stock1_name} vs {stock2_name} 相关性分析:")
        print(f"  价格相关系数: {results['corr_price']:.4f}")
        print(f"  收益率相关系数: {results['corr_ret']:.4f}")
        print(f"  价差均值: {results['spread_mean']:.4f}")
        print(f"  价差标差: {results['spread_std']:.4f}")
        print(f"  价差范围: [{results['spread_min']:.4f}, {results['spread_max']:.4f}]")
        print(f"  Z-Score > |2| 的比例: {results['zscore_extremes_pct']:.2f}%")
        print(f"  {stock1_name} 波动率: {results['price1_volatility']:.4f}")
        print(f"  {stock2_name} 波动率: {results['price2_volatility']:.4f}")
        
        return results
    
    def plot_pair_analysis(self, df, stock1_name='Stock1', stock2_name='Stock2', save_path=None):
        """绘制配对分析图表"""
        
        if df is None or len(df) < 30:
            print("❌ 数据不足，无法绘图")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 价格走势
        ax = axes[0]
        ax2 = ax.twinx()
        
        ax.plot(df['date'], df['price_1'], label=stock1_name, color='blue', alpha=0.7)
        ax2.plot(df['date'], df['price_2'], label=stock2_name, color='red', alpha=0.7)
        
        ax.set_ylabel(f'{stock1_name} 价格', color='blue')
        ax2.set_ylabel(f'{stock2_name} 价格', color='red')
        ax.set_title(f'{stock1_name} vs {stock2_name} 价格走势对比')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # 价差
        ax = axes[1]
        spread = df['price_1'].values - df['price_2'].values
        spread_ma = pd.Series(spread).rolling(20).mean()
        
        ax.plot(df['date'], spread, label='价差', alpha=0.5, color='gray')
        ax.plot(df['date'], spread_ma, label='20日MA', color='orange', linewidth=2)
        ax.axhline(y=spread.mean(), color='green', linestyle='--', label='均值', linewidth=1)
        ax.set_ylabel('价差')
        ax.set_title('价差与移动平均')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Z-Score
        ax = axes[2]
        spread_std = np.std(spread)
        zscore = (spread - spread.mean()) / (spread_std + 1e-8)
        
        ax.plot(df['date'], zscore, label='Z-Score', color='purple')
        ax.axhline(y=2, color='red', linestyle='--', linewidth=1, label='±2σ')
        ax.axhline(y=-2, color='red', linestyle='--', linewidth=1)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.fill_between(range(len(zscore)), -2, 2, alpha=0.1, color='green')
        ax.set_ylabel('Z-Score')
        ax.set_xlabel('日期')
        ax.set_title('价差 Z-Score')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ 图表已保存: {save_path}")
        
        plt.show()


def print_pair_candidates():
    """打印所有候选配对"""
    print("\n" + "="*80)
    print("📋 可用的股票配对候选库")
    print("="*80)
    
    for industry, pairs in PAIR_CANDIDATES.items():
        print(f"\n🏭 {industry.upper()}:")
        for pair_name, info in pairs.items():
            print(f"  {pair_name}")
            print(f"    {info['stock1_name']} ({info['stock1']}) vs {info['stock2_name']} ({info['stock2']})")
            print(f"    行业: {info['industry']}")
            print(f"    理由: {info['reason']}")


def main():
    parser = argparse.ArgumentParser(
        description='股票配对分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例用法：
  # 分析单个配对
  python analyze_pair_candidates.py --stock1 002074 --stock2 300014
  
  # 分析特定行业的所有配对
  python analyze_pair_candidates.py --industry battery
  
  # 列出所有可用配对
  python analyze_pair_candidates.py --list-all
        '''
    )
    
    parser.add_argument('--stock1', type=str, help='第一只股票代码')
    parser.add_argument('--stock2', type=str, help='第二只股票代码')
    parser.add_argument('--industry', type=str, choices=list(PAIR_CANDIDATES.keys()),
                       help='行业类别')
    parser.add_argument('--list-all', action='store_true', help='列出所有可用配对')
    parser.add_argument('--data-dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--output-dir', type=str, default='./figures', help='输出目录')
    
    args = parser.parse_args()
    
    # 列出所有配对
    if args.list_all:
        print_pair_candidates()
        return
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    analyzer = PairAnalyzer(data_dir=args.data_dir)
    
    # 分析特定行业
    if args.industry:
        print(f"\n分析行业: {args.industry}")
        for pair_name, pair_info in PAIR_CANDIDATES[args.industry].items():
            print(f"\n{'='*80}")
            print(f"配对: {pair_name}")
            print(f"{'='*80}")
            
            df = analyzer.load_pair_data(
                pair_info['stock1'],
                pair_info['stock2'],
                pair_info['stock1_name'],
                pair_info['stock2_name']
            )
            
            if df is not None:
                analyzer.analyze_correlation(
                    df,
                    pair_info['stock1_name'],
                    pair_info['stock2_name']
                )
                
                save_path = os.path.join(
                    args.output_dir,
                    f"pair_analysis_{pair_info['stock1']}_{pair_info['stock2']}.png"
                )
                analyzer.plot_pair_analysis(
                    df,
                    pair_info['stock1_name'],
                    pair_info['stock2_name'],
                    save_path=save_path
                )
    
    # 分析特定配对
    elif args.stock1 and args.stock2:
        df = analyzer.load_pair_data(args.stock1, args.stock2, f'Stock {args.stock1}', f'Stock {args.stock2}')
        
        if df is not None:
            analyzer.analyze_correlation(df, f'Stock {args.stock1}', f'Stock {args.stock2}')
            
            save_path = os.path.join(
                args.output_dir,
                f"pair_analysis_{args.stock1}_{args.stock2}.png"
            )
            analyzer.plot_pair_analysis(df, f'Stock {args.stock1}', f'Stock {args.stock2}', save_path=save_path)
    
    else:
        # 默认分析所有行业
        print_pair_candidates()
        print("\n请使用 --stock1 和 --stock2，或者使用 --industry 指定行业")


if __name__ == '__main__':
    main()
