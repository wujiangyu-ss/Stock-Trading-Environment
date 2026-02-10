"""
配对交易数据预处理管道
将两支股票的原始OHLCV数据转换为配对交易特征数据集
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class PairTradingDataPipeline:
    """配对交易数据预处理管道"""
    
    def __init__(self, stock1_path, stock2_path, stock1_name, stock2_name):
        """
        初始化管道
        
        Args:
            stock1_path: 第一只股票CSV文件路径
            stock2_path: 第二只股票CSV文件路径
            stock1_name: 第一只股票名称
            stock2_name: 第二只股票名称
        """
        self.stock1_path = stock1_path
        self.stock2_path = stock2_path
        self.stock1_name = stock1_name
        self.stock2_name = stock2_name
        self.pair_data = None
        
    def load_data(self):
        """加载两支股票数据"""
        print("=" * 60)
        print("第一步：加载数据")
        print("=" * 60)
        
        try:
            df1 = pd.read_csv(self.stock1_path)
            df2 = pd.read_csv(self.stock2_path)
            
            # 转换日期列为datetime类型
            df1['Date'] = pd.to_datetime(df1['Date'])
            df2['Date'] = pd.to_datetime(df2['Date'])
            
            print(f"✓ 加载 {self.stock1_name}: {len(df1)} 行数据")
            print(f"✓ 加载 {self.stock2_name}: {len(df2)} 行数据")
            
            return df1, df2
            
        except Exception as e:
            print(f"✗ 加载数据失败: {str(e)}")
            return None, None
    
    def align_data(self, df1, df2):
        """对齐两支股票的日期"""
        print("\n" + "=" * 60)
        print("第二步：对齐日期")
        print("=" * 60)
        
        # 内连接：只保留两个数据集都有的日期
        merged = pd.merge(
            df1[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(
                columns={'Open': f'{self.stock1_name}_Open',
                        'High': f'{self.stock1_name}_High',
                        'Low': f'{self.stock1_name}_Low',
                        'Close': f'{self.stock1_name}_Close',
                        'Volume': f'{self.stock1_name}_Volume'}
            ),
            df2[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].rename(
                columns={'Open': f'{self.stock2_name}_Open',
                        'High': f'{self.stock2_name}_High',
                        'Low': f'{self.stock2_name}_Low',
                        'Close': f'{self.stock2_name}_Close',
                        'Volume': f'{self.stock2_name}_Volume'}
            ),
            on='Date',
            how='inner'
        )
        
        merged = merged.sort_values('Date').reset_index(drop=True)
        
        print(f"✓ 对齐后数据行数: {len(merged)}")
        print(f"  日期范围: {merged['Date'].min().date()} 到 {merged['Date'].max().date()}")
        
        return merged
    
    def calculate_pair_features(self, merged_df):
        """计算配对交易特征"""
        print("\n" + "=" * 60)
        print("第三步：计算配对特征")
        print("=" * 60)
        
        df = merged_df.copy()
        
        # 1. 价格比（Close）
        df['price_ratio'] = (df[f'{self.stock1_name}_Close'] / 
                            df[f'{self.stock2_name}_Close'])
        print("✓ 计算价格比")
        
        # 2. 价差（Close）
        df['price_spread'] = (df[f'{self.stock1_name}_Close'] - 
                             df[f'{self.stock2_name}_Close'])
        print("✓ 计算价差")
        
        # 3. 价差的移动平均（5日和20日）
        df['spread_ma5'] = df['price_spread'].rolling(window=5, min_periods=1).mean()
        df['spread_ma20'] = df['price_spread'].rolling(window=20, min_periods=1).mean()
        print("✓ 计算5日和20日移动平均")
        
        # 计算滚动标准差（20日）
        df['spread_std'] = df['price_spread'].rolling(window=20, min_periods=1).std()
        # 处理NaN值
        df['spread_std'] = df['spread_std'].fillna(0)
        print("✓ 计算滚动标准差")
        
        # 5. Z-Score（价差相对于20日均线的标准化得分）
        df['zscore'] = np.where(
            df['spread_std'] > 0,
            (df['price_spread'] - df['spread_ma20']) / df['spread_std'],
            0
        )
        print("✓ 计算Z-Score")
        
        # 6. 布林带（20日均线 ± 2倍标准差）
        df['bollinger_high'] = df['spread_ma20'] + 2 * df['spread_std']
        df['bollinger_low'] = df['spread_ma20'] - 2 * df['spread_std']
        print("✓ 计算布林带")
        
        # 7. 成交量比
        df['volume_ratio'] = (df[f'{self.stock1_name}_Volume'] / 
                             (df[f'{self.stock2_name}_Volume'] + 1e-6))  # 避免除以0
        print("✓ 计算成交量比")
        
        return df
    
    def clean_data(self, df):
        """清洗和验证数据"""
        print("\n" + "=" * 60)
        print("第四步：数据清洗和验证")
        print("=" * 60)
        
        # 检查NaN值
        nan_count = df[['price_ratio', 'price_spread', 'zscore', 'spread_ma5', 
                        'spread_ma20', 'spread_std', 'bollinger_high', 
                        'bollinger_low', 'volume_ratio']].isna().sum().sum()
        
        if nan_count > 0:
            print(f"  检测到 {nan_count} 个NaN值，正在处理...")
            # 向前填充，然后向后填充
            df = df.ffill().bfill()
            # 如果还有NaN，用0填充
            df = df.fillna(0)
            print("  ✓ NaN值已处理")
        
        # 检查异常值（价格比应该是正数）
        invalid_ratio = (df['price_ratio'] <= 0).sum()
        if invalid_ratio > 0:
            print(f"  ✗ 发现 {invalid_ratio} 个无效的价格比")
            df = df[df['price_ratio'] > 0].reset_index(drop=True)
        
        print(f"✓ 数据清洗完成，最终行数: {len(df)}")
        
        return df
    
    def select_output_columns(self, df):
        """选择输出列"""
        output_columns = [
            'Date',
            'price_ratio',
            'price_spread',
            'zscore',
            'spread_ma5',
            'spread_ma20',
            'spread_std',
            'bollinger_high',
            'bollinger_low',
            'volume_ratio'
        ]
        
        return df[output_columns].copy()
    
    def save_data(self, df, output_path):
        """保存数据为CSV文件"""
        print("\n" + "=" * 60)
        print("第五步：保存数据")
        print("=" * 60)
        
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 转换Date为字符串格式
            df_save = df.copy()
            df_save['Date'] = df_save['Date'].dt.strftime('%Y-%m-%d')
            
            df_save.to_csv(output_path, index=False, encoding='utf-8')
            print(f"✓ 数据已保存: {output_path}")
            
            return True
        except Exception as e:
            print(f"✗ 保存失败: {str(e)}")
            return False
    
    def visualize(self, df, output_image_path):
        """绘制价差和Z-Score的可视化图表"""
        print("\n" + "=" * 60)
        print("第六步：数据可视化")
        print("=" * 60)
        
        try:
            fig, axes = plt.subplots(3, 1, figsize=(14, 10))
            
            # 转换Date为datetime（如果还不是）
            dates = pd.to_datetime(df['Date']) if isinstance(df['Date'].iloc[0], str) else df['Date']
            
            # 图1：价差及其布林带
            ax1 = axes[0]
            ax1.plot(dates, df['price_spread'], label='价差', color='blue', linewidth=1.5)
            ax1.plot(dates, df['spread_ma20'], label='20日均线', color='green', 
                    linewidth=1, linestyle='--', alpha=0.7)
            ax1.fill_between(dates, df['bollinger_high'], df['bollinger_low'], 
                            alpha=0.2, color='gray', label='布林带')
            ax1.set_ylabel('价差（元）', fontsize=11)
            ax1.set_title(f'{self.stock1_name} vs {self.stock2_name} - 价差与布林带', 
                         fontsize=12, fontweight='bold')
            ax1.legend(loc='best')
            ax1.grid(True, alpha=0.3)
            
            # 图2：Z-Score
            ax2 = axes[1]
            colors = ['red' if z > 2 else 'green' if z < -2 else 'blue' 
                     for z in df['zscore']]
            ax2.scatter(dates, df['zscore'], c=colors, s=10, alpha=0.6)
            ax2.axhline(y=2, color='red', linestyle='--', linewidth=1, alpha=0.7, label='超买 (Z=2)')
            ax2.axhline(y=-2, color='green', linestyle='--', linewidth=1, alpha=0.7, label='超卖 (Z=-2)')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
            ax2.set_ylabel('Z-Score', fontsize=11)
            ax2.set_title('价差Z-Score（标准化得分）', fontsize=12, fontweight='bold')
            ax2.legend(loc='best')
            ax2.grid(True, alpha=0.3)
            
            # 图3：价格比
            ax3 = axes[2]
            ax3.plot(dates, df['price_ratio'], color='purple', linewidth=1.5)
            ax3.set_xlabel('日期', fontsize=11)
            ax3.set_ylabel('价格比', fontsize=11)
            ax3.set_title(f'价格比 ({self.stock1_name}Close / {self.stock2_name}Close)', 
                         fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            os.makedirs(os.path.dirname(output_image_path) if os.path.dirname(output_image_path) else '.', 
                       exist_ok=True)
            plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
            print(f"✓ 可视化图表已保存: {output_image_path}")
            
            plt.close()
            
        except Exception as e:
            print(f"✗ 绘图失败: {str(e)}")
    
    def run(self, output_csv_path, output_image_path=None):
        """运行完整的管道"""
        print("\n")
        print("╔" + "=" * 58 + "╗")
        print("║" + " 配对交易数据预处理管道 ".center(58) + "║")
        print("║" + f" {self.stock1_name} vs {self.stock2_name} ".center(58) + "║")
        print("╚" + "=" * 58 + "╝")
        
        # 第一步：加载数据
        df1, df2 = self.load_data()
        if df1 is None or df2 is None:
            return False
        
        # 第二步：对齐数据
        merged_df = self.align_data(df1, df2)
        
        # 第三步：计算特征
        feature_df = self.calculate_pair_features(merged_df)
        
        # 第四步：数据清洗
        clean_df = self.clean_data(feature_df)
        
        # 第五步：选择输出列
        output_df = self.select_output_columns(clean_df)
        
        # 第六步：保存数据
        if not self.save_data(output_df, output_csv_path):
            return False
        
        # 第七步：可视化（可选）
        if output_image_path:
            self.visualize(output_df, output_image_path)
        
        # 显示数据统计
        self._print_statistics(output_df)
        
        print("\n" + "╔" + "=" * 58 + "╗")
        print("║" + " 处理完成！".center(58) + "║")
        print("╚" + "=" * 58 + "╝\n")
        
        self.pair_data = output_df
        return True
    
    def _print_statistics(self, df):
        """打印数据统计信息"""
        print("\n" + "=" * 60)
        print("数据统计信息")
        print("=" * 60)
        print(f"总行数: {len(df)}")
        print(f"日期范围: {df['Date'].min()} 到 {df['Date'].max()}")
        print(f"\n价差统计:")
        print(f"  最小值: {df['price_spread'].min():.4f}")
        print(f"  最大值: {df['price_spread'].max():.4f}")
        print(f"  平均值: {df['price_spread'].mean():.4f}")
        print(f"  标准差: {df['price_spread'].std():.4f}")
        print(f"\nZ-Score统计:")
        print(f"  最小值: {df['zscore'].min():.4f}")
        print(f"  最大值: {df['zscore'].max():.4f}")
        print(f"  平均值: {df['zscore'].mean():.4f}")
        print(f"\n价格比统计:")
        print(f"  最小值: {df['price_ratio'].min():.4f}")
        print(f"  最大值: {df['price_ratio'].max():.4f}")
        print(f"  平均值: {df['price_ratio'].mean():.4f}")


def main():
    """主函数"""
    
    # 配置参数
    DATA_DIR = './data'
    OUTPUT_DIR = './data'
    
    # 创建管道实例（可轻松更换为其他股票对）
    pipeline = PairTradingDataPipeline(
        stock1_path=os.path.join(DATA_DIR, '300750.csv'),
        stock2_path=os.path.join(DATA_DIR, '002594.csv'),
        stock1_name='宁德时代',
        stock2_name='比亚迪'
    )
    
    # 运行管道
    output_csv = os.path.join(OUTPUT_DIR, 'pair_NINGDE_BYD.csv')
    output_image = os.path.join(OUTPUT_DIR, 'pair_spread.png')
    
    success = pipeline.run(output_csv, output_image)
    
    if success:
        print(f"\n✓ 配对交易数据已准备好！")
        print(f"  CSV文件: {output_csv}")
        print(f"  图表文件: {output_image}")
    else:
        print(f"\n✗ 处理失败")


if __name__ == "__main__":
    main()
