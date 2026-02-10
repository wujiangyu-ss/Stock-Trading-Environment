"""
生成 宁德时代 (300750) vs 赣锋锂业 (002460) 的配对数据（使用已有的 PairTradingDataPipeline）
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from create_pair_trading_data import PairTradingDataPipeline

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_DIR = os.path.normpath(DATA_DIR)

stock1_path = os.path.join(DATA_DIR, '300750.csv')
stock2_path = os.path.join(DATA_DIR, '002460.csv')

pipeline = PairTradingDataPipeline(
    stock1_path=stock1_path,
    stock2_path=stock2_path,
    stock1_name='宁德时代',
    stock2_name='赣锋锂业'
)

output_csv = os.path.join(DATA_DIR, 'pair_NINGDE_GANFENG.csv')
output_image = os.path.join(DATA_DIR, 'pair_spread_300750_002460.png')

if __name__ == '__main__':
    ok = pipeline.run(output_csv, output_image)
    if not ok:
        raise SystemExit('配对数据生成失败')
    print('配对数据生成完成:', output_csv)
