import os
import sys

# 将研究脚本路径加入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from research.ICBC_CCB_research import run_strict_research as mod

if __name__ == '__main__':
    # 覆盖配对信息
    mod.STOCK1_CODE = '600519.SH'
    mod.STOCK1_NAME = '贵州茅台'
    mod.STOCK1_FILE = os.path.join(mod.DATA_DIR, '600519.csv')

    mod.STOCK2_CODE = '000858.SZ'
    mod.STOCK2_NAME = '五粮液'
    mod.STOCK2_FILE = os.path.join(mod.DATA_DIR, '000858.csv')

    mod.PAIR_NAME = f"{mod.STOCK1_NAME}_{mod.STOCK2_NAME}"
    # 将输出放到 pair 特定子目录，避免覆盖基准的 strict
    mod.STRICT_OUTPUT_DIR = os.path.join(mod.DATA_DIR, 'strict', f"{mod.STOCK1_CODE}_{mod.STOCK2_CODE}")
    os.makedirs(mod.STRICT_OUTPUT_DIR, exist_ok=True)

    # 执行步骤 1-4
    mod.setup_directories()
    mod.fetch_stock_data_akshare(mod.STOCK1_CODE, mod.STOCK1_NAME, mod.STOCK1_FILE)
    mod.fetch_stock_data_akshare(mod.STOCK2_CODE, mod.STOCK2_NAME, mod.STOCK2_FILE)
    mod.anti_leakage_prepare()
    mod.cointegration_test()
    mod.generate_comparison_report()

    print('\nAll steps 1-4 attempted for pair:', mod.PAIR_NAME)
