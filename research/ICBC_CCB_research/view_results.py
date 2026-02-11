"""
快速查看研究结果
"""
import os
import json
import pandas as pd

results_dir = os.path.join(os.path.dirname(__file__), 'results')
data_dir = os.path.join(os.path.dirname(__file__), 'data', 'strict')

print("\n" + "=" * 80)
print("研究结果概览 - 工商银行(601398) / 建设银行(601939)")
print("=" * 80)

# 数据划分
print("\n【数据划分】")
split_log = os.path.join(results_dir, 'data_split_log.json')
if os.path.exists(split_log):
    with open(split_log, 'r', encoding='utf-8') as f:
        split = json.load(f)
    print(f"  股票对: {split['stock1']} vs {split['stock2']}")
    print(f"  数据范围: {split['data_range']}")
    print(f"  训练集: {split['train_size']} 行 (2020-01-01 至 {split['train_end']})")
    print(f"  验证集: {split['val_size']} 行 ({split['train_end']} 至 {split['val_end']})")
    print(f"  测试集: {split['test_size']} 行 (之后)")

# 协整检验
print("\n【协整检验结果】")
coint_file = os.path.join(results_dir, 'coint_test_工商银行_建设银行_ICBC_CCB.json')
if os.path.exists(coint_file):
    with open(coint_file, 'r', encoding='utf-8') as f:
        coint = json.load(f)
    print(f"  测试方法: {coint['test_type']}")
    print(f"  检验统计量: {coint['test_statistic']:.6f}")
    print(f"  P值: {coint['p_value']:.6e}")
    print(f"  α=0.05显著: {coint['significant_at_0.05']}")
    print(f"  结论: {coint['interpretation']} {'✓' if coint['interpretation'] == 'Cointegrated' else '✗'}")

# 特征统计
print("\n【训练集特征统计】")
stats_file = os.path.join(data_dir, 'train_stats.json')
if os.path.exists(stats_file):
    with open(stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    features_to_show = ['zscore', 'price_spread', 'price_ratio', 'spread_ma20']
    for feat in features_to_show:
        if feat in stats:
            s = stats[feat]
            print(f"\n  {feat}:")
            print(f"    均值: {s['mean']:.6f}")
            print(f"    标准差: {s['std']:.6f}")
            print(f"    范围: [{s['min']:.6f}, {s['max']:.6f}]")

# 输出文件列表
print("\n【生成的文件】")
if os.path.exists(results_dir):
    files = sorted(os.listdir(results_dir))
    for f in files:
        full_path = os.path.join(results_dir, f)
        size = os.path.getsize(full_path)
        print(f"  {f:<50} ({size:>6} bytes)")

print("\n" + "=" * 80)
print("✓ 研究流程执行完毕，所有结果已保存")
print("=" * 80 + "\n")
