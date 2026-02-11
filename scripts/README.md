# Scripts 工具脚本文件夹

这个文件夹包含所有辅助工具脚本。按功能分为三类：

---

## 📁 data_prep/ - 数据准备

**功能**：从原始数据生成训练数据集

| 脚本 | 用途 |
|------|------|
| `create_clean_pair_dataset.py` | 生成清洁的配对数据集（计算特征、处理缺失值）|

**使用流程**：
```bash
# 1. 获取原始数据
python fetch_stock_data.py

# 2. 清理并生成配对数据集
python scripts/data_prep/create_clean_pair_dataset.py

# 3. 输出结果
# - data/pair_NINGDE_BYD.csv
# - config/train_stats.json（防止数据泄露）
```

---

## 📁 evaluation/ - 性能评估

**功能**：评估和分析模型性能

| 脚本 | 用途 |
|------|------|
| `benchmark_strategies.py` | 对比多种策略（随机、Buy&Hold、PPO） |
| `calc_performance_metrics.py` | 计算详细指标（收益率、回撤、夏普比） |
| `audit_retrained_model.py` | 深度分析交易日志和模型行为 |

**使用流程**：
```bash
# 1. 快速对比
python scripts/evaluation/benchmark_strategies.py

# 2. 详细指标
python scripts/evaluation/calc_performance_metrics.py

# 3. 深度分析
python scripts/evaluation/audit_retrained_model.py
```

---

## 🔄 其他脚本（根目录）

| 脚本 | 位置 | 用途 |
|------|------|------|
| `retrain_with_clean_data.py` | scripts/ | 基于清洁数据重新训练 |
| `run_pair_300750_002460.py` | scripts/ | 特定配对的交易脚本（实验） |

---

## 快速命令

```bash
# 完整的数据处理和训练流水线
cd ../
python fetch_stock_data.py && \
python scripts/data_prep/create_clean_pair_dataset.py && \
python scripts/retrain_with_clean_data.py && \
python scripts/evaluation/benchmark_strategies.py

# 只评估现有模型
python scripts/evaluation/benchmark_strategies.py && \
python scripts/evaluation/calc_performance_metrics.py
```

---

**建议**: 查看根目录的 `SCRIPTS.md` 了解更详细的说明。
