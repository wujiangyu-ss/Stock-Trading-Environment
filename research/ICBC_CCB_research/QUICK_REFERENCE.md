# 快速参考卡片

## 核心文件速查

### 🎯 立即查看
```bash
# 查看结果摘要（<1秒）
python research/ICBC_CCB_research/view_results.py

# 查看完整报告
cat research/ICBC_CCB_research/README.md
```

### 📊 关键数据
```
研究配对: 工商银行(601398) / 建设银行(601939)
数据范围: 2020-01-02 至 2025-12-31 (1455行)
训练集: 970行 (66.7%) | 验证集: 242行 (16.6%) | 测试集: 243行 (16.7%)
协整性: p=0.000339 ✓ Cointegrated
```

### 📁 数据路径
```
research/ICBC_CCB_research/data/strict/
├── pair_train.csv          # 训练数据 (970行, 11列)
├── pair_val.csv            # 验证数据 (242行, 11列)  
├── pair_test.csv           # 测试数据 (243行, 11列)
├── pair_all.csv            # 原始合并 (1455行, 6列)
└── train_stats.json        # 冻结参数
```

### 📄 文档位置
```
research/ICBC_CCB_research/
├── README.md               # 快速导航 ⭐
├── EXECUTION_REPORT.md     # 执行报告
└── RESEARCH_SUMMARY.md     # 详细总结
```

### 🔍 结果文件
```
research/ICBC_CCB_research/results/
├── 工商银行_建设银行_ICBC_CCB_report.json
├── research_summary_report.json
├── coint_test_*.json (协整检验)
├── data_split_log.json
├── cointegration_comparison.json
└── data_structure_comparison.csv
```

---

## 防泄露检查清单 ✓

| 检查项 | 状态 |
|--------|------|
| 时间边界分离 | ✓ (无重叠) |
| 参数冻结 | ✓ (仅训练集) |
| 验证集隔离 | ✓ (无参数泄露) |
| 测试集隔离 | ✓ (无参数泄露) |
| 因果性维护 | ✓ (滚动窗口历史) |
| 前向偏差检查 | ✓ (无未来数据) |

---

## 关键指标

```json
{
  "pair": "ICBC(601398) vs CCB(601939)",
  "data_samples": 1455,
  "train_samples": 970,
  "test_coverage": "70% train, 15% val, 15% test",
  
  "cointegration_test": {
    "statistic": -4.826360,
    "p_value": 0.000339,
    "significant": "Yes (α=0.05, α=0.01)",
    "conclusion": "Cointegrated"
  },
  
  "key_features": [
    "zscore",
    "price_spread", 
    "price_ratio",
    "volume_ratio",
    "spread_ma5",
    "spread_ma20",
    "spread_std",
    "bollinger_high",
    "bollinger_low"
  ]
}
```

---

## 使用特征集进行模型训练

### 推荐配置
```python
# 加载训练数据
import pandas as pd
X_train = pd.read_csv('research/ICBC_CCB_research/data/strict/pair_train.csv')
X_val = pd.read_csv('research/ICBC_CCB_research/data/strict/pair_val.csv')
X_test = pd.read_csv('research/ICBC_CCB_research/data/strict/pair_test.csv')

# 特征列（11列）
features = ['Date', 'price_ratio', 'price_spread', 'zscore', 
            'spread_ma5', 'spread_ma20', 'spread_std', 
            'bollinger_high', 'bollinger_low', 'volume_ratio']

# 时间戳
X_train['Date'] = pd.to_datetime(X_train['Date'])
```

### PPO模型训练
```bash
cd research/ICBC_CCB_research
python ../../03_train_ppo_model.py \
  --data_path data/strict/pair_train.csv \
  --total_timesteps 100000
```

---

## 对比基准案例

### 配对1: 宁德时代(300750) / 比亚迪(002594)
- 行业：新能源汽车
- 数据：6年
- 协整：✓ (已验证)
- 状态：基准案例

### 配对2: 工商银行(601398) / 建设银行(601939) ← **本研究**
- 行业：金融/银行
- 数据：6年
- 协整：✓ p=0.000339 (**更显著**)
- 状态：新研究完成

---

## 常见操作

### 查看训练集统计
```bash
cd research/ICBC_CCB_research
python -c "
import json
with open('data/strict/train_stats.json') as f:
    stats = json.load(f)
    for feat, vals in stats.items():
        print(f'{feat}: μ={vals[\"mean\"]:.4f}, σ={vals[\"std\"]:.4f}')
"
```

### 加载和检查数据
```bash
python -c "
import pandas as pd
df = pd.read_csv('research/ICBC_CCB_research/data/strict/pair_train.csv')
print(f'Shape: {df.shape}')
print(f'Columns: {list(df.columns)}')
print(f'Missing: {df.isnull().sum().sum()}')
"
```

### 重新执行研究
```bash
cd research/ICBC_CCB_research
python run_strict_research.py
```

---

## 质量指标

| 指标 | 数值 | 评分 |
|------|------|------|
| 数据完整性 | 100% (无缺失) | ✓✓✓ |
| 防泄露严谨性 | 完全隔离 | ✓✓✓ |
| 统计显著性 | p<0.001 | ✓✓✓ |
| 可复现性 | 开源+公开数据 | ✓✓✓ |
| 文档完整性 | 3份详细文档 | ✓✓✓ |

---

## 下一步行动 (优先级)

1. **🔴 立即** - 阅读 `research/ICBC_CCB_research/README.md`
2. **🟡 本周** - 使用 `pair_train.csv` 训练PPO模型
3. **🟢 后续** - 与基准模型进行性能对标
4. **🔵 可选** - 优化交易信号参数

---

**生成时间**: 2026-02-11  
**项目状态**: ✓ 完成，所有数据就绪  
**下一步**: 模型训练
