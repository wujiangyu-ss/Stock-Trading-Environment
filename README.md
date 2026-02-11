# 股票配对交易强化学习环境

基于 Gymnasium 框架的配对交易强化学习环境，用 PPO 算法训练配对交易策略。验证方法在不同股票配对上的普适性。

> **当前进度** 基准案例（宁德 vs 比亚迪）已完成 ✅ | 新配对研究进行中 ⏳

---

## 🎯 项目目标

1. **基准验证** ✅ 宁德时代 vs 比亚迪 - 配对交易策略验证
2. **普适性扩展** ⏳ 新配对研究 - 验证方法在其他股票对的有效性
3. **系统化研究** ⏳ 对比分析 - 理解哪些因素影响策略性能

---

## 📖 如何使用本项目

### 🟢 完全新手？

1. 阅读 **[PROJECT_NAVIGATION.md](PROJECT_NAVIGATION.md)** （5 分钟） - 了解项目结构
2. 查看 **[RESEARCH_STARTUP.md](RESEARCH_STARTUP.md)** （10 分钟） - 环境检查和快速启动
3. 运行第一个命令：`python fetch_stock_data.py --symbols 002466`

### 🟡 想快速上手？

1. 查看 **[PAIR_RESEARCH_QUICKSTART.md](PAIR_RESEARCH_QUICKSTART.md)** - 5 分钟快速指南
2. 执行推荐的配对研究流程

### 🔵 想深入理解？

1. 阅读 **[NEW_PAIR_RESEARCH_PLAN.md](NEW_PAIR_RESEARCH_PLAN.md)** - 完整研究计划
2. 查看 **[docs/WORKFLOW_SUMMARY.md](docs/WORKFLOW_SUMMARY.md)** - 详细工作流程

---

## 📁 项目结构

```
Stock-Trading-Environment/
├── 📚 核心文档（必读）
│   ├── PROJECT_NAVIGATION.md          🗺️ 项目导航（从这里开始）
│   ├── RESEARCH_STARTUP.md           🚀 快速启动指南
│   ├── PAIR_RESEARCH_QUICKSTART.md   ⚡ 5 分钟快速开始
│   ├── NEW_PAIR_RESEARCH_PLAN.md     📋 完整研究计划
│   └── RESEARCH_EXECUTION_LOG.md     📊 执行日志
│
├── 🎯 核心脚本（根目录）
│   ├── train_pairs.py                 ⭐ PPO 模型训练
│   ├── test_pairs_env.py              测试脚本
│   ├── fetch_stock_data.py            数据获取
│   └── create_pair_trading_data.py    数据准备
│
├── 🛠️ 工具脚本 (scripts/)
│   ├── data_prep/                     数据清洗和特征工程
│   ├── evaluation/                    性能评估工具
│   └── analyze_pair_candidates.py    配对分析工具
│
├── 📊 数据目录 (data/)
│   ├── pair_NINGDE_BYD.csv           ✅ 基准配对数据
│   ├── 002460.csv, 002466.csv        ⏳ 新配对数据
│   └── ...
│
├── 🤖 模型目录 (models/)
│   ├── ppo_pairs_trading.zip         基准模型
│   └── ppo_lithium_*.zip            🚀 新训练模型
│
├── 📈 结果目录 (results/)
│   └── 性能指标和对比分析
│
├── 🏗️ 环境实现 (env/)
│   ├── PairsTradingEnv.py            配对交易环境
│   └── StockTradingEnv.py            单股票环境
│
└── 📖 文档 (docs/)
    ├── WORKFLOW_SUMMARY.md            详细工作流程
    ├── QUICK_REFERENCE.md             快速参考
    └── PROMPT_TEMPLATES.md            AI Prompt 模板
```

---

## 🚀 快速开始（5 分钟）

### 第 1 步：获取新配对数据

```bash
python fetch_stock_data.py --symbols 002466
```

### 第 2 步：验证相关性

```bash
python scripts/analyze_pair_candidates.py --stock1 002460 --stock2 002466
```

### 第 3 步：生成训练数据

```bash
python scripts/data_prep/create_clean_pair_dataset.py \
  --stock1_file data/002460.csv \
  --stock2_file data/002466.csv \
  --output_pair data/pair_002460_002466.csv
```

### 第 4 步：训练模型（需要 30-45 分钟）

```bash
python train_pairs.py \
  --data_path data/pair_002460_002466.csv \
  --total_timesteps 100000 \
  --save_path models/ppo_lithium
```

### 第 5 步：评估性能

```bash
python scripts/evaluation/calc_performance_metrics.py \
  --model_path models/ppo_lithium.zip \
  --test_data data/pair_002460_002466.csv
```

> **💡 提示：** 需要详细步骤？查看 [RESEARCH_STARTUP.md](RESEARCH_STARTUP.md)

---

## 📊 项目概览

### 核心特性

| 特性 | 描述 |
|------|------|
| **算法** | PPO (Proximal Policy Optimization) |
| **框架** | Gymnasium + Stable Baselines3 |
| **策略** | 配对交易（同时持有两只股票的相反头寸） |
| **防护** | 内置防数据泄露机制 |
| **可视化** | TensorBoard 实时监控 |

### 关键指标

| 指标 | 基准案例 | 新配对 |
|------|---------|--------|
| 相关系数 | ~0.92 | TBD |
| 测试收益率 | TBD | TBD |
| 最大回撤 | TBD | TBD |
| 夏普比率 | TBD | TBD |
| 赢率 | TBD | TBD |

---

## 🔍 核心概念

### 什么是配对交易？

配对交易是一种统计套利策略：
- **同时做多和做空**：买入一只股票，做空相关性高的另一只股票
- **基于均值回归**：当价差偏离正常范围时，预期会回归
- **市场中性**：对整体市场方向的依赖较小

### 为什么用强化学习？

- **自动学习**：无需手工设计交易规则
- **动态适应**：模型可以学习不同市场环境下的策略
- **参数优化**：自动找到最优的进场/出场点

---

## 📚 文档导航

### 🆕 新用户必读

| 文档 | 时间 | 内容 |
|------|------|------|
| [PROJECT_NAVIGATION.md](PROJECT_NAVIGATION.md) | 10 min | 项目全景导航 |
| [RESEARCH_STARTUP.md](RESEARCH_STARTUP.md) | 15 min | 环境检查 + 快速启动 |
| [PAIR_RESEARCH_QUICKSTART.md](PAIR_RESEARCH_QUICKSTART.md) | 8 min | 5 分钟快速入门 |

### 📖 深度学习

| 文档 | 时间 | 内容 |
|------|------|------|
| [NEW_PAIR_RESEARCH_PLAN.md](NEW_PAIR_RESEARCH_PLAN.md) | 20 min | 完整研究计划 + 候选库 |
| [docs/WORKFLOW_SUMMARY.md](docs/WORKFLOW_SUMMARY.md) | 15 min | 详细工作流程 + API |
| [SCRIPTS.md](SCRIPTS.md) | 20 min | 所有脚本完整指南 |
| [ANTIDATA_LEAKAGE_DESIGN.md](ANTIDATA_LEAKAGE_DESIGN.md) | 15 min | 防数据泄露原理 |

### ⚡ 快速查询

| 文档 | 用途 |
|------|------|
| [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) | 常用命令速查 |
| [docs/PROMPT_TEMPLATES.md](docs/PROMPT_TEMPLATES.md) | AI 辅助 Prompts |

---

## 🛠️ 环境要求

```
Python >= 3.8
numpy >= 1.19
pandas >= 1.0
gymnasium >= 0.26
stable-baselines3 >= 2.0
matplotlib >= 3.0
tensorboard >= 2.0
```

**快速检查：**
```bash
python -c "import gymnasium; import stable_baselines3; print('✅ 环境就绪')"
```

---

## 🎯 研究进度

### Phase 1: 基础验证 ✅

- [x] 项目结构组织
- [x] 文档完善
- [x] 基准案例实现（宁德 vs 比亚迪）
- [x] 候选配对库筛选

### Phase 2: 新配对研究 ⏳

- [ ] 数据获取和验证
- [ ] 相关性分析
- [ ] 模型训练
- [ ] 性能评估

### Phase 3: 对比分析 ⏳

- [ ] 与基准对比
- [ ] 特征影响分析
- [ ] 参数迁移研究

### Phase 4: 扩展研究 ⏳

- [ ] 行业特征分析
- [ ] 跨配对泛化性能
- [ ] 参数优化框架

---

## 📊 输出示例

运行完整流程后，您将获得：

```
✅ 模型文件：models/ppo_lithium_002460_002466.zip
✅ 性能指标：results/metrics_lithium.json
✅ 对比分析：results/comparison_baseline_vs_lithium.json
✅ 可视化图表：figures/pair_analysis_002460_002466.png
✅ TensorBoard 日志：tb_logs/lithium_pair/
```

---

## 🤔 常见问题

### Q: 从哪开始？
**A:** 阅读 [RESEARCH_STARTUP.md](RESEARCH_STARTUP.md)，然后运行 `python fetch_stock_data.py --symbols 002466`

### Q: 需要多长时间？
**A:** 完整流程约 2 小时。详见 [RESEARCH_STARTUP.md](RESEARCH_STARTUP.md#-预期时间表)

### Q: 如何自定义？
**A:** 查看 [docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md) 中的参数表

### Q: 需要帮助？
**A:** 查看 [PROJECT_NAVIGATION.md](PROJECT_NAVIGATION.md#-需要帮助)

---

## 💡 推荐工作流程

**第一周：验证方法**
```
1. 重现基准案例结果
2. 理解环境和训练过程
3. 探索参数影响
```

**第二周：新配对研究**
```
1. 研究赣锋 vs 天齐
2. 与基准对比
3. 撰写初步报告
```

**第三周及以后：扩展研究**
```
1. 研究更多配对
2. 进行行业对比
3. 开发参数优化框架
```

---

## 📞 获取帮助

- 🗺️ **项目导航**：[PROJECT_NAVIGATION.md](PROJECT_NAVIGATION.md)
- 🚀 **快速启动**：[RESEARCH_STARTUP.md](RESEARCH_STARTUP.md)
- 📋 **脚本使用**：[SCRIPTS.md](SCRIPTS.md)
- 💻 **工作流程**：[docs/WORKFLOW_SUMMARY.md](docs/WORKFLOW_SUMMARY.md)
- ⚡ **快速参考**：[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)
- 🤖 **AI 辅助**：[docs/PROMPT_TEMPLATES.md](docs/PROMPT_TEMPLATES.md)

---

## 📝 许可证

See [LICENSE](LICENSE)

---

**准备开始？** 👉 [PROJECT_NAVIGATION.md](PROJECT_NAVIGATION.md)
python scripts/evaluation/benchmark_strategies.py
```

---

## 📋 文件导航

| 需求 | 文件 |
|------|------|
| **想训练？** | `train_pairs.py` + `docs/QUICK_REFERENCE.md` |
| **想理解流程？** | `docs/WORKFLOW_SUMMARY.md` |
| **想查看所有脚本？** | `SCRIPTS.md` |
| **想用工具脚本？** | `scripts/README.md` |
| **想查看旧代码？** | `archive/README.md` |

---

## ⚙️ 核心概念

**动作空间**（3 个离散动作）：
- 0 = 做多价差（买A卖B）
- 1 = 平仓
- 2 = 做空价差（卖A买B）

**观察特征**（6 个连续特征）：
- `zscore` - 价差标准分数
- `spread_ma5` - 5日移动平均
- `spread_std` - 标准差
- `bollinger_high/low` - 布林带
- `volume_ratio` - 成交量比率

**奖励函数**：基于净资产每步变化

---

## 🔧 常用命令

```bash
# 测试
python test_pairs_env.py

# 训练（基础）
python train_pairs.py

# 训练（自定义）
python train_pairs.py --total_timesteps 50000 --learning_rate 0.0001

# 性能对比
python scripts/evaluation/benchmark_strategies.py

# 详细指标
python scripts/evaluation/calc_performance_metrics.py

# 查看帮助
python train_pairs.py --help
```

---

## 📚 完整文档导航

**代码和脚本：**
- **[SCRIPTS.md](SCRIPTS.md)** - 所有 13 个脚本的详细说明和使用指南
- **[CODE_ORGANIZATION_REPORT.md](CODE_ORGANIZATION_REPORT.md)** - 代码整理总结

**功能文档：**
- **[docs/WORKFLOW_SUMMARY.md](docs/WORKFLOW_SUMMARY.md)** - 详细流程和 API
- **[docs/QUICK_REFERENCE.md](docs/QUICK_REFERENCE.md)** - 快速参考卡
- **[docs/PROMPT_TEMPLATES.md](docs/PROMPT_TEMPLATES.md)** - Prompt 模板库

**文件夹说明：**
- **[scripts/README.md](scripts/README.md)** - 工具脚本说明
- **[archive/README.md](archive/README.md)** - 旧版本代码说明

---

## ⚠️ 重要提示

- ✅ 核心脚本在根目录（train_pairs.py, test_pairs_env.py）
- ✅ 工具脚本在 scripts/ 按功能分类
- ✅ 旧版本代码在 archive/（仅参考，勿修改）
- ✅ 所有文件都有说明文档（README.md）

---

**版本**: Beta | **更新**: 2026.02 | **许可**: MIT
