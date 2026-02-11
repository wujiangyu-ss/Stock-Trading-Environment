# 配对交易强化学习环境：防止数据泄露的设计与实现

## 一、问题背景

在原始实现中，训练模型时所有特征（Z-Score、移动平均、布林带等）的归一化参数是从整个数据集计算得出的。这导致了**数据泄露**问题：
- 模型训练时，特征的均值/标准差包含了未来（测试集）的信息
- 这使得模型在训练阶段无意中"看到"了测试集数据的统计特性，违反了严格的时间序列机器学习原则
- 结果是模型在测试集上的表现被人为高估，实验结果不具有说服力

## 二、解决方案：严格的时间序列分割与因果特征工程

### 1. 时间序列划分（无随机划分）

将原始历史数据**严格按日期顺序**划分为三个不重叠的分区：

```
训练集 (Train)：   2020-01-02 ~ 2023-12-31  （4年）
验证集 (Val)：    2024-01-01 ~ 2024-12-31  （1年）
测试集 (Test)：   2025-01-01 ~ 2025-12-31  （1年）
```

**关键规则**：禁止任何形式的随机重排或时间跨越。训练数据始终早于验证数据，验证数据始终早于测试数据。

**实现代码**：`scripts/prepare_pair_data_strict.py::strict_time_split()`

### 2. 因果特征计算（仅使用历史信息）

#### 2.1 训练集的特征计算
对训练集内部使用标准滚动计算，但允许在时间序列初期使用少量未来窗口数据（因为那是序列开始处）：

```python
spread_ma20 = price_spread.rolling(window=20, min_periods=1).mean()
spread_std = price_spread.rolling(window=20, min_periods=1).std()
zscore = (price_spread - spread_ma20) / (spread_std + 1e-6)
```

#### 2.2 验证/测试集的特征计算
对验证/测试集，在计算滚动特征时：
- **前缀历史**：提供前一分区末尾的最后 `window_size-1` 行数据（如最后19行用于计算20日窗口）
- **计算范围**：使用"历史+当前分区"的连接，但**仅提取当前分区的结果**，从不使用分区后续数据
- **保证因果性**：每个时间点的特征值严格基于其过去信息，不涉及任何未来数据

**实现代码**：`scripts/prepare_pair_data_strict.py::causal_rolling_features()`

```python
def causal_rolling_features(partition_df, history_df=None, windows=(5,20)):
    """
    对单个分区计算滚动特征，保证因果性。
    如果提供 history_df（前一分区末尾），会拼接用于窗口计算历史上下文。
    返回仅属于 partition_df 长度的特征。
    """
    if history_df is not None and len(history_df) > 0:
        concat = pd.concat([history_df, partition_df], ignore_index=True)
        offset = len(history_df)  # 记录前缀偏移
    else:
        concat = partition_df.copy()
        offset = 0
    
    # 计算滚动特征
    result['spread_ma20'] = concat['price_spread'].rolling(20).mean()
    result['spread_std'] = concat['price_spread'].rolling(20).std()
    result['zscore'] = (concat['price_spread'] - result['spread_ma20']) / result['spread_std']
    
    # 仅返回当前分区部分
    return result.iloc[offset:].reset_index(drop=True)
```

### 3. 训练集统计量的固化与应用

#### 3.1 计算并保存训练集统计量
仅从训练集数据计算所有特征的统计参数（均值、标准差等），保存为 JSON：

```python
train_stats = {}
for col in feature_cols:
    col_series = train_feats[col].astype(float)
    train_stats[col] = {
        'mean': float(col_series.mean(skipna=True)),
        'std': float(col_series.std(skipna=True)),
        'min': float(col_series.min(skipna=True)),
        'max': float(col_series.max(skipna=True))
    }

# 保存为 JSON
with open('train_stats.json', 'w') as f:
    json.dump(train_stats, f, indent=2)
```

**生成文件**：`data/strict/train_stats.json`

#### 3.2 在环境中使用固化参数
强化学习环境 `PairsTradingEnv` 的初始化函数现在接受可选参数 `train_stats`：

```python
class PairsTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=10000.0, window_size=10, 
                 use_features=None, train_stats=None):
        """
        train_stats: 由训练集计算得到的归一化参数字典
                    若提供，则使用这些固定参数进行特征归一化
                    若不提供，回退到对传入df的计算（仅用于调试）
        """
        self.train_stats = train_stats
        self._calculate_normalization_params()  # 优先使用 train_stats
```

归一化参数计算逻辑：

```python
def _calculate_normalization_params(self):
    if self.train_stats is not None:
        # 优先使用外部提供的训练集统计量（无泄露）
        for feat in self.use_features:
            stats = self.train_stats.get(feat)
            self.feature_means[feat] = float(stats['mean'])
            self.feature_stds[feat] = float(stats['std']) if stats['std'] > 1e-6 else 1.0
    else:
        # 回退：仅在调试时，从当前df计算（存在泄露风险）
        for feat in self.use_features:
            self.feature_means[feat] = float(self.df[feat].mean())
            self.feature_stds[feat] = float(self.df[feat].std())
```

### 4. 完整的训练与评估流程

#### 4.1 训练阶段
```
输入：
  - 训练集特征：data/strict/pair_train.csv
  - 训练集统计：data/strict/train_stats.json

操作：
  1. 加载 pair_train.csv
  2. 初始化环境，传入 train_stats（已从训练集计算）
  3. 使用 PPO 算法训练 100,000 步

输出：
  - 模型权重：models/NINGDE_BYD_partitioned/ppo_partitioned.zip
  - 训练日志：logs/tb_logs/NINGDE_BYD_partitioned/
```

#### 4.2 评估阶段
```
输入：
  - 测试集特征：data/strict/pair_test.csv
  - 训练集统计：data/strict/train_stats.json（与训练时相同）
  - 训练好的模型：models/NINGDE_BYD_partitioned/ppo_partitioned.zip

操作：
  1. 加载 pair_test.csv
  2. 初始化环境，传入 train_stats（依然使用训练集统计）
  3. 运行训练好的模型进行推理（不进行梯度更新）
  4. 记录各时间步的净资产、奖励、动作等

输出：
  - 评估指标：results/NINGDE_BYD_partitioned/eval_results.json
    包含：初始资金、最终净资产、总收益率、夏普比率、最大回撤等
  - 推理日志：便于事后分析动作分布、交易频率等
```

**实现代码**：`scripts/train_with_partitions.py`

## 三、关键文件与代码位置

| 文件 | 功能 | 关键代码 |
|------|------|--------|
| `scripts/prepare_pair_data_strict.py` | 数据划分与因果特征工程 | `strict_time_split()`, `causal_rolling_features()`, `prepare_strict_pair()` |
| `env/PairsTradingEnv.py` | 强化学习环境（支持train_stats） | `__init__(train_stats=None)`, `_calculate_normalization_params()` |
| `scripts/train_with_partitions.py` | 分区数据的训练与评估脚本 | `make_vec_env_from_csv()`, `evaluate_model()` |
| `data/strict/pair_train.csv` | 训练集配对特征 | 2020-01-02 ~ 2023-12-31 |
| `data/strict/pair_val.csv` | 验证集配对特征 | 2024-01-01 ~ 2024-12-31 |
| `data/strict/pair_test.csv` | 测试集配对特征 | 2025-01-01 ~ 2025-12-31 |
| `data/strict/train_stats.json` | 训练集统计参数 | 用于验证/测试集的特征归一化 |

## 四、防泄露的验证清单

- [x] 时间划分无重叠，无随机排列
- [x] 训练集滚动特征计算仅使用历史数据（初期可用未来窗口）
- [x] 验证/测试集滚动特征计算提供历史前缀，但不使用其自身未来数据
- [x] 所有归一化参数（均值、标准差）仅从训练集计算
- [x] 环境类明确接受 `train_stats` 参数，优先使用训练集统计
- [x] 训练、验证、测试三个阶段使用完全相同的训练集统计参数
- [x] 模型评估仅使用推理，不更新模型参数

## 五、论文中的标准表述

> **防止数据泄露的因果特征工程**：为避免模型训练时无意中"看到"未来信息，我们采用严格的时间序列划分策略。原始股票数据按日期顺序划分为训练集（2020-2023年）、验证集（2024年）和测试集（2025年）。所有特征（包括Z-Score、移动平均、布林带等）的归一化参数仅从训练集数据计算得出，并在验证和测试阶段使用相同的固定参数。在计算验证/测试集的滚动特征时，我们提供前一分区末尾的历史数据作为窗口上下文，但严格避免使用当前分区后续的数据，确保每个时间点的特征计算仅基于其历史信息，不涉及任何未来信息。这种因果设计完全消除了数据泄露风险，使得模型的离线评估更加可靠。

## 六、快速开始命令

### 生成严格分区数据
```bash
python scripts/prepare_pair_data_strict.py
```

输出：
- `data/strict/pair_train.csv`
- `data/strict/pair_val.csv`
- `data/strict/pair_test.csv`
- `data/strict/train_stats.json`

### 训练并评估
```bash
python scripts/train_with_partitions.py \
  --train-csv ./data/strict/pair_train.csv \
  --test-csv ./data/strict/pair_test.csv \
  --train-stats ./data/strict/train_stats.json \
  --models-dir ./models/NINGDE_BYD_partitioned \
  --logs-dir ./logs/tb_logs/NINGDE_BYD_partitioned \
  --results-dir ./results/NINGDE_BYD_partitioned \
  --total-timesteps 100000 \
  --learning-rate 0.0001
```

## 七、预期结果对比

| 指标 | 原始实现（数据泄露） | 修复后实现（无泄露） |
|-----|------------------|------------|
| 测试集收益率 | +14.46% | 待验证（预期略低） |
| 夏普比率 | 1.0572 | 待验证 |
| 平仓比例 | 0% ❌ | 待观察（应>0） |
| 数据独立性 | ❌ 违反 | ✅ 满足 |

---

**附：重训模型绩效表（可直接复制到论文 LaTeX）**

```latex
\begin{table}[ht]
\centering
\caption{Performance metrics for retrained clean model}
\begin{tabular}{l r}
\hline
Metric & Value \\
\hline
累计收益率 (Total Return) & 0.04237 \\
年化收益率 (Annualized Return) & 0.04397 \\
年化波动率 (Annualized Volatility) & 0.03500 \\
夏普比率 (Sharpe Ratio) & 1.2563 \\
卡玛比率 (Calmar Ratio) & 3.6927 \\
索提诺比率 (Sortino Ratio) & 0.6432 \\
最大回撤 (Max Drawdown) & 0.01191 \\
最大回撤期 (Max Drawdown Duration days) & 65 \\
总交易次数 (Total Trades) & 3 \\
胜率 (Win Rate) & 0.6667 \\
平均盈亏比 (Average P/L Ratio) & 4.3985 \\
平均持仓时间 (Average Holding Period days) & 12.3333 \\
\hline
\end{tabular}
\end{table}
```

**文档版本**：v1.0 (2026-02-10)  
**配对组合**：宁德时代(300750) vs 比亚迪(002594)  
**训练框架**：Gymnasium + Stable-Baselines3 (PPO)  
