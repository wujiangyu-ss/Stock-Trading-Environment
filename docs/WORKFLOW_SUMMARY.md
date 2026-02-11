# 股票交易环境 - 完整操作流程文档

## 项目概述

**Stock-Trading-Environment** 是一个自定义 OpenAI Gym 环境，用于在历史股票价格数据上模拟股票交易。项目集成了 Stable Baselines3 的 PPO 算法进行强化学习训练。

---

## 核心组件架构

### 1. **自定义环境 (`env/StockTradingEnv.py`)**

#### 关键特性：
- **初始化参数**：
  - `df`: 股票数据 DataFrame（包含 OHLCV）
  - `initial_balance`: 初始账户余额（默认 10,000）
  - `window_size`: 观察窗口大小（默认 6）

- **行动空间** (`Action Space`):
  - 维度：`[action_type (0..3), amount (0..1)]`
  - `action_type 0-1`: 买入操作（按比例 `amount` 买入）
  - `action_type 1-2`: 卖出操作（按比例 `amount` 卖出）
  - `action_type 2-3`: 持有操作

- **观察空间** (`Observation Space`):
  - 形状：`(6, window_size)` 的 2D 观察
  - 行 0-4：OHLCV 价格数据（归一化）
  - 行 5：账户统计数据（6 个特征）
    - 当前余额 / 初始余额
    - 历史最高净值 / 初始余额
    - 持有股数
    - 成本基础
    - 总卖出股数
    - 总销售价值

#### 状态管理：
| 变量 | 含义 |
|------|------|
| `balance` | 当前现金余额 |
| `net_worth` | 净资产（现金 + 持股市值） |
| `shares_held` | 当前持有股数 |
| `cost_basis` | 平均成本价格 |
| `total_shares_sold` | 历史总卖出股数 |
| `total_sales_value` | 历史总销售价值 |

#### 奖励函数：
```python
step_return = (net_worth_t - net_worth_t-1) / net_worth_t-1
reward = tanh(step_return)  # 范围：[-1, 1]
```

#### 数值安全机制：
- NaN/Inf 保护：`np.nan_to_num()` 转换
- 归一化：所有价格相对于 `max_price`，体积相对于 `max_volume`
- 溢出防护：使用 `float()` 和 `np.clip()` 确保数值有效性

---

## 训练流程

### 2. **基础训练脚本 (`main.py`)**

```python
# 1. 数据加载和预处理
df = pd.read_csv('./data/AAPL.csv')
df = df.sort_values('Date')  # 按日期排序

# 2. 环境初始化
env = DummyVecEnv([lambda: StockTradingEnv(df)])

# 3. 模型创建与训练
model = PPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=20000)  # 训练 20,000 步

# 4. 测试推理
obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
```

#### 关键概念：
- **DummyVecEnv**：向量化环境包装器，支持并行环境运行
- **PPO（Proximal Policy Optimization）**：高效的策略梯度算法
- **MlpPolicy**：多层感知机策略网络

---

## 改进版本 (`main_v2.py`)

> **假设改进方向**（需查阅源代码确认）
- 可能包含更好的超参数配置
- 可能改进了奖励函数设计
- 可能增加了更多评估指标

---

## 数据格式

### 股票数据 (`data/AAPL.csv`)
```
Date,Open,High,Low,Close,Volume
2015-01-02,111.39,111.44,107.35,108.29,53204626
2015-01-05,108.10,108.65,105.41,106.69,64285491
...
```

**必需列**：`Date`, `Open`, `High`, `Low`, `Close`, `Volume`

---

## 训练日志

训练过程中的 TensorBoard 日志存储在 `tb_logs/` 中：
- `PPO_1/`：第一次训练运行
- `PPO_2/`：第二次训练运行

### 查看训练进度：
```bash
tensorboard --logdir=tb_logs/
```

---

## 完整操作流程

### 第一步：环境配置
```bash
# 安装依赖
pip install gymnasium stable-baselines3 pandas numpy

# 验证 AAPL.csv 在 data/ 目录
ls data/AAPL.csv
```

### 第二步：训练模型
```bash
python main.py
```

**预期输出**：
- PPO 训练日志（每 100 步输出一次）
- TensorBoard 事件文件保存到 `tb_logs/PPO_3/`

### 第三步：评估模型
- 模型在测试集上运行 2,000 步
- `env.render()` 输出当前步的交易统计

### 第四步：模型持久化（如需要）
```python
model.save("ppo_stock_trader")
```

### 第五步：加载已有模型
```python
from stable_baselines3 import PPO
model = PPO.load("ppo_stock_trader")
obs = env.reset()
action, _ = model.predict(obs)
```

---

## 常见问题与调试

### 问题 1：NaN 或 Inf 值
**原因**：数据中存在缺失值或极端值
**解决**：
```python
df = df.dropna()
df = df[(df['Close'] > 0) & (df['Volume'] > 0)]
```

### 问题 2：奖励为零
**原因**：网络未正确训练或数据缩放不当
**解决**：
- 增加 `total_timesteps`（例如 50,000）
- 调整超参数：`learning_rate`、`batch_size`

### 问题 3：账户余额为负
**原因**：逻辑错误或数值溢出
**解决**：检查 `_take_action()` 中的余额更新逻辑

---

## 性能优化建议

| 优化项 | 方法 |
|-------|------|
| **更快训练** | 使用 `SubprocVecEnv` 代替 `DummyVecEnv`；增加 `n_steps` |
| **更好收敛** | 调整学习率（默认 0.0003）；增加网络层数 |
| **防止过拟合** | 使用不同的股票数据或时间段验证 |
| **并行化** | 多个环境并行运行提高样本效率 |

---

## 参考资源

- [OpenAI Gym 文档](https://gymnasium.farama.org/)
- [Stable Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [原始 Medium 文章](https://medium.com/@adamjking3/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)
- [PPO 算法详解](https://arxiv.org/abs/1707.06347)

---

## 文件清单

```
Stock-Trading-Environment/
├── main.py                 # 基础训练脚本
├── main_v2.py             # 改进版训练脚本
├── test_train.py          # 单元测试脚本
├── README.md              # 项目说明
├── LICENSE                # 许可证
├── data/
│   └── AAPL.csv          # 苹果公司历史股票数据
├── env/
│   ├── __init__.py
│   ├── StockTradingEnv.py # 核心环境实现
│   └── __pycache__/
├── tb_logs/
│   ├── PPO_1/            # 第一次训练日志
│   └── PPO_2/            # 第二次训练日志
└── docs/
    └── WORKFLOW_SUMMARY.md # 本文档
```

---

**最后更新**：2024年
**版本**：1.0
