# 股票交易环境 - 快速参考卡

## 📋 核心对象速查表

### StockTradingEnv 状态变量
```python
env.balance           # 当前现金
env.net_worth         # 总净值（现金 + 持股市值）
env.shares_held       # 持有股数
env.cost_basis        # 平均持股成本
env.current_step      # 当前时间步
env.max_net_worth     # 历史最高净值
```

### 环境空间
```python
action_space = Box([0, 0], [3, 1])           # [操作类型 0-3, 比例 0-1]
observation_space = Box(-inf, inf, (6, 6))   # (特征数, 窗口大小)
```

### 操作解释
| action_type 范围 | 操作 | 参数含义 |
|-----------------|------|--------|
| 0.0 - 1.0 | 买入 | amount = 用账户余额的比例 |
| 1.0 - 2.0 | 卖出 | amount = 持有股数的比例 |
| 2.0 - 3.0 | 持有 | 无操作 |

---

## 🚀 快速启动命令

### 最小化训练脚本
```python
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env.StockTradingEnv import StockTradingEnv

# 1. 加载数据
df = pd.read_csv('./data/AAPL.csv').sort_values('Date')

# 2. 创建环境
env = DummyVecEnv([lambda: StockTradingEnv(df)])

# 3. 训练
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=20000)

# 4. 保存
model.save('model')
```

### 推理脚本
```python
from stable_baselines3 import PPO

model = PPO.load('model', env=env)
obs = env.reset()
for _ in range(100):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    print(f"Reward: {reward}")
```

---

## 🔍 观察空间详解

观察是一个 $(6, \text{window\_size})$ 的矩阵：

```
┌─────────────────────────────────┐
│ 行 0: Open 价格（归一化）        │
│ 行 1: High 价格（归一化）        │
│ 行 2: Low 价格（归一化）         │
│ 行 3: Close 价格（归一化）       │
│ 行 4: Volume（归一化）           │
│ 行 5: 账户统计（6 个特征）       │
└─────────────────────────────────┘
     列 0  列 1  列 2  ...  列 N
     ↓     ↓     ↓           ↓
   Day 0 Day 1 Day 2 ...  Day N-1
```

**行 5 的 6 个账户特征**：
```python
[balance/initial, max_net_worth/initial, shares_held, 
 cost_basis, total_shares_sold, total_sales_value]
```

---

## 📊 常用性能指标

### 1. 收益率 (Return)
$$\text{Return} = \frac{\text{Final NetWorth} - \text{Initial Balance}}{\text{Initial Balance}} \times 100\%$$

### 2. 最大回撤 (Max Drawdown)
$$\text{MDD} = \frac{\text{Max Equity} - \text{Equity}_{\text{trough}}}{\text{Max Equity}} \times 100\%$$

### 3. 夏普比率 (Sharpe Ratio)
$$\text{Sharpe} = \frac{\text{Mean Return} - \text{Risk-Free Rate}}{\text{Return Std Dev}}$$

### 4. 胜率 (Win Rate)
$$\text{Win Rate} = \frac{\text{Profitable Steps}}{\text{Total Steps}} \times 100\%$$

### 5. 平均奖励 (Mean Reward)
$$\bar{r} = \frac{1}{n}\sum_{i=1}^{n} r_i$$

---

## ⚙️ 超参数参考表

| 参数 | 推荐范围 | 说明 |
|------|--------|------|
| `learning_rate` | 1e-5 ~ 1e-3 | 越小收敛越慢但越稳定 |
| `batch_size` | 32 ~ 256 | 越大越稳定但需要更多内存 |
| `n_epochs` | 3 ~ 20 | PPO 内循环次数 |
| `gamma` | 0.9 ~ 0.999 | 折扣因子，越接近 1 越看重长期 |
| `gae_lambda` | 0.9 ~ 0.99 | GAE 系数 |
| `ent_coef` | 1e-4 ~ 1e-1 | 熵奖励系数，鼓励探索 |
| `window_size` | 2 ~ 20 | 观察窗口，太小失去历史，太大计算量大 |
| `initial_balance` | 1k ~ 100k | 初始资金 |

---

## 🐛 快速排查表

| 问题 | 症状 | 解决方案 |
|------|------|--------|
| **数据问题** | NaN/Inf 异常 | `df.dropna()`, `df[df['Close']>0]` |
| **环境问题** | step() 返回异常 | 检查 OHLCV 列，验证归一化 |
| **模型不学习** | 奖励 = 0 | 增加 timesteps，调整学习率 |
| **过度拟合** | 训练好，测试差 | 加入验证集，增加数据多样性 |
| **训练过慢** | 时间太长 | 使用 SubprocVecEnv，减少 window_size |
| **内存溢出** | OOM 错误 | 减少 batch_size，使用较短的时间序列 |
| **价格异常** | 出现极端值 | 检查数据是否有分股/并股事件 |

---

## 📈 训练检查清单

- [ ] 数据已加载并验证（非空，价格 > 0）
- [ ] 环境创建成功（action_space 和 obs_space 正确）
- [ ] 第一步推理可以运行（无错误）
- [ ] 模型开始训练（看到日志输出）
- [ ] TensorBoard 事件文件已生成
- [ ] 奖励不是全 0 或全 NaN
- [ ] 训练 1000 步后有可见的改进
- [ ] 模型可以成功保存和加载

---

## 🔗 关键类方法

### StockTradingEnv
```python
env.reset()                    # 重置环境，返回初始观察
env.step(action)               # 执行动作，返回 (obs, reward, done, info)
env.render()                   # 输出当前状态（用于调试）
env._take_action(action)       # 执行交易逻辑
env._next_observation()        # 生成观察
```

### PPO 模型
```python
model.learn(total_timesteps)   # 训练
model.predict(observation)    # 推理，返回 (action, state)
model.save(path)               # 保存模型
model = PPO.load(path, env)   # 加载模型
```

### DummyVecEnv
```python
env.reset()                    # 重置，返回 (n_envs, obs_size) 的观察
env.step(actions)              # 步进，actions 形状 (n_envs,)
env.render()                   # 渲染所有环境
env.close()                    # 关闭环境
```

---

## 💾 文件操作示例

### 保存训练结果
```python
import json
from datetime import datetime

results = {
    'timestamp': datetime.now().isoformat(),
    'total_return': 0.25,
    'sharpe_ratio': 1.5,
    'max_drawdown': 0.12,
    'config': {
        'learning_rate': 3e-4,
        'batch_size': 128,
        'total_timesteps': 50000
    }
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)
```

### 记录交易日志
```python
trades = []
for step in range(n_steps):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    
    trades.append({
        'step': step,
        'action': action[0],
        'balance': env.balance,
        'net_worth': env.net_worth,
        'shares_held': env.shares_held,
        'reward': reward
    })

import pandas as pd
pd.DataFrame(trades).to_csv('trades.csv', index=False)
```

---

## 🎯 典型工作流程

```
1️⃣ 数据准备
   └─ 加载 AAPL.csv
   └─ 验证数据质量
   └─ 数据排序

2️⃣ 环境设置
   └─ 创建 StockTradingEnv
   └─ 向量化包装
   └─ 验证空间

3️⃣ 模型创建
   └─ 初始化 PPO
   └─ 设置超参数

4️⃣ 训练执行
   └─ model.learn()
   └─ 监控 TensorBoard

5️⃣ 模型评估
   └─ 推理 2000 步
   └─ 计算指标

6️⃣ 结果保存
   └─ 保存模型
   └─ 保存交易记录
   └─ 生成报告
```

---

## 📚 学习资源链接

- [Gymnasium 官文](https://gymnasium.farama.org/)
- [Stable Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [PPO 论文 (1707.06347)](https://arxiv.org/abs/1707.06347)
- [原始 Medium 教程](https://medium.com/@adamjking3/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)

---

## 🆘 常见命令

```bash
# 终止训练
Ctrl+C

# 查看 tensorboard
tensorboard --logdir=tb_logs/

# 删除旧日志
rm -rf tb_logs/

# 列出模型文件
ls -lh models/

# 查看 Python 版本
python --version

# 列出包版本
pip show gymnasium stable-baselines3
```

---

**版本**: 1.0  
**适用**: Stock Trading Environment v2  
**更新日期**: 2024年
