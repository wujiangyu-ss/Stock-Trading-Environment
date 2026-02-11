# 股票对配对 Prompt 模板库

## 模板 1：基础数据预处理

### 场景
在训练前清理和验证股票数据

### Prompt 模板
```
我需要处理 {数据文件路径} 中的股票数据。请帮我：

1. 加载数据并检查：
   - 数据形状和列名
   - 缺失值统计
   - 数据类型

2. 数据清理（如果有缺失）：
   - 删除 NaN 值
   - 删除价格 ≤ 0 的行
   - 删除体积 = 0 的行

3. 数据排序：
   - 按日期升序排列
   - 重置索引

4. 生成预处理统计：
   - 日期范围
   - 价格范围（Open、High、Low、Close）
   - 体积统计

示例代码框架：
\`\`\`python
import pandas as pd
import numpy as np

df = pd.read_csv('{数据文件路径}')
# ... 清理逻辑
print(f"处理后数据形状: {df.shape}")
\`\`\`
```

### 示例使用
```
我需要处理 ./data/AAPL.csv 中的股票数据。请帮我：
1. 加载数据并检查...
2. 数据清理...
3. 数据排序...
4. 生成预处理统计...
```

---

## 模板 2：环境配置与初始化

### 场景
创建和配置自定义 Gym 环境

### Prompt 模板
```
基于 StockTradingEnv，我想创建一个配置化的环境初始化脚本。请帮我：

1. 创建环境工厂函数：
   参数：
   - df: 股票数据 DataFrame
   - initial_balance: 初始余额（默认 {默认值}）
   - window_size: 观察窗口（默认 {默认值}）
   - render_mode: 渲染模式（可选）

2. 包装环境：
   - 使用 DummyVecEnv（单进程）
   - 或使用 SubprocVecEnv（多进程，{进程数}）

3. 验证环境：
   - 打印 action_space 和 observation_space
   - 测试一步 step() 操作
   - 验证观察形状和范围

示例代码结构：
\`\`\`python
from stable_baselines3.common.vec_env import DummyVecEnv
from env.StockTradingEnv import StockTradingEnv

def create_env(df, initial_balance={默认值}, window_size={默认值}):
    def make_env():
        return StockTradingEnv(df, initial_balance, window_size)
    return DummyVecEnv([make_env])

env = create_env(df)
\`\`\`
```

### 示例使用
```
基于 StockTradingEnv，我想创建一个配置化的环境初始化脚本。请帮我：
1. 创建环境工厂函数（initial_balance=10000, window_size=6）
2. 包装环境为 DummyVecEnv
3. 验证环境（打印 shape 并测试一步）
```

---

## 模板 3：模型训练配置

### 场景
设计和执行 PPO 训练

### Prompt 模板
```
我想用 PPO 算法训练一个股票交易模型。请帮我配置和执行训练：

参数配置：
- 总时间步数: {时间步数}（如 20000, 50000, 100000）
- 学习率: {学习率}（如 3e-4, 1e-3）
- 批次大小: {批次大小}（如 64, 128, 256）
- 网络层数: {层数}（如 2, 3, 4）
- 网络隐层大小: {大小}（如 64, 128, 256）

训练脚本应该：

1. 初始化 PPO 模型：
   - policy: MlpPolicy（多层感知机）
   - 上述超参数设置
   - verbose=1（打印训练日志）

2. 训练过程：
   - model.learn(total_timesteps={时间步数})
   - TensorBoard 日志自动保存到 tb_logs/

3. 模型保存：
   - 训练完后保存到 models/ 目录
   - 文件名格式: ppo_stock_{时间戳}

4. 训练统计：
   - 总训练时间
   - 平均奖励（如果可获得）

示例代码：
\`\`\`python
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

model = PPO(
    MlpPolicy, 
    env,
    learning_rate={学习率},
    batch_size={批次大小},
    n_epochs=10,
    verbose=1
)
model.learn(total_timesteps={时间步数})
model.save("models/ppo_stock_v1")
\`\`\`
```

### 示例使用
```
我想用 PPO 算法训练一个股票交易模型。请帮我配置和执行训练：
- 总时间步数: 50000
- 学习率: 3e-4
- 批次大小: 128
- 网络层数: 3
```

---

## 模板 4：模型评估与回测

### 场景
测试训练好的模型在交易中的表现

### Prompt 模板
```
我需要对训练完的 PPO 模型进行评估。请帮我：

评估参数：
- 模型路径: {模型文件路径}
- 测试步数: {步数}（如 1000, 2000, 5000）
- 环境: {环境参数描述}

评估流程：

1. 加载模型：
   model = PPO.load("{模型路径}", env=env)

2. 初始化环境：
   obs = env.reset()

3. 运行评估循环 {步数} 步：
   - 记录每步的奖励
   - 记录每步的账户净值
   - 记录交易操作（买/卖）

4. 计算评估指标：
   - 总收益（final_net_worth - initial_balance）
   - 收益率 = 总收益 / 初始余额
   - 最大回撤
   - 夏普比率（如果有奖励序列）
   - 平均每步奖励

5. 输出报告：
   \`\`\`
   === 模型评估报告 ===
   初始余额: ${初始}
   最终净值: ${最终}
   总收益: ${收益} ({收益率}%)
   最大回撤: {回撤}
   总交易数: {交易数}
   胜率: {胜率}
   \`\`\`

示例代码：
\`\`\`python
model = PPO.load("models/ppo_stock_v1", env=env)
obs = env.reset()
rewards = []

for _ in range({步数}):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    if done[0]:
        obs = env.reset()

print(f"平均奖励: {np.mean(rewards):.4f}")
\`\`\`
```

### 示例使用
```
我需要对训练完的 PPO 模型进行评估。请帮我：
- 模型路径: models/ppo_stock_v1
- 测试步数: 2000
- 计算收益率、最大回撤等指标
```

---

## 模板 5：超参数调优

### 场景
进行系统的超参数搜索和对比

### Prompt 模板
```
我想进行超参数调优实验。请帮我创建一个超参数网格搜索脚本。

搜索空间：
学习率: {lr_list}（如 [1e-4, 3e-4, 1e-3]）
批次大小: {batch_list}（如 [32, 64, 128]）
网络大小: {net_list}（如 [64, 128, 256]）

实验配置：
- 每个配置训练 {时间步数} 步
- 评估 {评估步数} 步
- 保存最佳模型（基于评估奖励）

脚本应该：

1. 定义超参数网格
2. 对每个组合：
   a. 创建新环境
   b. 训练 PPO 模型
   c. 评估模型性能
   d. 记录结果

3. 生成对比表格：
   | 学习率 | 批大小 | 网络 | 平均奖励 | 收益 | 最大回撤 |
   |-------|-------|------|--------|------|--------|
   | ...   | ...   | ...  | ...    | ...  | ...    |

4. 保存最佳模型和参数

示例代码框架：
\`\`\`python
import itertools
import json

param_grid = {
    'learning_rate': {lr_list},
    'batch_size': {batch_list},
    'net_arch': {net_list}
}

results = []
for params in itertools.product(*param_grid.values()):
    config = dict(zip(param_grid.keys(), params))
    # 训练和评估...
    results.append({**config, 'score': score})

# 保存结果
with open('tuning_results.json', 'w') as f:
    json.dump(results, f, indent=2)
\`\`\`
```

### 示例使用
```
我想进行超参数调优实验。请帮我创建一个超参数网格搜索脚本。
- 学习率: [1e-4, 3e-4, 1e-3]
- 批次大小: [32, 64, 128]
- 网络大小: [64, 128, 256]
- 每个配置训练 20000 步，评估 1000 步
```

---

## 模板 6：数据增强与多资产训练

### 场景
扩展模型支持多只股票或创建合成数据

### Prompt 模板
```
我想扩展模型以支持多只股票。请帮我：

数据配置：
- 股票列表: {stock_list}（如 ['AAPL', 'GOOGL', 'MSFT']）
- 数据路径: {path_pattern}（如 ./data/{symbol}.csv）
- 训练/验证分割: {split_ratio}（如 0.8）

实现步骤：

1. 加载多只股票数据：
   - 分别读取每只股票的 CSV
   - 验证数据一致性（列名、日期范围等）

2. 数据对齐：
   - 选择共同的日期范围
   - 填补缺失日期（如节假日）

3. 创建数据生成器：
   - 随机选择股票
   - 随机选择时间段
   - 保证训练多样性

4. 训练策略：
   - 方案 A：分别训练每只股票
   - 方案 B：混合训练（在所有股票上）
   - 方案 C：迁移学习（先训练 AAPL，再微调其他）

5. 评估方法：
   - 在未见过的股票上测试（泛化能力）
   - 交叉验证

示例代码：
\`\`\`python
import glob

stocks = {stock_list}
data_dict = {}

for stock in stocks:
    path = f"./data/{stock}.csv"
    df = pd.read_csv(path)
    data_dict[stock] = df.sort_values('Date')

# 多资产环境包装...
\`\`\`
```

### 示例使用
```
我想扩展模型以支持多只股票。请帮我：
- 股票列表: ['AAPL', 'GOOGL', 'MSFT']
- 使用混合训练方法（在所有股票上）
```

---

## 模板 7：故障排查与调试

### 场景
诊断和修复常见问题

### Prompt 模板
```
我的模型训练出现了问题。请帮我诊断：

问题症状：
{问题描述}（如 "奖励始终为 0"、"模型不收敛"、"环境崩溃"）

诊断步骤：

1. 环境验证：
   - env.reset() 是否正常？
   - env.step(action) 返回值是否正确？
   - observation shape 和 dtype 是否一致？
   - reward 是否为有限数值？

2. 数据检查：
   - 数据是否存在 NaN 或 Inf？
   - 价格是否为正？
   - 体积是否为正？

3. 模型检查：
   - 模型参数是否在更新？
   - loss 是否在下降？
   - 策略网络输出是否在合理范围？

4. 日志收集：
   - TensorBoard 日志显示什么？
   - 有无警告或错误消息？

5. 修复建议：
   基于上述诊断结果

示例代码：
\`\`\`python
# 快速诊断
env = DummyVecEnv([lambda: StockTradingEnv(df)])
obs = env.reset()
print(f"obs shape: {obs.shape}, dtype: {obs.dtype}")
print(f"obs 范围: [{obs.min()}, {obs.max()}]")

action = env.action_space.sample()
obs, reward, done, info = env.step(action)
print(f"reward: {reward}, done: {done}")
print(f"action_space: {env.action_space}")
print(f"obs_space: {env.observation_space}")
\`\`\`
```

### 示例使用
```
我的模型训练出现了问题。请帮我诊断：
问题症状: 奖励始终为 0，模型似乎不在学习

- 请检查环境是否正常工作
- 验证数据是否有问题
- 检查奖励函数逻辑
```

---

## 快速参考

### 常用命令
```bash
# 安装依赖
pip install gymnasium stable-baselines3 pandas numpy tensorboard

# 运行训练
python main.py

# 查看 TensorBoard
tensorboard --logdir=tb_logs/

# 加载和推理
python -c "
from stable_baselines3 import PPO
model = PPO.load('models/ppo_stock_v1')
"
```

### 常用代码片段

**标准化观察**
```python
obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
```

**计算收益率**
```python
return_rate = (final_value - initial_value) / initial_value * 100
```

**记录指标**
```python
metrics = {
    'total_reward': sum(rewards),
    'mean_reward': np.mean(rewards),
    'std_reward': np.std(rewards),
    'max_reward': np.max(rewards),
    'min_reward': np.min(rewards)
}
```

---

## 使用建议

1. **选择合适的模板**：根据当前任务选择对应模板
2. **填充占位符**：用实际值替换 {占位符}
3. **逐步执行**：按照脚本的步骤顺序执行
4. **记录结果**：保存输出和日志用于后续分析
5. **迭代改进**：基于结果调整参数和策略

---

**版本**: 1.0  
**最后更新**: 2024年  
**适用环境**: Python 3.8+, Gymnasium, Stable-Baselines3
