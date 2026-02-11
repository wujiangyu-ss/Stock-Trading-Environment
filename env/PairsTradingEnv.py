"""
配对交易强化学习环境
基于Gymnasium框架，用于训练配对交易策略

配对交易逻辑：
- 状态空间：基于两支股票的特征（价差、Z-Score、布林带等）
- 动作空间：离散的3个操作（做多、平仓、做空）
- 收益计算：配对组合（多头+空头）的综合价值变化
"""

import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np


class PairsTradingEnv(gym.Env):
    """
    配对交易强化学习环境
    
    特点：
    - 同时操作两支相关股票
    - 基于价差特征进行交易
    - 离散动作空间（做多、平仓、做空）
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, df, initial_balance=10000.0, window_size=10, use_features=None, train_stats=None, mode='train', frozen_stats=None):
        """
        初始化环境
        
        Args:
            df: 包含配对特征的DataFrame（来自pair_NINGDE_BYD.csv）
            initial_balance: 初始账户余额
            window_size: 观察窗口大小（使用过去N个时间步）
            use_features: 使用的特征列表，默认为['zscore', 'spread_ma5', 'spread_std', 
                         'bollinger_high', 'bollinger_low', 'volume_ratio']
        """
        super().__init__()
        
        self.df = df.reset_index(drop=True)
        self.initial_balance = float(initial_balance)
        self.window_size = int(window_size)
        self.train_stats = train_stats  # 兼容旧参数名
        self.mode = mode  # 'train' / 'val' / 'test'
        self.frozen_stats = frozen_stats  # 来自 train_stats.json 的冻结整体参数（如 spread_mean/spread_std）
        
        # 选择用于观察的特征
        if use_features is None:
            self.use_features = ['zscore', 'spread_ma5', 'spread_std', 
                               'bollinger_high', 'bollinger_low', 'volume_ratio']
        else:
            self.use_features = use_features
        
        # 验证特征存在
        for feat in self.use_features:
            if feat not in self.df.columns:
                raise ValueError(f"特征 '{feat}' 不存在于数据中")
        
        # 计算特征的归一化参数
        self._calculate_normalization_params()
        
        # ===== 动作空间 =====
        # 0: 做多价差（买入股票1，卖出股票2）
        # 1: 平仓（平掉所有头寸）
        # 2: 做空价差（卖出股票1，买入股票2）
        self.action_space = spaces.Discrete(3)
        
        # ===== 观察空间 =====
        # shape: (特征数, window_size)
        obs_shape = (len(self.use_features), self.window_size)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        
        # ===== 状态变量 =====
        # 账户状态
        self.balance = None  # 现金余额
        self.net_worth = None  # 净资产
        self.max_net_worth = None  # 最大净资产
        
        # 头寸状态
        # stock1: 宁德时代，stock2: 比亚迪
        self.stock1_shares = None  # 股票1的持股数
        self.stock2_shares = None  # 股票2的持股数
        self.stock1_avg_price = None  # 股票1的平均成本价
        self.stock2_avg_price = None  # 股票2的平均成本价
        
        # 交易记录
        self.total_trades = None
        self.profitable_trades = None
        
        # 当前时间步
        self.current_step = None
        
        # 当前的股票价格（用于计算头寸价值）
        self.stock1_price = None
        self.stock2_price = None
    
    def _calculate_normalization_params(self):
        """计算特征的统计参数用于归一化"""
        self.feature_means = {}
        self.feature_stds = {}

        # 优先使用外部冻结的整体训练参数（frozen_stats）来处理 'zscore' 的计算/归一化
        if self.frozen_stats is not None:
            # 对于已经是 Z-Score 的列（如果数据预先计算了 zscore），我们不再对其做二次归一化
            for feat in self.use_features:
                if feat == 'zscore':
                    self.feature_means[feat] = 0.0
                    self.feature_stds[feat] = 1.0
                else:
                    # 其他特征优先尝试从 self.train_stats（按特征存储）读取
                    if self.train_stats is not None and isinstance(self.train_stats, dict):
                        stats = self.train_stats.get(feat)
                        if stats is not None:
                            self.feature_means[feat] = float(stats.get('mean', 0.0))
                            std = float(stats.get('std', 1.0))
                            self.feature_stds[feat] = std if std > 1e-6 else 1.0
                            continue
                    # 回退：从训练分区计算（如果有'dataset'列），否则从整个 df 计算
                    if 'dataset' in self.df.columns and self.mode == 'train':
                        tmp = self.df[self.df['dataset'] == 'train'][feat]
                    elif 'dataset' in self.df.columns:
                        # 始终从训练分区计算均值/方差以避免泄露
                        tmp = self.df[self.df['dataset'] == 'train'][feat]
                    else:
                        tmp = self.df[feat]
                    self.feature_means[feat] = float(tmp.mean()) if len(tmp) > 0 else 0.0
                    std = float(tmp.std()) if len(tmp) > 0 else 1.0
                    self.feature_stds[feat] = std if std > 1e-6 else 1.0
            return
        else:
            # 回退：从当前数据计算（存在数据泄露风险，仅用于快速调试）
            # 如果存在'dataset'列并且为训练模式，则仅使用训练分区来计算
            for feat in self.use_features:
                if 'dataset' in self.df.columns and self.mode == 'train':
                    tmp = self.df[self.df['dataset'] == 'train'][feat]
                else:
                    tmp = self.df[feat]
                self.feature_means[feat] = float(tmp.mean()) if len(tmp) > 0 else 0.0
                std = float(tmp.std()) if len(tmp) > 0 else 1.0
                self.feature_stds[feat] = std if std > 1e-6 else 1.0
    
    def _get_window_observation(self, current_idx):
        """
        获取当前时刻的窗口观察
        
        Args:
            current_idx: 当前在数据中的索引
            
        Returns:
            归一化后的观察矩阵，shape: (特征数, window_size)
        """
        # 确保窗口不越界
        start_idx = max(0, current_idx - self.window_size + 1)
        end_idx = current_idx
        
        # 如果窗口不足，用0填充（前面的数据）
        window_data = self.df.loc[start_idx:end_idx, self.use_features].values
        
        if len(window_data) < self.window_size:
            # 前面填充0
            padding = np.zeros((self.window_size - len(window_data), len(self.use_features)))
            window_data = np.vstack([padding, window_data])
        
        # 特征归一化
        normalized_window = np.zeros_like(window_data, dtype=np.float32)
        for i, feat in enumerate(self.use_features):
            normalized_window[:, i] = (
                (window_data[:, i] - self.feature_means[feat]) / self.feature_stds[feat]
            ).astype(np.float32)
        
        # 转置为 (特征数, 时间步)
        obs = normalized_window.T.astype(np.float32)
        
        # NaN/Inf处理
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        
        return obs
    
    def _update_portfolio_value(self, stock1_price, stock2_price):
        """
        更新投资组合价值
        
        配对交易中：
        - 做多价差：持有股票1正头寸，股票2负头寸
        - 做空价差：持有股票1负头寸，股票2正头寸
        
        Args:
            stock1_price: 股票1当前价格
            stock2_price: 股票2当前价格
        """
        self.stock1_price = stock1_price
        self.stock2_price = stock2_price
        
        # 计算头寸价值（考虑多空方向）
        stock1_value = self.stock1_shares * stock1_price
        stock2_value = self.stock2_shares * stock2_price
        
        # 配对组合净资产 = 现金 + 两个头寸的价值
        # 注意：stock1_shares > 0表示多头，< 0表示空头
        self.net_worth = self.balance + stock1_value + stock2_value
        
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
    
    def _take_action(self, action, stock1_price, stock2_price):
        """
        执行交易动作
        
        Args:
            action: 动作编号 (0: 做多, 1: 平仓, 2: 做空)
            stock1_price: 股票1价格
            stock2_price: 股票2价格
        """
        action_type = int(action)
        
        if action_type == 1:
            # 平仓：卖出所有头寸
            self._liquidate_all(stock1_price, stock2_price)
        
        elif action_type == 0:
            # 做多价差：买入股票1，卖出股票2
            if self.stock1_shares <= 0 and self.stock2_shares >= 0:
                # 当前没有做多头寸，可以建仓
                
                # 确定交易数量（基于账户余额的一定比例）
                available_cash = self.balance * 0.9  # 保留10%现金
                
                # 计算可以买入的股票1数量
                shares_to_trade = int(available_cash / (stock1_price + stock2_price + 1e-6))
                
                if shares_to_trade > 0:
                    # 买入股票1
                    cost1 = shares_to_trade * stock1_price
                    if cost1 <= self.balance:
                        self.balance -= cost1
                        prev_cost = self.stock1_avg_price * self.stock1_shares
                        self.stock1_shares += shares_to_trade
                        self.stock1_avg_price = (prev_cost + cost1) / (self.stock1_shares + 1e-6)
                        self.total_trades += 1
                    
                    # 卖出股票2
                    proceeds2 = shares_to_trade * stock2_price
                    self.balance += proceeds2
                    self.stock2_shares -= shares_to_trade
                    self.stock2_avg_price = stock2_price
                    self.total_trades += 1
        
        elif action_type == 2:
            # 做空价差：卖出股票1，买入股票2
            if self.stock1_shares >= 0 and self.stock2_shares <= 0:
                # 当前没有做空头寸，可以建仓
                
                available_cash = self.balance * 0.9
                shares_to_trade = int(available_cash / (stock1_price + stock2_price + 1e-6))
                
                if shares_to_trade > 0:
                    # 卖出股票1
                    proceeds1 = shares_to_trade * stock1_price
                    self.balance += proceeds1
                    self.stock1_shares -= shares_to_trade
                    self.stock1_avg_price = stock1_price
                    self.total_trades += 1
                    
                    # 买入股票2
                    cost2 = shares_to_trade * stock2_price
                    if cost2 <= self.balance:
                        self.balance -= cost2
                        prev_cost = self.stock2_avg_price * self.stock2_shares
                        self.stock2_shares += shares_to_trade
                        self.stock2_avg_price = (prev_cost + cost2) / (self.stock2_shares + 1e-6)
                        self.total_trades += 1
    
    def _liquidate_all(self, stock1_price, stock2_price):
        """
        平掉所有头寸
        
        Args:
            stock1_price: 股票1价格
            stock2_price: 股票2价格
        """
        if self.stock1_shares > 0:
            proceeds = self.stock1_shares * stock1_price
            self.balance += proceeds
            # 计算收益
            cost = self.stock1_avg_price * self.stock1_shares
            if proceeds > cost:
                self.profitable_trades += 1
            self.stock1_shares = 0
            self.stock1_avg_price = 0.0
            self.total_trades += 1
        
        elif self.stock1_shares < 0:
            # 做空头寸：回购
            cost = abs(self.stock1_shares) * stock1_price
            if cost <= self.balance:
                self.balance -= cost
                # 计算收益（做空是反向的）
                proceeds = abs(self.stock1_shares) * self.stock1_avg_price
                if proceeds > cost:
                    self.profitable_trades += 1
                self.stock1_shares = 0
                self.stock1_avg_price = 0.0
                self.total_trades += 1
        
        if self.stock2_shares > 0:
            proceeds = self.stock2_shares * stock2_price
            self.balance += proceeds
            cost = self.stock2_avg_price * self.stock2_shares
            if proceeds > cost:
                self.profitable_trades += 1
            self.stock2_shares = 0
            self.stock2_avg_price = 0.0
            self.total_trades += 1
        
        elif self.stock2_shares < 0:
            cost = abs(self.stock2_shares) * stock2_price
            if cost <= self.balance:
                self.balance -= cost
                proceeds = abs(self.stock2_shares) * self.stock2_avg_price
                if proceeds > cost:
                    self.profitable_trades += 1
                self.stock2_shares = 0
                self.stock2_avg_price = 0.0
                self.total_trades += 1
    
    def step(self, action):
        """
        执行一步环境交互
        
        Args:
            action: 动作 (0, 1, 2)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # 记录前一步的净资产
        prev_net_worth = float(self.net_worth)
        
        # 从数据中获取当前时刻的股票价格
        current_row = self.df.iloc[self.current_step]

        # 优先使用显式的收盘价列（如果存在）
        if 'NINGDE_Close' in self.df.columns and 'BYD_Close' in self.df.columns:
            stock1_price = float(current_row['NINGDE_Close'])
            stock2_price = float(current_row['BYD_Close'])
        else:
            # 回退：使用价差和价格比来反推两只股票的价格
            price_spread = float(current_row['price_spread'])
            price_ratio = float(current_row['price_ratio'])
            stock2_price = price_spread / (price_ratio - 1 + 1e-6)
            stock1_price = price_spread + stock2_price
        
        # 防止负价格
        stock1_price = max(1e-6, stock1_price)
        stock2_price = max(1e-6, stock2_price)
        
        # 执行动作
        self._take_action(action, stock1_price, stock2_price)
        
        # 更新投资组合价值
        self._update_portfolio_value(stock1_price, stock2_price)
        
        # 移动到下一时间步
        self.current_step += 1
        
        # 检查是否到达数据末尾
        truncated = False
        if self.current_step >= len(self.df):
            truncated = True
        
        # 计算收益
        reward = 0.0
        if prev_net_worth > 0:
            # 单步收益率
            step_return = (self.net_worth - prev_net_worth) / prev_net_worth
            # 使用tanh压缩到[-1, 1]范围
            reward = float(np.tanh(step_return))
        
        reward = float(np.clip(reward, -1.0, 1.0))
        
        # 破产检查
        terminated = self.net_worth <= 0
        
        # 获取观察
        obs = self._get_window_observation(self.current_step - 1)
        
        info = {
            'net_worth': self.net_worth,
            'balance': self.balance,
            'stock1_shares': self.stock1_shares,
            'stock2_shares': self.stock2_shares,
            'stock1_price': stock1_price,
            'stock2_price': stock2_price
        }
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 其他选项
            
        Returns:
            observation, info
        """
        super().reset(seed=seed)
        
        # 重置账户状态
        self.balance = float(self.initial_balance)
        self.net_worth = float(self.initial_balance)
        self.max_net_worth = float(self.initial_balance)
        
        # 重置头寸
        self.stock1_shares = 0
        self.stock2_shares = 0
        self.stock1_avg_price = 0.0
        self.stock2_avg_price = 0.0
        
        # 重置交易记录
        self.total_trades = 0
        self.profitable_trades = 0
        
        # 支持通过 options 指定 start_index（用于验证/测试的可复现评估）
        if options is not None and isinstance(options, dict) and 'start_index' in options:
            self.current_step = int(options.get('start_index', 0))
        else:
            # 随机选择起始时间步（避免从数据的最后部分开始，防止数据不足）
            max_start = max(0, len(self.df) - self.window_size - 100)
            self.current_step = np.random.randint(0, max_start) if max_start > 0 else 0
        
        # 初始化价格（优先使用显式收盘价）
        current_row = self.df.iloc[self.current_step]
        if 'NINGDE_Close' in self.df.columns and 'BYD_Close' in self.df.columns:
            self.stock1_price = float(current_row['NINGDE_Close'])
            self.stock2_price = float(current_row['BYD_Close'])
        else:
            price_spread = float(current_row['price_spread'])
            price_ratio = float(current_row['price_ratio'])
            self.stock2_price = price_spread / (price_ratio - 1 + 1e-6)
            self.stock1_price = price_spread + self.stock2_price
        self.stock1_price = max(1e-6, self.stock1_price)
        self.stock2_price = max(1e-6, self.stock2_price)
        
        # 获取初始观察
        obs = self._get_window_observation(self.current_step)
        
        info = {}
        return obs, info
    
    def render(self, mode='human'):
        """
        渲染环境状态
        """
        if self.current_step < len(self.df):
            current_row = self.df.iloc[self.current_step]
            zscore = float(current_row['zscore'])
            
            print(f"时间步: {self.current_step}")
            print(f"余额: ¥{self.balance:.2f}")
            print(f"股票1头寸: {self.stock1_shares} 股 (平均成本: ¥{self.stock1_avg_price:.2f})")
            print(f"股票2头寸: {self.stock2_shares} 股 (平均成本: ¥{self.stock2_avg_price:.2f})")
            print(f"当前价格: 股票1 ¥{self.stock1_price:.2f}, 股票2 ¥{self.stock2_price:.2f}")
            print(f"Z-Score: {zscore:.4f}")
            print(f"净资产: ¥{self.net_worth:.2f}")
            print(f"累计交易次数: {self.total_trades}")
            print(f"盈利交易: {self.profitable_trades}")
            print("-" * 60)
