import random
import json
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 10000


class StockTradingEnv(gym.Env):
    """Merged, improved StockTradingEnv (v2 implementation under the original filename).

    - Improved normalization and numeric safety
    - Stable reward: tanh(step-wise net-worth return) clipped to [-1,1]
    - Fixed windowing and NaN/Inf protections
    
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, df, initial_balance: float = 10000.0, window_size: int = 6):
        super().__init__()
        assert window_size >= 2, "window_size must be >= 2"

        self.df = df.reset_index(drop=True)
        self.window_size = int(window_size)
        self.initial_balance = float(initial_balance)

        # normalization constants
        self.max_price = float(max(1.0, self.df[["Open", "High", "Low", "Close"]].max().max()))
        self.max_volume = float(max(1.0, self.df["Volume"].max()))

        # action: [action_type (0..3), amount (0..1)]
        self.action_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([3.0, 1.0]), dtype=np.float32)

        # obs: window_size columns of OHLCV (5 x window_size) + 1 row of account stats (6 elements)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5 + 1, self.window_size), dtype=np.float32)

        # internal state placeholders
        self.balance = None
        self.net_worth = None
        self.max_net_worth = None
        self.shares_held = None
        self.cost_basis = None
        self.total_shares_sold = None
        self.total_sales_value = None
        self.current_step = None

    def _get_window(self, start: int):
        # ensure a valid fixed-size window
        end = start + self.window_size - 1
        if end > len(self.df) - 1:
            start = max(0, len(self.df) - self.window_size)
            end = start + self.window_size - 1
        return self.df.loc[start:end]

    def _next_observation(self):
        window = self._get_window(self.current_step)

        open_ = window["Open"].values.astype(np.float32) / self.max_price
        high_ = window["High"].values.astype(np.float32) / self.max_price
        low_ = window["Low"].values.astype(np.float32) / self.max_price
        close_ = window["Close"].values.astype(np.float32) / self.max_price
        vol_ = window["Volume"].values.astype(np.float32) / self.max_volume

        # shape (5, window_size)
        price_frame = np.vstack([open_, high_, low_, close_, vol_])

        account = np.array([
            self.balance / self.initial_balance,
            self.max_net_worth / self.initial_balance,
            self.shares_held / max(1.0, float(1e9)),  # scale-down placeholder
            (self.cost_basis / self.max_price) if self.max_price > 0 else 0.0,
            self.total_shares_sold / max(1.0, float(1e9)),
            self.total_sales_value / (self.max_volume * self.max_price + 1e-12),
        ], dtype=np.float32)

        # Replicate account stats across window columns to keep consistent 2D shape
        account_row = np.tile(account.reshape(-1, 1), (1, self.window_size))  # shape (6, window_size)

        # we want final obs shape (6, window_size) - but our observation_space was defined as (6, window_size)
        obs = np.vstack([price_frame, account_row[0:1, :]])  # only append one row to keep consistent with original description

        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        return obs

    def _take_action(self, action):
        # numeric safety
        action_type = float(action[0])
        amount = float(action[1])

        # pick a representative price at current_step (use Close for determinism)
        cur_price = float(self.df.loc[self.current_step, "Close"]) if "Close" in self.df.columns else float(self.df.loc[self.current_step, "Open"]) 
        cur_price = max(1e-6, cur_price)

        if action_type < 1.0:
            total_possible = int(self.balance // cur_price)
            shares_bought = int(total_possible * np.clip(amount, 0.0, 1.0))
            if shares_bought > 0:
                additional_cost = shares_bought * cur_price
                prev_cost = self.cost_basis * self.shares_held
                denom = (self.shares_held + shares_bought)
                self.balance -= additional_cost
                if denom > 0:
                    self.cost_basis = (prev_cost + additional_cost) / denom
                else:
                    self.cost_basis = 0.0
                self.shares_held += shares_bought

        elif action_type < 2.0:
            shares_sold = int(self.shares_held * np.clip(amount, 0.0, 1.0))
            if shares_sold > 0:
                self.balance += shares_sold * cur_price
                self.shares_held -= shares_sold
                self.total_shares_sold += shares_sold
                self.total_sales_value += shares_sold * cur_price

        # update net worth
        self.net_worth = float(self.balance + self.shares_held * cur_price)
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0.0

        # numerical clamp
        if not np.isfinite(self.cost_basis):
            self.cost_basis = 0.0
        if not np.isfinite(self.net_worth):
            self.net_worth = float(self.balance)

    def step(self, action):
        prev_net = float(self.net_worth)
        self._take_action(action)

        self.current_step += 1
        truncated = False
        if self.current_step > len(self.df) - self.window_size:
            truncated = True

        reward = 0.0
        if prev_net > 0:
            step_return = (self.net_worth - prev_net) / prev_net
            reward = float(np.tanh(step_return))
        reward = float(np.clip(reward, -1.0, 1.0))

        terminated = self.net_worth <= 0

        obs = self._next_observation()
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.balance = float(self.initial_balance)
        self.net_worth = float(self.initial_balance)
        self.max_net_worth = float(self.initial_balance)
        self.shares_held = 0
        self.cost_basis = 0.0
        self.total_shares_sold = 0
        self.total_sales_value = 0.0

        # set current step
        self.current_step = int(random.randint(0, max(0, len(self.df) - self.window_size)))

        obs = self._next_observation()
        return obs, {}

    def reset(self, seed=None, options=None):
        # Follow gymnasium API and reset RNGs for reproducibility
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Reset the state of the environment to an initial state
        self.balance = float(INITIAL_ACCOUNT_BALANCE)
        self.net_worth = float(INITIAL_ACCOUNT_BALANCE)
        self.max_net_worth = float(INITIAL_ACCOUNT_BALANCE)
        self.shares_held = 0
        self.cost_basis = 0.0
        self.total_shares_sold = 0
        self.total_sales_value = 0.0

        # Set the current step to a random point within the data frame
        self.current_step = int(random.randint(
            0, len(self.df.loc[:, 'Open'].values) - 6))

        obs = self._next_observation()
        return obs, {}

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
