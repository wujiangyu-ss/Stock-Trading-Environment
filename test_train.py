import pandas as pd
from env.StockTradingEnv import StockTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv

print('Loading data...')
df = pd.read_csv('./data/AAPL.csv').sort_values('Date')
print('Creating env...')
env = DummyVecEnv([lambda: StockTradingEnv(df)])
print('Creating model...')
model = PPO(MlpPolicy, env, verbose=1)
print('Start short learn...')
model.learn(total_timesteps=2048)
print('Done')
