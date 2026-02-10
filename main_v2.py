import pandas as pd
from env.StockTradingEnv import StockTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def main():
    df = pd.read_csv("./data/AAPL.csv").sort_values("Date")

    # create env and wrap with VecNormalize for obs normalization
    env = DummyVecEnv([lambda: StockTradingEnv(df)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    model = PPO("MlpPolicy", env, verbose=1, learning_rate=1e-4, tensorboard_log="./tb_logs/")

    # short test training - adjust total_timesteps for real training
    model.learn(total_timesteps=20000)

    # Save model and normalization statistics
    model.save("ppo_stock_v2")
    env.save("vecnormalize_stats.pkl")


if __name__ == "__main__":
    main()
