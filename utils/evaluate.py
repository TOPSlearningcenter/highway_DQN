import sys
sys.path.append("../env/highway_env")
import gymnasium as gym
# from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

def evaluate(env_name, model):
    """
        评估训练好的 DQN 模型性能。

        :param env_name: Gym 环境名称，例如 "highway-v0"。
        :param model：训练好的 DQN 模型。
    """
    print("-------------------------模型评估开始---------------------------------")

    # 创建 Gym 环境
    env = gym.make(env_name, render_mode='human')

    # 模型评估（mean_reward：平均奖励, std_reward：奖励标准差）
    mean_reward, std_reward = evaluate_policy(
        model,
        model.get_env(),
        deterministic=True,
        render=True,
        n_eval_episodes=10)

    print(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")


    # obs, info = model.reset()
    # done = truncated = False
    # while not (done or truncated):
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, done, truncated, info = model.step(action)

    print("-------------------------模型评估结束---------------------------------")
