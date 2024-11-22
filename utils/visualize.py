import sys
sys.path.append("../env/highway_env")
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

def visualize(env_name, model, step_num = 100):
    """
    使用 训练好的 DQN 模型进行预测并用highway_env 进行可视化。

    :param env_name: Gym 环境名称
    :param model: 训练好的模型。
    :param step_num: 预测的步数，默认为 100 步。
    """
    print("-------------------------可视化开始---------------------------------")
    # 创建 Gym 环境
    env = gym.make(env_name, render_mode='human')
    obs, info = env.reset()

    # 模型预测与可视化
    for i in range(step_num):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, dones, truncated, info = env.step(action)
        env.render()
        if dones:
            break
    env.close()