import sys
sys.path.append("../env/highway_env")
import gymnasium as gym
from stable_baselines3 import DQN

def train (env_name, total_timesteps=20000, save_path="model/model"):
    """
        使用 Stable-Baselines3 训练 DQN 模型。

        :param env_name: Gym 环境名称，例如 "highway-v0"。
        :param total_timesteps: 训练的总时间步数。
        :param save_path: 保存模型的路径。
        :return: 训练好的 DQN 模型。
     """
    print("-------------------------模型训练开始---------------------------------")
    # 创建 Gym 环境
    env = gym.make(env_name, render_mode='human')

    # 初始化 DQN 模型
    model = DQN('MlpPolicy', env,
                  policy_kwargs=dict(net_arch=[256, 256]),
                  learning_rate=5e-4,
                  buffer_size=15000,
                  learning_starts=200,
                  batch_size=32,
                  gamma=0.8,
                  train_freq=1,
                  gradient_steps=1,
                  target_update_interval=50,
                  verbose=1,
                  tensorboard_log="evaluator/")
    # 训练模型
    model.learn(int(total_timesteps))

    # 保存模型
    model.save(save_path)
    print(f"模型已保存到 {save_path}")

    print("-------------------------模型训练结束---------------------------------")

    # 返回训练好的模型
    return model
