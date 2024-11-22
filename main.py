# Demo
from utils.train import train
from utils.evaluate import evaluate
from utils.visualize import visualize

# 环境选择
env_name = 'highway-v0'

# 模型训练
trained_model = train(env_name, total_timesteps=20000)

# 模型评估
evaluate(env_name, trained_model)

# 模型可视化
visualize(env_name, trained_model, step_num=10)

# 模型训练阶段的 average reward 图可以打开tensorboard查看
# 在终端输入如下代码打开：
# tensorboard --logdir “训练模型log存储地址”



