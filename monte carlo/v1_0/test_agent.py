import torch
from ppo import PPO
import gym
import time
actor_lr = 1e-4
critic_lr = 1e-4#面对复杂任务时，这个值最好降低，调到过3e-4，但是结果不是很理想
hidden_dim = 128
gamma = 0.99#平衡过程奖励和结果奖励
lmbda = 0.95
epochs = 3#原来是10
eps = 0.2
minibatch_size=256#面对复杂任务时，这个值最好变大，详情可见ppo原文
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
fixed_epi=200
env_name = 'Pong-ram-v0'
env = gym.make(env_name)
env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, minibatch_size,device)


checkpoint = torch.load('ppo_agent.pth')
agent.actor.load_state_dict(checkpoint['actor_state_dict'])
agent.critic.load_state_dict(checkpoint['critic_state_dict'])
agent.actor.eval()  # 设置为评估模式
agent.critic.eval()  # 设置为评估模式

    # 渲染模型的表现
num_episodes = 50  # 渲染的轮次
done=False
for episode in range(num_episodes):
    state = env.reset()
    time.sleep(0.05)
    while not done:
        action = agent.take_action(state)  # 使用训练好的代理选择动作
        state, reward, done, _ = env.step(action)  # 执行动作
        env.render()  # 渲染当前环境
        time.sleep(0.05)
    # print(f"Episode {episode + 1} finished.")

env.close()  # 关闭环境