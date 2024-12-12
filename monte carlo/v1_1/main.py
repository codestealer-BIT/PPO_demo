import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from ppo import PPO
from online_collect import online_collect,bucket
from rl_utils import moving_average
import os
import random
seed=1
os.environ['PYTHONHASHSEED'] = str(seed) 
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1' #避免并行运算、GPU加速等带来的随机性

def main():
    actor_lr = 1e-3
    critic_lr = 1e-2#面对复杂任务时，这个值最好降低，调到过3e-4，但是结果不是很理想
    hidden_dim = 128
    gamma = 0.99#平衡过程奖励和结果奖励
    lmbda = 0.95
    epochs = 20#原来是10
    eps = 0.2 
    minibatch_size=128#面对复杂任务时，这个值最好变大，详情可见ppo原文，update耗时很严重
    fixed_epi=10#改的是这个
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    env.seed(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, minibatch_size)
    return_list = []

    for i in range(100):
        states,actions,returns=online_collect(agent,env,fixed_epi,return_list,gamma)
        agent.offline_train(np.array(states),np.array(actions),np.array(returns))#隔 个回合更新一次
        print(f"iteration{i}: average return of last 100 episodes is {np.mean(return_list[-100:])}")
    #agent.save()

    # episodes_list = list(range(len(return_list)))
    # plt.plot(episodes_list, return_list)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('PPO on {}'.format(env_name))
    # plt.show()

    # mv_return = moving_average(return_list, 9)
    # plt.plot(episodes_list, mv_return)
    # plt.xlabel('Episodes')
    # plt.ylabel('Returns')
    # plt.title('PPO on {}'.format(env_name))
    # plt.show()

     # 渲染模型的表现
    # num_episodes = 50  # 渲染的轮次
    # for episode in range(num_episodes):
    #     state = env.reset()
    #     done = False
    #     while not done:
    #         env.render()  # 渲染当前环境
    #         state=bucket(state)
    #         action = agent.take_action(state)  # 使用训练好的代理选择动作
    #         state, reward, done, _ = env.step(action)  # 执行动作
    #     print(f"Episode {episode + 1} finished.")

    # env.close()  # 关闭环境

if __name__=="__main__":
    main()