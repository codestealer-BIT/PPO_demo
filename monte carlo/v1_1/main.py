import gym
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from ppo import PPO,PolicyNet,ValueNet
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
    actor_net = PolicyNet(state_dim, hidden_dim, action_dim)
    critic_net = ValueNet(state_dim, hidden_dim)
    agent = PPO(actor_net,critic_net,actor_lr,critic_lr,eps)
    return_list = []

    for i in range(500):
        states,actions,old_log_probs,returns=online_collect(agent,env,fixed_epi,return_list,gamma)
        old_log_probs = tf.gather(np.array(old_log_probs), actions, axis=1, batch_dims=1)
        dataset = tf.data.Dataset.from_tensor_slices((np.array(states).astype(np.float32), np.array(actions).astype(np.int32).reshape(-1),old_log_probs,np.array(returns).astype(np.float32).reshape(-1)))
        dataset = dataset.shuffle(buffer_size=1024).batch(minibatch_size)
        
        for _ in range(epochs):
            for batch_states, batch_actions,batch_old_log_probs,batch_returns in dataset:
                agent.train_one_batch(batch_states,batch_actions,batch_old_log_probs,batch_returns)
        # print(agent.actor.weights)
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