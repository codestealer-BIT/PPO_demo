import gym
import torch
import matplotlib.pyplot as plt
import rl_utils
import numpy as np
from ppo import PPO
from online_collect import online_collect
def main():
    actor_lr = 1e-3
    critic_lr = 1e-2#面对复杂任务时，这个值最好降低，调到过3e-4，但是结果不是很理想
    hidden_dim = 128
    gamma = 0.99#平衡过程奖励和结果奖励
    lmbda = 0.95
    epochs = 5#原来是10
    eps = 0.2
    batch_size=128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    fixed_epi=10
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, batch_size,device)
    return_list = []

    for i in range(2000):
        transition_dict=online_collect(agent,env,fixed_epi,return_list)
        agent.offline_train(transition_dict)#隔 个回合更新一次
        print(f"iteration{i}: average return of last 100 episodes is {np.mean(return_list[-100:])}")

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()

    mv_return = rl_utils.moving_average(return_list, 9)
    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO on {}'.format(env_name))
    plt.show()

if __name__=="__main__":
    main()