import gym
import torch
import matplotlib.pyplot as plt
import rl_utils
from ppo import PPO
def main():
    actor_lr = 1e-3
    critic_lr = 1e-2
    num_episodes = 10000
    hidden_dim = 128
    gamma = 0.99#平衡过程奖励和结果奖励
    lmbda = 0.95
    epochs = 5#原来是10
    eps = 0.2
    batch_size=128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, batch_size,device)

    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
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