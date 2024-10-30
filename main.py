import gym
import torch
import matplotlib.pyplot as plt
import rl_utils
import numpy as np
from ppo import PPO
def main():
    actor_lr = 1e-3
    critic_lr = 1e-2
    hidden_dim = 128
    gamma = 0.99#平衡过程奖励和结果奖励
    lmbda = 0.95
    epochs = 5#原来是10
    eps = 0.2
    batch_size=128
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    fixed_step=2048
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, eps, gamma, batch_size,device)
    num_epi=0
    return_list = []
    for i in range(1000):
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        total_step=0
        episode_return = 0
        state = env.reset()
        done = False
        while total_step<=fixed_step:
            total_step+=1
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward
            if done:
                num_epi+=1
                return_list.append(episode_return)
                episode_return = 0
                state = env.reset()
                done = False
        print(f"iteration{i}: average return of last 100 episodes is {np.mean(return_list[-100:])},num_epi={num_epi}")
        agent.update(transition_dict)#隔 个回合更新一次
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