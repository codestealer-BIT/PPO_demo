import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
torch.manual_seed(0)
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, minibatch_size,device):
        self.minibatch_size=minibatch_size
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    def offline_train(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)


        td_target = torch.tensor(transition_dict['returns'], dtype=torch.float).view(-1, 1).to(self.device)
        advantage = td_target-self.critic(states).to(self.device)
        advantage=advantage.detach()#没有这一步的话，由于advantage参与了actor_loss的计算，在actor_loss.backward()的时候会
        #释放advantage中self.critic的计算图，然后critic_loss.backward()的时候中critic的计算图就不存在了。所以必须先把critic的计算图detach掉
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # 创建 TensorDataset
        dataset = TensorDataset(states, actions, advantage,old_log_probs,td_target)

        # 创建 DataLoader
        dataloader = DataLoader(dataset, self.minibatch_size, shuffle=True)

        for _ in range(self.epochs):
            for batch_states,batch_actions,batch_advantage,batch_old_log_probs,batch_td_target in dataloader:

                log_probs = torch.log(self.actor(batch_states).gather(1, batch_actions))
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantage
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * batch_advantage  # 截断
                actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
                critic_loss = torch.mean(F.mse_loss(self.critic(batch_states), batch_td_target.detach()))

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()
    
    def save(self):
        torch.save({ 'actor_state_dict': self.actor.state_dict(),'critic_state_dict': self.critic.state_dict(),}, 'ppo_agent.pth') 
        