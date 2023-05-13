import torch
import torch.nn as nn
import torch.optim as optim
import gym

class Policy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(Policy, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.softmax(dim=-1)
#策略网络与价值网络都是3层结构，且共享前两层
    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        action_prob = self.softmax(self.fc3(x))#策略网络输出
        value = self.fc4(x)#价值网络输出
        return action_prob, value

class PPO:
    def __init__(self, obs_dim, action_dim, hidden_dim, lr, betas, gamma, K, eps_clip):
        self.policy = Policy(obs_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.gamma = gamma#折扣率，计算return时使用
        self.K = K#这是什么？
        self.eps_clip = eps_clip

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0)
        action_prob, _ = self.policy(obs)
        action_dist = torch.distributions.Normal(action_prob, 1)
        action = action_dist.sample()
        return action.item()

    def update(self, obs, action, old_action_prob, advantage, target_value):
        obs = torch.FloatTensor(obs)
        action = torch.FloatTensor([action])
        old_action_prob = torch.FloatTensor([old_action_prob])
        advantage = torch.FloatTensor([advantage])
        target_value = torch.FloatTensor([target_value])

        action_prob, value = self.policy(obs)
        action_dist = torch.distributions.Normal(action_prob, 1)
        entropy = action_dist.entropy().mean()

        # calculate advantage
        delta = target_value - value
        delta = delta.detach().numpy()
        advantage_lst = []
        advantage = 0
        for i in reversed(range(len(delta))):
            advantage = self.gamma * self.K * advantage + delta[i][0]
            advantage_lst.append([advantage])
        advantage_lst.reverse()
        advantage = torch.FloatTensor(advantage_lst)

        # calculate surrogate loss
        ratio = torch.exp(action_dist.log_prob(action) - torch.log(old_action_prob))
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        policy_loss = -torch.min(surr1, surr2).mean()

        # calculate value loss
        value_loss = nn.MSELoss()(value, target_value)

        # update policy
        self.optimizer.zero_grad()
        (policy_loss + 0.5 * value_loss - 0.01 * entropy).backward()
        self.optimizer.step()

class MAML:
    def __init__(self, policy, alpha, beta, num_tasks):
        self.policy = policy
        self.alpha = alpha
        self.beta = beta
        self.num_tasks = num_tasks

    def meta_update(self, envs):
        for env in envs:
            obs = env.reset()
            ppo = PPO(obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0],
                      hidden_dim=64, lr=0.001, betas=(0.9, 0.999), gamma=0.99, K=0.5, eps_clip=0.2)
            for i in range(self.alpha):
                action = ppo.select_action(obs)
                new_obs, reward, done, _ = env.step(action)
                old_action_prob, value = ppo.policy(torch.FloatTensor(obs))
                advantage = reward + (1 - done) * ppo.gamma * ppo.policy(torch.FloatTensor(new_obs))[1] - value.item()
                ppo.update(obs, action, old_action_prob.item(), advantage, ppo.policy(torch.FloatTensor(obs))[1].item())
                obs = new_obs
                if done:
                    obs = env.reset()

            # calculate task loss
            task_loss = 0
            for i in range(self.beta):
                action = ppo.select_action(obs)
                new_obs, reward, done, _ = env.step(action)
                old_action_prob, value = ppo.policy(torch.FloatTensor(obs))
                advantage = reward + (1 - done) * ppo.gamma * ppo.policy(torch.FloatTensor(new_obs))[1] - value.item()
                task_loss += advantage
                obs = new_obs
                if done:
                    obs = env.reset()

            # calculate meta gradients
            self.policy.zero_grad()
            task_loss.backward()
        for param in self.policy.parameters():
            param.grad = param.grad / self.num_tasks

        return self.policy.parameters()

# example usage