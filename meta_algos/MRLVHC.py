import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np
import env.VehicleEnv.VehicularHoneypotEnv

# 定义环境和任务
env = gym.make('CartPole-v0')
state_dim = env.observation_space.shape[0] #
action_dim = env.action_space.n #四个动作

# 定义元策略网络
class MetaPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=32):
        super(MetaPolicy, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 定义PPO算法
class PPO:
    def __init__(self, state_dim, action_dim, hidden_size=32, lr=1e-3):
        self.policy = MetaPolicy(state_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def train(self, episodes, K=3, eps=0.2, gamma=0.99, lam=0.95):
        for episode in range(episodes):
            state = env.reset()
            done = False
            rewards = []
            log_probs = []
            values = []
            while not done:#这当作一个回合
                # 采样动作
                state = torch.FloatTensor(state)
                action_logits = self.policy(state)
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = self.policy(state).detach()

                # 执行动作
                next_state, reward, done, _ = env.step(action.item())

                # 保存经验
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)

                state = next_state

            # 计算回报和优势值
            returns = []
            advantages = []
            G = 0
            for r in reversed(rewards):#reversed反转，从后往前遍历
                G = gamma * G + r
                returns.insert(0, G)#returns.insert(0, G)是将当前时间步的回报累积和G插入到returns列表的最前面（即索引为0的位置），并将原有的元素向后移动。相当于头插法，顺序又反过来了
            returns = torch.FloatTensor(returns)
            for t in range(len(rewards)):
                advantage = returns[t] - values[t]#Advantage = G_t - V_t，其中G_t是从时间步t开始的回报的累积和，V_t是策略网络在时间步t处的值函数预测值。python列表是可以跟pytorch张量做运算的，确保数据类型一致就可。
                advantages.append(advantage)
            #advantages = torch.FloatTensor(advantages)#创建一个新的PyTorch张量，其中的元素值和advantages列表中的元素值相同，但是数据类型为float32。
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)#对advantages张量进行标准化处理，使其均值为0，标准差为1
            #使用advantages.std()计算advantages张量的标准差
            #使用advantages.mean()计算advantage张量的均值

            # 更新策略
            for k in range(K):
                for i in range(len(rewards)):
                    state = torch.FloatTensor(state) #貌似没有状态更新，不需要更新，就是从当前状态开始的
                    action_logits = self.policy(state)
                    dist = torch.distributions.Categorical(logits=action_logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    value = self.policy(state).detach()

                    ratio = torch.exp(log_prob - log_probs[i])
                    advantage = advantages[i]

                    # 计算PPO损失
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.MSELoss()(value, returns[i])

                    # 计算总损失
                    loss = policy_loss + 0.5 * value_loss

                    # 反向传播更新参数
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

# 定义MAML算法
class MAML:
    def __init__(self, state_dim, action_dim, hidden_size=32, alpha=0.1, beta=0.1, lr=1e-3):
        self.policy = MetaPolicy(state_dim, action_dim, hidden_size)
        self.alpha = alpha
        self.beta = beta
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def fast_adapt(self, task, K=3, eps=0.2, gamma=0.99, lam=0.95):
        state_dim, action_dim = task.observation_space.shape[0], task.action_space.n

        # 复制元策略网络参数作为快速适应的起点
        policy = MetaPolicy(state_dim, action_dim)
        policy.load_state_dict(self.policy.state_dict())

        # 定义PPO算法
        ppo = PPO(state_dim, action_dim)

        # 在当前任务上进行快速适应
        for k in range(K):
            state = task.reset()
            done = False
            rewards = []
            log_probs = []
            values = []
            while not done:
                # 采样动作
                state = torch.FloatTensor(state)
                action_logits = policy(state)
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = policy(state).detach()

                # 执行动作
                next_state, reward, done, _ = task.step(action.item())

                # 保存经验
                rewards.append(reward)
                log_probs.append(log_prob)
                values.append(value)

                state = next_state

            # 计算回报和优势值
            returns = []
            advantages = []
            G = 0
            for r in reversed(rewards):
                G = gamma * G + r
                returns.insert(0, G)
            #returns = torch.FloatTensor(returns)
            for t in range(len(rewards)):
                advantage = returns[t] - values[t]#
                advantages.append(advantage)#advantages是列表
            advantages = torch.FloatTensor([item.detach().numpy() for item in advantages])#服了
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # 更新策略
            for i in range(len(rewards)):
                state = torch.FloatTensor(state)
                action_logits = policy(state)
                dist = torch.distributions.Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                value = policy(state).detach()

                ratio = torch.exp(log_prob - log_probs[i])
                advantage = advantages[i]

                # 计算PPO损失
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * advantage
                policy_loss = -torch.min(surr1, surr2).mean()

                #value = torch.tensor([value])
                returns_i = torch.tensor([returns[i]])
                value_loss = nn.MSELoss()(value, returns_i)

                # 计算总损失
                loss = policy_loss + 0.5 * value_loss

                # 反向传播更新参数
                ppo.optimizer.zero_grad()
                loss.backward()
                ppo.optimizer.step()

        # 计算元梯度
        meta_grads = {}
        for name, parameter in self.policy.named_parameters():
            meta_grads[name] = (parameter - ppo.policy.state_dict()[name]) / self.alpha #alpha是元学习率，为啥子元梯度是这个样子

        return meta_grads

    def train(self, tasks, episodes, K=3, eps=0.2, gamma=0.99, lam=0.95):
        for episode in range(episodes):
            # 在所有任务上进行元训练
            meta_grads = {}
            meta_all_grads = {}
            for task in tasks:
                meta_all_grads[task] = self.fast_adapt(task, K, eps, gamma, lam)

            #字典meta_grads的键是任务名称，值则是字典（包含各个参数名称及其值）

            # 计算元梯度的平均值，意思是所有任务的元梯度都加起来求个平均
            for name, parameter in self.policy.named_parameters():
                meta_grads[name] = torch.stack([grads[name] for grads in meta_all_grads.values()]).mean(dim=0)

            # 更新元策略网络，这个有问题吗
            for name, parameter in self.policy.named_parameters():
                parameter.grad = meta_grads[name]
            self.optimizer.step()
            self.optimizer.zero_grad()

# 定义训练集和测试集
train_tasks = [gym.make('CartPole-v0') for _ in range(10)]
test_tasks = [gym.make('CartPole-v0') for _ in range(10)]

# 训练MAML算法
maml = MAML(state_dim, action_dim)
maml.train(train_tasks, episodes=1000)

# 在测试集上测试MAML算法
total_rewards = []
for task in test_tasks:
    state = task.reset()
    done = False
    rewards = 0
    while not done:
        state = torch.FloatTensor(state)
        action_logits = maml.policy(state)
        dist = torch.distributions.Categorical(logits=action_logits)
        action = dist.sample()
        next_state, reward, done, _ = task.step(action.item())
        rewards += reward
        state = next_state
    total_rewards.append(rewards)
print('Average reward: {}'.format(np.mean(total_rewards)))