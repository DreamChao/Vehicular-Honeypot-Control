import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from multiprocessing import Process, Pipe
import argparse
import gym
import matplotlib.pyplot as plt
import seaborn as sns
import easydict
from common.multiprocessing_env import SubprocVecEnv
from env.VehicularHoneypotEnv import VehicularHoneypotEnv



class ActorCritic(nn.Module):
    ''' A2C网络模型，包含一个Actor和Critic
    '''

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist = Categorical(probs)
        return dist, value

class A2C:
    ''' A2C算法
    '''
    def __init__(self,n_states,n_actions,cfg) -> None:
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.model = ActorCritic(n_states, n_actions, cfg.hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters())

    def compute_returns(self,next_value, rewards, masks):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + self.gamma * R * masks[step]
            returns.insert(0, R)
        return returns

def make_envs(env_name):
    def _thunk():
        env = VehicularHoneypotEnv()
        return env
    return _thunk
def test_env(env,model,vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(cfg.device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward

def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def train(cfg,envs):
    print('Start training!')
    print(f'Env:{cfg.env_name}, Algorithm:{cfg.algo_name}, Device:{cfg.device}')
    #env = gym.make(cfg.env_name) # a single env
    env = VehicularHoneypotEnv()
    n_states  = sum(space.shape[0] if isinstance(space, gym.spaces.Box) else 1 for space in env.observation_space)
    n_actions = envs.action_space.n
    model = ActorCritic(n_states, n_actions, cfg.hidden_dim).to(cfg.device)
    optimizer = optim.Adam(model.parameters())
    step_idx    = 0
    test_rewards = []
    test_ma_rewards = []
    state = envs.reset()
    while step_idx < cfg.max_steps:
        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        entropy = 0
        # rollout trajectory
        for _ in range(cfg.n_steps):
            state = torch.FloatTensor(state).to(cfg.device)
            dist, value = model(state)
            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())
            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(cfg.device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(cfg.device))
            state = next_state
            step_idx += 1
            if step_idx % 200 == 0:
                test_reward = np.mean([test_env(env,model) for _ in range(10)])
                print(f"step_idx:{step_idx}, test_reward:{test_reward}")
                test_rewards.append(test_reward)
                if test_ma_rewards:
                    test_ma_rewards.append(0.9*test_ma_rewards[-1]+0.1*test_reward)
                else:
                    test_ma_rewards.append(test_reward)
                # plot(step_idx, test_rewards)
        next_state = torch.FloatTensor(next_state).to(cfg.device)
        _, next_value = model(next_state)
        returns = compute_returns(next_value, rewards, masks)
        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)
        advantage = returns - values
        actor_loss  = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Finish training！')
    return test_rewards, test_ma_rewards

def plot_rewards(rewards, ma_rewards, cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title("learning curve on {} of {} for {}".format(
        cfg.device, cfg.algo_name, cfg.env_name))
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(ma_rewards, label='ma rewards')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    cfg = easydict.EasyDict({
        "algo_name": 'A2C',
        "env_name": 'Vehicular-honeypot',
        "n_envs": 8,
        "max_steps": 20000,
        "n_steps": 5,
        "gamma": 0.99,
        "lr": 1e-3,
        "hidden_dim": 256,
        "device": torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    })
    envs = [make_envs(cfg.env_name) for i in range(cfg.n_envs)]
    envs = SubprocVecEnv(envs)
    rewards, ma_rewards = train(cfg, envs)
    plot_rewards(rewards, ma_rewards, cfg, tag="train")  # 画出结果