import random
import gym
import numpy as np
import tensorflow as tf
from collections import deque
from env.VehicularHoneypotEnv import VehicularHoneypotEnv
import matplotlib.pyplot as plt


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减率
        self.learning_rate = 0.001  # 学习率
        self.model = self._build_model()

        # 创建奖励列表，用于可视化奖励
        self.reward_list = []

    def _build_model(self):
        # 创建神经网络，使用两层全连接层
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='softmax'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # 以 epsilon 的概率随机选择动作，以 (1 - epsilon) 的概率选择最佳动作
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state)[0])

    def replay(self, batch_size):
        # 从记忆库中随机取出 batch_size 个样本进行更新
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if state is None:
                print("Warning: state is None!")
                continue
            target = reward
            if not done:
                # 使用神经网络预测下一个状态的最大 Q 值
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            # 使用神经网络预测当前状态的 Q 值
            target_f = self.model.predict(state)
            # 更新 Q 值
            target_f[0][action] = target
            # 使用梯度下降法更新神经网络
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def train(self, env, num_episodes=1000, max_steps=100):
        for episode in range(num_episodes):
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            total_reward = 0
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
                if len(self.memory) > batch_size:
                    self.replay(batch_size)

            # 将每个 episode 的总奖励添加到奖励列表中
            self.reward_list.append(total_reward)

            # 打印每个 episode 的奖励
            print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

        # 可视化奖励列表
        plt.plot(self.reward_list)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('DQN Training')

    plt.show()


if __name__ == "__main__":
    env = VehicularHoneypotEnv()
    observation = env.reset()
    print(observation)
    state_size = sum(space.shape[0] if isinstance(space, gym.spaces.Box) else 1 for space in env.observation_space)
    print(state_size)
    action_size = env.action_space.n
    print(action_size)
    agent = DQNAgent(state_size, action_size)
    done = False
    batch_size = 32

    agent.train(env, num_episodes=10, max_steps=10)

    agent.save("vehicular_honeypot_dqn.h5")