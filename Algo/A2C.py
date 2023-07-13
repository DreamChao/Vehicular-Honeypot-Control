import random
import gym
from gym import spaces
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from env.VehicularHoneypotEnv import VehicularHoneypotEnv

class A2C:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        # 策略网络
        self.actor = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(self.action_size, activation='softmax')
        ])

        # 值网络
        self.critic = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(self.value_size, activation=None)
        ])

        # 优化器
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    def get_action(self, state):
        policy = self.actor.predict(state)[0]
        action = np.random.choice(self.action_size, 1, p=policy)[0]
        return action

    def train(self, state, action, reward, next_state, done):
        target = reward + (1 - done) * 0.95 * self.critic.predict(next_state)[0]
        advantage = target - self.critic.predict(state)[0]

        # 计算策略梯度
        with tf.GradientTape() as tape:
            policy = self.actor(state)
            log_prob = tf.math.log(policy[0, action])
            loss_actor = -log_prob * advantage

        # 计算价值函数梯度
        with tf.GradientTape() as tape:
            value = self.critic(state)
            loss_critic = tf.keras.losses.mean_squared_error(target, value)

        # 计算总损失
        total_loss = loss_actor + loss_critic

        # 计算梯度并更新网络参数
        grads = tape.gradient(total_loss, self.actor.trainable_variables + self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables + self.critic.trainable_variables))

# 训练函数
def train(env, agent, episodes):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        state_size = sum(space.shape[0] if isinstance(space, gym.spaces.Box) else 1 for space in env.observation_space)
        state = np.reshape(state, [1, state_size])
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.train(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        if episode % 10 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")

    # 绘制训练曲线
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

# 创建环境和 A2C 算法对象，然后训练模型
if __name__ == '__main__':
    env = VehicularHoneypotEnv()
    state_size = sum(space.shape[0] if isinstance(space, gym.spaces.Box) else 1 for space in env.observation_space)
    action_size = env.action_space.n
    agent = A2C(state_size, action_size)
    train(env, agent, 200)