import gym
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from env.VehicularHoneypotEnv import VehicularHoneypotEnv
import matplotlib.pyplot as plt

class ActorCriticPPO:
    def __init__(self, env):
        self.env = env
        self.state_dim = sum(space.shape[0] if isinstance(space, gym.spaces.Box) else 1 for space in env.observation_space)
        self.action_dim = env.action_space.n
        self.lr_actor = 0.001
        self.lr_critic = 0.001
        self.gamma = 0.99
        self.epsilon_clip = 0.2
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = Adam(learning_rate=self.lr_actor)
        self.critic_optimizer = Adam(learning_rate=self.lr_critic)

    def build_actor(self):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = Dense(256, activation='sigmoid')(inputs)
        x = BatchNormalization()(x)  # 添加批量归一化层

        x = Dense(128, activation='sigmoid')(x)
        x = BatchNormalization()(x)  # 添加批量归一化层

        x = Dense(64, activation='sigmoid')(x)
        x = BatchNormalization()(x)  # 添加批量归一化层

        # x = Dense(500, activation='sigmoid')(inputs)
        # x = BatchNormalization()(x)  # 添加批量归一化层
        # x = Dropout(0.2)(x)
        # x = Dense(500, activation='sigmoid')(x)
        # x = BatchNormalization()(x)  # 添加批量归一化层
        # x = Dropout(0.2)(x)
        # x = Dense(500, activation='sigmoid')(inputs)
        # x = BatchNormalization()(x)  # 添加批量归一化层
        # x = Dropout(0.2)(x)
        # x = Dense(500, activation='sigmoid')(x)
        # x = BatchNormalization()(x)  # 添加批量归一化层
        # x = Dropout(0.2)(x)
        outputs = Dense(self.action_dim, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def build_critic(self):
        inputs = tf.keras.Input(shape=(self.state_dim,))
        x = Dense(256, activation='sigmoid')(inputs)
        x = BatchNormalization()(x)  # 添加批量归一化层

        x = Dense(128, activation='sigmoid')(x)
        x = BatchNormalization()(x)  # 添加批量归一化层

        x = Dense(64, activation='sigmoid')(x)
        x = BatchNormalization()(x)  # 添加批量归一化层

        # x = Dense(500, activation='sigmoid')(inputs)
        # x = BatchNormalization()(x)  # 添加批量归一化层
        # x = Dropout(0.2)(x)
        # x = Dense(500, activation='sigmoid')(inputs)
        # x = BatchNormalization()(x)  # 添加批量归一化层
        # x = Dropout(0.2)(x)
        # x = Dense(500, activation='sigmoid')(x)
        # x = BatchNormalization()(x)  # 添加批量归一化层
        # x = Dropout(0.2)(x)
        # x = Dense(500, activation='sigmoid')(x)
        # x = BatchNormalization()(x)  # 添加批量归一化层
        # x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)
        probabilities = self.actor.predict(state)[0]
        action = np.random.choice(self.action_dim, p=probabilities)
        return action

    def compute_returns(self, rewards):
        returns = np.zeros_like(rewards)
        discounted_sum = 0
        for t in reversed(range(len(rewards))):
            discounted_sum = rewards[t] + self.gamma * discounted_sum
            returns[t] = discounted_sum
        returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        return returns

    def train(self, episodes=100):
        start_time = time.time()
        rewards_history = []
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            states = []
            actions = []
            rewards = []

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state
                total_reward += reward

            rewards_history.append(total_reward)
            returns = self.compute_returns(rewards)

            states = np.array(states)
            actions = np.array(actions)
            returns = np.array(returns)

            with tf.GradientTape() as tape:
                # Compute actor loss
                logits = self.actor(states)
                action_masks = tf.one_hot(actions, self.action_dim)
                action_probs = tf.reduce_sum(action_masks * logits, axis=1)
                old_logits = tf.stop_gradient(logits)
                old_action_probs = tf.reduce_sum(action_masks * old_logits, axis=1)

                ratio = action_probs / (old_action_probs + 1e-8)
                surrogate1 = ratio * returns
                surrogate2 = tf.clip_by_value(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * returns
                actor_loss = -tf.reduce_mean(tf.minimum(surrogate1, surrogate2))

            actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

            # Create a new tape for computing critic gradients
            with tf.GradientTape() as tape:
                # Compute critic loss
                values = self.critic(states)
                critic_loss = tf.reduce_mean(tf.square(values - returns[:, np.newaxis]))

            critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

            print("Episode:", episode, "Total Reward:", total_reward)

        end_time = time.time()
        train_time = (end_time - start_time)%60
        print(train_time)
        # 绘制奖励变化曲线
        time.strftime("%Y-%m-%d %p %H:%M:%S", time.localtime())
        plt.plot(rewards_history)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig('img/AC_PPO-'+ time.strftime("%Y-%m-%d %p %H:%M:%S"+'.png', time.localtime()), dpi=300)
        plt.show()

if __name__ == "__main__":
    env = VehicularHoneypotEnv()
    agent = ActorCriticPPO(env)
    agent.train(episodes=100)