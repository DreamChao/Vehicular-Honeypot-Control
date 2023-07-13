import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from env.VehicularHoneypotEnv import VehicularHoneypotEnv

class ActorCriticModel(Model):
    def __init__(self, num_actions):
        super(ActorCriticModel, self).__init__()
        self.dense1 = Dense(64, activation='relu')
        self.policy_logits = Dense(num_actions)
        self.dense2 = Dense(64, activation='relu')
        self.values = Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        logits = self.policy_logits(x)
        v = self.dense2(inputs)
        values = self.values(v)
        return logits, values

class A2CAgent:
    def __init__(self, env):
        self.env = env
        self.num_actions = self.env.action_space.n
        self.model = ActorCriticModel(self.num_actions)
        self.optimizer = Adam(learning_rate=0.001)

    def compute_loss(self, logits, values, actions, rewards, next_values, dones):
        advantage = rewards + next_values * (1 - dones) - values
        action_prob = tf.nn.softmax(logits)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
        actor_loss = tf.reduce_mean(cross_entropy * tf.stop_gradient(advantage))
        critic_loss = tf.reduce_mean(tf.square(advantage))
        return actor_loss + critic_loss

    def train(self, num_episodes=1000, gamma=0.99):
        episode_rewards = []

        for episode in range(num_episodes):
            observation = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                observation = tf.convert_to_tensor([observation], dtype=tf.float32)
                with tf.GradientTape() as tape:
                    logits, values = self.model(observation)
                    action = tf.random.categorical(logits, num_samples=1)[0, 0]
                    next_observation, reward, done, _ = self.env.step(action.numpy())

                    next_observation = tf.convert_to_tensor([next_observation], dtype=tf.float32)
                    _, next_values = self.model(next_observation)

                    episode_reward += reward

                    loss = self.compute_loss(logits, values, action, reward, next_values, done)

                grads = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                observation = next_observation.numpy()[0]

            episode_rewards.append(episode_reward)
            print("Episode:", episode + 1, "Reward:", episode_reward)

        return episode_rewards

    def play(self, num_episodes=10):
        for episode in range(num_episodes):
            observation = self.env.reset()
            done = False

            while not done:
                observation = tf.convert_to_tensor([observation], dtype=tf.float32)
                logits, _ = self.model(observation)
                action = tf.argmax(logits, axis=1)[0].numpy()
                observation, _, done, _ = self.env.step(action)
                self.env.render()

        self.env.close()

if __name__ == '__main__':
    env = VehicularHoneypotEnv(render=True)
    agent = A2CAgent(env)
    episode_rewards = agent.train(num_episodes=1000)

    # Plot the rewards over episodes
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('A2C Training')
    plt.show()

    # Play the trained agent
    agent.play(num_episodes=10)
