import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.models import Model
from env.VehicularHoneypotEnv import VehicularHoneypotEnv

class PPO:
    def __init__(self, env):
        self.env = env

        self.state_size = env.observation_space['prev_action'].n + 1
        self.action_size = env.action_space.n

        self.gamma = 0.99
        self.alpha = 0.2
        self.beta = 0.2
        self.epsilon = 0.2

        self.actor_net = self.build_actor_net()
        self.critic_net = self.build_critic_net()

        self.actor_optimizer = Adam(learning_rate=0.0003)
        self.critic_optimizer = Adam(learning_rate=0.0003)

        self.action_probs_history = []
        self.critic_value_history = []
        self.rewards_history = []
        self.entropy_history = []

    def build_actor_net(self):
        inputs = tf.keras.layers.Input(shape=(self.state_size,))
        x = Dense(32, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        action_probs = Dense(self.action_size, activation='softmax')(x)
        actor_net = Model(inputs=inputs, outputs=action_probs)
        return actor_net

    def build_critic_net(self):
        inputs = tf.keras.layers.Input(shape=(self.state_size,))
        x = Dense(32, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        critic_value = Dense(1, activation=None)(x)
        critic_net = Model(inputs=inputs, outputs=critic_value)
        return critic_net

    def choose_action(self, state):
        state = np.expand_dims(state, axis=0)
        action_probs = self.actor_net.predict(state)[0]
        action = np.random.choice(self.action_size, p=action_probs)
        return action, action_probs

    def learn(self):
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(self.rewards_history):
            cumulative_reward = reward + self.gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)

        states = np.array([s[:-1] for s in self.action_probs_history])
        action_probs = np.array(self.action_probs_history)
        critic_values = np.array(self.critic_value_history)
        discounted_rewards = np.array(discounted_rewards)

        advantages = discounted_rewards - critic_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            action_probs_history = tf.math.log(action_probs)
            entropy = -tf.reduce_mean(action_probs * tf.math.log(action_probs))
            critic_values_history = tf.squeeze(critic_values)

            new_action_probs = self.actor_net(states)
            new_critic_values = self.critic_net(states)

            actor_loss = -tf.reduce_mean(tf.minimum(
                tf.exp(action_probs_history - tf.math.log(tf.clip_by_value(new_action_probs, 1e-10, 1.0))) * advantages,
                tf.exp(tf.math.log(tf.clip_by_value(new_action_probs, 1e-10, 1.0)) - action_probs_history) * advantages * self.beta
            ))
            entropy_loss = -self.alpha * entropy
            critic_loss = Huber()(discounted_rewards, critic_values_history)

            total_loss = actor_loss + entropy_loss + critic_loss

        actor_grads = tape1.gradient(total_loss, self.actor_net.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_net.trainable_variables))

        critic_grads = tape2.gradient(critic_loss, self.critic_net.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_net.trainable_variables))

        self.action_probs_history = []
        self.critic_value_history = []
        self.rewards_history = []
        self.entropy_history = []

    def train(self, max_episodes):
        for episode in range(max_episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, action_probs = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)

                self.action_probs_history.append(action_probs)
                self.critic_value_history.append(self.critic_net.predict(np.array([state]))[0])
                self.rewards_history.append(reward)
                self.entropy_history.append(-np.sum(action_probs * np.log(action_probs)))

                state = next_state


env = VehicularHoneypotEnv(render=False)
ppo = PPO(env)

max_episodes = 1000
for episode in range(max_episodes):
    state = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, action_probs = ppo.choose_action(state)
        next_state, reward, done, info = env.step(action)

        ppo.action_probs_history.append(action_probs)
        ppo.critic_value_history.append(ppo.critic_net.predict(np.array([state]))[0])
        ppo.rewards_history.append(reward)
        ppo.entropy_history.append(-np.sum(action_probs * np.log(action_probs)))

        state = next_state
        episode_reward += reward

        if done:
            ppo.learn()

            print(f"Episode {episode}/{max_episodes}: reward={episode_reward}")
            break