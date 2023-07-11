


import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from env.HoneypotEnv import VehicularHoneypotEnv

class ActorCriticAgent:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.actor_model = self.build_actor_model()
        self.critic_model = self.build_critic_model()

    def build_actor_model(self):
        inputs = Input(shape=self.observation_space.shape)
        x = Dense(32, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(self.action_space.n, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def build_critic_model(self):
        inputs = Input(shape=self.observation_space.shape)
        x = Dense(32, activation='relu')(inputs)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1, activation='linear')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    def train(self, num_episodes=1000, gamma=0.99):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                # Choose an action using the actor model
                action_probabilities = self.actor_model.predict(np.array([state]))[0]
                action = np.random.choice(self.action_space.n, p=action_probabilities)

                # Take the chosen action and observe the new state and reward
                next_state, reward, done, _ = self.env.step(action)

                # Calculate the TD error
                value = self.critic_model.predict(np.array([state]))[0][0]
                next_value = self.critic_model.predict(np.array([next_state]))[0][0]
                td_error = reward + gamma * next_value - value

                # Train the critic model
                with tf.GradientTape() as tape:
                    value_loss = tf.math.square(td_error)
                grads = tape.gradient(value_loss, self.critic_model.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))

                # Train the actor model
                with tf.GradientTape() as tape:
                    logits = self.actor_model(np.array([state]))
                    action_mask = tf.one_hot(action, self.action_space.n)
                    log_prob = tf.reduce_sum(action_mask * tf.math.log(logits))
                    entropy = -tf.reduce_sum(logits * tf.math.log(logits))
                    actor_loss = -(log_prob * td_error + 0.01 * entropy)
                grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))

                state = next_state
                total_reward += reward

            print(f'Episode {episode + 1}: Total reward = {total_reward}')

    def act(self, state):
        action_probabilities = self.actor_model.predict(np.array([state]))[0]
        action = np.random.choice(self.action_space.n, p=action_probabilities)
        return action

env = VehicularHoneypotEnv()
agent = ActorCriticAgent(env)
agent.train(num_episodes=1000)