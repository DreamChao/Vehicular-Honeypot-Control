import gym
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from env.VehicularHoneypotEnv import VehicularHoneypotEnv

# experiences replay buffer size
REPLAY_SIZE = 10000
# size of minibatch
small_BATCH_SIZE = 16
big_BATCH_SIZE = 128
BATCH_SIZE_door = 1000

# these are the hyper Parameters for DQN
# discount factor for target Q to caculate the TD aim value
GAMMA = 0.9
# the start value of epsilon
INITIAL_EPSILON = 0.5
# the final value of epsilon
FINAL_EPSILON = 0.01

class DQN():
    def __init__(self, observation_space, action_space):
        # the state is the input vector of network, in this env, it has four dimensions
        self.state_dim = sum(space.shape[0] if isinstance(space, gym.spaces.Box) else 1 for space in observation_space)
        # the action is the output vector and it has two dimensions
        self.action_dim = action_space.n
        # init experience replay, the deque is a list that first-in & first-out
        self.replay_buffer = deque()
        # you can create the network by the two parameters
        self.create_Q_network()
        # after create the network, we can define the training methods
        self.create_updating_method()
        # set the value in choose_action
        self.epsilon = INITIAL_EPSILON
        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        self.episode_rewards = []  # 用于存储每个训练周期的累计奖励值

    # the function to create the network
    # we set the network with four layers
    # (self.state_dim[4]-->50-->20-->self.action_dim[1])
    # there are two networks, the one is action_value and the other is target_action_value
    # these two networks has same architecture
    def create_Q_network(self):
        self.state_input = tf.placeholder(tf.float32, [None, self.state_dim])

        with tf.variable_scope('current_net'):
            h1 = tf.layers.dense(self.state_input, units=512, activation=tf.nn.relu)
            bn1 = tf.layers.batch_normalization(h1)
            h2 = tf.layers.dense(bn1, units=256, activation=tf.nn.relu)
            bn2 = tf.layers.batch_normalization(h2)
            h3 = tf.layers.dense(bn2, units=256, activation=tf.nn.relu)
            bn3 = tf.layers.batch_normalization(h3)
            h4 = tf.layers.dense(bn3, units=256, activation=tf.nn.relu)
            bn4 = tf.layers.batch_normalization(h4)
            h5 = tf.layers.dense(bn4, units=128, activation=tf.nn.relu)
            bn5 = tf.layers.batch_normalization(h5)
            logits = tf.layers.dense(bn5, units=self.action_dim)
            self.Q_value = tf.nn.softmax(logits)

            #self.Q_value = tf.layers.dense(bn3, units=self.action_dim)

        with tf.variable_scope('target_net'):
            h1_target = tf.layers.dense(self.state_input, units=512, activation=tf.nn.relu)
            bn1_target = tf.layers.batch_normalization(h1_target)
            h2_target = tf.layers.dense(bn1_target, units=256, activation=tf.nn.relu)
            bn2_target = tf.layers.batch_normalization(h2_target)
            h3_target = tf.layers.dense(bn2_target, units=256, activation=tf.nn.relu)
            bn3_target = tf.layers.batch_normalization(h3_target)
            h4_target = tf.layers.dense(bn3_target, units=256, activation=tf.nn.relu)
            bn4_target = tf.layers.batch_normalization(h4_target)
            h5_target = tf.layers.dense(bn4_target, units=128, activation=tf.nn.relu)
            bn5_target = tf.layers.batch_normalization(h5_target)
            logits_target = tf.layers.dense(bn5_target, units=self.action_dim)
            self.target_Q_value = tf.nn.softmax(logits_target)
            #self.target_Q_value = tf.layers.dense(bn3_target, units=self.action_dim)

        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    # the function that give the weight initial value
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    # the function that give the bias initial value
    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    # this the function that define the method to update the current_net's parameters
    def create_updating_method(self):
        # this the input action, use one hot presentation
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        # this the TD aim value
        self.y_input = tf.placeholder("float", [None])
        # this the action's Q_value
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        # this is the lost
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        # use the loss to optimize the network
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    # this is the function that use the network output the action
    def Choose_Action(self, state):
        # the output is a tensor, so the [0] is to get the output as a list
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]
        # use epsilon greedy to get the action
        if random.random() <= self.epsilon:
            # if lower than epsilon, give a random value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            # if bigger than epsilon, give the argmax value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)

    # this the function that store the data in replay memory
    def Store_Data(self, state, action, reward, next_state, done):
        # generate a list with all 0,and set the action is 1
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        # store all the elements
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        # if the length of replay_buffer is bigger than REPLAY_SIZE
        # delete the left value, make the len is stable
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

    # train the network, update the parameters of Q_value
    def Train_Network(self, BATCH_SIZE):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate TD aim value
        y_batch = []
        # give the next_state_batch flow to target_Q_value and caculate the next state's Q_value
        Q_value_batch = self.target_Q_value.eval(feed_dict={self.state_input: next_state_batch})
        # caculate the TD aim value by the formulate
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                # the Q value caculate use the max directly
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        # step 3: update the network
        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def Update_Target_Network(self):
        # update target Q netowrk
        self.session.run(self.target_replace_op)

    # use for test
    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])



EPISODES = 1000
STEPS = 300
# steps that copy the current_net's parameters to target_net
UPDATE_STEP = 50
# times that evaluate the network
TEST = 5

def main():
    # first, create the envrioment
    env = VehicularHoneypotEnv()
    observation_space = env.observation_space
    action_space = env.action_space
    agent = DQN(observation_space, action_space)
    for episode in range(EPISODES):
        # get the initial state
        state = env.reset()
        episode_reward = 0  # 当前训练周期的累计奖励值
        for step in range(STEPS):
            # get the action by state
            action = agent.Choose_Action(state)
            # step the env forward and get the new state
            next_state, reward, done, info = env.step(action)
            # store the data in order to update the network in future
            agent.Store_Data(state, action, reward, next_state, done)
            if len(agent.replay_buffer) > big_BATCH_SIZE:
                agent.Train_Network(big_BATCH_SIZE)
            # update the target_network
            if step % UPDATE_STEP == 0:
                agent.Update_Target_Network()
            state = next_state
            episode_reward += reward
            if done:
                break
        print("episode", episode)
        print("episode_reward:", episode_reward)
        agent.episode_rewards.append(episode_reward)  # 保存当前训练周期的累计奖励值
        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEPS):
                    env.render()
                    action = agent.action(state)  # direct action for test
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
        # 绘制奖励曲线
    plt.plot(agent.episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Training Progress')
    plt.savefig('img/DQN-' + time.strftime("%Y-%m-%d %p %H:%M:%S" + '.png', time.localtime()), dpi=300)
    plt.show()

if __name__ == '__main__':
  main()
