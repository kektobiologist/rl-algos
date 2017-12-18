import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import numpy as np
from policy_gradient.REINFORCEWithBaseline import REINFORCEWithBaseline
from collections import deque

env = gym.make('CartPole-v1')

sess = tf.Session()
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)

observation_shape = env.observation_space.shape[0]
num_actions = env.action_space.n

def policy_network_slim(states):
    net = slim.stack(states, slim.fully_connected, [24, 24], activation_fn=tf.nn.tanh, scope='stack')
    net = slim.fully_connected(net, num_actions, activation_fn=None, scope='full')
    return net
def value_network_slim(states):
    net = slim.stack(states, slim.fully_connected, [24, 24], activation_fn=tf.nn.tanh, scope='stack')
    net = slim.fully_connected(net, 1, activation_fn=None, scope='full')
    return net

pg_reinforce = REINFORCEWithBaseline(policy_network_slim,
                              value_network_slim,
                              optimizer,
                              sess,
                              num_actions,
                              observation_shape)

MAX_EPISODES = 10000
MAX_STEPS    = 200

episode_history = deque(maxlen=100)
for e in range(MAX_EPISODES):
  state = env.reset()
  episode_data=[]
  cum_reward = 0
  for time_t in range(500):
    action = pg_reinforce.sampleAction(state)
    next_state, reward, done, _ = env.step(action)
    cum_reward += reward
    reward = -100 if done else reward # normalize reward
    episode_data.append((state, action, reward))
    state = next_state
    if done:
      # train agent
      # print the score and break out of the loop
      episode_history.append(cum_reward)
      print("episode: {}/{}, score: {}, avg score for 100 runs: {:.2f}".format(e, MAX_EPISODES, cum_reward, np.mean(episode_history)))
      break
  pg_reinforce.updateModel(episode_data)
