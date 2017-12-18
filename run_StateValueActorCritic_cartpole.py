import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import numpy as np
from policy_gradient.StateValueActorCritic import Actor, Critic
from collections import deque

env = gym.make('CartPole-v1')

sess = tf.Session()
actor_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
critic_optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

observation_shape = env.observation_space.shape[0]
num_actions = env.action_space.n

def actor_network(states):
    net = slim.stack(states, slim.fully_connected, [24], activation_fn=tf.nn.tanh, scope='stack')
    net = slim.fully_connected(net, num_actions, activation_fn=None, scope='full')
    return net

def critic_network(states):
    net = slim.stack(states, slim.fully_connected, [24], activation_fn=tf.nn.relu, scope='stack')
    net = slim.fully_connected(net, 1, activation_fn=None, scope='full')
    net = tf.squeeze(net, [1])
    return net

actor = Actor(actor_network, actor_optimizer, sess, num_actions, observation_shape)
critic = Critic(critic_network, critic_optimizer, sess, observation_shape)

MAX_EPISODES = 10000
MAX_STEPS    = 200

sess.run(tf.global_variables_initializer())

episode_history = deque(maxlen=100)
for e in range(MAX_EPISODES):
  state = env.reset()
  cum_reward = 0
  for time_t in range(500):
    action = actor.sampleAction(state)
    next_state, reward, done, _ = env.step(action)
    reward = -20 if done and time_t < 499 else reward # normalize reward
    cum_reward += reward
    td_error = critic.learn(state, reward, next_state)
    actor.learn(state, action, td_error)
    if done:
      # train agent
      # print the score and break out of the loop
      episode_history.append(cum_reward)
      print("episode: {}/{}, score: {}, avg score for 100 runs: {:.2f}".format(e, MAX_EPISODES, cum_reward, np.mean(episode_history)))
      break
    state = next_state
