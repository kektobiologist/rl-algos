import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import numpy as np
from policy_gradient.ActionValueActorCritic import Actor, Critic
from collections import deque

env = gym.make('CartPole-v1')

sess = tf.Session()
actor_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
critic_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

observation_shape = env.observation_space.shape[0]
num_actions = env.action_space.n

def actor_network(states):
    net = slim.stack(states, slim.fully_connected, [24, 24], activation_fn=tf.nn.tanh, scope='stack')
    net = slim.fully_connected(net, num_actions, activation_fn=None, scope='full')
    return net

def critic_network(states):
    net = slim.stack(states, slim.fully_connected, [24, 24], activation_fn=tf.nn.relu, scope='stack')
    net = slim.fully_connected(net, num_actions, activation_fn=None, scope='full')
    return net

actor = Actor(actor_network, actor_optimizer, sess, num_actions, observation_shape)
critic = Critic(critic_network, critic_optimizer, sess, num_actions, observation_shape)

MAX_EPISODES = 10000
MAX_STEPS    = 200

sess.run(tf.global_variables_initializer())

episode_history = deque(maxlen=100)
for e in range(MAX_EPISODES):
  state = env.reset()
  cum_reward = 0
  action = actor.sampleAction(state)
  for time_t in range(500):
    # env.render()
    next_state, reward, done, _ = env.step(action)
    cum_reward += reward
    reward = -100 if done else reward # normalize reward
    next_action = actor.sampleAction(next_state)
    td_error = critic.learn(state, action, reward, next_state, next_action)
    actor.learn(state, action, td_error)
    if done:
      # train agent
      # print the score and break out of the loop
      episode_history.append(cum_reward)
      print("episode: {}/{}, score: {}, avg score for 100 runs: {:.2f}".format(e, MAX_EPISODES, cum_reward, np.mean(episode_history)))
      break
    state = next_state
    action = next_action