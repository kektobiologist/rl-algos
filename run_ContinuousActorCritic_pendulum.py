import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import numpy as np
from policy_gradient.StateValueActorCritic import ContinuousActor, Critic
from collections import deque

env = gym.make('Pendulum-v0')

sess = tf.Session()
actor_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
critic_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

observation_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]

def actor_network(states, actions):
  # should return predicted_actions, action_probs
  with tf.variable_scope('mu'):
    net = slim.stack(states, slim.fully_connected, [24], activation_fn=tf.nn.tanh, scope='stack')
    mu = slim.fully_connected(net, action_shape, activation_fn=None, scope='full')

  with tf.variable_scope('std'):
    net = slim.stack(states, slim.fully_connected, [24], activation_fn=tf.nn.tanh, scope='stack')
    logstd = slim.fully_connected(net, action_shape, activation_fn=None, scope='full')
    std = tf.exp(logstd)

  prob_dist = tf.distributions.Normal(loc=mu, scale=std)
  batch_size = tf.shape(states)[0]
  # predicted actions
  predicted_actions = prob_dist.sample(batch_size)
  # action_probs. each row of shape action_shape, needs to be reduced by multiplication
  action_probs = prob_dist.prob(actions)
  action_probs = tf.reduce_prod(action_probs, axis=1)
  return predicted_actions, action_probs

def critic_network(states):
  net = slim.stack(states, slim.fully_connected, [24,], activation_fn=tf.nn.tanh, scope='stack')
  net = slim.fully_connected(net, 1, activation_fn=None, scope='full')
  net = tf.squeeze(net, [1])
  return net

actor = ContinuousActor(actor_network, actor_optimizer, sess, action_shape, observation_shape)
critic = Critic(critic_network, critic_optimizer, sess, observation_shape)

MAX_EPISODES = 10000
MAX_STEPS    = 200

sess.run(tf.global_variables_initializer())

episode_history = deque(maxlen=100)
for e in range(MAX_EPISODES):
  state = env.reset()
  cum_reward = 0
  for time_t in range(200):
    action = [actor.sampleAction(state)]
    # print action
    next_state, reward, done, _ = env.step(action)
    cum_reward += reward
    # reward = -100 if done and time_t < 499 else reward # normalize reward
    td_error = critic.learn(state, reward, next_state)
    actor.learn(state, action, td_error)
    if done:
      # train agent
      # print the score and break out of the loop
      episode_history.append(cum_reward)
      print("episode: {}/{}, score: {}, avg score for 100 runs: {:.2f}".format(e, MAX_EPISODES, cum_reward, np.mean(episode_history)))
      break
    state = next_state
