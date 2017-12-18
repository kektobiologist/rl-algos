# Action value actor critic
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import numpy as np
import random
from collections import deque

class ActionValueActorCritic:
  def __init__(self,
               policy_network,
               action_value_network,
               optimizer,
               session,
               num_actions,
               observation_shape):
    self.session = session
    self.policy_network = policy_network
    self.action_value_network = action_value_network
    self.optimizer = optimizer
    self.num_actions = num_actions
    self.observation_shape = observation_shape
    self.discount_factor = 0.99
    self.createVariables()
    var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.session.run(tf.variables_initializer(var_lists))

    # make sure all variables are initialized
    self.session.run(tf.assert_variables_initialized())

    self.all_rewards = []
    self.max_reward_length = 1000000
    self.memory = deque(maxlen=10000)


  def getActionValues(self, action_value_outputs, taken_actions, batch_size):
    # converts action value network outputs to list of action values provided the taken_actions
    range_tensor = tf.range(batch_size)
    stacked_taken_actions = tf.stack([range_tensor, taken_actions], axis=1)
    action_values = tf.gather_nd(action_value_outputs, stacked_taken_actions)
    return action_values

  def createVariables(self):
    self.states = tf.placeholder(tf.float32, [None, self.observation_shape])
    with tf.variable_scope('policy_network'):
      policy_outputs = self.policy_network(self.states)
    with tf.variable_scope('action_value_network'):
      action_value_outputs = self.action_value_network(self.states)
    
    # sample action variable
    self.predicted_actions = tf.multinomial(policy_outputs, 1)
    
    # policy network loss
    batch_size = tf.shape(self.states)[0]
    self.taken_actions = tf.placeholder(tf.int32, (None,))
    action_values = self.getActionValues(action_value_outputs, self.taken_actions, batch_size)
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy_outputs, labels=self.taken_actions)
    neg_log_loss = tf.reduce_mean(cross_entropy_loss * action_values)
    
    # action value network loss
    self.rewards = tf.placeholder(tf.float32, (None,))
    self.next_states = tf.placeholder(tf.float32, [None, self.observation_shape])
    with tf.variable_scope('policy_network', reuse=True):
      next_states_policy_outputs = self.policy_network(self.next_states)
    next_predicted_actions = tf.to_int32(tf.squeeze(tf.multinomial(next_states_policy_outputs, 1), [1]))
    with tf.variable_scope('action_value_network', reuse=True):
      next_states_action_value_outputs = self.action_value_network(self.next_states)
    next_states_action_values = self.getActionValues(next_states_action_value_outputs, next_predicted_actions, batch_size)
    target_action_values = self.rewards + self.discount_factor * next_states_action_values
    action_value_loss = tf.losses.mean_squared_error(target_action_values, action_values)

    self.policy_network_train_op = self.optimizer.minimize(neg_log_loss, var_list=tf.trainable_variables(scope='policy_network'))
    self.action_value_network_train_op = self.optimizer.minimize(action_value_loss, var_list=tf.trainable_variables(scope='action_value_network'))

  def sampleAction(self, state):
    return self.session.run(self.predicted_actions, feed_dict={self.states: [state]}).squeeze()

  def updateModel(self, state, action, reward, next_state):
    self.session.run(self.policy_network_train_op, feed_dict = {
      self.states: [state],
      self.taken_actions: [action],
      })
    self.memory.append((state, action, reward, next_state))
    batch_size = min(64, len(self.memory))
    (states, actions, rewards, next_states) = zip(*random.sample(self.memory, batch_size))
    self.session.run(self.action_value_network_train_op, feed_dict = {
      self.states: states,
      self.taken_actions: actions,
      self.rewards: rewards,
      self.next_states: next_states
      })



class Actor:
  def __init__(self,
               actor_network,
               optimizer,
               session,
               num_actions,
               observation_shape):
    self.actor_network = actor_network
    self.optimizer = optimizer
    self.session = session
    self.num_actions = num_actions
    self.observation_shape = observation_shape

    self.discount_factor = 0.99
    self.createVariables()


  def createVariables(self):
    self.states = tf.placeholder(tf.float32, [None, self.observation_shape])
    self.td_error = tf.placeholder(tf.float32, [None,])
    self.actions = tf.placeholder(tf.int32, [None,])

    with tf.variable_scope('actor_network'):
      actor_outputs = self.actor_network(self.states)

    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=actor_outputs, labels=self.actions)
    neg_log_loss = tf.reduce_mean(cross_entropy_loss * self.td_error)

    self.train_op = self.optimizer.minimize(neg_log_loss, var_list=tf.trainable_variables(scope='actor_network'))

    self.predicted_actions = tf.squeeze(tf.multinomial(actor_outputs, 1), [1])

  def sampleAction(self, state):
    return self.session.run(self.predicted_actions, {self.states: [state] }).squeeze()

  def learn(self, state, action, td_error):
    self.session.run(self.train_op, {
        self.states: [state],
        self.actions: [action],
        self.td_error: [td_error]
      })


class Critic:
  def __init__(self,
               critic_network,
               optimizer,
               session,
               num_actions,
               observation_shape):
    self.critic_network = critic_network
    self.optimizer = optimizer
    self.session = session
    self.num_actions = num_actions
    self.observation_shape = observation_shape

    self.discount_factor = 0.99
    self.createVariables()

  def getQValues(self, critic_outputs, actions, batch_size):
    # converts action value network outputs to list of action values provided the taken_actions
    range_tensor = tf.range(batch_size)
    stacked_taken_actions = tf.stack([range_tensor, actions], axis=1)
    action_values = tf.gather_nd(critic_outputs, stacked_taken_actions)
    return action_values

  def createVariables(self):
    self.states = tf.placeholder(tf.float32, [None, self.observation_shape])
    self.actions = tf.placeholder(tf.int32, [None,])
    self.rewards = tf.placeholder(tf.float32, [None,])
    self.q_next = tf.placeholder(tf.float32, [None,])
    batch_size = tf.shape(self.states)[0]

    with tf.variable_scope('critic_network', reuse=tf.AUTO_REUSE):
      critic_outputs = self.critic_network(self.states)
    self.q = self.getQValues(critic_outputs, self.actions, batch_size)

    self.error = self.rewards + self.discount_factor * self.q_next - self.q
    loss = tf.reduce_mean(tf.square(self.error))

    self.train_op = self.optimizer.minimize(loss, var_list=tf.trainable_variables(scope='critic_network'))

  def learn(self, state, action, reward, next_state, next_action):
    q_next = self.session.run(self.q, {self.states: [next_state], self.actions: [next_action]}).squeeze()
    td_error, _ = self.session.run([self.q, self.train_op], {
        self.states: [state],
        self.rewards: [reward],
        self.actions: [action],
        self.q_next: [q_next]
      })
    return td_error.squeeze()