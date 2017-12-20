import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import numpy as np
import random
from collections import deque


class StateValueActorCritic:
  def __init__(self,
               actor_network,
               critic_network,
               optimizer,
               session,
               num_actions,
               observation_shape):
    self.actor_network = actor_network
    self.critic_network = critic_network
    self.optimizer = optimizer
    self.session = session
    self.num_actions = num_actions
    self.observation_shape = observation_shape

    self.discount_factor = 0.99
    self.createVariables()

    var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.session.run(tf.variables_initializer(var_lists))

    # make sure all variables are initialized
    self.session.run(tf.assert_variables_initialized())

  def createVariables(self):
    self.states = tf.placeholder(tf.float32, [None, self.observation_shape])
    self.next_states = tf.placeholder(tf.float32, [None, self.observation_shape])
    self.rewards = tf.placeholder(tf.float32, [None])
    self.actions = tf.placeholder(tf.int32, [None,])

    with tf.variable_scope('actor_network'):
      actor_outputs = self.actor_network(self.states)
    with tf.variable_scope('critic_network', reuse=tf.AUTO_REUSE):
      states_critic_outputs = self.critic_network(self.states)
      next_states_critic_outputs = self.critic_network(self.states)

    # sample action variable
    self.predicted_actions = tf.squeeze(tf.multinomial(actor_outputs, 1), [1])

    # td_error
    td_error = self.rewards + self.discount_factor * next_states_critic_outputs - states_critic_outputs
    # critic loss
    critic_loss = tf.reduce_mean(tf.square(td_error))

    # actor loss
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=actor_outputs, labels=self.actions)
    actor_loss = tf.reduce_mean(cross_entropy_loss * td_error)

    self.train_op = [self.optimizer.minimize(actor_loss, var_list=tf.trainable_variables(scope='actor_network')),
                     self.optimizer.minimize(critic_loss, var_list=tf.trainable_variables(scope='critic_network'))]

  def sampleAction(self, state):
    return self.session.run(self.predicted_actions, {self.states: [state]}).squeeze()


  def updateModel(self, state, action, reward, next_state):
    self.session.run(self.train_op, {
        self.states: [state],
        self.actions: [action],
        self.rewards: [reward],
        self.next_states: [next_state]
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

    log_prob = tf.log(tf.nn.softmax(actor_outputs)[0][self.actions[0]])
    neg_log_loss = -tf.reduce_mean(self.td_error * log_prob)
    # cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=actor_outputs, labels=self.actions)
    # neg_log_loss = tf.reduce_mean(cross_entropy_loss * self.td_error)

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


class ContinuousActor:
  def __init__(self,
               actor_network,
               optimizer,
               session,
               action_shape,
               observation_shape):
    self.actor_network = actor_network
    self.optimizer = optimizer
    self.session = session
    self.action_shape = action_shape
    self.observation_shape = observation_shape

    self.discount_factor = 0.99
    self.createVariables()

  def createVariables(self):
    self.states = tf.placeholder(tf.float32, [None, self.observation_shape])
    self.actions = tf.placeholder(tf.float32, [None, self.action_shape])
    self.td_error = tf.placeholder(tf.float32, [None,])

    with tf.variable_scope('actor_network'):
      self.predicted_actions, action_probs = self.actor_network(self.states, self.actions)

    log_probs = tf.log(action_probs)
    neg_log_loss = -tf.reduce_mean(self.td_error * log_probs)

    self.train_op = self.optimizer.minimize(neg_log_loss, var_list=tf.trainable_variables(scope='actor_network'))

  def sampleAction(self, state):
    return self.session.run(self.predicted_actions, {self.states: [state]}).squeeze()

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
               observation_shape):
    self.critic_network = critic_network
    self.optimizer = optimizer
    self.session = session
    self.observation_shape = observation_shape

    self.discount_factor = 0.99
    self.createVariables()

  def createVariables(self):
    self.states = tf.placeholder(tf.float32, [None, self.observation_shape])
    self.rewards = tf.placeholder(tf.float32, [None,])
    # self.next_states = tf.placeholder(tf.float32, [None, self.observation_shape])
    self.v_next = tf.placeholder(tf.float32, [None,])

    with tf.variable_scope('critic_network', reuse=tf.AUTO_REUSE):
      self.v = self.critic_network(self.states)
      # self.v_next = self.critic_network(self.next_states)

    self.td_error = self.rewards + self.discount_factor * self.v_next - self.v
    loss = tf.reduce_mean(tf.square(self.td_error))

    self.train_op = self.optimizer.minimize(loss, var_list=tf.trainable_variables(scope='critic_network'))

  def learn(self, state, reward, next_state):
    v_next = self.session.run(self.v, {self.states: [next_state]}).squeeze()
    td_error, _ = self.session.run([self.td_error, self.train_op], {
        self.states: [state],
        self.rewards: [reward],
        # self.next_states: [next_state]
        self.v_next: [v_next]
      })
    return td_error.squeeze()