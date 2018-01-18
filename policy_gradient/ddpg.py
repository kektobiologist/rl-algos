import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
from collections import deque

DISCOUNT_FACTOR = 0.99

class Actor:
  def __init__(self,
               actor_network,
               optimizer,
               session,
               observation_shape,
               action_shape,
               tau=1e-3):
    self.actor_network = actor_network
    self.optimizer = optimizer
    self.session = session
    self.observation_shape = observation_shape
    self.action_shape = action_shape

    self.discount_factor = DISCOUNT_FACTOR
    self.tau = tau
    self.createVariables()


  def createVariables(self):
    with tf.name_scope('actor_scope'):
      # actor network: states => action
      self.states = tf.placeholder(tf.float32, [None, self.observation_shape], 'states')

      with tf.variable_scope('actor_network'):
        self.actor_outputs = self.actor_network(self.states)
      with tf.variable_scope('target_actor_network'):
        self.target_actor_outputs = self.actor_network(self.states)

      actor_variables = tf.trainable_variables(scope='actor_network')
      target_actor_variables = tf.trainable_variables(scope='target_actor_network')

      # del-a Q(s,a) | a = mu(s)
      self.action_gradients = tf.placeholder(tf.float32, [None, self.action_shape], 'action_gradients')
      # del-theta mu(s) * del-a Q(s,a) | a = mu(s)
      # minus sign because apply_gradients negates the gradient before applying
      self.unnormalized_actor_gradients = tf.gradients(self.actor_outputs, actor_variables, -self.action_gradients, name='unnormalized_actor_gradients')
      # removing normalization of actor gradients
      # batch_size = tf.cast(tf.shape(self.states)[0], tf.float32)
      # self.actor_gradients = [tf.div(gradient, batch_size) for gradient in self.unnormalized_actor_gradients]

      # use unnormalized gradients
      self.train_op = self.optimizer.apply_gradients(zip(self.unnormalized_actor_gradients, actor_variables), name='actor_train_op')

      # copy variables op
      with tf.variable_scope('actor_target_update'):
        self.update_target_variables_op = [target_variable.assign(actor_variable * self.tau + target_variable * (1. - self.tau))
          for (target_variable, actor_variable) in zip(target_actor_variables, actor_variables) ]

        self.hard_update_target_variables_op = [target_variable.assign(actor_variable) 
          for (target_variable, actor_variable) in zip(target_actor_variables, actor_variables)]


  def predict(self, states):
    return self.session.run(self.actor_outputs, {self.states: states})

  def predict_target(self, states):
    return self.session.run(self.target_actor_outputs, {self.states: states})

  def train(self, states, action_gradients):
    self.session.run(self.train_op, {self.states: states, self.action_gradients: action_gradients})

  def update_target(self):
    self.session.run(self.update_target_variables_op)

  def hard_update(self):
    self.session.run(self.hard_update_target_variables_op)

class Critic:
  def __init__(self,
               critic_network,
               optimizer,
               session,
               observation_shape,
               action_shape,
               tau=1e-3):
    self.critic_network = critic_network
    self.optimizer = optimizer
    self.session = session
    self.observation_shape = observation_shape
    self.action_shape = action_shape

    self.discount_factor = DISCOUNT_FACTOR
    self.tau = tau
    self.createVariables()

  def createVariables(self):
    # critic network: states, actions => action values
    with tf.name_scope('critic_scope'):
      self.states = tf.placeholder(tf.float32, [None, self.observation_shape], name='states')
      self.actions = tf.placeholder(tf.float32, [None, self.action_shape], name='actions')
      self.next_states = tf.placeholder(tf.float32, [None, self.observation_shape], name='next_states')
      self.next_actions = tf.placeholder(tf.float32, [None, self.action_shape], name='next_actions')
      self.rewards = tf.placeholder(tf.float32, [None,], name='rewards')
      self.notdones = tf.placeholder(tf.float32, [None,], name='notdones')

      with tf.variable_scope('critic_network'):
        self.critic_outputs = self.critic_network(self.states, self.actions)
      with tf.variable_scope('target_critic_network'):
        self.target_critic_outputs = self.critic_network(self.next_states, self.next_actions)

      critic_variables = tf.trainable_variables(scope='critic_network')
      target_critic_variables = tf.trainable_variables(scope='target_critic_network')

      with tf.variable_scope('target_qs'):
        targets = self.rewards + self.discount_factor * self.notdones * self.target_critic_outputs
      self.loss = tf.losses.mean_squared_error(targets, self.critic_outputs)

      # self.train_op = slim.learning.create_train_op(loss, self.optimizer, var_list=critic_variables)
      # use normal train op
      self.train_op = self.optimizer.minimize(self.loss, var_list=critic_variables, name='critic_train_op')

      with tf.variable_scope('critic_target_update'):
        # copy variables op
        self.update_target_variables_op = [target_variable.assign(critic_variable * self.tau + target_variable * (1. - self.tau))
          for (target_variable, critic_variable) in zip(target_critic_variables, critic_variables) ]

        self.hard_update_target_variables_op = [target_variable.assign(critic_variable) 
          for (target_variable, critic_variable) in zip(target_critic_variables, critic_variables)]
        # action gradients op, why is it [1, batch_size, 1]?
      self.action_gradients = tf.gradients(self.critic_outputs, self.actions, name='action_gradients')[0]

  def train(self, states, actions, rewards, next_states, next_actions, notdones):
    return self.session.run([self.critic_outputs, self.loss, self.train_op], feed_dict={
      self.states: states,
      self.actions: actions,
      self.rewards: rewards,
      self.next_states: next_states,
      self.next_actions: next_actions,
      self.notdones: notdones
      })

  def update_target(self):
    self.session.run(self.update_target_variables_op)

  def hard_update(self):
    self.session.run(self.hard_update_target_variables_op)

  def get_action_gradients(self, states, actions):
    return self.session.run(self.action_gradients, feed_dict={self.states: states, self.actions: actions})


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
  def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
    self.theta = theta
    self.mu = mu
    self.sigma = sigma
    self.dt = dt
    self.x0 = x0
    self.reset()

  def __call__(self):
    x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
         self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
    self.x_prev = x
    return x

  def reset(self):
    self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

  def __repr__(self):
    return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
