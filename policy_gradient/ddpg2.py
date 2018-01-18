import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
from collections import deque

DISCOUNT_FACTOR = 0.99


class Agent:
  def __init__(self,
               actor_network,
               critic_network,
               actor_optimizer,
               critic_optimizer,
               session,
               observation_shape,
               action_shape,
               tau=1e-3):
    self.session = session
    with tf.name_scope('agent'):
      # all placeholders
      self.states = tf.placeholder(tf.float32, [None, observation_shape], name='states')
      self.actions = tf.placeholder(tf.float32, [None, action_shape], name='actions')
      self.rewards = tf.placeholder(tf.float32, [None], name='rewards')
      self.next_states = tf.placeholder(tf.float32, [None, observation_shape], name='next_states')
      self.terminals = tf.placeholder(tf.float32, [None], name='terminals')

      # make the network
      with tf.variable_scope('actor_net'):
        self.actor = actor_network(self.states)
      with tf.variable_scope('actor_target_net'):
        self.actor_target = actor_network(self.next_states)

      self.next_actions = self.actor_target

      with tf.variable_scope('critic_net') as critic_net_scope:
        self.critic = critic_network(self.states, self.actions)
      with tf.variable_scope('critic_target_net'):
        self.critic_target = critic_network(self.next_states, self.next_actions)

      with tf.variable_scope('target_q_values'):
        self.target_q_values = self.rewards + DISCOUNT_FACTOR * self.terminals * self.critic_target

      # critic loss
      self.critic_loss = tf.losses.mean_squared_error(self.target_q_values, self.critic)
      self.critic_train_op = critic_optimizer.minimize(self.critic_loss, var_list=tf.trainable_variables(scope='critic_net'), name='critic_train_op')

      # action gradients
      # del-a Q(s,a) | a = mu(s)
      with tf.variable_scope('critic_net', reuse=tf.AUTO_REUSE):
        self.q_vals = critic_network(self.states, self.actor)
      self.action_gradients = tf.gradients(self.q_vals, self.actor, name='action_gradients')[0]
      # neg sign because apply_gradients applies negative too
      self.policy_gradient = tf.gradients(self.actor, tf.trainable_variables(scope='actor_net'), -self.action_gradients, name='policy_gradient') 

      # use policy loss = -mean(critic(s, mu(s)))
      # this is equivalent to multiply action gradients approach
      self.actor_train_op = actor_optimizer.minimize(-tf.reduce_mean(self.q_vals), var_list=tf.trainable_variables(scope='actor_net'), name='actor_train_op')
      # self.actor_train_op = actor_optimizer.apply_gradients(zip(self.policy_gradient, tf.trainable_variables(scope='actor_net')), name='actor_train_op')

      self.variables = tf.trainable_variables(scope='critic_net') + tf.trainable_variables(scope='actor_net')
      self.target_variables = tf.trainable_variables(scope='critic_target_net') + tf.trainable_variables(scope='actor_target_net')

      with tf.variable_scope('target_update_ops'):
        self.soft_update_ = [target_variable.assign(variable * tau + target_variable * (1. - tau))
            for (variable, target_variable) in zip(self.variables, self.target_variables) ]

        self.hard_update_ = [target_variable.assign(variable) for (variable, target_variable) in zip(self.variables, self.target_variables)]


  def hard_update(self):
    return self.session.run(self.hard_update_)

  def soft_update(self):
    return self.session.run(self.soft_update_)

  def train(self, states, actions, rewards, next_states, terminals):
    return self.session.run([self.critic_train_op, self.actor_train_op, self.critic_loss, self.critic], feed_dict={
        self.states: states,
        self.actions: actions,
        self.rewards: rewards,
        self.next_states: next_states,
        self.terminals: terminals
      })

  def sample_action(self, state):
    return self.session.run(self.actor, feed_dict={self.states: [state]})[0]