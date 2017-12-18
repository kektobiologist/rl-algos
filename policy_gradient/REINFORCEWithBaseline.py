  # REINFORCE algo for discrete action space.
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
import numpy as np


class REINFORCEWithBaseline:
  def __init__(self,
               policy_network,
               value_network,
               optimizer,
               session,
               num_actions,
               observation_shape):
    self.session = session
    self.policy_network = policy_network
    self.value_network = value_network
    self.optimizer = optimizer
    self.num_actions = num_actions
    self.observation_shape = observation_shape
    self.createVariables()
    var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.session.run(tf.variables_initializer(var_lists))

    # make sure all variables are initialized
    self.session.run(tf.assert_variables_initialized())

    self.all_rewards = []
    self.max_reward_length = 1000000

    self.discount_factor = 0.99

  def createVariables(self):
    # placeholder for state
    self.states = tf.placeholder(tf.float32, [None, self.observation_shape])
    with tf.variable_scope('policy_network'):
      policy_outputs = self.policy_network(self.states)
    with tf.variable_scope('value_network'):
      value_outputs = tf.squeeze(self.value_network(self.states), [1])
    
    self.predicted_actions = tf.multinomial(policy_outputs, 1)
    
    self.taken_actions = tf.placeholder(tf.int32, (None,))
    self.discounted_rewards = tf.placeholder(tf.float32, (None,))
    cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy_outputs, labels=self.taken_actions)
    neg_log_loss = tf.reduce_mean(cross_entropy_loss * (self.discounted_rewards - value_outputs))

    value_network_loss = tf.losses.mean_squared_error(labels=self.discounted_rewards, predictions=value_outputs)
    self.train_op = [self.optimizer.minimize(neg_log_loss, var_list=tf.trainable_variables(scope='policy_network')),
                     self.optimizer.minimize(value_network_loss, var_list=tf.trainable_variables(scope='value_network'))]

  def sampleAction(self, state):
    return self.session.run(self.predicted_actions, feed_dict={self.states: [state]}).squeeze()

  def updateModel(self, episode_data):
    N = len(episode_data)
    r = 0 # use discounted reward to approximate Q value

    # compute discounted future rewards
    discounted_rewards = np.zeros(N)
    for t in reversed(range(N)):
      state, action, reward = episode_data[t]
      # future discounted reward from now on
      r = reward + self.discount_factor * r
      discounted_rewards[t] = r

    # reduce gradient variance by normalization
    self.all_rewards += discounted_rewards.tolist()
    self.all_rewards = self.all_rewards[:self.max_reward_length]
    # don't
    # discounted_rewards -= np.mean(self.all_rewards)
    # discounted_rewards /= np.std(self.all_rewards)

    states, actions, _ = zip(*episode_data)
    # train at once
    self.session.run(self.train_op, feed_dict = {
      self.states: states,
      self.taken_actions: actions,
      self.discounted_rewards: discounted_rewards
      })