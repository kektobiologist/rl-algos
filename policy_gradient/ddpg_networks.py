import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import tflearn

def actor_network_tflearn(states, action_shape, action_bounds):
  with tf.variable_scope('actor'):
    net = tflearn.fully_connected(states, 400)
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.relu(net)
    net = tflearn.fully_connected(net, 300)
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.relu(net)
    # Final layer weights are init to Uniform[-3e-3, 3e-3]
    w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
    out = tflearn.fully_connected(
        net, action_shape, activation='tanh', weights_init=w_init)
    # Scale output to -action_bound to action_bound
    scaled_out = tf.multiply(out, action_bounds)
    return scaled_out

def critic_network_tflearn(states, actions, action_shape):
  with tf.variable_scope('critic'):
    net = tflearn.fully_connected(states, 400)
    # net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.relu(net)

    # Add the action tensor in the 2nd hidden layer
    # Use two temp layers to get the corresponding weights and biases
    t1 = tflearn.fully_connected(net, 300)
    t2 = tflearn.fully_connected(actions, 300)
    print t1.W, t2.W
    net = tflearn.activation(
        tf.matmul(net, t1.W) + tf.matmul(actions, t2.W) + t2.b, activation='relu')

    # linear layer connected to 1 output representing Q(s,a)
    # Weights are init to Uniform[-3e-3, 3e-3]
    w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
    out = tflearn.fully_connected(net, 1, weights_init=w_init)
    # out = tflearn.fully_connected(net, 1)
    out = tf.squeeze(out, axis=[1])
    return out


def actor_network(states, action_shape, action_bounds):
  net = slim.stack(states, slim.fully_connected, [100, 100], activation_fn=tf.nn.relu, scope='stack')
  net = slim.fully_connected(net, action_shape, activation_fn=tf.nn.tanh, scope='full')
  # mult with action bounds
  net = action_bounds * net
  return net

def critic_network(states, actions, action_shape):
  with tf.variable_scope('critic'):
    # state_net = tflearn.fully_connected(states, 300, activation='relu', scope='full_state')
    # action_net = tflearn.fully_connected(actions, 300, activation='relu', scope='full_action')
    state_net = slim.stack(states, slim.fully_connected, [300], activation_fn=tf.nn.relu, scope='stack_state')
    action_net = slim.stack(actions, slim.fully_connected, [300], activation_fn=tf.nn.relu, scope='stack_action')
    # net = tf.contrib.layers.fully_connected(states, 400, scope='full_state')
    # net = tflearn.fully_connected(states, 400)
    # net = tflearn.layers.normalization.batch_normalization(net)
    # net = tflearn.activations.relu(net)
    net = tf.concat([state_net, action_net], 1)
    # net = tf.contrib.layers.fully_connected(net, 400)
    net = slim.fully_connected(net, 400, activation_fn=tf.nn.relu, scope='full')
    # w1 = tf.get_variable('w1', shape=[400, 300], dtype=tf.float32)
    # w2 = tf.get_variable('w2', shape=[1, 300], dtype=tf.float32)
    # b = tf.get_variable('b', shape=[300], dtype=tf.float32)
    # t1 = tflearn.fully_connected(net, 300)
    # t2 = tflearn.fully_connected(actions, 300)
    # print t1.W, t2.W
    # net = tflearn.activation(
    #     tf.matmul(net, t1.W) + tf.matmul(actions, t2.W) + t2.b, activation='relu')

    # net = tf.matmul(net, w1) + tf.matmul(actions, w2) + b
    # net = tf.nn.relu(net)
    # net = slim.stack(net, slim.fully_connected, [5], activation_fn=tf.nn.relu, scope='stack')
    # net = slim.fully_connected(net, 1, activation_fn=tf.nn.relu, scope='full')
    # net = tf.contrib.layers.fully_connected(net, 1, scope='last')
    # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
    # net = slim.stack(net, slim.fully_connected, [24, 1], scope='final', biases_initializer=tf.zeros_initializer())
    # net = tf.layers.dense(net, 1, activation=tf.nn.relu, use_bias=True, name='last')
    net = tflearn.fully_connected(net, 1)
    net = tf.squeeze(net, axis=[1])
    return net