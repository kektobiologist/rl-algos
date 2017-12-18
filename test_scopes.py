import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def getNet(x):
  net = slim.stack(x, slim.fully_connected, [24, 24], activation_fn=tf.nn.tanh, scope='full')
  net = slim.fully_connected(net, 2, activation_fn=None, scope='last')
  return net

x1 = tf.placeholder(tf.float32, [None, 1])
x2 = tf.placeholder(tf.float32, [None, 1])

with tf.variable_scope('sc'):
  y1 = getNet(x1)
# tf.get_variable_scope().reuse_variables()
with tf.variable_scope('sc', reuse=True):
  y2 = getNet(x2)

y_ = tf.placeholder(tf.float32, [None, 2])
loss = tf.losses.mean_squared_error(y_, y1)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(loss)

sess = tf.Session()
tf.global_variables_initializer().run(session=sess)

print sess.run([y1, y2], {x1: [[1]], x2: [[1]]})

sess.run(train_op, {x1: [[1]], y_: [[1,2]]})

print sess.run([y1, y2], {x1: [[1]], x2: [[1]]})
