import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

tf.set_random_seed(0)
np.random.seed(0)

def test1():
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

# test1()

def test2():
  def getNet(x, initW, initB):
    w = slim.variable('w', initializer=tf.constant(initW))
    b = slim.variable('b', initializer=tf.constant(initB))
    y = w * x * x + b
    return w, b, y
  x = tf.placeholder(tf.float32, None)
  with tf.variable_scope('untrained'):
    w1, b1, y1 = getNet(x, 5., 7.)
  with tf.variable_scope('to_train'):
    w2, b2, y2 = getNet(x, 0., 0.)

  y2_grad = tf.gradients(y2, x)
  # y_ = tf.placeholder(tf.float32, None)
  loss = tf.losses.mean_squared_error(y1, y2)
  optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
  train_op = optimizer.minimize(loss, var_list=tf.trainable_variables(scope='to_train'))

  sess = tf.Session()
  tf.global_variables_initializer().run(session=sess)
  for _ in range(1000):
    x1 = np.random.choice(10, 10).astype(float)
    # target = sess.run(y1, {x: x1})
    w1_, b1_, w2_, b2_, y2_, loss_, y2_grad_, _ = sess.run([w1, b1, w2, b2, y2, loss, y2_grad, train_op], feed_dict={
      x: x1
      # y_: target
      })
    print w1_, b1_, w2_, b2_, loss_, 2 * x1 * w2_ - y2_grad_

test2()