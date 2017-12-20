import tensorflow as tf
import numpy as np

target_dist = tf.distributions.Normal(loc=[3.,2.], scale=[.5, 1.0])

mu = tf.get_variable('mu', initializer=tf.constant([0.0, 0.0]))
logstd = tf.get_variable('logstd', initializer=tf.constant([1., 1.]))
std = tf.exp(logstd)

fun = tf.distributions.Normal(loc=mu, scale=std)

optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1, decay=0.9)

xs = tf.placeholder(tf.float32, [None, 2])
ys = fun.prob(xs)
y_s = tf.placeholder(tf.float32, [None, 2])

loss = tf.losses.mean_squared_error(ys, y_s)
train_op = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# sess.run([mu.initializer, logstd.initializer])
# print sess.run([mu, std])

for x in range(100):
  x = sess.run(target_dist.sample(100))
  y_ = sess.run(target_dist.prob(x))
  # print x, y_
  # mean, deviation = sess.run([mu, std])
  mean, deviation, _ = sess.run([mu, std, train_op], {xs: x, y_s: y_})
  print mean, deviation
