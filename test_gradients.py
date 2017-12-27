import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

tf.set_random_seed(0)
np.random.seed(0)


w = tf.get_variable('w', [5], tf.float32)
x = tf.placeholder(tf.float32, [None, 5])
w2 = tf.transpose(tf.stack([w,w,w]))
y = tf.matmul(x, w2)

grads = tf.gradients(y, w)


sess = tf.Session()
tf.global_variables_initializer().run(session=sess)


print sess.run(grads, {x: [[1,2,3,4,5], [6,7,8,9,10]]})