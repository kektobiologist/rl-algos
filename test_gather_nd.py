import tensorflow as tf

x = tf.placeholder(tf.float32, [None, 4])
indices = tf.placeholder(tf.int32, [None, ])

range_tensor = tf.range(tf.shape(x)[0])
stacked_indices = tf.stack([range_tensor, indices], axis=1)

output = tf.gather_nd(x, stacked_indices)


sess = tf.Session()

print sess.run(output, {x: [[1,2,3,4], [5,6,7,8], [10, 11, 23, 34]], indices: [1,0,3]})