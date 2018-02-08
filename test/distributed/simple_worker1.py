"""How to run:
python main.py --job_name 'ps' --task_index 0 &
python main.py --job_name 'worker' --task_index 0 &
python main.py --job_name 'worker' --task_index 1
"""

# https://stackoverflow.com/questions/42986653/distributed-tensorflow-not-running-some-workers
# this causes worker 1 to start with delay of 30 seconds

import tensorflow as tf
import time
flags = tf.app.flags
flags.DEFINE_string('job_name', 'ps', 'ps or worker')
flags.DEFINE_integer('task_index', 0, 'task index')
FLAGS = flags.FLAGS

ps = ['localhost:2220']; worker = ['localhost:2221', 'localhost:2222']

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=(0.0001 if FLAGS.job_name == 'ps' else 0.3))

cluster = tf.train.ClusterSpec({'worker': worker, 'ps': ps})

server = tf.train.Server(cluster, 
  job_name=FLAGS.job_name, 
  task_index=FLAGS.task_index, 
  config=tf.ConfigProto(gpu_options=gpu_options))

if FLAGS.job_name == 'ps':
  server.join()

elif FLAGS.job_name == 'worker':
  worker_device = '/job:worker/task:{}/gpu:0'.format(FLAGS.task_index)
  with tf.device(tf.train.replica_device_setter(cluster=cluster, ps_device='/job:ps/task:0/cpu:0', worker_device=worker_device)) as dev:
    print dev
    W = tf.Variable(2.0, dtype=tf.float32, name='W')
    b = tf.Variable(5.0, dtype=tf.float32, name='b')

  # use worker 0 as the variable server
  # with tf.device(worker_device):
    inp = tf.placeholder(tf.float32, [None], name='input')
    output = W * inp + b
    y = tf.placeholder(tf.float32, [None], name='y')
    loss = tf.reduce_mean(tf.square(y - output))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss)

  # with tf.device(worker_device):
  #   with tf.variable_scope('local'):
  #     assign_op = global_step.assign(global_step + 1)


  with tf.train.MonitoredTrainingSession(master=server.target, is_chief=(FLAGS.task_index == 0)) as mon_sess:
    while True:
      if FLAGS.task_index == 0:
        _, loss_ = mon_sess.run([train_op, loss], feed_dict={inp: [1,2,3], y: [5,7,9]})
      else:
        loss_ = mon_sess.run( loss, feed_dict={inp: [1,2,3], y: [5,7,9]})
      print ('{}\t| {}'.format(worker_device, loss_))
      time.sleep(1)



  # with tf.Session(server.target) as sess:
  #   step = 0
  #   sess.run(init_op)
  #   while step < 10000:
  #     _, step = sess.run([assign_op, global_step])
  #     print('{}\t| {}'.format(worker_device, step))