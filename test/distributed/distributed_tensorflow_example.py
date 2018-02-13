"""How to run:
python main.py --job_name 'ps' --task_index 0 &
python main.py --job_name 'worker' --task_index 0 &
python main.py --job_name 'worker' --task_index 1
"""
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('job_name', 'ps', 'ps or worker')
flags.DEFINE_integer('task_index', 0, 'task index')
FLAGS = flags.FLAGS

ps = ['localhost:2220']; worker = ['localhost:2221', 'localhost:2222']

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

cluster = tf.train.ClusterSpec({'worker': worker, 'ps': ps})
server = tf.train.Server(cluster, 
  job_name=FLAGS.job_name, 
  task_index=FLAGS.task_index, 
  config=tf.ConfigProto(gpu_options=gpu_options))

if FLAGS.job_name == 'ps':
  server.join()
elif FLAGS.job_name == 'worker':
  worker_device = '/job:worker/task:{}/gpu:0'.format(FLAGS.task_index)
  with tf.device(tf.train.replica_device_setter(ps_tasks=1, 
                                                ps_device='/job:ps/task:0/cpu:0',
                                                worker_device=worker_device)):

    with tf.variable_scope('global'):
      global_step = tf.get_variable('global_step', [], tf.int32,
          initializer=tf.constant_initializer(0, dtype=tf.int32),
          trainable=False)
      init_op = tf.global_variables_initializer()

  with tf.device(worker_device):
    with tf.variable_scope('local'):
      assign_op = global_step.assign(global_step + 1)

  with tf.Session(server.target) as sess:
    step = 0
    sess.run(init_op)
    while step < 10000:
      _, step = sess.run([assign_op, global_step])
      print('{}\t| {}'.format(worker_device, step))