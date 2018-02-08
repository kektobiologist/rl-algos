"""How to run:
python main.py --job_name 'ps' --task_index 0 &
python main.py --job_name 'worker' --task_index 0 &
python main.py --job_name 'worker' --task_index 1
"""

# https://stackoverflow.com/questions/42986653/distributed-tensorflow-not-running-some-workers
# this causes worker 1 to start with delay of 30 seconds

import tensorflow as tf
import numpy as np
import multiprocessing
import time

ps_spec = ['localhost:2220']; worker_spec = ['localhost:2221', 'localhost:2222']
cluster = tf.train.ClusterSpec({'worker': worker_spec, 'ps': ps_spec})

def worker_process(idx, q):
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
  server = tf.train.Server(cluster, 
    job_name='worker', 
    task_index=idx, 
    config=tf.ConfigProto(gpu_options=gpu_options))
  worker_device = '/job:worker/task:{}/gpu:0'.format(idx)
  with tf.device(tf.train.replica_device_setter(cluster=cluster, ps_device='/job:ps/task:0/cpu:0', worker_device=worker_device)) as dev:
    W = tf.Variable(2.0, dtype=tf.float32, name='W')
    b = tf.Variable(5.0, dtype=tf.float32, name='b')

    inp = tf.placeholder(tf.float32, [None], name='input')
    output = W * inp + b
    y = tf.placeholder(tf.float32, [None], name='y')
    loss = tf.reduce_mean(tf.square(y - output))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss)

  # making both workers chief: this will initialize graph twice?
  with tf.train.MonitoredTrainingSession(master=server.target, is_chief=True) as mon_sess:
    while True:
      if idx == 0:
        inp_, y_ = q.get()
        _, loss_ = mon_sess.run([train_op, loss], feed_dict={inp: inp_, y: y_})
        print ('{}\t| {}'.format(worker_device, loss_))
      else:
        q.put(([1,2,3], [5,7,9]))
        print('{}\t| sent data '.format(worker_device))
        time.sleep(1)

def ps_process():
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)
  server = tf.train.Server(cluster, 
    job_name='ps', 
    task_index=0, 
    config=tf.ConfigProto(gpu_options=gpu_options))
  server.join()

def main(_):
  q = multiprocessing.Queue()

  ps_process0 = multiprocessing.Process(target=ps_process, args=())
  ps_process0.start()
  worker_process0 = multiprocessing.Process(target=worker_process, args=(0, q))
  worker_process0.start()
  # sleep here so that process 1 has time to setup...
  worker_process1 = multiprocessing.Process(target=worker_process, args=(1, q))
  worker_process1.start()

if __name__ == '__main__':
  tf.app.run()

