import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import tflearn
from multiprocessing import Process, Queue
# from Queue import Empty
from collections import deque
from policy_gradient.ddpg2_distributed import Agent
from policy_gradient.noise import OrnsteinUhlenbeckActionNoise
from policy_gradient.memory import SequentialMemory
import time
from flask import Flask, abort, request 
import json
import psutil, os
# make this process high priority?
from util.timer import timer

tf.app.flags.DEFINE_string('checkpoint',  '', 'load a checkpoint file for model')
tf.app.flags.DEFINE_string('save_checkpoint_dir', './models/ddpg2_dota/', 'dir for storing checkpoints')
tf.app.flags.DEFINE_string('summaries_dir', './summaries/ddpg2_dota/', 'dir for storing summaries')
tf.app.flags.DEFINE_boolean('dont_save', False, 'whether to save checkpoints')
tf.app.flags.DEFINE_boolean('only_critic', False, 'whether to train only critic')
tf.app.flags.DEFINE_boolean('render', False, 'render of not')
tf.app.flags.DEFINE_boolean('train', True, 'train or not')
tf.app.flags.DEFINE_integer('seed', 0, 'seed for tf and numpy')
tf.app.flags.DEFINE_float('actor_lr', 0.0001, 'learning rate for actor')
tf.app.flags.DEFINE_float('critic_lr', 0.001, 'learning rate for critic')
tf.app.flags.DEFINE_float('tau', 0.001, 'tau')
FLAGS = tf.app.flags.FLAGS

ps_spec = ['localhost:2220']; worker_spec = ['localhost:2221', 'localhost:2222']
cluster = tf.train.ClusterSpec({'worker': worker_spec, 'ps': ps_spec})

# seconds after which tensorflow computation will be timed out and following exp considered invalid
DISCARD_TIMEOUT = 0.2/3.0
# globals required by flask.getAction....
f_prevState = None
f_prevAction = None
f_episode_history = deque(maxlen=100)
f_cum_reward = 0
f_episode = 0
f_t = 0
f_maxTime = 0
f_shouldDiscardExperience = False


def queue_get_all(q, maxItemsToRetreive=10):
  items = []
  for numOfItemsRetrieved in range(0, maxItemsToRetreive):
    try:
      if numOfItemsRetrieved == maxItemsToRetreive:
        break
      items.append(q.get_nowait())
    except:
      break
  return items
  
count = 0

def ps_process():
  if os.name == 'nt':
    p = psutil.Process(os.getpid())
    p.nice(psutil.REALTIME_PRIORITY_CLASS)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)
  server = tf.train.Server(cluster, 
    job_name='ps', 
    task_index=0, 
    config=tf.ConfigProto(gpu_options=gpu_options))
  server.join()

def worker_process(isTrainer, q):
  if os.name == 'nt':
    p = psutil.Process(os.getpid())
    p.nice(psutil.REALTIME_PRIORITY_CLASS)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
  server = tf.train.Server(cluster, 
    job_name='worker', 
    task_index=isTrainer, 
    config=tf.ConfigProto(gpu_options=gpu_options))

  np.random.seed(FLAGS.seed)
  tf.set_random_seed(FLAGS.seed)

  # env = gym.make('Pendulum-v0')
  worker_device = '/job:worker/task:{}/gpu:0'.format(isTrainer)

  with tf.device(tf.train.replica_device_setter(cluster=cluster, ps_device='/job:ps/task:0/cpu:0', worker_device=worker_device)):
    global_step = tf.train.get_or_create_global_step()

    actor_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.actor_lr)
    critic_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.critic_lr)

    # observation_shape = env.observation_space.shape[0]
    # action_shape = env.action_space.shape[0]
    observation_shape = 12 # x, y of 4 creeps and hero + sin(t), cos(t) of hero angle
    action_shape = 4 # x, y of target location, and 2 params for deciding to move or not
    action_scale = [200., 200.] # limit x, y to 200 units ?
    # ACTION_SCALE_MAX = [2.0]
    # ACTION_SCALE_MIN = [-2.0]
    # ACTION_SCALE_VALID = [True]
    BATCH_SIZE = 64


    # ddpg actor output is always in [-1, 1], need to scale action when applying it!
    def actor_network(states):
      with tf.variable_scope('actor', reuse=tf.AUTO_REUSE):
        net = slim.stack(states, slim.fully_connected, [400, 300], activation_fn=tf.nn.relu, scope='stack')
        net = slim.fully_connected(net, action_shape, activation_fn=tf.nn.tanh, scope='full', weights_initializer=tf.random_uniform_initializer(-3e-4, 3e-4))
        # mult with action bounds
        # net = action_scale * net
        return net

    def critic_network(states, actions):
      with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
        state_net = slim.stack(states, slim.fully_connected, [400], activation_fn=tf.nn.relu, scope='stack_state')
        net = tf.concat([state_net, actions], 1)
        net = slim.fully_connected(net, 300, activation_fn=tf.nn.relu, scope='full')
        net = slim.fully_connected(net, 1, activation_fn=None, scope='last', weights_initializer=tf.random_uniform_initializer(-3e-4, 3e-4))
        net = tf.squeeze(net, axis=[1])
        return net

    agent = Agent(global_step, actor_network, critic_network, actor_optimizer, critic_optimizer, observation_shape, action_shape, tau=FLAGS.tau)
  actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_shape), sigma=0.2)

  MAX_EPISODES = 10000
  MAX_STEPS    = 1000

  with tf.train.MonitoredTrainingSession(master=server.target, 
      is_chief=True, 
      checkpoint_dir=FLAGS.save_checkpoint_dir,
      save_checkpoint_secs=3600) as sess:

    if isTrainer:
      agent.hard_update(sess)
      critic_losses = []
      critic_values = []
      max_critic_values = []
      actor_diffs, critic_diffs = [], []
      memory = SequentialMemory(limit=1000000, window_length=1)

      while True:
        # get some experience data
        items = queue_get_all(q, 15)
        # print ('{}\t | got {} items'.format(worker_device, len(items)))
        for (state, action, reward, done) in items:
          memory.append(state, action, reward, done)

        if memory.nb_entries > BATCH_SIZE and FLAGS.train:
          states, actions, rewards, next_states, terminals = memory.sample_and_split(BATCH_SIZE)
          rewards, terminals = [np.squeeze(x) for x in [rewards, terminals]]
          if not FLAGS.only_critic:
            _, _, qloss, qvals = agent.train(sess, states, actions, rewards, next_states, terminals)
          else:
            _, qloss, qvals = agent.train_critic_only(sess, states, actions, rewards, next_states, terminals)
          critic_losses.append(qloss)
          critic_values.append(np.mean(qvals))
          max_critic_values.append(np.amax(qvals))
          _, actor_diff, critic_diff = agent.soft_update(sess)
          actor_diffs.append(actor_diff)
          critic_diffs.append(critic_diff)
        # if done:
        #   print('{}\t | episode {}/{}: avg qloss = {:.5f}, avg qvals = {:.5f}, avg maxQ = {:.5f}, actor_diff = {:.5f}, critic_diff = {:.5f}'.format(
        #     worker_device,
        #     e, 
        #     MAX_EPISODES, 
        #     np.mean(critic_losses), 
        #     np.mean(critic_values),
        #     np.mean(max_critic_values),
        #     np.mean(actor_diffs),
        #     np.mean(critic_diffs)))
    else:
      # run a cold start computation
      print(agent.sample_action(sess, np.array([0.] * observation_shape)))
      train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)

      app = Flask('my_flask')
      @app.route('/', methods=['POST'])
      def getAction():
        startTime = timer()
        global f_prevState, f_prevAction, f_episode_history, f_cum_reward, f_episode, f_t, f_maxTime, f_shouldDiscardExperience
        # increment total number of steps
        f_t += 1
        data = request.get_json()
        state = data['state']
        t = data['t']
        ep = data['ep']
        # print(len(state))
        noise = actor_noise() if FLAGS.train else np.zeros(action_shape)
        actionWithoutNoise = agent.sample_action(sess, np.array(state))
        action = actionWithoutNoise + noise
        endComputationTime = timer()
        prevReward = data['prevReward']
        prevTerminal = data['prevTerminal']
        if endComputationTime - startTime > DISCARD_TIMEOUT:
          # discard all following experience
          f_shouldDiscardExperience = True
        # discard experience that comes after a lag spike
        if f_prevState is not None and f_prevAction is not None and not f_shouldDiscardExperience:
          q.put((f_prevState, f_prevAction, prevReward, prevTerminal))
        f_prevState = state
        f_prevAction = action
        f_cum_reward += prevReward
        f_maxTime = max(f_maxTime, endComputationTime - startTime)
        # print(data)
        print('{}| {}, reward = {}, terminal = {}, took {:.2f} sec, max is {:.2f} sec, noise = ({:.2f}, {:.2f}), discarded = {}'.format(
          ep, 
          t, 
          prevReward, 
          prevTerminal, 
          (endComputationTime - startTime), 
          f_maxTime,
          noise[0],
          noise[1],
          f_shouldDiscardExperience
          ))
        if prevTerminal:
          # start using experience again from next episode
          f_shouldDiscardExperience = False
          # reset noise!
          actor_noise.reset()
          f_episode_history.append(f_cum_reward)
          print('{}\t | episode: {}|{} score: {}, avg score for 100 runs: {:.2f}'.format(worker_device, f_episode, data['ep'], f_cum_reward, np.mean(f_episode_history)))
          # add summaries
          def getScalarSummary(tag, value):
            return tf.Summary.Value(tag=tag, simple_value=value)
          summary = tf.Summary(value=[getScalarSummary('return', f_cum_reward), getScalarSummary('timestamp', endComputationTime), getScalarSummary('episode', ep)])
          train_writer.add_summary(summary, f_t)
          f_episode += 1
          f_cum_reward = 0
        # scale action here
        if action[2] < action[3]:
          action = np.zeros([4])
        action = action[0:2]
        scaledAction = action_scale * action
        return json.dumps({'action': {
            'x': scaledAction[0],
            'y': scaledAction[1]
          }})
      app.run(debug=False)
      # while True:
      #   time.sleep(1)
      # episode_history = deque(maxlen=100)
      # for e in range(MAX_EPISODES):
      #   cum_reward = 0
      #   state = env.reset()
      #   for j in range(MAX_STEPS):
      #     if FLAGS.render:
      #       env.render()
      #     noise = actor_noise() if FLAGS.train else 0
      #     action = agent.sample_action(sess, state) + noise
      #     next_state, reward, done, _ = env.step(action)
      #     cum_reward += reward
      #     q.put((state, action, reward, done))
      #     state = next_state
      #     if done:
      #       episode_history.append(cum_reward)
      #       print('{}\t |episode {}/{}: score = {}, avg score for 100 runs = {:.2f}, '.format(
      #         worker_device,
      #         e,
      #         MAX_EPISODES,
      #         cum_reward,
      #         np.mean(episode_history)
      #         ))
      #       break

# def server(serverQ):
#   app = Flask(__name__)
#   @app.route("/")
#   def hello():
#     global count
#     count = count + 1
#     serverQ.put(count)
#     # time.sleep(1)
#     return json.dumps("Hello World! {}".format(count))
#   app.run(debug=True)


def main(_):
  q = Queue()
  ps_process0 = Process(target=ps_process, args=())
  ps_process0.start()
  worker_process0 = Process(target=worker_process, args=(0, q))
  worker_process0.start()
  worker_process1 = Process(target=worker_process, args=(1, q))
  worker_process1.start()

if __name__ == '__main__':
  tf.app.run()


