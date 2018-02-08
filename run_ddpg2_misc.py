import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import tflearn

from collections import deque
from policy_gradient.ddpg2 import Agent
from policy_gradient.noise import OrnsteinUhlenbeckActionNoise
from policy_gradient.memory import SequentialMemory

tf.app.flags.DEFINE_string('checkpoint',  '', 'load a checkpoint file for model')
tf.app.flags.DEFINE_string('save_checkpoint_dir', './models/ddpg2_misc/', 'dir for storing checkpoints')
tf.app.flags.DEFINE_boolean('dont_save', False, 'whether to save checkpoints')
tf.app.flags.DEFINE_boolean('only_critic', False, 'whether to train only critic')
tf.app.flags.DEFINE_boolean('render', False, 'render of not')
tf.app.flags.DEFINE_boolean('train', True, 'train or not')
tf.app.flags.DEFINE_integer('seed', 0, 'seed for tf and numpy')
tf.app.flags.DEFINE_float('actor_lr', 0.0001, 'learning rate for actor')
tf.app.flags.DEFINE_float('critic_lr', 0.001, 'learning rate for critic')
tf.app.flags.DEFINE_float('sigma', 0.2, 'sigma for Ornstein Uhlenbeck noise')
tf.app.flags.DEFINE_float('tau', 0.001, 'tau')
tf.app.flags.DEFINE_string('env', 'Pendulum-v0', 'gym environment')
FLAGS = tf.app.flags.FLAGS

np.random.seed(FLAGS.seed)
tf.set_random_seed(FLAGS.seed)

env = gym.make(FLAGS.env)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session()

actor_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.actor_lr)
critic_optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.critic_lr)

observation_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]
action_scale = env.action_space.high

print action_scale, action_shape, observation_shape
# ACTION_SCALE_MAX = [2.0]
# ACTION_SCALE_MIN = [-2.0]
# ACTION_SCALE_VALID = [True]
BATCH_SIZE = 64


def actor_network(states):
  with tf.variable_scope('actor', reuse=tf.AUTO_REUSE):
    net = slim.stack(states, slim.fully_connected, [400, 300], activation_fn=tf.nn.relu, scope='stack')
    net = slim.fully_connected(net, action_shape, activation_fn=tf.nn.tanh, scope='full', weights_initializer=tf.random_uniform_initializer(-3e-4, 3e-4))
    # mult with action bounds
    net = action_scale * net
    return net

def critic_network(states, actions):
  with tf.variable_scope('critic', reuse=tf.AUTO_REUSE):
    state_net = slim.stack(states, slim.fully_connected, [400], activation_fn=tf.nn.relu, scope='stack_state')
    net = tf.concat([state_net, actions], 1)
    net = slim.fully_connected(net, 300, activation_fn=tf.nn.relu, scope='full')
    net = slim.fully_connected(net, 1, activation_fn=None, scope='last', weights_initializer=tf.random_uniform_initializer(-3e-4, 3e-4))
    net = tf.squeeze(net, axis=[1])
    return net

def main(_):
  agent = Agent(actor_network, critic_network, actor_optimizer, critic_optimizer, sess, observation_shape, action_shape, tau=FLAGS.tau)
  actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_shape), sigma=FLAGS.sigma)
  writer = tf.summary.FileWriter("logs/ddpg2_misc", sess.graph)

  MAX_EPISODES = 10000
  MAX_STEPS    = 1000

  saver = tf.train.Saver()
  if FLAGS.checkpoint:
    saver.restore(sess, FLAGS.checkpoint)
  else:
    sess.run(tf.global_variables_initializer())

  agent.hard_update()
  episode_history = deque(maxlen=100)
  critic_losses = []
  critic_values = []
  max_critic_values = []
  actor_diffs, critic_diffs = [], []
  memory = SequentialMemory(limit=1000000, window_length=1)
  for e in range(MAX_EPISODES):
    cum_reward = 0
    state = env.reset()
    for j in range(MAX_STEPS):
      if FLAGS.render:
        env.render()
      noise = actor_noise() if FLAGS.train else 0
      action = agent.sample_action(state) + noise
      next_state, reward, done, _ = env.step(action)
      # if done and j < MAX_STEPS - 1:
      #   reward = -20
      cum_reward += reward
      memory.append(state, action, reward, done)
      if memory.nb_entries > BATCH_SIZE and FLAGS.train:
        states, actions, rewards, next_states, terminals = memory.sample_and_split(BATCH_SIZE)
        rewards, terminals = [np.squeeze(x) for x in [rewards, terminals]]
        if not FLAGS.only_critic:
          _, _, qloss, qvals, _, actor_diff, critic_diff = agent.train_and_soft_update(states, actions, rewards, next_states, terminals)
        else:
          _, qloss, qvals = agent.train_critic_only(states, actions, rewards, next_states, terminals)
          _, actor_diff, critic_diff = agent.soft_update()
        critic_losses.append(qloss)
        critic_values.append(np.mean(qvals))
        max_critic_values.append(np.amax(qvals))
        actor_diffs.append(actor_diff)
        critic_diffs.append(critic_diff)
      if done:
        episode_history.append(cum_reward)
        print('episode {}/{}: score = {}, avg score for 100 runs = {:.2f}, avg qloss = {:.5f}, avg qvals = {:.5f}, avg maxQ = {:.5f}, actor_diff = {:.5f}, critic_diff = {:.5f}'.format(
          e, 
          MAX_EPISODES, 
          cum_reward, 
          np.mean(episode_history), 
          np.mean(critic_losses), 
          np.mean(critic_values),
          np.mean(max_critic_values),
          np.mean(actor_diffs),
          np.mean(critic_diffs)))
        break
      state = next_state
    if e%100 == 0 and not FLAGS.dont_save:
      save_path = saver.save(sess, FLAGS.save_checkpoint_dir + 'model-' + str(e) + '.ckpt')
      print 'saved model ' + save_path


if __name__ == '__main__':
  tf.app.run()
