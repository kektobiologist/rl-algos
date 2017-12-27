import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import tflearn

from collections import deque
from policy_gradient.ddpg import Actor, Critic, OrnsteinUhlenbeckActionNoise
from policy_gradient.memory import SequentialMemory

env = gym.make('Pendulum-v0')
np.random.seed(0)
tf.set_random_seed(0)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

actor_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
critic_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

observation_shape = env.observation_space.shape[0]
action_shape = env.action_space.shape[0]

ACTION_BOUNDS = [2.0]
BATCH_SIZE = 128


tf.app.flags.DEFINE_string('checkpoint',  '', 'load a checkpoint file for model')
tf.app.flags.DEFINE_string('save_checkpoint_dir', './models/ddpg_pendulum/', 'dir for storing checkpoints')
tf.app.flags.DEFINE_boolean('dont_save', False, 'whether to save checkpoints')
tf.app.flags.DEFINE_boolean('render', False, 'render of not')
FLAGS = tf.app.flags.FLAGS

def actor_network(states):
  net = slim.stack(states, slim.fully_connected, [400, 300], activation_fn=tf.nn.relu, scope='stack')
  net = slim.fully_connected(net, action_shape, activation_fn=tf.nn.tanh, scope='full')
  # mult with action bounds
  net = ACTION_BOUNDS * net
  return net

def critic_network(states, actions):
  with tf.variable_scope('critic'):
    # state_net = tflearn.fully_connected(states, 300, activation='relu', scope='full_state')
    # action_net = tflearn.fully_connected(actions, 300, activation='relu', scope='full_action')
    state_net = slim.stack(states, slim.fully_connected, [300], activation_fn=tf.nn.relu, scope='stack_state')
    # action_net = slim.stack(actions, slim.fully_connected, [300], activation_fn=tf.nn.relu, scope='stack_action')
    # net = tf.contrib.layers.fully_connected(states, 400, scope='full_state')
    # net = tflearn.fully_connected(states, 400)
    # net = tflearn.layers.normalization.batch_normalization(net)
    # net = tflearn.activations.relu(net)
    net = tf.concat([state_net, actions], 1)
    # net = tf.contrib.layers.fully_connected(net, 400)
    net = slim.fully_connected(net, 400, activation_fn=tf.nn.relu, scope='full')
    # w1 = tf.get_variable('w1', shape=[400, 300], dtype=tf.float32)
    # w2 = tf.get_variable('w2', shape=[1, 300], dtype=tf.float32)
    # b = tf.get_variable('b', shape=[300], dtype=tf.float32)
    # t1 = tflearn.fully_connected(net, 300)
    # t2 = tflearn.fully_connected(actions, 300)
    # print t1.W, t2.W
    # net = tflearn.activation(
    #     tf.matmul(net, t1.W) + tf.matmul(actions, t2.W) + t2.b, activation='relu')

    # net = tf.matmul(net, w1) + tf.matmul(actions, w2) + b
    # net = tf.nn.relu(net)
    # net = slim.stack(net, slim.fully_connected, [5], activation_fn=tf.nn.relu, scope='stack')
    # net = slim.fully_connected(net, 1, activation_fn=tf.nn.relu, scope='full')
    # net = tf.contrib.layers.fully_connected(net, 1, scope='last')
    # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
    # net = slim.stack(net, slim.fully_connected, [24, 1], scope='final', biases_initializer=tf.zeros_initializer())
    # net = tf.layers.dense(net, 1, activation=tf.nn.relu, use_bias=True, name='last')
    net = tflearn.fully_connected(net, 1)
    net = tf.squeeze(net, axis=[1])
    return net

def actor_network_tflearn(states):
  with tf.variable_scope('actor'):
    net = tflearn.fully_connected(states, 400)
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.relu(net)
    net = tflearn.fully_connected(net, 300)
    net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.relu(net)
    # Final layer weights are init to Uniform[-3e-3, 3e-3]
    w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
    out = tflearn.fully_connected(
        net, action_shape, activation='tanh', weights_init=w_init)
    # Scale output to -action_bound to action_bound
    scaled_out = tf.multiply(out, env.action_space.high)
    return scaled_out

def critic_network_tflearn(states, actions):
  with tf.variable_scope('critic'):
    net = tflearn.fully_connected(states, 400)
    # net = tflearn.layers.normalization.batch_normalization(net)
    net = tflearn.activations.relu(net)

    # Add the action tensor in the 2nd hidden layer
    # Use two temp layers to get the corresponding weights and biases
    t1 = tflearn.fully_connected(net, 300)
    t2 = tflearn.fully_connected(actions, 300)
    print t1.W, t2.W
    net = tflearn.activation(
        tf.matmul(net, t1.W) + tf.matmul(actions, t2.W) + t2.b, activation='relu')

    # linear layer connected to 1 output representing Q(s,a)
    # Weights are init to Uniform[-3e-3, 3e-3]
    w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
    out = tflearn.fully_connected(net, 1, weights_init=w_init)
    # out = tflearn.fully_connected(net, 1)
    out = tf.squeeze(out, axis=[1])
    return out


def main(_):
  actor = Actor(actor_network, actor_optimizer, sess, observation_shape, action_shape)
  critic = Critic(critic_network, critic_optimizer, sess, observation_shape, action_shape)
  actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_shape))

  MAX_EPISODES = 10000
  MAX_STEPS    = 1000

  saver = tf.train.Saver()
  if FLAGS.checkpoint:
    saver.restore(sess, FLAGS.checkpoint)
  else:
    sess.run(tf.global_variables_initializer())

  # hard update target networks
  actor.hard_update()
  critic.hard_update()

  episode_history = deque(maxlen=100)
  memory = SequentialMemory(limit=600000, window_length=1)

  tot_rewards = deque(maxlen=10000)
  numsteps = 0
  for e in range(MAX_EPISODES):
    state = env.reset()
    cum_reward = 0
    ep_ave_max_q = 0
    tot_loss = 0
    for j in range(MAX_STEPS):
      if FLAGS.render:
        env.render()
      action = actor.predict([state])[0] 
        #+ actor_noise()
      next_state, reward, done, _ = env.step(action)
      cum_reward += reward
      tot_rewards.append(reward)
      # memory_buffer.append((state, action, reward, next_state, 1.0 if not done else 0.0))
      memory.append(state, action, reward, done)
      numsteps += 1
      if numsteps > BATCH_SIZE:
        # indices = np.random.choice(len(memory_buffer), BATCH_SIZE, replace=False)
        # indices = range(64)
        states, actions, rewards, next_states, notdones = memory.sample_and_split(BATCH_SIZE)

        rewards, notdones = [np.squeeze(x) for x in [rewards, notdones]]
        next_actions = actor.predict_target(next_states)
        qs, qloss, _ = critic.train(states=states, 
          actions=actions, 
          rewards=rewards,
          next_states=next_states,
          next_actions=next_actions,
          notdones=notdones
          )
        # print target_net_qs
        # print qs
        # print np.mean(np.square(target_qs-qs)) - qloss
        # print qloss
        ep_ave_max_q += np.amax(qs)
        tot_loss += qloss
        predicted_actions = actor.predict(states)
        action_gradients = critic.get_action_gradients(states, predicted_actions)
        actor.train(states=states, action_gradients=action_gradients)

        # update targets
        actor.update_target()
        critic.update_target()

      if done:
        # train agent
        # print the score and break out of the loop
        episode_history.append(cum_reward)
        print("episode: {}/{}, score: {}, avg score for 100 runs: {:.2f}, maxQ: {:.2f}, avg loss: {:.5f}".format(e, MAX_EPISODES, cum_reward, np.mean(episode_history), ep_ave_max_q / float(j), tot_loss / float(j)))
        break
      state = next_state
    if e%100 == 0 and not FLAGS.dont_save:
      save_path = saver.save(sess, FLAGS.save_checkpoint_dir + 'model-' + str(e) + '.ckpt')
      print 'saved model ' + save_path


if __name__ == '__main__':
  tf.app.run()