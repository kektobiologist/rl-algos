import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import gym_soccer
import tflearn

from collections import deque
from policy_gradient.ddpg import Actor, Critic, OrnsteinUhlenbeckActionNoise
from policy_gradient.memory import SequentialMemory

env = gym.make('SoccerEmptyGoal-v0')

BATCH_SIZE = 128

tf.app.flags.DEFINE_string('checkpoint',  '', 'load a checkpoint file for model')
tf.app.flags.DEFINE_string('save_checkpoint_dir', './models/ddpg_soccer/', 'dir for storing checkpoints')
tf.app.flags.DEFINE_boolean('dont_save', False, 'whether to save checkpoints')
tf.app.flags.DEFINE_boolean('render', False, 'render of not')
tf.app.flags.DEFINE_boolean('train', True, 'if false, don\'t train')
FLAGS = tf.app.flags.FLAGS


np.random.seed(0)
tf.set_random_seed(0)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session()

actor_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
critic_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

action_shape = 2 #  move speed, move angle, turn angle
observation_shape = 59 # ball ang (2), ball dist (1)

ACTION_SCALE_MAX = [ 100., 180.]
ACTION_SCALE_MIN = [ 0., -180.]
ACTION_SCALE_VALID = [ True, True]
def actor_network(states):
  with tf.variable_scope('actor'):
    net = slim.stack(states, slim.fully_connected, [1024, 512, 256, 128], activation_fn=tf.nn.relu, scope='stack1')
    # first 2 ouputs as linear (for logprob later), rest 5 as tanh
    # no, use tanh for all
    # net1 = slim.fully_connected(net, 2, activation_fn=None, scope='net1full')
    # net2 = slim.fully_connected(net, 5, activation_fn=tf.nn.tanh, scope='net2full')
    # net = tf.concat([net1, net2], 1)
    # net_actiontype = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, scope='full_actiontype')
    # net_dash = slim.fully_connected(net, 1, activation_fn=None, scope='full_dash')
    # net_turn = slim.fully_connected(net, 1, activation_fn=None, scope='full_turn')
    # net_movespeed = slim.fully_connected(net, 1, activation_fn=tf.nn.sigmoid, scope='full_movespeed')
    # net_moveangle = slim.fully_connected(net, 1, activation_fn=tf.nn.tanh, scope='full_moveangle')
    net_movespeed = slim.fully_connected(net, 1, activation_fn=None, scope='full_movespeed')
    net_moveangle = slim.fully_connected(net, 1, activation_fn=None, scope='full_moveangle')
    # net_turnangle = slim.fully_connected(net, 1, activation_fn=None, scope='full_turnangle')
    net = tf.concat([  net_movespeed, net_moveangle], 1)
    # no need to scale
    # net = ACTION_SCALE_MAX * net
    return net

def critic_network(states, actions):
  with tf.variable_scope('critic'):
    # state_net = slim.stack(states, slim.fully_connected, [512], activation_fn=tf.nn.relu, scope='stack_state')
    # action_net = slim.stack(actions, slim.fully_connected, [300], activation_fn=tf.nn.relu, scope='stack_action')

    net = tf.concat([states, actions], 1)

    net = slim.stack(net, slim.fully_connected, [1024, 512, 256, 128], activation_fn=tf.nn.relu, scope='stack_all')
    net = tflearn.fully_connected(net, 1)
    net = tf.squeeze(net, axis=[1])
    return net

def getHFOAction(action):
  movespeed, moveangle = action
  # actiontype = np.random.choice(2, 1, p=[actiontype, 1-actiontype])
  # actiontype = np.squeeze(actiontype)
  hfoAction = (0 , [movespeed], [moveangle], [0.], [0.], [0.])
  # print hfoAction
  return hfoAction
  # cont = tuple(np.expand_dims(action[1:], axis=1))
  # # disc = 1 if action[0] > 0 else 0
  # # always dash
  # disc = 0
  # return (disc,) + cont

def fromHFOState(hfoState):
  return hfoState
  # ballAng = hfoState[51:53]
  # ballDist = hfoState[53]
  # ret = [ballAng[0], ballAng[1], ballDist]
  # return ret

def main(_):
  actor = Actor(actor_network, actor_optimizer, sess, observation_shape, action_shape, tau=1e-3)
  critic = Critic(critic_network, critic_optimizer, sess, observation_shape, action_shape, tau=1e-3)
  actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.array([ 50., 0.]))
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
    state = fromHFOState(env.reset())
    cum_reward = 0
    ep_ave_max_q = 0
    tot_loss = 0
    for j in range(MAX_STEPS):
      if FLAGS.render:
        env.render()
      action = actor.predict([state])[0] + actor_noise()
        # *ACTION_SCALE
      next_state, reward, done, _ = env.step(getHFOAction(action))
      next_state = fromHFOState(next_state)
      cum_reward += reward
      tot_rewards.append(reward)
      # memory_buffer.append((state, action, reward, next_state, 1.0 if not done else 0.0))
      memory.append(state, action, reward, done)
      numsteps += 1

      if numsteps > BATCH_SIZE and FLAGS.train:
        # indices = np.random.choice(len(memory_buffer), BATCH_SIZE, replace=False)
        # indices = range(64)
        # states, actions, rewards, next_states, notdones = zip(*[memory_buffer[idx] for idx in indices])
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
        # modify action gradients for bounded parameters https://arxiv.org/pdf/1511.04143.pdf
        inverted_grads = []
        for grad, action in zip(action_gradients, predicted_actions):
          # inverting gradients approach
          newgrad = []
          for delp, p, pmin, pmax, valid in zip(grad, action, ACTION_SCALE_MIN, ACTION_SCALE_MAX, ACTION_SCALE_VALID):
            if not valid:
              newgrad.append(delp)
            else:
              if delp > 0:
                newgrad.append(delp * (pmax - p) / (pmax - pmin))
              else:
                newgrad.append(delp * (p - pmin) / (pmax - pmin))
          inverted_grads.append(newgrad)

        # nope, use normal grads 
        actor.train(states=states, action_gradients=np.array(inverted_grads))
        # actor.train(states=states, action_gradients=action_gradients)

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