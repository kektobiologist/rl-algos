import gym
import numpy as np
from pyswarm import pso

env = gym.make('Pendulum-v0')

def select_action(ob, weights):
  b1 = np.reshape(weights[0], (1, 1))
  w1 = np.reshape(weights[1:4], (1, 3))
  b2 = np.reshape(weights[4:7], (3, 1))
  w2 = np.reshape(weights[7:16], (3, 3))
  w3 = np.reshape(weights[16:25], (3, 3))
  b3 = np.reshape(weights[25:], (3, 1))
  ob = np.reshape(ob, (3, 1))
  action = np.dot(w1, np.tanh(np.dot(w2, np.tanh(np.dot(w3, ob) - b3)) - b2)) - b1
  return np.tanh(action) * 2

def evaluate_weights(weights):
  # just evaluate it once
  acc_rewards = []
  for _ in range(10):
    observation = env.reset()
    cum_reward = 0
    while True:
      # env.render()
      action = select_action(observation, weights)
      observation, reward, done, _ = env.step(action)
      cum_reward += reward
      if done:
        break
    acc_rewards.append(cum_reward)
  return -np.mean(acc_rewards)

weights_dim = 3*3+3*3+3*1+3*1+3*1+1
lb = np.full(weights_dim, -10)
ub = np.full(weights_dim, 10)
# print lb, ub
# xopt, fopt = pso(func=evaluate_weights, lb=lb, ub=ub, maxiter=100, debug=True, swarmsize=100)

# print xopt, fopt
xopt = [ 4.25770395,  5.92293508,  7.35912901, -2.86909565, -5.63476377,  4.0117739,
  6.22163174,  5.77538554,  0.05525834, -3.95992838, -6.2024729,  -0.22040324,
 -4.6904221,   8.81396937, -0.9009028,  -0.20577123, -4.30760346, -0.77449623,
  6.51810038, -3.06869128,  2.8503578,   7.64745174, -8.06377678,  9.71477048,
  2.02402009, -0.83918242,  6.11991193, -7.0909533 ]

def test(weights):
  acc_rewards = []
  for _ in range(100):
    observation = env.reset()
    cum_reward = 0
    for t in range(2000):
      env.render()
      action = select_action(observation, weights)
      observation, reward, done, _ = env.step(action)
      cum_reward += reward
      if done:
        break
    acc_rewards.append(cum_reward)
  print np.mean(acc_rewards)


test(xopt)