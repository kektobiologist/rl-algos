import gym
import numpy as np
from pyswarm import pso

env = gym.make('CartPole-v1')

def select_action(ob, weights):
  b1 = np.reshape(weights[0], (1, 1))
  w1 = np.reshape(weights[1:4], (1, 3))
  b2 = np.reshape(weights[4:7], (3, 1))
  w2 = np.reshape(weights[7:19], (3, 4))
  ob = np.reshape(ob, (4, 1))

  l1 = np.tanh(np.dot(w2, ob) - b2)
  l2 = np.tanh(np.dot(w1, l1) - b1)
  action = np.squeeze(l2)
  action = 1 if action > 0 else 0
  return action

def evaluate_weights(weights):
  # just evaluate it once
  observation = env.reset()
  cum_reward = 0
  for t in range(500):
    # env.render()
    action = select_action(observation, weights)
    observation, reward, done, _ = env.step(action)
    reward = -20 if done and t < 499 else reward
    cum_reward += reward
    if done:
      break
  return -cum_reward

weights_dim = 19
lb = np.full(weights_dim, -10)
ub = np.full(weights_dim, 10)
# print lb, ub
xopt, fopt = pso(func=evaluate_weights, lb=lb, ub=ub, maxiter=10, debug=True, swarmsize=100)

print xopt, fopt

def test(weights):
  observation = env.reset()
  cum_reward = 0
  for t in range(5000):
    env.render()
    action = select_action(observation, weights)
    observation, reward, done, _ = env.step(action)
    reward = -20 if done and t < 499 else reward
    cum_reward += reward
    if done:
      break
  return cum_reward


test(xopt)