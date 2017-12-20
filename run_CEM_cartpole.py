import gym
import numpy as np

class CEMOptimizer:

  def __init__(self, weights_dim, batch_size=1000, deviation=10, deviation_lim=100, rho=0.1, eta=0.1, mean=None):
    self.rho = rho
    self.eta = eta
    self.weights_dim = weights_dim
    self.mean = mean if mean!=None else np.zeros(weights_dim)
    self.deviation = np.full(weights_dim, deviation)
    self.batch_size = batch_size
    self.select_num = int(batch_size * rho)
    self.deviation_lim = deviation_lim

    assert(self.select_num > 0)

  def update_weights(self, weights, rewards):
    rewards = np.array(rewards).flatten()
    weights = np.array(weights)
    sorted_idx = (-rewards).argsort()[:self.select_num]
    top_weights = weights[sorted_idx]
    top_weights = np.reshape(top_weights, (self.select_num, self.weights_dim))
    self.mean = np.sum(top_weights, axis=0) / self.select_num
    self.deviation = np.std(top_weights, axis=0)
    self.deviation[self.deviation > self.deviation_lim] = self.deviation_lim
    if(len(self.deviation)!=self.weights_dim):
      print("dim error")
      print(len(self.deviation))
      print(self.weights_dim)
      exit()


  def sample_batch_weights(self):
    return [np.random.normal(self.mean, self.deviation * (1 + self.eta)) \
        for _ in range(self.batch_size)]

  def get_weights(self):
    return self.mean



def train():

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

  opt = CEMOptimizer(19, 500, rho=0.01, eta=0.3, deviation=10, deviation_lim=20)
  env = gym.make("CartPole-v0")
  env = gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-3', force=True)
  epoch = 80
  run_times = 200

  def test():
    W = opt.get_weights()
    observation = env.reset()
    accreward = 0
    while True:
      env.render()
      action = select_action(observation, W)
      observation, reward, done, info = env.step(action)
      accreward += reward
      if done:
        print("test end with reward: {}".format(accreward))
        break

  for ep in range(epoch):
    print("start epoch {}".format(ep))
    weights = opt.sample_batch_weights()
    rewards = []
    opt.eta *= 0.99
    print("deviation mean = {}".format(np.mean(opt.deviation)))
    for b in range(opt.batch_size):
      accreward = 0
      for t in range(run_times):  
        observation = env.reset()  
        while True:
          action = select_action(observation, weights[b])
          observation, reward, done, info = env.step(action)
          reward = -20 if done and t < 199 else reward
          accreward += reward
          if done:
            break
      rewards.append(accreward)
    opt.update_weights(weights, rewards)
    test()

if __name__ == '__main__':
  train()