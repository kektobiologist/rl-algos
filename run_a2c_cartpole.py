import os, logging, gym
from baselines import logger
from baselines.common import set_global_seeds
from baselines.a2c.a2c import learn
from baselines.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy

seed = 0
set_global_seeds(seed)
MAX_EPISODES = 10000

policy_fn = LstmPolicy
env = gym.make('CartPole-v1')

learn(policy_fn, env, seed, total_timesteps=10000, lrschedule='constant')

